#pragma once

// ============================================================================
// StatisticsProcessor — главный ROCm-фасад модуля stats (Layer 6 Ref03)
//
// ЧТО:    Тонкий Facade per-beam статистики на complex<float> GPU-данных.
//         Координирует 7 Layer-5 Op'ов:
//           - MeanReductionOp           — иерархический complex mean
//           - WelfordFusedOp            — single-pass (complex → mean+var+std)
//           - WelfordFloatOp            — Welford по float-магнитудам
//           - MedianRadixSortOp         — rocPRIM segmented sort (n ≤ 100K)
//           - MedianHistogramOp         — 4-pass byte histogram (n > 100K, float)
//           - MedianHistogramComplexOp  — то же на complex (без отдельного |z|)
//           - SnrEstimatorOp            — SNR-CFAR pipeline (SNR_05/06)
//         Auto-select median strategy: kHistogramThreshold = 100'000.
//         Input layout: beam_count × n_point complex<float> (interleaved beams).
//
// ЗАЧЕМ:  Это публичный API модуля stats — Python-биндинги (py_statistics.hpp)
//         и тесты обращаются только сюда. Facade прячет 6-слойную модель
//         (GpuContext + Op'ы + kernel modules), вызывающему видны только
//         result-структуры (MeanResult, StatisticsResult, MedianResult,
//         FullStatisticsResult, SnrEstimationResult). API НЕ меняется при
//         внутренних рефакторингах — Python работает как был.
//         ComputeAll объединяет Welford + Median в один upload (уменьшает
//         H2D в 2× vs последовательных ComputeStatistics + ComputeMedian).
//
// ПОЧЕМУ: - Layer 6 Ref03 (Facade) — НЕ делает kernel-launch'и сам, делегирует
//           всё Op'ам. Op'ы — value-члены (не unique_ptr), trivially-movable.
//         - GpuContext ctx_ (Layer 1) — единая точка для compile/cache kernels
//           и shared-buffer pool (kInput, kMagnitudes, kResult, kMediansCompact,
//           SNR-slots — см. statistics_types.hpp::shared_buf).
//         - Compile lazy через EnsureCompiled() — kernel source строится
//           конкатенацией нескольких источников (statistics_kernels_rocm +
//           gather_decimated_kernel + peak_cfar_kernel) → один hiprtcCompile.
//         - SnrEstimatorOp создаёт свой FFTProcessorROCm (через SetupFft) —
//           ленивая инициализация (snr_op_initialized_ flag). FFT — отдельный
//           модуль со своим GpuContext, не конфликтует с stats ctx_.
//         - Move noexcept, без copy — facade owns kernel modules + device
//           buffers (через ctx_), копирование = chaos с lifetime.
//         - Result НЕ содержит BranchType (классификация Low/Mid/High SNR) —
//           её делает caller через `BranchSelector` (stateful + hysteresis,
//           SOLID: facade остаётся stateless).
//
// Использование:
//   statistics::StatisticsProcessor proc(rocm_backend);
//   statistics::StatisticsParams p{.beam_count=256, .n_point=4096};
//   auto stats = proc.ComputeStatistics(iq_data, p);
//   auto full  = proc.ComputeAll(iq_data, p);     // mean + var + std + median
//   // SNR (CA-CFAR pipeline):
//   statistics::SnrEstimationConfig cfg;          // defaults уже калиброваны
//   auto snr_res = proc.ComputeSnrDb(iq_data, n_ant, n_samp, cfg);
//   // Классификация ветки:
//   statistics::BranchSelector sel;
//   auto branch = sel.Select(snr_res.snr_db_global, cfg.thresholds);
//
// История:
//   - Создан:  2026-02-23 (v1, ROCm Facade монолитный)
//   - Изменён: 2026-04-09 (SNR_06: добавлены ComputeSnrDb + SnrEstimatorOp)
// ============================================================================

#if ENABLE_ROCM

#include <stats/statistics_types.hpp>
#include <core/interface/gpu_context.hpp>
#include <core/services/profiling_types.hpp>

// Op classes (Layer 5)
#include <stats/operations/mean_reduction_op.hpp>
#include <stats/operations/welford_fused_op.hpp>
#include <stats/operations/welford_float_op.hpp>
#include <stats/operations/median_radix_sort_op.hpp>
#include <stats/operations/median_histogram_op.hpp>
#include <stats/operations/median_histogram_complex_op.hpp>
#include <stats/operations/snr_estimator_op.hpp>  // SNR_05

#include <core/interface/i_backend.hpp>

#include <complex>
#include <vector>
#include <cstdint>

namespace statistics {

/// ROCm profiling events для ComputeAll-методов.
/// Vector пар (event_name, timing_data) — тот же паттерн, что HeterodyneROCmProfEvents.
using StatisticsROCmProfEvents =
    std::vector<std::pair<const char*, drv_gpu_lib::ROCmProfilingData>>;

/**
 * @class StatisticsProcessor
 * @brief Layer 6 Ref03 Facade: per-beam статистика (mean / median / Welford / SNR) на ROCm.
 *
 * @note Move-only (copy=delete, move noexcept). Owns GpuContext, Op'ы, FFTProcessorROCm.
 * @note Требует #if ENABLE_ROCM. Backend* — non-owning (передаётся снаружи).
 * @note PUBLIC API НЕ меняется — Python bindings (py_statistics.hpp) стабильны.
 * @see statistics::BranchSelector — stateful классификатор Low/Mid/High по SNR.
 * @see statistics::shared_buf — слоты GpuContext для этого модуля.
 * @ingroup grp_statistics
 */
class StatisticsProcessor {
public:
  // =========================================================================
  // Constructor / Destructor
  // =========================================================================

  /**
   * @brief Конструктор.
   * @param backend Указатель на IBackend (non-owning, обязан быть ROCm-backend).
   */
  explicit StatisticsProcessor(drv_gpu_lib::IBackend* backend);

  ~StatisticsProcessor();

  // No copying
  StatisticsProcessor(const StatisticsProcessor&) = delete;
  StatisticsProcessor& operator=(const StatisticsProcessor&) = delete;

  // Move semantics
  StatisticsProcessor(StatisticsProcessor&& other) noexcept;
  StatisticsProcessor& operator=(StatisticsProcessor&& other) noexcept;

  // =========================================================================
  // Public API -- CPU data (upload -> compute -> download)
  // =========================================================================

  /**
   * @brief Complex mean per beam из CPU-данных. H2D → MeanReductionOp → D2H.
   *
   * @param data CPU complex<float> [beam_count × n_point] interleaved beams.
   *   @test { size=[100..1300000], value=6000, unit="elements" }
   * @param params Параметры обработки (beam_count, n_point, memory_limit).
   *   @test_ref StatisticsParams
   *
   * @return Массив [beam_count] MeanResult с complex mean per beam.
   *   @test_check result.size() == params.beam_count
   */
  std::vector<MeanResult> ComputeMean(
      const std::vector<std::complex<float>>& data,
      const StatisticsParams& params);

  /**
   * @brief Median(|z|) per beam из CPU-данных. Стратегия выбирается по n_point (kHistogramThreshold=100K).
   *
   * @param data CPU complex<float> [beam_count × n_point] interleaved beams.
   *   @test { size=[100..1300000], value=6000, unit="elements" }
   * @param params Параметры обработки (beam_count, n_point, memory_limit).
   *   @test_ref StatisticsParams
   *
   * @return Массив [beam_count] MedianResult с median(|z|) per beam.
   *   @test_check result.size() == params.beam_count
   */
  std::vector<MedianResult> ComputeMedian(
      const std::vector<std::complex<float>>& data,
      const StatisticsParams& params);

  /**
   * @brief Welford mean+variance+std per beam из CPU-данных (single-pass через WelfordFusedOp).
   *
   * @param data CPU complex<float> [beam_count × n_point] interleaved beams.
   *   @test { size=[100..1300000], value=6000, unit="elements" }
   * @param params Параметры обработки (beam_count, n_point, memory_limit).
   *   @test_ref StatisticsParams
   *
   * @return Массив [beam_count] StatisticsResult: complex mean + var(|z|) + std(|z|) + mean(|z|).
   *   @test_check result.size() == params.beam_count
   */
  std::vector<StatisticsResult> ComputeStatistics(
      const std::vector<std::complex<float>>& data,
      const StatisticsParams& params);

  // =========================================================================
  // Public API -- GPU data (already on device)
  // =========================================================================

  /**
   * @brief Complex mean per beam из GPU-данных (D2D → MeanReductionOp → D2H), без H2D upload.
   *
   * @param gpu_data GPU complex<float>* [beam_count × n_point] interleaved beams.
   *   @test { pattern=gpu_pointer, values=["valid_alloc", nullptr] }
   * @param params Параметры обработки (beam_count, n_point, memory_limit).
   *   @test_ref StatisticsParams
   *
   * @return Массив [beam_count] MeanResult с complex mean per beam.
   *   @test_check result.size() == params.beam_count
   */
  std::vector<MeanResult> ComputeMean(
      void* gpu_data,
      const StatisticsParams& params);

  /**
   * @brief Median(|z|) per beam из GPU-данных (без H2D). Стратегия выбирается по n_point.
   *
   * @param gpu_data GPU complex<float>* [beam_count × n_point] interleaved beams.
   *   @test { pattern=gpu_pointer, values=["valid_alloc", nullptr] }
   * @param params Параметры обработки (beam_count, n_point, memory_limit).
   *   @test_ref StatisticsParams
   *
   * @return Массив [beam_count] MedianResult с median(|z|) per beam.
   *   @test_check result.size() == params.beam_count
   */
  std::vector<MedianResult> ComputeMedian(
      void* gpu_data,
      const StatisticsParams& params);

  /**
   * @brief Welford mean+variance+std per beam из GPU-данных (без H2D), single-pass complex.
   *
   * @param gpu_data GPU complex<float>* [beam_count × n_point] interleaved beams.
   *   @test { pattern=gpu_pointer, values=["valid_alloc", nullptr] }
   * @param params Параметры обработки (beam_count, n_point, memory_limit).
   *   @test_ref StatisticsParams
   *
   * @return Массив [beam_count] StatisticsResult: complex mean + var(|z|) + std(|z|) + mean(|z|).
   *   @test_check result.size() == params.beam_count
   */
  std::vector<StatisticsResult> ComputeStatistics(
      void* gpu_data,
      const StatisticsParams& params);

  // =========================================================================
  // Public API -- ComputeAll (Statistics + Median in one call)
  // =========================================================================

  /// CPU complex: один upload → Welford + Median → FullStatisticsResult per beam.
  /// Убирает двойной H2D vs последовательных ComputeStatistics + ComputeMedian.
  /**
   * @brief Welford + Median за один H2D upload из CPU-данных. Возвращает FullStatisticsResult.
   *
   * @param data CPU complex<float> [beam_count × n_point] interleaved beams.
   *   @test { size=[100..1300000], value=6000, unit="elements" }
   * @param params Параметры обработки (beam_count, n_point, memory_limit).
   *   @test_ref StatisticsParams
   * @param prof_events Сборщик ROCm-событий профилирования (опционально).
   *   @test { values=[nullptr] }
   *
   * @return Массив [beam_count] FullStatisticsResult: mean + var + std + median(|z|).
   *   @test_check result.size() == params.beam_count
   */
  std::vector<FullStatisticsResult> ComputeAll(
      const std::vector<std::complex<float>>& data,
      const StatisticsParams& params,
      StatisticsROCmProfEvents* prof_events = nullptr);

  /// GPU complex (production-путь): D2D один раз → Welford + Median.
  /**
   * @brief Welford + Median за один D2D из GPU-данных (production-путь, без H2D).
   *
   * @param gpu_data GPU complex<float>* [beam_count × n_point] interleaved beams.
   *   @test { pattern=gpu_pointer, values=["valid_alloc", nullptr] }
   * @param params Параметры обработки (beam_count, n_point, memory_limit).
   *   @test_ref StatisticsParams
   * @param prof_events Сборщик ROCm-событий профилирования (опционально).
   *   @test { values=[nullptr] }
   *
   * @return Массив [beam_count] FullStatisticsResult: mean + var + std + median(|z|).
   *   @test_check result.size() == params.beam_count
   */
  std::vector<FullStatisticsResult> ComputeAll(
      void* gpu_data,
      const StatisticsParams& params,
      StatisticsROCmProfEvents* prof_events = nullptr);

  /// GPU float magnitudes: kMagnitudes → WelfordFloat + Median.
  /// Note: mean field всегда {0, 0} (float-путь не имеет complex mean).
  /**
   * @brief WelfordFloat + Median по уже-вычисленным GPU float-магнитудам. mean всегда {0,0}.
   *
   * @param gpu_float_data GPU float* [beam_count × n_point] (магнитуды |z|, готовы).
   *   @test { pattern=gpu_pointer, values=["valid_alloc", nullptr] }
   * @param params Параметры обработки (beam_count, n_point, memory_limit).
   *   @test_ref StatisticsParams
   * @param prof_events Сборщик ROCm-событий профилирования (опционально).
   *   @test { values=[nullptr] }
   *
   * @return Массив [beam_count] FullStatisticsResult с mean={0,0}, остальное заполнено.
   *   @test_check result.size() == params.beam_count && result[0].mean == complex(0,0)
   */
  std::vector<FullStatisticsResult> ComputeAllFloat(
      void* gpu_float_data,
      const StatisticsParams& params,
      StatisticsROCmProfEvents* prof_events = nullptr);

  /// CPU float magnitudes: convenience-обёртка, делает upload и вызывает GPU-overload.
  /**
   * @brief Convenience-обёртка: H2D upload float-магнитуд → GPU-overload ComputeAllFloat.
   *
   * @param data CPU float [beam_count × n_point] (магнитуды |z|).
   *   @test { size=[100..1300000], value=6000, unit="elements" }
   * @param params Параметры обработки (beam_count, n_point, memory_limit).
   *   @test_ref StatisticsParams
   *
   * @return Массив [beam_count] FullStatisticsResult с mean={0,0}, остальное заполнено.
   *   @test_check result.size() == params.beam_count
   */
  std::vector<FullStatisticsResult> ComputeAllFloat(
      const std::vector<float>& data,
      const StatisticsParams& params);

  // =========================================================================
  // Public API -- GPU float data (magnitudes already computed)
  // =========================================================================

  /**
   * @brief Welford по уже-вычисленным GPU float-магнитудам. mean всегда {0,0}.
   *
   * @param gpu_float_data GPU float* [beam_count × n_point] (магнитуды |z|, готовы).
   *   @test { pattern=gpu_pointer, values=["valid_alloc", nullptr] }
   * @param params Параметры обработки (beam_count, n_point, memory_limit).
   *   @test_ref StatisticsParams
   *
   * @return Массив [beam_count] StatisticsResult с mean={0,0}, var/std/mean_mag заполнено.
   *   @test_check result.size() == params.beam_count
   */
  std::vector<StatisticsResult> ComputeStatisticsFloat(
      void* gpu_float_data,
      const StatisticsParams& params);

  /**
   * @brief Median по уже-вычисленным GPU float-магнитудам (без compute_magnitudes стадии).
   *
   * @param gpu_float_data GPU float* [beam_count × n_point] (магнитуды |z|, готовы).
   *   @test { pattern=gpu_pointer, values=["valid_alloc", nullptr] }
   * @param params Параметры обработки (beam_count, n_point, memory_limit).
   *   @test_ref StatisticsParams
   *
   * @return Массив [beam_count] MedianResult с median(|z|) per beam.
   *   @test_check result.size() == params.beam_count
   */
  std::vector<MedianResult> ComputeMedianFloat(
      void* gpu_float_data,
      const StatisticsParams& params);

  // =========================================================================
  // Public API -- CPU float data (vector<float> wrappers for tests)
  // =========================================================================

  /**
   * @brief Welford по CPU float-магнитудам (convenience: H2D upload → GPU-overload).
   *
   * @param data CPU float [beam_count × n_point] (магнитуды |z|).
   *   @test { size=[100..1300000], value=6000, unit="elements" }
   * @param params Параметры обработки (beam_count, n_point, memory_limit).
   *   @test_ref StatisticsParams
   *
   * @return Массив [beam_count] StatisticsResult с mean={0,0}, var/std/mean_mag заполнено.
   *   @test_check result.size() == params.beam_count
   */
  std::vector<StatisticsResult> ComputeStatisticsFloat(
      const std::vector<float>& data,
      const StatisticsParams& params);

  /**
   * @brief Median по CPU float-магнитудам (convenience: H2D upload → GPU-overload).
   *
   * @param data CPU float [beam_count × n_point] (магнитуды |z|).
   *   @test { size=[100..1300000], value=6000, unit="elements" }
   * @param params Параметры обработки (beam_count, n_point, memory_limit).
   *   @test_ref StatisticsParams
   *
   * @return Массив [beam_count] MedianResult с median(|z|) per beam.
   *   @test_check result.size() == params.beam_count
   */
  std::vector<MedianResult> ComputeMedianFloat(
      const std::vector<float>& data,
      const StatisticsParams& params);

  // =========================================================================
  // Public API -- SNR estimation (SNR_06)
  // =========================================================================

  /**
   * @brief Вычислить SNR (dB) из CPU-данных через CA-CFAR.
   *
   * Pipeline: upload → gather → FFT(Hann)|X|² → CFAR → median.
   *
   * @param data        CPU complex<float> [n_antennas × n_samples] (row-major).
   *   @test { size=[100..1300000], value=6000, unit="elements" }
   * @param n_antennas  Число антенн.
   *   @test { range=[1..50000], value=128, unit="лучей/каналов" }
   * @param n_samples   Сэмплов на антенну.
   *   @test { range=[100..1300000], value=6000 }
   * @param config      Конфиг SNR-estimator (см. snr_defaults::).
   *   @test_ref SnrEstimationConfig
   * @return SnrEstimationResult с snr_db_global, used_antennas, used_bins, n_actual.
   *
   * @note Result НЕ содержит BranchType — классификация через BranchSelector.
   *   @test_check std::isfinite(result.snr_db_global) && result.used_antennas > 0 && result.used_bins > 0
   */
  SnrEstimationResult ComputeSnrDb(
      const std::vector<std::complex<float>>& data,
      uint32_t n_antennas,
      uint32_t n_samples,
      const SnrEstimationConfig& config);

  /**
   * @brief Вычислить SNR (dB) из GPU-данных (production-путь).
   *
   * Pipeline: gather → FFT(Hann)|X|² → CFAR → median (данные уже на GPU).
   *
   * @param gpu_data    GPU complex<float>* [n_antennas × n_samples].
   *   @test { pattern=gpu_pointer, values=["valid_alloc", nullptr] }
   * @param n_antennas  Число антенн.
   *   @test { range=[1..50000], value=128, unit="лучей/каналов" }
   * @param n_samples   Сэмплов на антенну.
   *   @test { range=[100..1300000], value=6000 }
   * @param config      Конфиг SNR-estimator.
   *   @test_ref SnrEstimationConfig
   * @return SnrEstimationResult с snr_db_global, used_antennas, used_bins, n_actual.
   *   @test_check std::isfinite(result.snr_db_global) && result.used_antennas > 0 && result.used_bins > 0
   */
  SnrEstimationResult ComputeSnrDb(
      void* gpu_data,
      uint32_t n_antennas,
      uint32_t n_samples,
      const SnrEstimationConfig& config);

private:
  // =========================================================================
  // Ref03: GpuContext + Op instances
  // =========================================================================

  /// Ensure kernels are compiled (lazy, one-time)
  void EnsureCompiled();

  /// Upload CPU complex data to shared kInput buffer
  void UploadComplexData(const std::complex<float>* data, size_t count);

  /// Copy GPU complex data to shared kInput buffer (D2D)
  void CopyComplexGpuData(void* src, size_t count);

  /// Copy GPU float data to shared kMagnitudes buffer (D2D)
  void CopyFloatGpuData(void* src, size_t count);

  /// Upload CPU float data to shared kMagnitudes buffer (H2D async, reuses buffer)
  void UploadFloatData(const float* data, size_t count);

  /// Read MeanResult from kResult buffer
  std::vector<MeanResult> ReadMeanResults(uint32_t beam_count);

  /// Read StatisticsResult from kResult buffer
  std::vector<StatisticsResult> ReadStatisticsResults(uint32_t beam_count);

  /// Read MedianResult from kMediansCompact buffer
  std::vector<MedianResult> ReadMedianResults(uint32_t beam_count);

  /// Merge StatisticsResult + MedianResult into FullStatisticsResult (1-to-1 by index)
  std::vector<FullStatisticsResult> MergeResults(
      const std::vector<StatisticsResult>& stats,
      const std::vector<MedianResult>& medians);

  // ── Members ───────────────────────────────────────────────────────────

  drv_gpu_lib::IBackend* backend_ = nullptr;  ///< сохранён из конструктора (нужен для SnrEstimatorOp::SetupFft)
  drv_gpu_lib::GpuContext ctx_;     ///< Per-module context (stream, kernels, shared bufs)

  // Op instances (Layer 5) — member variables, not unique_ptr
  MeanReductionOp         mean_op_;
  WelfordFusedOp          welford_fused_op_;
  WelfordFloatOp          welford_float_op_;
  MedianRadixSortOp       median_sort_op_;
  MedianHistogramOp       median_hist_op_;
  MedianHistogramComplexOp median_hist_complex_op_;

  // SNR-estimator (SNR_05/06) — ленивая инициализация
  SnrEstimatorOp snr_estimator_op_;
  bool snr_op_initialized_ = false;

  bool compiled_ = false;

  /// Auto-select threshold: n_point > kHistogramThreshold → histogram, else radix sort
  static constexpr size_t kHistogramThreshold = 100'000;
};

}  // namespace statistics

#endif  // ENABLE_ROCM
