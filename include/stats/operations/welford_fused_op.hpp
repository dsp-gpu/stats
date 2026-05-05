#pragma once

// ============================================================================
// WelfordFusedOp — single-pass Welford по complex-входу (Layer 5 Ref03)
//
// ЧТО:    Concrete Op (наследник GpuKernelOp): за ОДИН проход по входным
//         данным считает sum_re, sum_im, mean(|z|), variance(|z|), std(|z|).
//         Один блок на beam, kBlockSize=256, 5 LDS-массивов с +1 padding.
//         Результат — 5 floats per beam в kResult: mean_re, mean_im,
//         mean_mag, variance, std_dev.
//
// ЗАЧЕМ:  Заменяет наивный двухпроходный pipeline: «compute_magnitudes →
//         welford по float magnitudes». Один проход = вдвое меньше memory
//         transactions + не нужен промежуточный buffer на beam_count×n_point
//         floats. Главный путь для StatisticsProcessor::ComputeStatistics
//         и ComputeAll (complex input).
//
// ПОЧЕМУ: - Layer 5 Ref03 (один Op = один kernel «полная Welford-статистика»).
//         - Welford'овская формула online: M2 += (x − mean)·(x − new_mean).
//           Численно устойчива vs naive sum(x²) − (sum(x))²/n при больших n.
//         - LDS +1 padding на каждый из 5 массивов — устранение bank
//           conflicts при tree reduction (P3-B optimization).
//         - Один блок на beam (`bc` × 1 × 1 grid) — простая модель,
//           поскольку n_point ≤ 4-16K в типичных сценариях; для n_point >
//           65K стоит думать о hierarchical reduction (TODO, не блокер).
//         - BufferSet<0> — нет private buffers, статус Stateless Op.
//
// Использование:
//   statistics::WelfordFusedOp wel;
//   wel.Initialize(ctx);
//   wel.Execute(beam_count, n_point);
//   // kResult теперь содержит beam_count × {mean_re, mean_im, mean_mag, var, std}
//
// История:
//   - Создан:  2026-03-14 (Ref03 Layer 5, оптимизация P0-A: один pass)
//   - Изменён: 2026-05-01 (унификация формата шапки под dsp-asst RAG-индексер)
// ============================================================================

#if ENABLE_ROCM

#include <core/services/gpu_kernel_op.hpp>
#include <core/services/buffer_set.hpp>
#include <core/interface/gpu_context.hpp>
#include <stats/statistics_types.hpp>

#include <hip/hip_runtime.h>
#include <stdexcept>
#include <string>

namespace statistics {

/**
 * @class WelfordFusedOp
 * @brief Layer 5 Ref03 Op: single-pass Welford по complex-входу (online mean+var+std).
 *
 * @note Stateless (BufferSet<0>, нет private buffers).
 * @note Требует #if ENABLE_ROCM. Зависит от kernel `welford_fused`.
 * @note Численно устойчив (online Welford vs naive sum(x²)−(sum(x))²/n).
 * @see statistics::WelfordFloatOp — аналог по уже-вычисленным float magnitudes.
 * @see statistics::MeanReductionOp — отдельный mean (если нужно ТОЛЬКО среднее).
 */
class WelfordFusedOp : public drv_gpu_lib::GpuKernelOp {
public:
  /**
   * @brief Возвращает имя Op'а для логирования и профилирования.
   *
   * @return C-строка "WelfordFused" (статический литерал).
   *   @test_check std::string(result) == "WelfordFused"
   */
  const char* Name() const override { return "WelfordFused"; }

  /**
   * @brief Выполнить single-pass Welford по complex-входу.
   * @param beam_count Число beam'ов.
   *   @test { range=[1..50000], value=128, unit="лучей/каналов", error_values=[-1, 100000, 3.14] }
   * @param n_point    Сэмплов на beam.
   *   @test { range=[100..1300000], value=6000, error_values=[-1, 3000000, 3.14] }
   *
   * Читает kInput (complex<float>), пишет kResult (5 floats per beam:
   * mean_re, mean_im, mean_mag, variance, std_dev).
   * @throws std::runtime_error при сбое hipModuleLaunchKernel("welford_fused").
   *   @test_check throws on hipModuleLaunchKernel != hipSuccess
   */
  void Execute(size_t beam_count, size_t n_point) {
    unsigned int bc = static_cast<unsigned int>(beam_count);
    unsigned int np = static_cast<unsigned int>(n_point);

    void* input_buf  = ctx_->GetShared(shared_buf::kInput);
    void* result_buf = ctx_->RequireShared(
        shared_buf::kResult,
        beam_count * 5 * sizeof(float));

    // 5 shared arrays × (kBlockSize+1) floats: sum_re, sum_im, mean_mag, M2, count
    size_t shared_mem = 5 * (kBlockSize + 1) * sizeof(float);

    void* args[] = { &input_buf, &result_buf, &bc, &np };

    hipError_t err = hipModuleLaunchKernel(
        kernel("welford_fused"),
        bc, 1, 1,           // one block per beam
        kBlockSize, 1, 1,
        shared_mem, stream(),
        args, nullptr);

    if (err != hipSuccess) {
      throw std::runtime_error("WelfordFusedOp: " +
                                std::string(hipGetErrorString(err)));
    }
  }

private:
  static constexpr unsigned int kBlockSize = 256;
  drv_gpu_lib::BufferSet<0> bufs_;  // no private buffers
};

}  // namespace statistics

#endif  // ENABLE_ROCM
