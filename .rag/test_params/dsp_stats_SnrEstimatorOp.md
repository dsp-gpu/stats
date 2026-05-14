---
schema_version: 1
repo: stats
class_fqn: dsp::stats::SnrEstimatorOp
file: /home/alex/DSP-GPU/stats/include/dsp/stats/operations/snr_estimator_op.hpp
line: 89
brief: "/**  * @class SnrEstimatorOp  * @brief Layer 5 Ref03 composite Op: полный SNR-CFAR pipeline (gather → FFT → CFAR → median).  *  * @note Owns FFTProcessorROCm (через unique_ptr, SetupFft до Initialize)"
methods_total: 2
methods_with_doxygen: 2
ai_generated: false
human_verified: false
parser_version: 1
---

# `dsp::stats::SnrEstimatorOp` — карточка класса

> **Этот файл генерируется автоматически** командой `dsp-asst rag cards build --repo stats --class SnrEstimatorOp`.
> Не править руками — правки потеряются при следующем refresh.
> Источник правды — Doxygen-теги в `.hpp` + секции в `Doc/*.md`.

---

## Описание класса

<!-- rag-block: id=stats__snr_estimator_op__class_overview__v1 -->

/**
 * @class SnrEstimatorOp
 * @brief Layer 5 Ref03 composite Op: полный SNR-CFAR pipeline (gather → FFT → CFAR → median).
 *
 * @note Owns FFTProcessorROCm (через unique_ptr, SetupFft до Initialize).
 * @note Требует #if ENABLE_ROCM. Зависит от kernels gather_decimated + peak_cfar.
 * @note Калибровано Python Эксп.5 (P_correct=97.9% для Hann + CA-CFAR mean).
 * @note Lifecycle: SetupFft(backend) → Initialize(ctx) → Execute(...) → Release.
 * @see ::dsp::stats::BranchSelector — классификация result.snr_db_global → Low/Mid/High.
 * @see ::dsp::spectrum::FFTProcessorROCm — внутренний FFT-фасад (свой GpuContext).
 * @see ::dsp::stats::MedianRadixSortOp — переиспользуется для медианы по антеннам.
 */

<!-- /rag-block -->

## Связанные секции из Doc/

- `stats__meta__claude_card__v1` (meta_claude): <!-- type:meta_claude repo:stats source:stats/CLAUDE.md -->  # stats — Repository Card  _Источник: `stats/CLAUDE.md`_  # 🤖 CLAUDE — `stats`  > Статистика на GPU: mean, std, variance, median, histogram…
- `stats__patterns__operation_002__v1` (operation): - **`dsp::stats::MeanReductionOp`** — `stats/include/stats/operations/mean_reduction_op.hpp:31`   - Concrete Op (наследник GpuKernelOp): два-фазная reduce-сумма complex<float> по beam'у с делением на …

## Public-методы (2)

## Method 1: `Name`

**Сигнатура** (`snr_estimator_op.hpp:97`):
```cpp
const char* Name() const override { return "SnrEstimator";
```

**Возвращает**: `char`

**Doxygen-источник**:
```cpp
/**
   * @brief Возвращает имя Op'а для логирования и профилирования.
   *
   * @return C-строка "SnrEstimator" (статический литерал).
   *   @test_check std::string(result) == "SnrEstimator"
   */
```

## Method 2: `Execute`

**Сигнатура** (`snr_estimator_op.hpp:128`):
```cpp
void Execute(void* gpu_input, uint32_t n_antennas, uint32_t n_samples, const SnrEstimationConfig& config, SnrEstimationResult& out_result) { config.Validate(); if (!gpu_input) { throw std::invalid_argument("SnrEstimatorOp::Execute: gpu_input is null"); } if (!fft_processor_) { throw std::runtime_error( "SnrEstimatorOp::Execute: SetupFft() не был вызван"); } // 1. Compute auto-parameters (0 → default from snr_defaults) const uint32_t target_n_fft = (config.target_n_fft > 0) ? config.target_n_fft : ::dsp::stats::snr_defaults::kTargetNFft; const uint32_t step_samples = (config.step_samples > 0) ? config.step_samples : CeilDiv(n_samples, target_n_fft); const uint32_t step_antennas = (config.step_antennas > 0) ? config.step_antennas : CeilDiv(n_antennas, ::dsp::stats::snr_defaults::kTargetAntennasMedian); const uint32_t n_actual = n_samples / step_samples; const uint32_t n_ant_out = CeilDiv(n_antennas, step_antennas); if (n_actual == 0 || n_ant_out == 0) { throw std::invalid_argument( "SnrEstimatorOp: degenerate sizes — n_actual=" + std::to_string(n_actual) + " n_ant_out=" + std::to_string(n_ant_out)); } // Дополнительная проверка на фактическом n_actual после децимации if (2u * (config.guard_bins + config.ref_bins) + 1u >= n_actual) { throw std::invalid_argument( "SnrEstimatorOp: ref window (2*(guard+ref)+1) >= n_actual=" + std::to_string(n_actual)); } // 2. Allocate shared buffers (slots из ::dsp::stats::shared_buf) const size_t gather_bytes = (size_t)n_ant_out * (size_t)n_actual * sizeof(float) * 2; ctx_->RequireShared(shared_buf::kGatherOutput, gather_bytes); // nFFT pre-allocation: используем NextPowerOf2(n_actual) как оценку. // Фактический nFFT берём из fft_processor_->GetNFFT() после вызова. const uint32_t n_fft_est = NextPowerOf2(n_actual); const size_t mag_bytes = (size_t)n_ant_out * (size_t)n_fft_est * sizeof(float); ctx_->RequireShared(shared_buf::kFftMagSquared, mag_bytes); ctx_->RequireShared(shared_buf::kSnrPerAntenna, (size_t)n_ant_out * sizeof(float)); // 3. Stage 1: gather_decimated kernel ExecuteGather(gpu_input, n_antennas, n_samples, step_antennas, step_samples, n_ant_out, n_actual); // 4. Stage 2: FFT → |X|² через FFTProcessorROCm (с Hann window!) ::dsp::spectrum::FFTProcessorParams fft_params; fft_params.beam_count = n_ant_out; fft_params.n_point = n_actual; fft_params.repeat_count = 1; fft_params.sample_rate = 1.0f; // не используется — мы не читаем частоты // КРИТИЧНО: window=config.window (default Hann из snr_defaults). // Калибровано Python Эксп.5 — без Hann −27 dB bias от sinc sidelobes! fft_processor_->ProcessMagnitudesToGPU( ctx_->GetShared(shared_buf::kGatherOutput), ctx_->GetShared(shared_buf::kFftMagSquared), fft_params, /*squared=*/true, config.window); const uint32_t n_fft = fft_processor_->GetNFFT(); // 5. Stage 3: peak_cfar kernel ExecutePeakCfar(n_ant_out, n_fft, config.guard_bins, config.ref_bins, config.search_full_spectrum); // 6. Stage 4: median по антеннам через MedianRadixSortOp::ExecuteFloat // median_op_ читает из shared_buf::kMagnitudes, пишет в kMediansCompact. // Копируем SNR-per-antenna → kMagnitudes (D2D, async). const size_t snr_bytes = (size_t)n_ant_out * sizeof(float); ctx_->RequireShared(shared_buf::kMagnitudes, snr_bytes); hipError_t err = hipMemcpyAsync( ctx_->GetShared(shared_buf::kMagnitudes), ctx_->GetShared(shared_buf::kSnrPerAntenna), snr_bytes, hipMemcpyDeviceToDevice, stream()); if (err != hipSuccess) { throw std::runtime_error( "SnrEstimatorOp: D2D copy snr→magnitudes failed: " + std::string(hipGetErrorString(err))); } // Один "beam" с n_ant_out сэмплами → медиана по антеннам. median_op_.ExecuteFloat(/*beam_count=*/1, /*n_point=*/n_ant_out); // 7. D2H — читаем один float (медиана SNR_db) float median_snr_db = 0.0f; err = hipMemcpyAsync( &median_snr_db, ctx_->GetShared(shared_buf::kMediansCompact), sizeof(float), hipMemcpyDeviceToHost, stream()); if (err != hipSuccess) { throw std::runtime_error( "SnrEstimatorOp: D2H median read failed: " + std::string(hipGetErrorString(err))); } hipStreamSynchronize(stream()); // 8. Populate result struct (БЕЗ BranchType — его считает BranchSelector) out_result.snr_db_global = median_snr_db; out_result.used_antennas = n_ant_out; out_result.used_bins = n_fft; out_result.actual_step_samples = step_samples; out_result.n_actual = n_actual; out_result.snr_db_per_antenna.clear(); // опционально — пока не заполняем
```

**Параметры**:
- `gpu_input` — `void*` *(pointer)* *(void\*)*
- `n_antennas` — `uint32_t`
- `n_samples` — `uint32_t`
- `config` — `const SnrEstimationConfig&`
- `out_result` — `SnrEstimationResult&`

**Doxygen-источник**:
```cpp
/**
   * @brief Выполнить полный SNR pipeline: gather → FFT|X|² → CFAR → median.
   *
   * @param gpu_input    Complex<float>* [n_antennas × n_samples] на GPU.
   *   @test { pattern=gpu_pointer, values=["valid_alloc", nullptr], error_values=[0xDEADBEEF, null] }
   * @param n_antennas   Число входных антенн.
   *   @test { range=[1..50000], value=128, unit="лучей/каналов", error_values=[-1, 100000, 3.14] }
   * @param n_samples    Сэмплов на антенну.
   *   @test { range=[100..1300000], value=6000, error_values=[-1, 3000000, 3.14] }
   * @param config       Validated SnrEstimationConfig (поля 0 → auto-defaults).
   *   @test_ref SnrEstimationConfig
   * @param out_result   Result struct (заполняется этим методом, БЕЗ BranchType).
   * @throws std::invalid_argument если gpu_input null, или вырожденные размеры,
   *         или ref window ≥ n_actual после auto-decimation.
   * @throws std::runtime_error если SetupFft не был вызван, или kernel-launch упал.
   *   @test_check throws std::invalid_argument on (gpu_input==nullptr) || (n_actual==0) || (n_ant_out==0) || (2*(guard+ref)+1 >= n_actual); throws std::runtime_error if SetupFft не вызван
   */
```

