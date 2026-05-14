---
schema_version: 1
repo: stats
class_fqn: dsp::stats::StatisticsProcessor
file: /home/alex/DSP-GPU/stats/include/dsp/stats/statistics_processor.hpp
line: 100
brief: "/**  * @class StatisticsProcessor  * @brief Layer 6 Ref03 Facade: per-beam статистика (mean / median / Welford / SNR) на ROCm.  *  * @note Move-only (copy=delete, move noexcept). Owns GpuContext, Op'ы"
methods_total: 16
methods_with_doxygen: 16
ai_generated: false
human_verified: false
parser_version: 1
---

# `dsp::stats::StatisticsProcessor` — карточка класса

> **Этот файл генерируется автоматически** командой `dsp-asst rag cards build --repo stats --class StatisticsProcessor`.
> Не править руками — правки потеряются при следующем refresh.
> Источник правды — Doxygen-теги в `.hpp` + секции в `Doc/*.md`.

---

## Описание класса

<!-- rag-block: id=stats__statistics_processor__class_overview__v1 -->

/**
 * @class StatisticsProcessor
 * @brief Layer 6 Ref03 Facade: per-beam статистика (mean / median / Welford / SNR) на ROCm.
 *
 * @note Move-only (copy=delete, move noexcept). Owns GpuContext, Op'ы, FFTProcessorROCm.
 * @note Требует #if ENABLE_ROCM. Backend* — non-owning (передаётся снаружи).
 * @note PUBLIC API НЕ меняется — Python bindings (py_statistics.hpp) стабильны.
 * @see ::dsp::stats::BranchSelector — stateful классификатор Low/Mid/High по SNR.
 * @see ::dsp::stats::shared_buf — слоты GpuContext для этого модуля.
 * @ingroup grp_statistics
 */

<!-- /rag-block -->

## Связанные секции из Doc/

- `stats__meta__claude_card__v1` (meta_claude): <!-- type:meta_claude repo:stats source:stats/CLAUDE.md -->  # stats — Repository Card  _Источник: `stats/CLAUDE.md`_  # 🤖 CLAUDE — `stats`  > Статистика на GPU: mean, std, variance, median, histogram…
- `stats__api__statisticsprocessor_001__v1` (statisticsprocessor): ### `StatisticsProcessor`  ```cpp // stats/include/stats/statistics_processor.hpp // Требует: #if ENABLE_ROCM  namespace dsp::stats {  class StatisticsProcessor { public:     // Конструктор (non-ownin…
- `stats__gpu__api_002__v1` (api): ### Python API  ```python import sys sys.path.insert(0, './DSP/Python/lib') import dsp_stats import numpy as np  # 1. Контекст и процессор (ROCmGPUContext — не GPUContext!) ctx = dsp_stats.ROCmGPUCont…
- `stats__gpu__section_005__v1` (section): ## Файловое дерево модуля  ``` stats/ ├── CMakeLists.txt                          # ROCm-only; rocprim REQUIRED ├── include/ │   ├── statistics_processor.hpp            # StatisticsProcessor (весь пуб…
- `stats__gpu__section_006__v1` (section): ## Важные нюансы  1. **ROCm-only**: весь модуль обёрнут в `#if ENABLE_ROCM`. На Windows компиляция не выполняется. Сборка только с `-DENABLE_ROCM=ON` (Linux + AMD GPU + ROCm SDK).  2. **Population var…

## Public-методы (16)

## Method 1: `ComputeMean`

**Сигнатура** (`statistics_processor.hpp:137`):
```cpp
std::vector<MeanResult> ComputeMean( const std::vector<std::complex<float>>& data, const StatisticsParams& params)
```

**Параметры**:
- `data` — `const std::vector<std::complex<float>>&`
- `params` — `const StatisticsParams&`

**Возвращает**: `std::vector<MeanResult>`

**Doxygen-источник**:
```cpp
/**
   * @brief Complex mean per beam из CPU-данных. H2D → MeanReductionOp → D2H.
   *
   * @param data CPU complex<float> [beam_count × n_point] interleaved beams.
   *   @test { size=[100..1300000], value=6000, unit="elements", error_values=[-1, 3000000, 3.14] }
   * @param params Параметры обработки (beam_count, n_point, memory_limit).
   *   @test_ref StatisticsParams
   *
   * @return Массив [beam_count] MeanResult с complex mean per beam.
   *   @test_check result.size() == params.beam_count
   */
```

## Method 2: `ComputeMedian`

**Сигнатура** (`statistics_processor.hpp:152`):
```cpp
std::vector<MedianResult> ComputeMedian( const std::vector<std::complex<float>>& data, const StatisticsParams& params)
```

**Параметры**:
- `data` — `const std::vector<std::complex<float>>&`
- `params` — `const StatisticsParams&`

**Возвращает**: `std::vector<MedianResult>`

**Doxygen-источник**:
```cpp
/**
   * @brief Median(|z|) per beam из CPU-данных. Стратегия выбирается по n_point (kHistogramThreshold=100K).
   *
   * @param data CPU complex<float> [beam_count × n_point] interleaved beams.
   *   @test { size=[100..1300000], value=6000, unit="elements", error_values=[-1, 3000000, 3.14] }
   * @param params Параметры обработки (beam_count, n_point, memory_limit).
   *   @test_ref StatisticsParams
   *
   * @return Массив [beam_count] MedianResult с median(|z|) per beam.
   *   @test_check result.size() == params.beam_count
   */
```

## Method 3: `ComputeStatistics`

**Сигнатура** (`statistics_processor.hpp:167`):
```cpp
std::vector<StatisticsResult> ComputeStatistics( const std::vector<std::complex<float>>& data, const StatisticsParams& params)
```

**Параметры**:
- `data` — `const std::vector<std::complex<float>>&`
- `params` — `const StatisticsParams&`

**Возвращает**: `std::vector<StatisticsResult>`

**Doxygen-источник**:
```cpp
/**
   * @brief Welford mean+variance+std per beam из CPU-данных (single-pass через WelfordFusedOp).
   *
   * @param data CPU complex<float> [beam_count × n_point] interleaved beams.
   *   @test { size=[100..1300000], value=6000, unit="elements", error_values=[-1, 3000000, 3.14] }
   * @param params Параметры обработки (beam_count, n_point, memory_limit).
   *   @test_ref StatisticsParams
   *
   * @return Массив [beam_count] StatisticsResult: complex mean + var(|z|) + std(|z|) + mean(|z|).
   *   @test_check result.size() == params.beam_count
   */
```

## Method 4: `ComputeMean`

**Сигнатура** (`statistics_processor.hpp:186`):
```cpp
std::vector<MeanResult> ComputeMean( void* gpu_data, const StatisticsParams& params)
```

**Параметры**:
- `gpu_data` — `void*` *(pointer)* *(void\*)*
- `params` — `const StatisticsParams&`

**Возвращает**: `std::vector<MeanResult>`

**Doxygen-источник**:
```cpp
/**
   * @brief Complex mean per beam из GPU-данных (D2D → MeanReductionOp → D2H), без H2D upload.
   *
   * @param gpu_data GPU complex<float>* [beam_count × n_point] interleaved beams.
   *   @test { pattern=gpu_pointer, values=["valid_alloc", nullptr], error_values=[0xDEADBEEF, null] }
   * @param params Параметры обработки (beam_count, n_point, memory_limit).
   *   @test_ref StatisticsParams
   *
   * @return Массив [beam_count] MeanResult с complex mean per beam.
   *   @test_check result.size() == params.beam_count
   */
```

## Method 5: `ComputeMedian`

**Сигнатура** (`statistics_processor.hpp:201`):
```cpp
std::vector<MedianResult> ComputeMedian( void* gpu_data, const StatisticsParams& params)
```

**Параметры**:
- `gpu_data` — `void*` *(pointer)* *(void\*)*
- `params` — `const StatisticsParams&`

**Возвращает**: `std::vector<MedianResult>`

**Doxygen-источник**:
```cpp
/**
   * @brief Median(|z|) per beam из GPU-данных (без H2D). Стратегия выбирается по n_point.
   *
   * @param gpu_data GPU complex<float>* [beam_count × n_point] interleaved beams.
   *   @test { pattern=gpu_pointer, values=["valid_alloc", nullptr], error_values=[0xDEADBEEF, null] }
   * @param params Параметры обработки (beam_count, n_point, memory_limit).
   *   @test_ref StatisticsParams
   *
   * @return Массив [beam_count] MedianResult с median(|z|) per beam.
   *   @test_check result.size() == params.beam_count
   */
```

## Method 6: `ComputeStatistics`

**Сигнатура** (`statistics_processor.hpp:216`):
```cpp
std::vector<StatisticsResult> ComputeStatistics( void* gpu_data, const StatisticsParams& params)
```

**Параметры**:
- `gpu_data` — `void*` *(pointer)* *(void\*)*
- `params` — `const StatisticsParams&`

**Возвращает**: `std::vector<StatisticsResult>`

**Doxygen-источник**:
```cpp
/**
   * @brief Welford mean+variance+std per beam из GPU-данных (без H2D), single-pass complex.
   *
   * @param gpu_data GPU complex<float>* [beam_count × n_point] interleaved beams.
   *   @test { pattern=gpu_pointer, values=["valid_alloc", nullptr], error_values=[0xDEADBEEF, null] }
   * @param params Параметры обработки (beam_count, n_point, memory_limit).
   *   @test_ref StatisticsParams
   *
   * @return Массив [beam_count] StatisticsResult: complex mean + var(|z|) + std(|z|) + mean(|z|).
   *   @test_check result.size() == params.beam_count
   */
```

## Method 7: `ComputeAll`

**Сигнатура** (`statistics_processor.hpp:239`):
```cpp
std::vector<FullStatisticsResult> ComputeAll( const std::vector<std::complex<float>>& data, const StatisticsParams& params, StatisticsROCmProfEvents* prof_events = nullptr)
```

**Параметры**:
- `data` — `const std::vector<std::complex<float>>&`
- `params` — `const StatisticsParams&`
- `prof_events` — `StatisticsROCmProfEvents*` *(pointer)*

**Возвращает**: `std::vector<FullStatisticsResult>`

**Doxygen-источник**:
```cpp
/**
   * @brief Welford + Median за один H2D upload из CPU-данных. Возвращает FullStatisticsResult.
   *
   * @param data CPU complex<float> [beam_count × n_point] interleaved beams.
   *   @test { size=[100..1300000], value=6000, unit="elements", error_values=[-1, 3000000, 3.14] }
   * @param params Параметры обработки (beam_count, n_point, memory_limit).
   *   @test_ref StatisticsParams
   * @param prof_events Сборщик ROCm-событий профилирования (опционально).
   *   @test { values=[nullptr], error_values=[0xDEADBEEF, null] }
   *
   * @return Массив [beam_count] FullStatisticsResult: mean + var + std + median(|z|).
   *   @test_check result.size() == params.beam_count
   */
```

## Method 8: `ComputeAll`

**Сигнатура** (`statistics_processor.hpp:258`):
```cpp
std::vector<FullStatisticsResult> ComputeAll( void* gpu_data, const StatisticsParams& params, StatisticsROCmProfEvents* prof_events = nullptr)
```

**Параметры**:
- `gpu_data` — `void*` *(pointer)* *(void\*)*
- `params` — `const StatisticsParams&`
- `prof_events` — `StatisticsROCmProfEvents*` *(pointer)*

**Возвращает**: `std::vector<FullStatisticsResult>`

**Doxygen-источник**:
```cpp
/**
   * @brief Welford + Median за один D2D из GPU-данных (production-путь, без H2D).
   *
   * @param gpu_data GPU complex<float>* [beam_count × n_point] interleaved beams.
   *   @test { pattern=gpu_pointer, values=["valid_alloc", nullptr], error_values=[0xDEADBEEF, null] }
   * @param params Параметры обработки (beam_count, n_point, memory_limit).
   *   @test_ref StatisticsParams
   * @param prof_events Сборщик ROCm-событий профилирования (опционально).
   *   @test { values=[nullptr], error_values=[0xDEADBEEF, null] }
   *
   * @return Массив [beam_count] FullStatisticsResult: mean + var + std + median(|z|).
   *   @test_check result.size() == params.beam_count
   */
```

## Method 9: `ComputeAllFloat`

**Сигнатура** (`statistics_processor.hpp:278`):
```cpp
std::vector<FullStatisticsResult> ComputeAllFloat( void* gpu_float_data, const StatisticsParams& params, StatisticsROCmProfEvents* prof_events = nullptr)
```

**Параметры**:
- `gpu_float_data` — `void*` *(pointer)* *(void\*)*
- `params` — `const StatisticsParams&`
- `prof_events` — `StatisticsROCmProfEvents*` *(pointer)*

**Возвращает**: `std::vector<FullStatisticsResult>`

**Doxygen-источник**:
```cpp
/**
   * @brief WelfordFloat + Median по уже-вычисленным GPU float-магнитудам. mean всегда {0,0}.
   *
   * @param gpu_float_data GPU float* [beam_count × n_point] (магнитуды |z|, готовы).
   *   @test { pattern=gpu_pointer, values=["valid_alloc", nullptr], error_values=[0xDEADBEEF, null] }
   * @param params Параметры обработки (beam_count, n_point, memory_limit).
   *   @test_ref StatisticsParams
   * @param prof_events Сборщик ROCm-событий профилирования (опционально).
   *   @test { values=[nullptr], error_values=[0xDEADBEEF, null] }
   *
   * @return Массив [beam_count] FullStatisticsResult с mean={0,0}, остальное заполнено.
   *   @test_check result.size() == params.beam_count && result[0].mean == complex(0,0)
   */
```

## Method 10: `ComputeAllFloat`

**Сигнатура** (`statistics_processor.hpp:295`):
```cpp
std::vector<FullStatisticsResult> ComputeAllFloat( const std::vector<float>& data, const StatisticsParams& params)
```

**Параметры**:
- `data` — `const std::vector<float>&`
- `params` — `const StatisticsParams&`

**Возвращает**: `std::vector<FullStatisticsResult>`

**Doxygen-источник**:
```cpp
/**
   * @brief Convenience-обёртка: H2D upload float-магнитуд → GPU-overload ComputeAllFloat.
   *
   * @param data CPU float [beam_count × n_point] (магнитуды |z|).
   *   @test { size=[100..1300000], value=6000, unit="elements", error_values=[-1, 3000000, 3.14] }
   * @param params Параметры обработки (beam_count, n_point, memory_limit).
   *   @test_ref StatisticsParams
   *
   * @return Массив [beam_count] FullStatisticsResult с mean={0,0}, остальное заполнено.
   *   @test_check result.size() == params.beam_count
   */
```

## Method 11: `ComputeStatisticsFloat`

**Сигнатура** (`statistics_processor.hpp:314`):
```cpp
std::vector<StatisticsResult> ComputeStatisticsFloat( void* gpu_float_data, const StatisticsParams& params)
```

**Параметры**:
- `gpu_float_data` — `void*` *(pointer)* *(void\*)*
- `params` — `const StatisticsParams&`

**Возвращает**: `std::vector<StatisticsResult>`

**Doxygen-источник**:
```cpp
/**
   * @brief Welford по уже-вычисленным GPU float-магнитудам. mean всегда {0,0}.
   *
   * @param gpu_float_data GPU float* [beam_count × n_point] (магнитуды |z|, готовы).
   *   @test { pattern=gpu_pointer, values=["valid_alloc", nullptr], error_values=[0xDEADBEEF, null] }
   * @param params Параметры обработки (beam_count, n_point, memory_limit).
   *   @test_ref StatisticsParams
   *
   * @return Массив [beam_count] StatisticsResult с mean={0,0}, var/std/mean_mag заполнено.
   *   @test_check result.size() == params.beam_count
   */
```

## Method 12: `ComputeMedianFloat`

**Сигнатура** (`statistics_processor.hpp:329`):
```cpp
std::vector<MedianResult> ComputeMedianFloat( void* gpu_float_data, const StatisticsParams& params)
```

**Параметры**:
- `gpu_float_data` — `void*` *(pointer)* *(void\*)*
- `params` — `const StatisticsParams&`

**Возвращает**: `std::vector<MedianResult>`

**Doxygen-источник**:
```cpp
/**
   * @brief Median по уже-вычисленным GPU float-магнитудам (без compute_magnitudes стадии).
   *
   * @param gpu_float_data GPU float* [beam_count × n_point] (магнитуды |z|, готовы).
   *   @test { pattern=gpu_pointer, values=["valid_alloc", nullptr], error_values=[0xDEADBEEF, null] }
   * @param params Параметры обработки (beam_count, n_point, memory_limit).
   *   @test_ref StatisticsParams
   *
   * @return Массив [beam_count] MedianResult с median(|z|) per beam.
   *   @test_check result.size() == params.beam_count
   */
```

## Method 13: `ComputeStatisticsFloat`

**Сигнатура** (`statistics_processor.hpp:348`):
```cpp
std::vector<StatisticsResult> ComputeStatisticsFloat( const std::vector<float>& data, const StatisticsParams& params)
```

**Параметры**:
- `data` — `const std::vector<float>&`
- `params` — `const StatisticsParams&`

**Возвращает**: `std::vector<StatisticsResult>`

**Doxygen-источник**:
```cpp
/**
   * @brief Welford по CPU float-магнитудам (convenience: H2D upload → GPU-overload).
   *
   * @param data CPU float [beam_count × n_point] (магнитуды |z|).
   *   @test { size=[100..1300000], value=6000, unit="elements", error_values=[-1, 3000000, 3.14] }
   * @param params Параметры обработки (beam_count, n_point, memory_limit).
   *   @test_ref StatisticsParams
   *
   * @return Массив [beam_count] StatisticsResult с mean={0,0}, var/std/mean_mag заполнено.
   *   @test_check result.size() == params.beam_count
   */
```

## Method 14: `ComputeMedianFloat`

**Сигнатура** (`statistics_processor.hpp:363`):
```cpp
std::vector<MedianResult> ComputeMedianFloat( const std::vector<float>& data, const StatisticsParams& params)
```

**Параметры**:
- `data` — `const std::vector<float>&`
- `params` — `const StatisticsParams&`

**Возвращает**: `std::vector<MedianResult>`

**Doxygen-источник**:
```cpp
/**
   * @brief Median по CPU float-магнитудам (convenience: H2D upload → GPU-overload).
   *
   * @param data CPU float [beam_count × n_point] (магнитуды |z|).
   *   @test { size=[100..1300000], value=6000, unit="elements", error_values=[-1, 3000000, 3.14] }
   * @param params Параметры обработки (beam_count, n_point, memory_limit).
   *   @test_ref StatisticsParams
   *
   * @return Массив [beam_count] MedianResult с median(|z|) per beam.
   *   @test_check result.size() == params.beam_count
   */
```

## Method 15: `ComputeSnrDb`

**Сигнатура** (`statistics_processor.hpp:389`):
```cpp
SnrEstimationResult ComputeSnrDb( const std::vector<std::complex<float>>& data, uint32_t n_antennas, uint32_t n_samples, const SnrEstimationConfig& config)
```

**Параметры**:
- `data` — `const std::vector<std::complex<float>>&`
- `n_antennas` — `uint32_t`
- `n_samples` — `uint32_t`
- `config` — `const SnrEstimationConfig&`

**Возвращает**: `SnrEstimationResult`

**Doxygen-источник**:
```cpp
/**
   * @brief Вычислить SNR (dB) из CPU-данных через CA-CFAR.
   *
   * Pipeline: upload → gather → FFT(Hann)|X|² → CFAR → median.
   *
   * @param data        CPU complex<float> [n_antennas × n_samples] (row-major).
   *   @test { size=[100..1300000], value=6000, unit="elements", error_values=[-1, 3000000, 3.14] }
   * @param n_antennas  Число антенн.
   *   @test { range=[1..50000], value=128, unit="лучей/каналов", error_values=[-1, 100000, 3.14] }
   * @param n_samples   Сэмплов на антенну.
   *   @test { range=[100..1300000], value=6000, error_values=[-1, 3000000, 3.14] }
   * @param config      Конфиг SNR-estimator (см. ::dsp::stats::snr_defaults::).
   *   @test_ref SnrEstimationConfig
   * @return SnrEstimationResult с snr_db_global, used_antennas, used_bins, n_actual.
   *
   * @note Result НЕ содержит BranchType — классификация через BranchSelector.
   *   @test_check std::isfinite(result.snr_db_global) && result.used_antennas > 0 && result.used_bins > 0
   */
```

## Method 16: `ComputeSnrDb`

**Сигнатура** (`statistics_processor.hpp:411`):
```cpp
SnrEstimationResult ComputeSnrDb( void* gpu_data, uint32_t n_antennas, uint32_t n_samples, const SnrEstimationConfig& config)
```

**Параметры**:
- `gpu_data` — `void*` *(pointer)* *(void\*)*
- `n_antennas` — `uint32_t`
- `n_samples` — `uint32_t`
- `config` — `const SnrEstimationConfig&`

**Возвращает**: `SnrEstimationResult`

**Doxygen-источник**:
```cpp
/**
   * @brief Вычислить SNR (dB) из GPU-данных (production-путь).
   *
   * Pipeline: gather → FFT(Hann)|X|² → CFAR → median (данные уже на GPU).
   *
   * @param gpu_data    GPU complex<float>* [n_antennas × n_samples].
   *   @test { pattern=gpu_pointer, values=["valid_alloc", nullptr], error_values=[0xDEADBEEF, null] }
   * @param n_antennas  Число антенн.
   *   @test { range=[1..50000], value=128, unit="лучей/каналов", error_values=[-1, 100000, 3.14] }
   * @param n_samples   Сэмплов на антенну.
   *   @test { range=[100..1300000], value=6000, error_values=[-1, 3000000, 3.14] }
   * @param config      Конфиг SNR-estimator.
   *   @test_ref SnrEstimationConfig
   * @return SnrEstimationResult с snr_db_global, used_antennas, used_bins, n_actual.
   *   @test_check std::isfinite(result.snr_db_global) && result.used_antennas > 0 && result.used_bins > 0
   */
```


## Python API

**Pybind модуль**: `dsp_stats` · **Класс Python**: `StatisticsProcessor` · **Wrapper C++**: `PyStatisticsProcessor`

_Источник биндинга_: `stats/python/py_statistics.hpp`

**Конструктор**: `py::init<ROCmGPUContext&>()`

| Python | C++ | Overload |
|---|---|---|
| `compute_mean` | `PyStatisticsProcessor::compute_mean` | — |
| `compute_median` | `PyStatisticsProcessor::compute_median` | — |
| `compute_statistics` | `PyStatisticsProcessor::compute_statistics` | — |
| `compute_all` | `PyStatisticsProcessor::compute_all` | — |
| `compute_all_float` | `PyStatisticsProcessor::compute_all_float` | — |
| `compute_statistics_float` | `PyStatisticsProcessor::compute_statistics_float` | — |
| `compute_median_float` | `PyStatisticsProcessor::compute_median_float` | — |
| `compute_snr_db` | `PyStatisticsProcessor::compute_snr_db` | — |

