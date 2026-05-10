# Архитектурные паттерны репо `stats`

> **Источник истины:** `stats/.rag/_RAG.md` (теги `#pattern:Type:Class`, auto-inferred RAG_CLAUDE_C4 от 9.05).
> Brief'ы — из `key_classes:` того же манифеста (fallback из `rag_dsp.symbols`).
>
> Используется как источник для `dataset_v4` (collect_doc_deep подхватит Doc/Patterns.md).
> Alex: проверить + добавить руками то что не размечено в `_RAG.md tags:`.

## Facade

> Тонкий публичный API над набором операций. Стабильный → Python-биндинги не ломаются.


- **`statistics::StatisticsProcessor`** — `stats/include/stats/statistics_processor.hpp:61`
  - Facade репо `stats`: mean / std / variance (Welford online), median (histogram / radix-sort), SNR-estimator. Pipeline на rocPRIM. Стабильный публичный API → Python-биндинги не ломаются.

## Pipeline

> Композиция операций в цепочку. Конфиг → Pipeline объект.


- **`statistics::StatisticsProcessor`** — `stats/include/stats/statistics_processor.hpp:61`
  - Facade репо `stats`: mean / std / variance (Welford online), median (histogram / radix-sort), SNR-estimator. Pipeline на rocPRIM. Стабильный публичный API → Python-биндинги не ломаются.

## Adapter

> Тонкая pybind-обёртка над C++ Facade: адаптирует API под Python (numpy↔GPU, GIL release).


- **`PyStatisticsProcessor`** — `stats/python/py_statistics.hpp:31`
  - Pybind-Adapter над `statistics::StatisticsProcessor`: numpy↔GPU, GIL-release в `Compute*`. Beam-major flat layout `[beam_count × n_point]`, exposes mean/std/variance/median/SNR.

## Operation

> Атомарная GPU-операция Ref03 Layer 5: `Initialize() / IsReady() / Release()`.


- **`statistics::MeanReductionOp`** — `stats/include/stats/operations/mean_reduction_op.hpp:31`
  - Concrete Op (наследник GpuKernelOp): два-фазная reduce-сумма complex<float> по beam'у с делением на n. Phase 1 — block-level partial sums (2D grid: blocks_per_beam × beam_count, double-load), Phase 2 — финальная reduce одного блока на beam 
- **`statistics::MedianHistogramOp`** — `stats/include/stats/operations/median_histogram_op.hpp:37`
  - Concrete Op (наследник GpuKernelOp): считает median(|z|) per beam через 4 прохода byte-wise гистограммы по IEEE-754 float-битам. На каждом проходе: histogram (256 bins) → find_median_bucket сужает диапазон по одному байту (MSB → LSB). После
- **`statistics::MedianHistogramComplexOp`** — `stats/include/stats/operations/median_histogram_complex_op.hpp:37`
  - Concrete Op (наследник GpuKernelOp): то же что MedianHistogramOp, но kernel `histogram_median_pass_complex` считает |z|=√(re²+im²) on-the-fly внутри гистограммы. Промежуточный buffer магнитуд (beam_count × n_point × float) НЕ выделяется.
- **`statistics::MedianRadixSortOp`** — `stats/include/stats/operations/median_radix_sort_op.hpp:37`
  - Concrete Op: универсальная медиана через полную segmented sort всех элементов. Two execution paths: - Execute(beam, n_point) — complex input: magnitudes + sort + extract - ExecuteFloat(beam, n_point) — float magnitudes готовы: только sort +
- **`statistics::SnrEstimatorOp`** — `stats/include/stats/operations/snr_estimator_op.hpp:52`
  - SnrEstimatorOp — Layer 5 Op for full SNR CFAR pipeline.
- **`statistics::WelfordFloatOp`** — `stats/include/stats/operations/welford_float_op.hpp:34`
  - Concrete Op (наследник GpuKernelOp): считает mean(|z|), variance(|z|), std(|z|) по float-входу (магнитуды уже вычислены другим этапом). 3 LDS-массива × (kBlockSize+1) floats: mean_mag, M2, count. Результат — те же 5 floats per beam, что у W
- **`statistics::WelfordFusedOp`** — `stats/include/stats/operations/welford_fused_op.hpp:34`
  - Concrete Op (наследник GpuKernelOp): за ОДИН проход по входным данным считает sum_re, sum_im, mean(|z|), variance(|z|), std(|z|). Один блок на beam, kBlockSize=256, 5 LDS-массивов с +1 padding. Результат — 5 floats per beam в kResult: mean_


## См. также

- `stats/.rag/arch/C2_container.md`
- `stats/.rag/arch/C3_component.md`
- `stats/.rag/arch/C4_code.md`
- `MemoryBank/.architecture/DSP-GPU_Design_C4_Full.md`

---

*Сгенерировано из `_RAG.md` тегов. Alex редактирует руками + коммитит.*
