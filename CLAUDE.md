# 🤖 CLAUDE — `stats`

> Статистика на GPU: mean, std, variance, median, histogram, radix sort, SNR-estimator.
> Зависит от: `core` + `spectrum` + `rocprim`. Глобальные правила → `../CLAUDE.md` + `.claude/rules/*.md`.

## 🎯 Что здесь

| Класс | Что делает |
|-------|-----------|
| `StatisticsProcessor` | Facade: mean / std / variance (Welford online) |
| `MedianHistogramOp` | Median через histogram (быстрый для uint16) |
| `MedianStrategy` | GoF Strategy: выбор histogram vs sort в зависимости от данных |
| `RadixSortOp` | Обёртка rocPRIM radix sort |
| `SNREstimator` | SNR по спектру (peak vs noise floor) |

## 📁 Структура

```
stats/
├── include/dsp/stats/
│   ├── statistics_processor.hpp
│   ├── gpu_context.hpp
│   ├── operations/    # WelfordOp, MedianHistogramOp, RadixSortOp, SNREstimatorOp
│   └── strategies/    # MedianStrategy
├── src/
├── kernels/rocm/      # welford.hip, histogram.hip
├── tests/
└── python/dsp_stats_module.cpp
```

## ⚠️ Специфика

- **Welford online** для численно-устойчивого mean/variance — не naive two-pass.
- **Median через histogram** — только для целочисленных типов с ограниченным диапазоном.
- **rocPRIM** для reduce/scan/sort — не писать свои, не использовать CUB.
- **SNR-оценка** требует FFT → зависимость от `spectrum`.

## 🚫 Запреты

- Не писать свои sort/reduce/scan — только rocPRIM.
- Не использовать CUB / thrust — это CUDA.
- Не смешивать стратегии median в одном вызове — выбор через `MedianStrategy`.

<!-- BEGIN: RAG_CLAUDE_C4 (auto) -->
## 🏗️ Архитектура (C4 — компактно)

- **C1 System Context:** репо `stats` (layer=compute). Полный C4 → `MemoryBank/.architecture/DSP-GPU_Design_C4_Full.md` §`stats`
- **C2 Container:** namespace из top key_classes (см. `.rag/_RAG.md`)
- **C3 Component:** `key_classes` в `.rag/_RAG.md` (top по test_params)
- **C4 Code:** SnrEstimatorOp · MedianRadixSortOp · BranchSelector · MedianHistogramComplexOp

## 🏷️ RAG теги


`#layer:compute` `#repo:stats` `#namespace:statistics` `#namespace:test_snr_estimator` `#namespace:test_statistics_compute_all_benchmark` `#pattern:Pipeline:StatisticsProcessor` `#pattern:Pipeline:PyStatisticsProcessor` `#pattern:Facade:StatisticsProcessor` `#pattern:Operation:MeanReductionOp` `#pattern:Operation:MedianHistogramOp` `#pattern:Operation:MedianHistogramComplexOp` `#pattern:Operation:MedianRadixSortOp` `#pattern:Operation:SnrEstimatorOp` `#pattern:Operation:WelfordFloatOp` `#pattern:Operation:WelfordFusedOp`

## 🔗 Правила (path-scoped автоматически)

- `09-rocm-only.md` — rocPRIM обязателен
- `05-architecture-ref03.md` — Strategy + Facade
- `14-cpp-style.md` + `15-cpp-testing.md`
- `11-python-bindings.md`
