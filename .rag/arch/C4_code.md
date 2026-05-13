---
schema_version: 1
repo: stats
arch_level: c4
tags:
  - "#level:c4"
  - "#repo:stats"
  - "#layer:compute"
  - "#pattern:Pipeline:StatisticsProcessor"
description: "C4 Code — реальные классы с паттернами GoF/SOLID для репо stats."
---

# C4 Code — `stats`

## Классы с паттернами проектирования

| Класс | Паттерн | Brief |
|-------|---------|-------|
| `StatisticsProcessor` | **Pipeline** |  |

## HIP-ядра (`kernels/rocm/`)

*kernels/rocm/ пуст или отсутствует.*

## Все key_classes (FQN список)

- `dsp::stats::StatisticsProcessor` (32 методов)
- `dsp::stats::SnrEstimatorOp` (8 методов)
- `dsp::stats::BranchSelector` (4 методов)
- `dsp::stats::MedianRadixSortOp` (7 методов)
- `dsp::stats::MedianHistogramComplexOp` (4 методов)
- `dsp::stats::MedianHistogramOp` (4 методов)
- `dsp::stats::MeanReductionOp` (2 методов)
- `dsp::stats::WelfordFloatOp` (1 методов)
- `dsp::stats::WelfordFusedOp` (1 методов)
- `dsp::stats::SnrEstimationConfig` (1 методов)
