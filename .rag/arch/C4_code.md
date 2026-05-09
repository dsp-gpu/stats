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

- `statistics::StatisticsProcessor` (32 методов)
- `statistics::SnrEstimatorOp` (8 методов)
- `statistics::BranchSelector` (4 методов)
- `statistics::MedianRadixSortOp` (7 методов)
- `statistics::MedianHistogramComplexOp` (4 методов)
- `statistics::MedianHistogramOp` (4 методов)
- `statistics::MeanReductionOp` (2 методов)
- `statistics::WelfordFloatOp` (1 методов)
- `statistics::WelfordFusedOp` (1 методов)
- `statistics::SnrEstimationConfig` (1 методов)
