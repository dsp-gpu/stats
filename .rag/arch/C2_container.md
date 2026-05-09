---
schema_version: 1
repo: stats
arch_level: c2
tags:
  - "#level:c2"
  - "#repo:stats"
  - "#layer:compute"
  - "#namespace:statistics"
  - "#namespace:test_snr_estimator"
  - "#namespace:test_snr_estimator_benchmark"
description: "C2 Container — namespace tree и зависимости репо stats."
---

# C2 Container — `stats` (layer=compute)

## Namespaces (top по числу классов)

- `statistics`
- `test_snr_estimator`
- `test_snr_estimator_benchmark`
- `test_statistics_compute_all_benchmark`

## Public modules (`include/stats/`)

- `kernels/`
- `operations/`

## Зависимости (depends_on)

`core`

## Используется (used_by)

`strategies`, `DSP`

## Top key_classes

| Class | Namespace | Methods | TestParams |
|-------|-----------|--------:|-----------:|
| `StatisticsProcessor` | `statistics` | 32 | 36 |
| `SnrEstimatorOp` | `statistics` | 8 | 6 |
| `BranchSelector` | `statistics` | 4 | 5 |
| `MedianRadixSortOp` | `statistics` | 7 | 4 |
| `MedianHistogramComplexOp` | `statistics` | 4 | 3 |
