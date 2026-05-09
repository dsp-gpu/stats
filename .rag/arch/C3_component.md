---
schema_version: 1
repo: stats
arch_level: c3
tags:
  - "#level:c3"
  - "#repo:stats"
  - "#layer:compute"
  - "#namespace:statistics"
description: "C3 Component — key classes и интерфейсы репо stats."
---

# C3 Component — `stats`

## Key classes (top-10 по test_params)

### `statistics::StatisticsProcessor`

- **Namespace:** `statistics`
- **Методы:** 32, **test_params rows:** 36
- **Brief:** *(описание не задано)*

### `statistics::SnrEstimatorOp`

- **Namespace:** `statistics`
- **Методы:** 8, **test_params rows:** 6
- **Brief:** — Layer 5 Op for full SNR CFAR pipeline. Lifecycle: 1. op.SetupFft(backend) — создать FFTProcessorROCm (один раз) 2. op.Initialize(ctx)

### `statistics::BranchSelector`

- **Namespace:** `statistics`
- **Методы:** 4, **test_params rows:** 5
- **Brief:** branch selector with hysteresis. Применяется после `StatisticsProcessor::ComputeSnrDb()`. Хранит текущую ветку между вызовами. Переход происходит только когда SNR п

### `statistics::MedianRadixSortOp`

- **Namespace:** `statistics`
- **Методы:** 7, **test_params rows:** 4
- **Brief:** *(описание не задано)*

### `statistics::MedianHistogramComplexOp`

- **Namespace:** `statistics`
- **Методы:** 4, **test_params rows:** 3
- **Brief:** *(описание не задано)*

### `statistics::MedianHistogramOp`

- **Namespace:** `statistics`
- **Методы:** 4, **test_params rows:** 3
- **Brief:** *(описание не задано)*

### `statistics::MeanReductionOp`

- **Namespace:** `statistics`
- **Методы:** 2, **test_params rows:** 3
- **Brief:** *(описание не задано)*

### `statistics::WelfordFloatOp`

- **Namespace:** `statistics`
- **Методы:** 1, **test_params rows:** 3
- **Brief:** *(описание не задано)*

### `statistics::WelfordFusedOp`

- **Namespace:** `statistics`
- **Методы:** 1, **test_params rows:** 3
- **Brief:** *(описание не задано)*

### `statistics::SnrEstimationConfig`

- **Namespace:** `statistics`
- **Методы:** 1, **test_params rows:** 1
- **Brief:** для SNR-estimator. Все поля с `= 0` обозначают auto-режим: - target_n_fft = 0 → snr_defaults::kTargetNFft (2048) - step_samples = 0 → ceil(n_samples / targe

