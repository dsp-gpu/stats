---
schema_version: 1
kind: use_case
id: statistics_float_rocm
repo: stats
title: "Вычисление статистик массива float на GPU"
synonyms:
  ru:
    - "вычислить статные характеристики float на GPU"
    - "расчет среднего и стандартного отклонения массива"
    - "обработка массива float с GPU"
    - "вычисление статистик для батча float"
    - "анализ массива float на GPU"
    - "расчет параметров распределения float"
    - "вычисление статистик с использованием ROCm"
    - "параллельная статистика для float"
  en:
    - "compute float statistics on GPU"
    - "calculate mean and standard deviation array"
    - "processing float array with GPU"
    - "compute statistics for float batch"
    - "analysis of float array on GPU"
    - "calculation of distribution parameters float"
    - "compute statistics using ROCm"
    - "parallel statistics for float"
primary_class: (unknown)
primary_method: (unknown)
related_classes:
related_use_cases:
  - stats__helpers_rocm__usecase__v1
  - spectrum__moving_average_rocm__usecase__v1
  - spectrum__lch_farrow_rocm__usecase__v1
maturity: stable
language: cpp
tags: [stats, rocm, float, statistics, batch_processing, gpu_computing, parallel_processing, numerical_analysis, array_processing]
ai_generated: true
human_verified: false
operator: ai
updated_at: 2026-05-06
---

# Use-case: Вычисление статистик массива float на GPU

## Когда применять

Когда требуется параллельная обработка больших массивов float с высокой точностью на GPU с использованием ROCm

## Решение

Класс — `(unknown)`, метод `(unknown)`.

```cpp
  int gpu_id = 0;

  ROCmBackend backend;
  backend.Initialize(gpu_id);
  StatisticsProcessor stats(&backend);
  fft_processor::ComplexToMagPhaseROCm mag_proc(&backend);

  TestRunner runner(&backend, "StatsFloat", gpu_id);

  // ── T1: ComputeStatisticsFloat(vector<float>) ────────────────────

  runner.test("stats_vector_float", [&]() -> TestResult {
    constexpr uint32_t N = 4096;
    std::vector<float> data(N);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.5f, 5.0f);
    for (float& x : data) x = dist(rng);

    StatisticsParams params;
    params.beam_count = 1;
    params.n_point = N;

    auto results = stats.ComputeStatisticsFloat(data, params);

    float ref_mean = refs::CpuMeanFloat(data.data(), N);
    float ref_std  = refs::CpuStdFloat(data.data(), N);

    TestResult tr{"stats_vector_float"};
    tr.add(ScalarRelError(results[0].mean_magnitude, ref_mean, 1e-2, "mean"));
    tr.add(ScalarRelError(results[0].std_dev, ref_std, 1e-2, "std"));
// ... (truncated)
```

## Граничные случаи

_Не определены (нет `@throws` в Doxygen primary_method)._

## Что делать дальше

- См. [stats__helpers_rocm__usecase__v1](./helpers_rocm.md)
- См. [spectrum__moving_average_rocm__usecase__v1](./moving_average_rocm.md)
- См. [spectrum__lch_farrow_rocm__usecase__v1](./lch_farrow_rocm.md)

## Ссылки

- Источник кода: `E:/DSP-GPU/stats/tests/test_statistics_float_rocm.hpp:1`
