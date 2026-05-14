---
schema_version: 1
kind: use_case
id: statistics_float_rocm
repo: stats
title: "Statistics Float Rocm"
synonyms:
  ru:
    - []
  en:
    - []
primary_class: (unknown)
primary_method: (unknown)
related_classes:
related_use_cases:
  - spectrum__moving_average_rocm__usecase__v1
  - spectrum__filters_rocm__usecase__v1
  - core__rocm_external_context__usecase__v1
maturity: stable
language: cpp
tags: []
ai_generated: false
human_verified: false
operator: alex
updated_at: 2026-05-13
---

# Use-case: Statistics Float Rocm

## Когда применять

_LLM-fallback: см. описание класса._

## Решение

Класс — `(unknown)`, метод `(unknown)`.

```cpp
  int gpu_id = 0;

  ROCmBackend backend;
  backend.Initialize(gpu_id);
  StatisticsProcessor stats(&backend);
  dsp::spectrum::ComplexToMagPhaseROCm mag_proc(&backend);

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

- См. [spectrum__moving_average_rocm__usecase__v1](./moving_average_rocm.md)
- См. [spectrum__filters_rocm__usecase__v1](./filters_rocm.md)
- См. [core__rocm_external_context__usecase__v1](./rocm_external_context.md)

## Ссылки

- Источник кода: `/home/alex/DSP-GPU/stats/tests/test_statistics_float_rocm.hpp:1`
