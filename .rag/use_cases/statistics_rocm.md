---
schema_version: 1
kind: use_case
id: statistics_rocm
repo: stats
title: "Вычисление статистических характеристик на GPU"
synonyms:
  ru:
    - "расчет среднего значения на GPU"
    - "обработка данных с использованием ROCm"
    - "вычисление статистики для массива"
    - "параллельная обработка данных на GPU"
    - "вычисление среднего для батча"
    - "статистика на GPU с ROCm"
    - "обработка сигналов с использованием GPU"
    - "вычисление характеристик на GPU"
  en:
    - "compute mean on GPU"
    - "statistical processing with ROCm"
    - "batch data processing on GPU"
    - "GPU-based statistical calculation"
    - "ROCm statistical computation"
    - "mean calculation for array"
    - "parallel data processing with GPU"
    - "statistical analysis on GPU"
primary_class: (unknown)
primary_method: (unknown)
related_classes:
related_use_cases:
  - stats__helpers_rocm__usecase__v1
  - spectrum__moving_average_rocm__usecase__v1
  - spectrum__lch_farrow_rocm__usecase__v1
maturity: stable
language: cpp
tags: [stats, rocm, gpu, statistics, batch, processing, computation, parallel, data]
ai_generated: true
human_verified: false
operator: ai
updated_at: 2026-05-06
---

# Use-case: Вычисление статистических характеристик на GPU

## Когда применять

Когда требуется выполнять статистические вычисления на GPU с использованием ROCm для обработки больших массивов данных

## Решение

Класс — `(unknown)`, метод `(unknown)`.

```cpp
  int gpu_id = 0;

  // Check for ROCm devices
  int device_count = ROCmCore::GetAvailableDeviceCount();
  if (device_count == 0) {
    auto& con = ConsoleOutput::GetInstance();
    con.Print(gpu_id, "Stats ROCm", "[!] No ROCm devices found -- skipping tests");
    return;
  }

  ROCmBackend backend;
  backend.Initialize(gpu_id);
  StatisticsProcessor stats(&backend);

  TestRunner runner(&backend, "Stats ROCm", gpu_id);

  // ── Test 1: ComputeMean — single beam, sinusoid ──────────────────

  runner.test("mean_single_beam", [&]() -> TestResult {
    auto data = refs::GenerateSinusoid(100.f, 1000.f, 4096);

    StatisticsParams params;
    params.beam_count = 1;
    params.n_point = 4096;

    auto results = stats.ComputeMean(data, params);
    if (results.empty()) return TestResult{"mean_single_beam"}.add(FailResult("size", 0, 1));

    auto cpu_mean = refs::CpuMean(data.data(), data.size());

// ... (truncated)
```

## Граничные случаи

_Не определены (нет `@throws` в Doxygen primary_method)._

## Что делать дальше

- См. [stats__helpers_rocm__usecase__v1](./helpers_rocm.md)
- См. [spectrum__moving_average_rocm__usecase__v1](./moving_average_rocm.md)
- См. [spectrum__lch_farrow_rocm__usecase__v1](./lch_farrow_rocm.md)

## Ссылки

- Источник кода: `E:/DSP-GPU/stats/tests/test_statistics_rocm.hpp:1`
