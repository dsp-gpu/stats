---
schema_version: 1
kind: use_case
id: statistics_rocm
repo: stats
title: "Statistics Rocm"
synonyms:
  ru:
    - []
  en:
    - []
primary_class: (unknown)
primary_method: (unknown)
related_classes:
related_use_cases:
  - stats__statistics_float_rocm__usecase__v1
  - spectrum__moving_average_rocm__usecase__v1
  - core__rocm_backend__usecase__v1
maturity: stable
language: cpp
tags: []
ai_generated: false
human_verified: false
operator: alex
updated_at: 2026-05-13
---

# Use-case: Statistics Rocm

## Когда применять

_LLM-fallback: см. описание класса._

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

- См. [stats__statistics_float_rocm__usecase__v1](./statistics_float_rocm.md)
- См. [spectrum__moving_average_rocm__usecase__v1](./moving_average_rocm.md)
- См. [core__rocm_backend__usecase__v1](./rocm_backend.md)

## Ссылки

- Источник кода: `/home/alex/DSP-GPU/stats/tests/test_statistics_rocm.hpp:1`
