---
schema_version: 1
kind: use_case
id: statistics_compute_all_benchmark
repo: stats
title: "Statistics Compute All Benchmark"
synonyms:
  ru:
    - []
  en:
    - []
primary_class: (unknown)
primary_method: (unknown)
related_classes:
related_use_cases:
  - stats__snr_estimator_benchmark__usecase__v1
  - spectrum__lch_farrow_benchmark_rocm__usecase__v1
  - spectrum__filters_benchmark_rocm__usecase__v1
maturity: stable
language: cpp
tags: []
ai_generated: false
human_verified: false
operator: alex
updated_at: 2026-05-13
---

# Use-case: Statistics Compute All Benchmark

## Когда применять

_LLM-fallback: см. описание класса._

## Решение

Класс — `(unknown)`, метод `(unknown)`.

```cpp
  auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();

  con.Print(0, "Stats Bench", "");
  con.Print(0, "Stats Bench", "============================================================");
  con.Print(0, "Stats Bench", "  Statistics ComputeAll Benchmark — ROCm");
  con.Print(0, "Stats Bench", "  (4 beams x 65536 complex float, CPU path)");
  con.Print(0, "Stats Bench", "============================================================");

  if (drv_gpu_lib::ROCmCore::GetAvailableDeviceCount() == 0) {
    con.Print(0, "Stats Bench", "  [SKIP] No AMD GPU available");
    return 0;
  }

  try {
    // ── Backend init ──────────────────────────────────────────────────────
    auto backend = std::make_unique<drv_gpu_lib::ROCmBackend>();
    backend->Initialize(0);

    // ── Test data ─────────────────────────────────────────────────────────
    const uint32_t beam_count = 4;
    const uint32_t n_point    = 65536;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<std::complex<float>> data(beam_count * n_point);
    for (auto& v : data) v = std::complex<float>(dist(rng), dist(rng));

    dsp::stats::StatisticsParams params;
    params.beam_count = beam_count;
    params.n_point    = n_point;
// ... (truncated)
```

## Граничные случаи

_Не определены (нет `@throws` в Doxygen primary_method)._

## Что делать дальше

- См. [stats__snr_estimator_benchmark__usecase__v1](./snr_estimator_benchmark.md)
- См. [spectrum__lch_farrow_benchmark_rocm__usecase__v1](./lch_farrow_benchmark_rocm.md)
- См. [spectrum__filters_benchmark_rocm__usecase__v1](./filters_benchmark_rocm.md)

## Ссылки

- Источник кода: `/home/alex/DSP-GPU/stats/tests/test_statistics_compute_all_benchmark.hpp:1`
