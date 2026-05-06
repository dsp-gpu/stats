---
schema_version: 1
kind: use_case
id: statistics_compute_all_benchmark
repo: stats
title: "Как вычислить статистики для массива данных на GPU"
synonyms:
  ru:
    - "вычисление статистик батчем"
    - "fft для массива антенн"
    - "обработка сигналов на GPU"
    - "статистика для батча данных"
    - "вычисление характеристик массива"
    - "аналитика сигналов на GPU"
    - "статистические вычисления батчем"
    - "обработка данных с GPU"
  en:
    - "compute statistics batch"
    - "fft for antenna array"
    - "signal processing on gpu"
    - "statistics for batch data"
    - "calculate array characteristics"
    - "signal analytics on gpu"
    - "statistical computations batch"
    - "data processing with gpu"
primary_class: (unknown)
primary_method: (unknown)
related_classes:
related_use_cases:
  - stats__helpers_rocm__usecase__v1
  - spectrum__lch_farrow_rocm__usecase__v1
  - spectrum__lch_farrow_benchmark_rocm__usecase__v1
maturity: stable
language: cpp
tags: [stats, rocml, fft, batch, antenna_array, gpu_computing, statistics, signal_processing, benchmark]
ai_generated: true
human_verified: false
operator: ai
updated_at: 2026-05-06
---

# Use-case: Как вычислить статистики для массива данных на GPU

## Когда применять

Когда требуется выполнять вычисления статистических характеристик для больших массивов данных с использованием GPU, особенно в контексте обработки сигналов и антенных массивов.

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

    statistics::StatisticsParams params;
    params.beam_count = beam_count;
    params.n_point    = n_point;
// ... (truncated)
```

## Граничные случаи

_Не определены (нет `@throws` в Doxygen primary_method)._

## Что делать дальше

- См. [stats__helpers_rocm__usecase__v1](./helpers_rocm.md)
- См. [spectrum__lch_farrow_rocm__usecase__v1](./lch_farrow_rocm.md)
- См. [spectrum__lch_farrow_benchmark_rocm__usecase__v1](./lch_farrow_benchmark_rocm.md)

## Ссылки

- Источник кода: `E:/DSP-GPU/stats/tests/test_statistics_compute_all_benchmark.hpp:1`
