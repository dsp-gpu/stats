---
schema_version: 1
kind: use_case
id: snr_estimator_rocm
repo: stats
title: "Как рассчитать SNR для антенн на GPU"
synonyms:
  ru:
    - "оценка SNR для антенн на GPU"
    - "расчет уровня шума с использованием GPU"
    - "обработка сигналов антенн с оценкой SNR"
    - "аналоговая обработка сигналов с GPU"
    - "оценка отношения сигнал-шум для массивов антенн"
    - "вычисление SNR с использованием GPU"
    - "обработка шума в реальном времени на GPU"
    - "анализ сигналов антенн с GPU"
  en:
    - "SNR estimation for antenna arrays on GPU"
    - "noise level calculation using GPU"
    - "signal processing for antennas with SNR estimation"
    - "analog signal processing with GPU"
    - "SNR evaluation for antenna arrays"
    - "compute SNR using GPU acceleration"
    - "real-time noise processing on GPU"
    - "antenna signal analysis with GPU"
primary_class: (unknown)
primary_method: (unknown)
related_classes:
related_use_cases:
  - spectrum__moving_average_rocm__usecase__v1
  - stats__snr_estimator_benchmark__usecase__v1
  - spectrum__lch_farrow_rocm__usecase__v1
maturity: stable
language: cpp
tags: [stats, snr, gpu, antenna, signal_processing, rocm, batch_processing, cfar, noise_estimation, real_time]
ai_generated: true
human_verified: false
operator: ai
updated_at: 2026-05-06
---

# Use-case: Как рассчитать SNR для антенн на GPU

## Когда применять

Когда требуется оценка уровня шума в реальном времени с использованием GPU для обработки сигналов с антенн

## Решение

Класс — `(unknown)`, метод `(unknown)`.

```cpp
  TestPrint("[test_01] Noise only — CFAR artifact");
  auto* backend = snr_test_helpers::GetTestBackend();
  dsp::stats::StatisticsProcessor proc(backend);

  const uint32_t n_ant = 1, n_samp = 5000;
  auto data = snr_test_helpers::MakeNoise(n_samp, /*noise_power=*/1.0f, /*seed=*/42u);

  dsp::stats::SnrEstimationConfig cfg;  // defaults: target_n_fft=0→2048, Hann, guard=5, ref=16

  auto result = proc.ComputeSnrDb(data, n_ant, n_samp, cfg);

  // H0 артефакт: E[SNR_fft_db | noise] ≈ 10·log10(ln(N_fft) + γ) ≈ 8-10 dB.
  // Широкий диапазон — CFAR имеет разброс по реализациям.
  assert(result.snr_db_global > 3.0f && result.snr_db_global < 18.0f);
  assert(result.used_bins >= 1024u && result.used_bins <= 4096u);

  // BranchSelector с откалиброванными порогами: шум должен быть Low.
  dsp::stats::BranchSelector selector;
  auto branch = selector.Select(result.snr_db_global, cfg.thresholds);
  assert(branch == dsp::stats::BranchType::Low);

  TestPrint("[test_01] PASS — snr_db=" + std::to_string(result.snr_db_global));
```

## Граничные случаи

_Не определены (нет `@throws` в Doxygen primary_method)._

## Что делать дальше

- См. [spectrum__moving_average_rocm__usecase__v1](./moving_average_rocm.md)
- См. [stats__snr_estimator_benchmark__usecase__v1](./snr_estimator_benchmark.md)
- См. [spectrum__lch_farrow_rocm__usecase__v1](./lch_farrow_rocm.md)

## Ссылки

- Источник кода: `E:/DSP-GPU/stats/tests/test_snr_estimator_rocm.hpp:1`
