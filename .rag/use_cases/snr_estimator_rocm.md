---
schema_version: 1
kind: use_case
id: snr_estimator_rocm
repo: stats
title: "Snr Estimator Rocm"
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
  - spectrum__moving_average_rocm__usecase__v1
  - spectrum__filters_benchmark_rocm__usecase__v1
maturity: stable
language: cpp
tags: []
ai_generated: false
human_verified: false
operator: alex
updated_at: 2026-05-13
---

# Use-case: Snr Estimator Rocm

## Когда применять

_LLM-fallback: см. описание класса._

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

- См. [stats__snr_estimator_benchmark__usecase__v1](./snr_estimator_benchmark.md)
- См. [spectrum__moving_average_rocm__usecase__v1](./moving_average_rocm.md)
- См. [spectrum__filters_benchmark_rocm__usecase__v1](./filters_benchmark_rocm.md)

## Ссылки

- Источник кода: `/home/alex/DSP-GPU/stats/tests/test_snr_estimator_rocm.hpp:1`
