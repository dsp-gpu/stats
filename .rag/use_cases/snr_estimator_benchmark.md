---
schema_version: 1
kind: use_case
id: snr_estimator_benchmark
repo: stats
title: "Snr Estimator Benchmark"
synonyms:
  ru:
    - []
  en:
    - []
primary_class: test_snr_estimator::SnrEstimatorBenchmark
primary_method: SnrEstimatorBenchmark
related_classes:
  - strategies::statistics_processor
  - strategies::all_maxima_pipeline_rocm
  - stats::snr_estimator_op
  - spectrum::fft_processor_rocm
  - stats::statistics_processor
related_use_cases:
  - core__profile_analyzer__usecase__v1
  - spectrum__filters_benchmark_rocm__usecase__v1
  - spectrum__fft_maxima_benchmark_rocm__usecase__v1
maturity: stable
language: cpp
tags: []
ai_generated: false
human_verified: false
operator: alex
updated_at: 2026-05-13
---

# Use-case: Snr Estimator Benchmark

## Когда применять

_LLM-fallback: см. описание класса._

## Решение

Класс — `test_snr_estimator::SnrEstimatorBenchmark`, метод `SnrEstimatorBenchmark`.

_Пример кода не найден в `tests/` или `examples/`._

## Граничные случаи

_Не определены (нет `@throws` в Doxygen primary_method)._

## Что делать дальше

- См. [core__profile_analyzer__usecase__v1](./profile_analyzer.md)
- См. [spectrum__filters_benchmark_rocm__usecase__v1](./filters_benchmark_rocm.md)
- См. [spectrum__fft_maxima_benchmark_rocm__usecase__v1](./fft_maxima_benchmark_rocm.md)

## Ссылки

- Источник кода: `/home/alex/DSP-GPU/stats/tests/test_snr_estimator_benchmark.hpp:1`
