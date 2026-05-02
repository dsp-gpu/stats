#pragma once

// ============================================================================
// stats_all_test — агрегатор тестов модуля stats
//
// ЧТО:    Единая точка подключения всех test_*.hpp модуля stats.
//         Все тесты под #if ENABLE_ROCM.
// ЗАЧЕМ:  main.cpp вызывает только этот файл — не отдельные test_*.hpp.
//         Закомментированный include = выключенный тест без правки main.cpp.
// ПОЧЕМУ: Паттерн all_test.hpp (правило 15-cpp-testing.md).
//
// История: Создан: 2026-04-12
// ============================================================================

/**
 * @file all_test.hpp
 * @brief Индекс тестов модуля stats — единая точка включения для main.cpp.
 * @note Test fixture, не публичный API. Запускается через main.cpp → statistics_all_test::run().
 *       ROCm-only: все тесты под #if ENABLE_ROCM. Закомментированные include — отключённые сценарии.
 */

#if ENABLE_ROCM
#include "test_statistics_rocm.hpp"
#include "test_statistics_float_rocm.hpp"
#include "statistics_compute_all_benchmark.hpp"
#include "test_statistics_compute_all_benchmark.hpp"
// SNR-estimator (SNR_08, SNR_09) — запуск в понедельник на Debian/AMD
#include "test_snr_estimator_rocm.hpp"
#include "snr_estimator_benchmark.hpp"
#include "test_snr_estimator_benchmark.hpp"
#endif

namespace statistics_all_test {

inline void run() {
  // StatisticsProcessor: mean, median, variance, std (ROCm only)
#if ENABLE_ROCM
  test_statistics_rocm::run();
  // Statistics float API + ProcessMagnitude→Statistics pipeline
  test_statistics_float_rocm::run();
  // ComputeAll benchmark (раскомментировать для профилирования):
  // test_statistics_compute_all_benchmark::run();

  // SNR-estimator: написано 2026-04-09, запуск в понедельник на Debian/AMD
  // test_snr_estimator_rocm::run_all();
  // test_snr_estimator_benchmark::run_benchmark();
#endif
}

}  // namespace statistics_all_test
