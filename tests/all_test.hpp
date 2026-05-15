#pragma once

// ============================================================================
// stats_all_test — агрегатор тестов модуля stats
//
// ЧТО:    Единая точка подключения всех test_*.hpp модуля stats.
//         Все тесты ROCm-only.
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
 *       ROCm-only: все тесты ROCm-only. Закомментированные include — отключённые сценарии.
 */

#include "test_statistics_rocm.hpp"
#include "test_statistics_float_rocm.hpp"
#include "statistics_compute_all_benchmark.hpp"
#include "test_statistics_compute_all_benchmark.hpp"
// SNR-estimator (SNR_08, SNR_09) — запуск в понедельник на Debian/AMD
#include "test_snr_estimator_rocm.hpp"
#include "snr_estimator_benchmark.hpp"
#include "test_snr_estimator_benchmark.hpp"

namespace statistics_all_test {

inline void run() {
  // StatisticsProcessor: mean, median, variance, std (ROCm only)
  test_statistics_rocm::run();
  // Statistics float API + ProcessMagnitude→Statistics pipeline
  test_statistics_float_rocm::run();
  // ComputeAll benchmark (раскомментировать для профилирования):
  // test_statistics_compute_all_benchmark::run();

  // SNR-estimator: написано 2026-04-09, активировано 2026-05-14 после
  // миграции assert(...range...) на gpu_test_utils::InRangeError.
  test_snr_estimator_rocm::run_all();
  // test_snr_estimator_benchmark::run_benchmark();  // benchmark включать отдельно
}

}  // namespace statistics_all_test
