#pragma once

// ============================================================================
// test_snr_estimator_rocm — тесты SNREstimator (SNR_08, 7 сценариев)
//
// ЧТО:    noise-only (CFAR artifact 8-15 dB), CW+AWGN (SNR>40 dB),
//         negative freq, Scenario A/B/C (до 9000×10000).
// ЗАЧЕМ:  SNREstimator — в radar pipeline. Ошибка SNR = неверное обнаружение цели.
// ПОЧЕМУ: Assert'ы — диапазоны, не точные значения (зависит от noise seed).
//
// История: Создан: 2026-04-12
// ============================================================================

/**
 * @file test_snr_estimator_rocm.hpp
 * @brief Тесты SNR-estimator (SNR_08) — 7 сценариев (test_01..test_06b).
 * @note Test fixture, не публичный API. Запускается через all_test.hpp. ROCm-only.
 *       Сценарии: noise-only (CFAR artifact 8-15 dB), CW+AWGN (SNR>40 dB),
 *       negative freq, Scenario A/B/C (до 9000×10000). Assert'ы — диапазоны, не точные.
 *       Запуск на Debian/AMD (нет GPU под Windows).
 */


#include "snr_test_helpers.hpp"
#include <dsp/stats/statistics_processor.hpp>
#include <dsp/stats/branch_selector.hpp>

#include <core/services/console_output.hpp>
#include <test_utils/validators/numeric.hpp>  // gpu_test_utils::InRangeError

#include <cassert>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#include <complex>

namespace test_snr_estimator_rocm {

using cx = std::complex<float>;

inline void TestPrint(const std::string& msg) {
  drv_gpu_lib::ConsoleOutput::GetInstance().Print(0, "SNR", msg);
}

// ============================================================================
// test_01 — Только шум (нет сигнала). Проверяем артефакт CFAR ≈ 8-15 dB
// ============================================================================
inline void test_01_noise_only_artifact() {
  TestPrint("[test_01] Noise only — CFAR artifact");
  auto* backend = snr_test_helpers::GetTestBackend();
  dsp::stats::StatisticsProcessor proc(backend);

  const uint32_t n_ant = 1, n_samp = 5000;
  auto data = snr_test_helpers::MakeNoise(n_samp, /*noise_power=*/1.0f, /*seed=*/42u);

  dsp::stats::SnrEstimationConfig cfg;  // defaults: target_n_fft=0→2048, Hann, guard=5, ref=16

  auto result = proc.ComputeSnrDb(data, n_ant, n_samp, cfg);

  // H0 артефакт: E[SNR_fft_db | noise] ≈ 10·log10(ln(N_fft) + γ) ≈ 8-10 dB.
  // Широкий диапазон — CFAR имеет разброс по реализациям.
  {
    auto v = gpu_test_utils::InRangeError(
        result.snr_db_global, 3.0f, 18.0f, "snr_db_global_noise_only");
    if (!v.passed) {
      throw std::runtime_error("test_01_noise_only_artifact: " +
                               v.metric_name + " " + v.message);
    }
  }
  {
    auto v = gpu_test_utils::InRangeInclusiveError(
        result.used_bins, 1024u, 4096u, "used_bins_default_cfg");
    if (!v.passed) {
      throw std::runtime_error("test_01_noise_only_artifact: " +
                               v.metric_name + " " + v.message);
    }
  }

  // BranchSelector с откалиброванными порогами: шум должен быть Low.
  dsp::stats::BranchSelector selector;
  auto branch = selector.Select(result.snr_db_global, cfg.thresholds);
  {
    auto v = gpu_test_utils::ScalarEqError(
        static_cast<int>(branch),
        static_cast<int>(dsp::stats::BranchType::Low),
        "branch_is_low_for_noise");
    if (!v.passed) {
      throw std::runtime_error("test_01_noise_only_artifact: " +
                               v.metric_name + " " + v.message);
    }
  }

  TestPrint("[test_01] PASS — snr_db=" + std::to_string(result.snr_db_global));
}

// ============================================================================
// test_02 — CW + шум, SNR_in = 20 dB, 1 антенна
// ============================================================================
inline void test_02_basic_signal() {
  TestPrint("[test_02] CW + noise, SNR_in=20 dB");
  auto* backend = snr_test_helpers::GetTestBackend();
  dsp::stats::StatisticsProcessor proc(backend);

  const uint32_t n_ant = 1, n_samp = 5000;
  // SNR_in = 20 dB → A² / σ² = 100 → A = 10 при σ² = 1
  const float A = 10.0f, noise_power = 1.0f;
  auto signal = snr_test_helpers::MakeDechirpedCW(n_samp, /*freq_norm=*/0.15f, A);
  snr_test_helpers::AddNoise(signal, noise_power, /*seed=*/42u);

  dsp::stats::SnrEstimationConfig cfg;
  auto result = proc.ComputeSnrDb(signal, n_ant, n_samp, cfg);

  // SNR_fft = SNR_in + 10·log10(N_actual) ≈ 20 + 32 = 52 dB.
  // С учётом Hann processing loss ~1.76 dB и CFAR bias ~−5 dB: 40-55 dB диапазон.
  {
    auto v = gpu_test_utils::LowerBoundError(
        result.snr_db_global, 38.0f, "snr_db_basic_signal");
    if (!v.passed) {
      throw std::runtime_error("test_02_basic_signal: " +
                               v.metric_name + " " + v.message);
    }
  }

  TestPrint("[test_02] PASS — snr_db=" + std::to_string(result.snr_db_global));
}

// ============================================================================
// test_03 — Отрицательная частота + search_full_spectrum toggle
// ============================================================================
inline void test_03_negative_freq() {
  TestPrint("[test_03] Negative freq + search_full_spectrum toggle");
  auto* backend = snr_test_helpers::GetTestBackend();
  dsp::stats::StatisticsProcessor proc(backend);

  const uint32_t n_ant = 1, n_samp = 5000;
  // Отрицательная частота: −0.2 → f * n_samples даёт пик в [nFFT/2..nFFT)
  auto signal = snr_test_helpers::MakeDechirpedCW(n_samp, /*freq_norm=*/-0.2f, /*A=*/10.0f);
  snr_test_helpers::AddNoise(signal, /*noise_power=*/1.0f, /*seed=*/42u);

  // Дефолт `search_full_spectrum = true` — CFAR ищет пик по всему спектру.
  // Для radar это единственно корректный режим: знак Doppler / направление цели
  // заранее неизвестны, цель может оказаться на любой стороне FFT (в т.ч. на
  // отрицательной нормированной частоте, как здесь). Поэтому проверка
  // `search_full_spectrum=false` (legacy half-search) удалена 2026-05-14 как
  // anti-pattern: для radar half-search теряет половину сцены.
  dsp::stats::SnrEstimationConfig cfg;  // search_full_spectrum=true по умолчанию
  auto result = proc.ComputeSnrDb(signal, n_ant, n_samp, cfg);
  {
    auto v = gpu_test_utils::LowerBoundError(
        result.snr_db_global, 30.0f, "snr_db_neg_freq_full_spectrum");
    if (!v.passed) {
      throw std::runtime_error("test_03_negative_freq: " +
                               v.metric_name + " " + v.message);
    }
  }

  TestPrint("[test_03] PASS — snr_db=" + std::to_string(result.snr_db_global));
}

// ============================================================================
// test_04 — Сценарий A (2500 × 5000, auto params)
// ============================================================================
inline void test_04_scenario_a() {
  TestPrint("[test_04] Scenario A: 2500 ant x 5000 samp");
  auto* backend = snr_test_helpers::GetTestBackend();
  dsp::stats::StatisticsProcessor proc(backend);

  const uint32_t n_ant = 2500, n_samp = 5000;
  // Генерируем сигнал со случайными частотами в диапазоне 0.05..0.3
  // SNR_in ~ 15 dB → A ≈ 5.6 при σ² = 1
  std::vector<cx> data(static_cast<size_t>(n_ant) * n_samp);
  uint32_t state = 1337u;
  auto rng = [&]() -> float {
    state = state * 1664525u + 1013904223u;
    return static_cast<float>(state) / static_cast<float>(0xFFFFFFFFu);
  };
  for (uint32_t ant = 0; ant < n_ant; ++ant) {
    float freq_norm = 0.05f + 0.25f * rng();
    auto sig = snr_test_helpers::MakeDechirpedCW(n_samp, freq_norm, /*A=*/5.6f);
    snr_test_helpers::AddNoise(sig, /*noise_power=*/1.0f, /*seed=*/ant + 1u);
    std::copy(sig.begin(), sig.end(), data.begin() + static_cast<size_t>(ant) * n_samp);
  }

  dsp::stats::SnrEstimationConfig cfg;  // auto: step_antennas → ceil(2500/50) = 50
  auto result = proc.ComputeSnrDb(data, n_ant, n_samp, cfg);

  // n_ant_out = ceil(2500 / 50) = 50
  {
    auto v = gpu_test_utils::ScalarEqError(
        result.used_antennas, 50u, "used_antennas_scenario_a");
    if (!v.passed) {
      throw std::runtime_error("test_04_scenario_a: " +
                               v.metric_name + " " + v.message);
    }
  }
  // SNR_in=15 dB + 10·log10(2500) ≈ 15 + 34 = 49 dB, с bias → 35-52 dB
  {
    auto v = gpu_test_utils::InRangeError(
        result.snr_db_global, 30.0f, 55.0f, "snr_db_global_scenario_a");
    if (!v.passed) {
      throw std::runtime_error("test_04_scenario_a: " +
                               v.metric_name + " " + v.message);
    }
  }

  TestPrint("[test_04] PASS — snr_db=" + std::to_string(result.snr_db_global) +
            " used_ant=" + std::to_string(result.used_antennas));
}

// ============================================================================
// test_05 — Сценарий B (256 × 1.3M) — 2.66 GB, GPU-only путь
// ============================================================================
inline void test_05_scenario_b() {
  TestPrint("[test_05] Scenario B: 256 ant x 1.3M samp (2.66 GB)");
  auto* backend = snr_test_helpers::GetTestBackend();
  dsp::stats::StatisticsProcessor proc(backend);

  const uint32_t n_ant = 256, n_samp = 1'300'000u;

  // Для экономии RAM: генерируем одну антенну (сигнал+шум) и тиражируем
  // в GPU через hipMemcpy. Тест проверяет pipeline, не физическую уникальность.
  std::vector<cx> one_ant =
      snr_test_helpers::MakeDechirpedCW(n_samp, /*freq_norm=*/0.1f, /*A=*/3.2f);
  snr_test_helpers::AddNoise(one_ant, /*noise_power=*/1.0f, /*seed=*/1u);

  void* gpu_data = nullptr;
  const size_t total_bytes = static_cast<size_t>(n_ant) * n_samp * sizeof(cx);
  hipError_t err = hipMalloc(&gpu_data, total_bytes);
  if (err != hipSuccess) {
    TestPrint("[test_05] SKIP — hipMalloc failed (" +
              std::to_string(total_bytes / (1024u * 1024u)) + " MB)");
    return;
  }

  for (uint32_t ant = 0; ant < n_ant; ++ant) {
    char* dst = static_cast<char*>(gpu_data) +
                static_cast<size_t>(ant) * n_samp * sizeof(cx);
    hipMemcpy(dst, one_ant.data(), n_samp * sizeof(cx), hipMemcpyHostToDevice);
  }

  dsp::stats::SnrEstimationConfig cfg;  // auto: step_antennas → ceil(256/50) = 6
  auto result = proc.ComputeSnrDb(gpu_data, n_ant, n_samp, cfg);

  snr_test_helpers::FreeGpu(gpu_data);

  // n_ant_out = ceil(256 / 6) = 43
  {
    auto v = gpu_test_utils::ScalarEqError(
        result.used_antennas, 43u, "used_antennas_scenario_b");
    if (!v.passed) {
      throw std::runtime_error("test_05_scenario_b: " +
                               v.metric_name + " " + v.message);
    }
  }
  // SNR_in = 10 dB (A=3.2, σ²=1 → A²/σ² ≈ 10) + 10·log10(~1666) ≈ 10 + 32 = 42 dB
  // После CFAR bias и Hann loss: > 25 dB
  {
    auto v = gpu_test_utils::LowerBoundError(
        result.snr_db_global, 25.0f, "snr_db_scenario_b");
    if (!v.passed) {
      throw std::runtime_error("test_05_scenario_b: " +
                               v.metric_name + " " + v.message);
    }
  }

  TestPrint("[test_05] PASS — snr_db=" + std::to_string(result.snr_db_global) +
            " used_ant=" + std::to_string(result.used_antennas));
}

// ============================================================================
// test_06 — Scenario B только шум — CFAR artifact стабилен
// ============================================================================
inline void test_06_scenario_b_noise() {
  TestPrint("[test_06] Scenario B noise only — CFAR artifact stable");
  auto* backend = snr_test_helpers::GetTestBackend();
  dsp::stats::StatisticsProcessor proc(backend);

  const uint32_t n_ant = 256, n_samp = 1'300'000u;

  // Только шум, одна антенна тиражируется в GPU memory
  auto one_ant = snr_test_helpers::MakeNoise(n_samp, /*noise_power=*/1.0f, /*seed=*/7u);

  void* gpu_data = nullptr;
  const size_t total_bytes = static_cast<size_t>(n_ant) * n_samp * sizeof(cx);
  hipError_t err = hipMalloc(&gpu_data, total_bytes);
  if (err != hipSuccess) {
    TestPrint("[test_06] SKIP — hipMalloc failed");
    return;
  }
  for (uint32_t ant = 0; ant < n_ant; ++ant) {
    char* dst = static_cast<char*>(gpu_data) +
                static_cast<size_t>(ant) * n_samp * sizeof(cx);
    hipMemcpy(dst, one_ant.data(), n_samp * sizeof(cx), hipMemcpyHostToDevice);
  }

  dsp::stats::SnrEstimationConfig cfg;
  auto result = proc.ComputeSnrDb(gpu_data, n_ant, n_samp, cfg);
  snr_test_helpers::FreeGpu(gpu_data);

  // Только noise → CFAR artifact ≈ 8-15 dB
  {
    auto v = gpu_test_utils::InRangeError(
        result.snr_db_global, 3.0f, 18.0f, "snr_db_scenario_b_noise");
    if (!v.passed) {
      throw std::runtime_error("test_06_scenario_b_noise: " +
                               v.metric_name + " " + v.message);
    }
  }

  TestPrint("[test_06] PASS — snr_db=" + std::to_string(result.snr_db_global));
}

// ============================================================================
// test_06b — Сценарий C (9000 × 10000)
// ============================================================================
inline void test_06b_scenario_c() {
  TestPrint("[test_06b] Scenario C: 9000 ant x 10000 samp");
  auto* backend = snr_test_helpers::GetTestBackend();
  dsp::stats::StatisticsProcessor proc(backend);

  const uint32_t n_ant = 9000, n_samp = 10000;

  // 9000 × 10000 × 8 = 720 MB — помещается в CPU
  std::vector<cx> data(static_cast<size_t>(n_ant) * n_samp);
  uint32_t state = 2026u;
  auto rng = [&]() -> float {
    state = state * 1664525u + 1013904223u;
    return static_cast<float>(state) / static_cast<float>(0xFFFFFFFFu);
  };
  // SNR_in = 10 dB → A²/σ² = 10 → A ≈ 3.16
  for (uint32_t ant = 0; ant < n_ant; ++ant) {
    float freq_norm = 0.05f + 0.25f * rng();
    auto sig = snr_test_helpers::MakeDechirpedCW(n_samp, freq_norm, /*A=*/3.16f);
    snr_test_helpers::AddNoise(sig, /*noise_power=*/1.0f, /*seed=*/ant + 100u);
    std::copy(sig.begin(), sig.end(), data.begin() + static_cast<size_t>(ant) * n_samp);
  }

  dsp::stats::SnrEstimationConfig cfg;  // auto: step_antennas → ceil(9000/50) = 180
  auto result = proc.ComputeSnrDb(data, n_ant, n_samp, cfg);

  // n_ant_out = ceil(9000 / 180) = 50
  {
    auto v = gpu_test_utils::ScalarEqError(
        result.used_antennas, 50u, "used_antennas_scenario_c");
    if (!v.passed) {
      throw std::runtime_error("test_06b_scenario_c: " +
                               v.metric_name + " " + v.message);
    }
  }
  // SNR_in=10 + 10·log10(10000) ≈ 10 + 40 = 50 dB, с bias → > 30
  {
    auto v = gpu_test_utils::LowerBoundError(
        result.snr_db_global, 30.0f, "snr_db_scenario_c");
    if (!v.passed) {
      throw std::runtime_error("test_06b_scenario_c: " +
                               v.metric_name + " " + v.message);
    }
  }

  TestPrint("[test_06b] PASS — snr_db=" + std::to_string(result.snr_db_global) +
            " used_ant=" + std::to_string(result.used_antennas));
}

// ============================================================================
// run_all — вызывает все 7 тестов (запуск в понедельник!)
// ============================================================================
inline void run_all() {
  TestPrint("=== SNR Estimator C++ Tests ===");

  // Проверка что есть ROCm device
  int devices = drv_gpu_lib::ROCmCore::GetAvailableDeviceCount();
  if (devices == 0) {
    TestPrint("[!] No ROCm devices — skipping SNR tests");
    return;
  }

  // Wrap each test in try/catch — один FAIL не должен валить остальные.
  // Включено 2026-05-14 после миграции ranges на gpu_test_utils::InRangeError
  // (бросает std::runtime_error). Single-bound assert'ы при failure всё ещё
  // abort'ят процесс — это известное поведение <cassert>.
  int passed = 0, failed = 0;
  auto run = [&](const std::string& name, void (*fn)()) {
    try {
      fn();
      ++passed;
    } catch (const std::exception& e) {
      ++failed;
      TestPrint("[FAIL] " + name + ": " + e.what());
    }
  };

  run("test_01_noise_only_artifact",  &test_01_noise_only_artifact);
  run("test_02_basic_signal",         &test_02_basic_signal);
  run("test_03_negative_freq",        &test_03_negative_freq);
  run("test_04_scenario_a",           &test_04_scenario_a);
  run("test_05_scenario_b",           &test_05_scenario_b);
  run("test_06_scenario_b_noise",     &test_06_scenario_b_noise);
  run("test_06b_scenario_c",          &test_06b_scenario_c);

  TestPrint("=== SNR Estimator Tests DONE: " +
            std::to_string(passed) + " passed, " +
            std::to_string(failed) + " failed ===");
}

}  // namespace test_snr_estimator_rocm

