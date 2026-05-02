#pragma once

// ============================================================================
// test_statistics_float_rocm — тесты StatisticsProcessor float API (ROCm)
//
// ЧТО:    6 сценариев: ComputeStatisticsFloat, ComputeMedianFloat (vector<float>
//         и void* GPU), pipeline ProcessMagnitude→Statistics/Median.
// ЗАЧЕМ:  Float API — отдельный путь (не complex). Ошибки здесь = неверная
//         статистика в float-only pipeline (radar power estimation).
// ПОЧЕМУ: GPU-to-GPU pipeline без промежуточного D2H copy. test_utils.
//
// История: Создан: 2026-04-12
// ============================================================================

/**
 * @file test_statistics_float_rocm.hpp
 * @brief Тесты StatisticsProcessor float API + связки ProcessMagnitude→Statistics.
 * @note Test fixture, не публичный API. Запускается через all_test.hpp. ROCm-only.
 *       6 сценариев: ComputeStatisticsFloat / ComputeMedianFloat (vector<float> и void* GPU)
 *       + GPU-to-GPU pipeline ProcessMagnitude → Statistics/Median.
 *       Мигрирован на test_utils (CppTest-06).
 */

#if ENABLE_ROCM

#include <stats/statistics_processor.hpp>
#include <spectrum/complex_to_mag_phase_rocm.hpp>
#include "test_helpers_rocm.hpp"
#include <core/backends/rocm/rocm_backend.hpp>

// test_utils — единая тестовая инфраструктура
#include "test_utils/test_utils.hpp"

#include <hip/hip_runtime.h>

#include <vector>
#include <complex>
#include <cmath>
#include <string>
#include <algorithm>
#include <numeric>
#include <random>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace test_statistics_float_rocm {

using namespace statistics;
using namespace drv_gpu_lib;
using namespace test_helpers_rocm;
using namespace gpu_test_utils;

// =========================================================================
// run() — TestRunner (функциональный стиль)
// =========================================================================

inline void run() {
  int gpu_id = 0;

  ROCmBackend backend;
  backend.Initialize(gpu_id);
  StatisticsProcessor stats(&backend);
  fft_processor::ComplexToMagPhaseROCm mag_proc(&backend);

  TestRunner runner(&backend, "StatsFloat", gpu_id);

  // ── T1: ComputeStatisticsFloat(vector<float>) ────────────────────

  runner.test("stats_vector_float", [&]() -> TestResult {
    constexpr uint32_t N = 4096;
    std::vector<float> data(N);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.5f, 5.0f);
    for (float& x : data) x = dist(rng);

    StatisticsParams params;
    params.beam_count = 1;
    params.n_point = N;

    auto results = stats.ComputeStatisticsFloat(data, params);

    float ref_mean = refs::CpuMeanFloat(data.data(), N);
    float ref_std  = refs::CpuStdFloat(data.data(), N);

    TestResult tr{"stats_vector_float"};
    tr.add(ScalarRelError(results[0].mean_magnitude, ref_mean, 1e-2, "mean"));
    tr.add(ScalarRelError(results[0].std_dev, ref_std, 1e-2, "std"));
    return tr;
  });

  // ── T2: ComputeMedianFloat(vector<float>) ────────────────────────

  runner.test("median_vector_float", [&]() {
    constexpr uint32_t N = 1024;
    std::vector<float> data(N);
    std::iota(data.begin(), data.end(), 0.0f);  // 0..1023

    StatisticsParams params;
    params.beam_count = 1;
    params.n_point = N;

    auto results = stats.ComputeMedianFloat(data, params);
    float ref_median = refs::CpuMedianFloat(data.data(), N);

    return ScalarAbsError(results[0].median_magnitude, ref_median, 1.0, "median");
  });

  // ── T3: ComputeStatisticsFloat(void*) — managed memory ───────────

  runner.test("stats_gpu_float_managed", [&]() -> TestResult {
    constexpr uint32_t N = 2048;
    auto managed = AllocateManagedForTest(N * sizeof(float));
    auto* ptr = static_cast<float*>(managed);

    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(1.0f, 4.0f);
    std::vector<float> src(N);
    for (size_t i = 0; i < N; ++i) { ptr[i] = dist(rng); src[i] = ptr[i]; }

    auto input = MakeManagedMagnitudeInput(managed, 1, N);

    StatisticsParams params;
    params.beam_count = 1;
    params.n_point = N;

    auto results = stats.ComputeStatisticsFloat(input.data, params);
    (void)hipFree(managed);

    float ref_mean = refs::CpuMeanFloat(src.data(), N);
    float ref_std  = refs::CpuStdFloat(src.data(), N);

    TestResult tr{"stats_gpu_float_managed"};
    tr.add(ScalarRelError(results[0].mean_magnitude, ref_mean, 1e-2, "mean"));
    tr.add(ScalarRelError(results[0].std_dev, ref_std, 1e-2, "std"));
    return tr;
  });

  // ── T4: ComputeMedianFloat(void*) — managed memory ───────────────

  runner.test("median_gpu_float_managed", [&]() {
    constexpr uint32_t N = 512;
    auto managed = AllocateManagedForTest(N * sizeof(float));
    auto* ptr = static_cast<float*>(managed);

    std::vector<float> src(N);
    std::iota(src.begin(), src.end(), 0.0f);
    std::mt19937 rng(13);
    std::shuffle(src.begin(), src.end(), rng);
    for (size_t i = 0; i < N; ++i) ptr[i] = src[i];

    auto input = MakeManagedMagnitudeInput(managed, 1, N);

    StatisticsParams params;
    params.beam_count = 1;
    params.n_point = N;

    auto results = stats.ComputeMedianFloat(input.data, params);
    (void)hipFree(managed);

    float ref_median = refs::CpuMedianFloat(src.data(), N);
    return ScalarAbsError(results[0].median_magnitude, ref_median, 1.0, "median");
  });

  // ── T5: Pipeline Magnitude → Stats (GPU-to-GPU) ──────────────────

  runner.test("pipeline_mag_to_stats", [&]() -> TestResult {
    constexpr uint32_t kBeams = 2;
    constexpr uint32_t kN = 4096;
    constexpr size_t kTotal = kBeams * kN;

    auto managed = AllocateManagedForTest(kTotal * sizeof(std::complex<float>));
    auto* cptr = static_cast<std::complex<float>*>(managed);

    for (uint32_t b = 0; b < kBeams; ++b) {
      float amp = static_cast<float>(b + 1);
      for (uint32_t k = 0; k < kN; ++k) {
        float ph = 2.0f * static_cast<float>(M_PI) * 440.0f * k / 44100.0f;
        cptr[b * kN + k] = {amp * std::cos(ph), amp * std::sin(ph)};
      }
    }

    fft_processor::MagPhaseParams mp;
    mp.beam_count = kBeams;
    mp.n_point = kN;
    mp.norm_coeff = 1.0f;

    void* gpu_mag = mag_proc.ProcessMagnitudeToGPU(managed, mp, kTotal * sizeof(std::complex<float>));
    (void)hipFree(managed);

    StatisticsParams sp;
    sp.beam_count = kBeams;
    sp.n_point = kN;

    auto results = stats.ComputeStatisticsFloat(gpu_mag, sp);
    (void)hipFree(gpu_mag);

    TestResult tr{"pipeline_mag_to_stats"};
    for (uint32_t b = 0; b < kBeams; ++b) {
      float expected_mean = static_cast<float>(b + 1);
      tr.add(ScalarAbsError(results[b].mean_magnitude, expected_mean,
                             0.05, "beam" + std::to_string(b) + "_mean"));
    }
    return tr;
  });

  // ── T6: Pipeline Magnitude → Median (GPU-to-GPU) ─────────────────

  runner.test("pipeline_mag_to_median", [&]() {
    constexpr uint32_t kN = 2048;

    auto managed = AllocateManagedForTest(kN * sizeof(std::complex<float>));
    auto* cptr = static_cast<std::complex<float>*>(managed);
    for (uint32_t k = 0; k < kN; ++k)
      cptr[k] = {3.0f, 0.0f};

    fft_processor::MagPhaseParams mp;
    mp.beam_count = 1;
    mp.n_point = kN;
    mp.norm_coeff = 1.0f;

    void* gpu_mag = mag_proc.ProcessMagnitudeToGPU(managed, mp, kN * sizeof(std::complex<float>));
    (void)hipFree(managed);

    StatisticsParams sp;
    sp.beam_count = 1;
    sp.n_point = kN;

    auto results = stats.ComputeMedianFloat(gpu_mag, sp);
    (void)hipFree(gpu_mag);

    return ScalarAbsError(results[0].median_magnitude, 3.0f, 1e-3, "median_3.0");
  });

  // ── Summary ──────────────────────────────────────────────────────

  runner.print_summary();
  runner.export_json("Results/JSON/test_statistics_float_rocm.json");
}

}  // namespace test_statistics_float_rocm

#endif  // ENABLE_ROCM
