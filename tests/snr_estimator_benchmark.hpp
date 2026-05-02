#pragma once

// ============================================================================
// SnrEstimatorBenchmark — GPU benchmark для StatisticsProcessor::ComputeSnrDb (SNR_09)
//
// ЧТО:    Измеряет end-to-end время SNR-estimator pipeline через hipEvent-пары,
//         собирает события через ProfilingFacade::RecordROCmEvent.
//         Pre-allocated GPU input на инстанс — без upload в каждой итерации.
//
// ЗАЧЕМ:  В сценариях SNR_09 (256×1.3M = 2.66 GB) разовый ComputeSnrDb идёт
//         десятки мс. Нужно стабильное измерение чистой работы оператора
//         (warmup → N runs → агрегация) без шума от аллокаций / upload.
//
// ПОЧЕМУ: Наследник GpuBenchmarkBase (Template Method GoF) — invariant skeleton
//         (Run/Report) задан базой, конкретика в ExecuteKernel / ExecuteKernelTimed.
//         hipEvent — единственный точный GPU-таймер; ScopedHipEvent гарантирует
//         RAII-освобождение при exception в hot-path.
//
// История: Создан: 2026-04-09
// ============================================================================

#if ENABLE_ROCM

#include <stats/statistics_processor.hpp>
#include "snr_test_helpers.hpp"
#include <core/services/gpu_benchmark_base.hpp>
#include <core/services/console_output.hpp>
#include <core/services/profiling_types.hpp>
#include <core/services/scoped_hip_event.hpp>
#include <core/services/profiling/profiling_facade.hpp>

#include <hip/hip_runtime.h>

#include <complex>
#include <string>
#include <vector>

namespace test_snr_estimator {

/**
 * @class SnrEstimatorBenchmark
 * @brief GpuBenchmarkBase-наследник для замера ComputeSnrDb() через hipEvent.
 *
 * @note ROCm-only (#if ENABLE_ROCM). Caller владеет pre-allocated gpu_input.
 * @see drv_gpu_lib::GpuBenchmarkBase, statistics::StatisticsProcessor::ComputeSnrDb
 */
class SnrEstimatorBenchmark : public drv_gpu_lib::GpuBenchmarkBase {
public:
  SnrEstimatorBenchmark(
      drv_gpu_lib::IBackend* backend,
      statistics::StatisticsProcessor& proc,
      void* gpu_input,                                  // pre-allocated, owned by caller
      uint32_t n_antennas,
      uint32_t n_samples,
      const statistics::SnrEstimationConfig& cfg,
      std::string instance_name,
      GpuBenchmarkBase::Config bench_cfg = {
          .n_warmup   = 3,
          .n_runs     = 10,
          .output_dir = "Results/Profiler/GPU_00_SNR_Estimator_ROCm"})
    : GpuBenchmarkBase(backend, instance_name, bench_cfg),
      proc_(proc),
      gpu_input_(gpu_input),
      n_antennas_(n_antennas),
      n_samples_(n_samples),
      cfg_(cfg) {}

protected:
  /// Warmup — один вызов, без measurement
  void ExecuteKernel() override {
    (void)proc_.ComputeSnrDb(gpu_input_, n_antennas_, n_samples_, cfg_);
  }

  /// Timed — записываем e2e время через hipEvent пару + RecordROCmEvent
  void ExecuteKernelTimed() override {
    // RAII: события освобождаются автоматически при выходе из scope
    // (в т.ч. при исключении в ComputeSnrDb).
    drv_gpu_lib::ScopedHipEvent ev_start, ev_end;
    ev_start.Create();
    ev_end.Create();

    hipEventRecord(ev_start.get(), 0);  // stream 0 (default)
    (void)proc_.ComputeSnrDb(gpu_input_, n_antennas_, n_samples_, cfg_);
    hipEventRecord(ev_end.get(), 0);

    hipEventSynchronize(ev_end.get());

    float elapsed_ms = 0.0f;
    hipEventElapsedTime(&elapsed_ms, ev_start.get(), ev_end.get());

    // Заполним ROCmProfilingData: start/end в наносекундах (для ProfilingStats)
    drv_gpu_lib::ROCmProfilingData data;
    data.start_ns = 0;
    data.end_ns   = static_cast<uint64_t>(elapsed_ms * 1e6);  // ms → ns
    data.op_string = "ComputeSnrDb total (gather → FFT|X|² → CFAR → median)";

    drv_gpu_lib::profiling::ProfilingFacade::GetInstance()
        .Record(gpu_id_, "stats/snr_estimator", "ComputeSnrDb_total", data);
    // hipEventDestroy не нужен — RAII ~ScopedHipEvent
  }

private:
  statistics::StatisticsProcessor&    proc_;
  void*                                gpu_input_ = nullptr;
  uint32_t                             n_antennas_ = 0;
  uint32_t                             n_samples_  = 0;
  statistics::SnrEstimationConfig      cfg_;
};

}  // namespace test_snr_estimator

#endif  // ENABLE_ROCM
