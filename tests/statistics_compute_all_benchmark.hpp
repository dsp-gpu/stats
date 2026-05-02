#pragma once

// ============================================================================
// ComputeAllBenchmarkROCm — GPU benchmark для StatisticsProcessor::ComputeAll
//
// ЧТО:    Замер ComputeAll() на CPU-path (4 beams × 65536 complex float).
//         Per-event breakdown через ProfilingFacade: Upload | Welford_Fused | Median.
//         Pre-allocated GPU buffers переиспользуются между итерациями.
//
// ЗАЧЕМ:  ComputeAll объединяет 3 стадии (mean+std через Welford, медиану) в один
//         pipeline — нужно подтвердить выигрыш vs раздельные ComputeStatistics +
//         ComputeMedian (меньше launch overhead, общий upload).
//
// ПОЧЕМУ: Наследник GpuBenchmarkBase (Template Method GoF) — общий skeleton
//         warmup/runs/report задан базой, специфика в ExecuteKernel*.
//         BatchRecord для всех событий (W1: меньше contention с воркером
//         ProfilingFacade чем N × Record).
//
// История: Создан: 2026-03-20
// ============================================================================

#if ENABLE_ROCM

#include <stats/statistics_processor.hpp>
#include <core/services/gpu_benchmark_base.hpp>
#include <core/services/profiling/profiling_facade.hpp>

#include <complex>
#include <vector>

namespace test_statistics_compute_all_benchmark {

// ─── Benchmark: StatisticsProcessor::ComputeAll() ─────────────────────────

/**
 * @class ComputeAllBenchmarkROCm
 * @brief GpuBenchmarkBase-наследник: замер ComputeAll() с per-stage breakdown.
 *
 * @note ROCm-only (#if ENABLE_ROCM).
 * @see drv_gpu_lib::GpuBenchmarkBase, statistics::StatisticsProcessor::ComputeAll
 */
class ComputeAllBenchmarkROCm : public drv_gpu_lib::GpuBenchmarkBase {
public:
  ComputeAllBenchmarkROCm(
      drv_gpu_lib::IBackend* backend,
      statistics::StatisticsProcessor& proc,
      const statistics::StatisticsParams& params,
      const std::vector<std::complex<float>>& data,
      GpuBenchmarkBase::Config cfg = {
          .n_warmup   = 5,
          .n_runs     = 20,
          .output_dir = "Results/Profiler/GPU_00_Statistics"})
    : GpuBenchmarkBase(backend, "Statistics_ComputeAll_ROCm", cfg),
      proc_(proc), params_(params), data_(data) {}

protected:
  /// Warmup — ComputeAll без profiling events
  void ExecuteKernel() override {
    proc_.ComputeAll(data_, params_);
  }

  /// Замер — ComputeAll с StatisticsROCmProfEvents → ProfilingFacade::BatchRecord
  void ExecuteKernelTimed() override {
    statistics::StatisticsROCmProfEvents events;
    proc_.ComputeAll(data_, params_, &events);
    drv_gpu_lib::profiling::ProfilingFacade::GetInstance()
        .BatchRecord(gpu_id_, "stats/compute_all", events);
  }

private:
  statistics::StatisticsProcessor&         proc_;
  statistics::StatisticsParams             params_;
  std::vector<std::complex<float>>         data_;
};

}  // namespace test_statistics_compute_all_benchmark

#endif  // ENABLE_ROCM
