/**
 * @file dsp_stats_module.cpp
 * @brief pybind11 bindings for dsp::stats
 *
 * Python API:
 *   import dsp_stats
 *   proc = dsp_stats.StatisticsProcessor(ctx)
 *   results = proc.compute_statistics(data, beam_count=4)
 *
 * Экспортируемые классы:
 *   StatisticsProcessor — mean/median/variance/std/SNR (ROCm)
 */

#include "py_helpers.hpp"

#if ENABLE_ROCM
#include "py_gpu_context.hpp"
#include "py_statistics.hpp"
#endif

PYBIND11_MODULE(dsp_stats, m) {
    m.doc() = "dsp::stats — statistics on GPU (ROCm)\n\n"
              "Classes:\n"
              "  ROCmGPUContext          - GPU context (AMD ROCm)\n"
              "  StatisticsProcessor     - Welford/median/SNR estimator\n";

#if ENABLE_ROCM
    // ROCmGPUContext зарегистрирован в dsp_core (один раз глобально).
    py::module_::import("dsp_core");

    register_statistics(m);
#endif
}
