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
#include "py_statistics.hpp"
#endif

PYBIND11_MODULE(dsp_stats, m) {
    m.doc() = "dsp::stats — statistics on GPU (ROCm)\n\n"
              "Classes:\n"
              "  StatisticsProcessor - Welford/median/SNR estimator (ROCm)\n";

#if ENABLE_ROCM
    register_statistics(m);
#endif
}
