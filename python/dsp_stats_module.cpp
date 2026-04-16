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
    py::class_<ROCmGPUContext>(m, "ROCmGPUContext",
        "ROCm GPU context (creates HIP backend for AMD GPU).")
        .def(py::init<int>(), py::arg("device_index") = 0)
        .def_property_readonly("device_name", &ROCmGPUContext::device_name)
        .def_property_readonly("device_index", &ROCmGPUContext::device_index);

    register_statistics(m);
#endif
}
