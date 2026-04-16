# ── DspStats Python bindings ───────────────────────────────────────
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
execute_process(
  COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
  OUTPUT_VARIABLE pybind11_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(dsp_stats dsp_stats_module.cpp)
target_link_libraries(dsp_stats PRIVATE DspStats::DspStats)
target_include_directories(dsp_stats PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/src/statistics/include
  # py_gpu_context.hpp — ROCmGPUContext shared header
  ${FETCHCONTENT_SOURCE_DIR_DSPCORE}/python)
target_compile_definitions(dsp_stats PRIVATE
  $<$<BOOL:${ENABLE_ROCM}>:ENABLE_ROCM=1>)
install(TARGETS dsp_stats DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
