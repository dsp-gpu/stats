cmake_minimum_required(VERSION 3.25)
project(DspStats VERSION 0.1.0
  DESCRIPTION "GPU statistics: Welford, medians, radix sort"
  LANGUAGES CXX HIP)

# ── Опции ──────────────────────────────────────────────────────────
option(DSP_STATS_BUILD_TESTS  "Build tests"            ON)
option(DSP_STATS_BUILD_PYTHON "Build Python bindings"  OFF)

# ── Зависимости ────────────────────────────────────────────────────
find_package(hip     REQUIRED)   # lowercase — Linux case-sensitive!
find_package(rocprim REQUIRED)

include(cmake/fetch_deps.cmake)
fetch_dsp_core()
fetch_dsp_spectrum()  # SNR_05: SnrEstimatorOp использует FFTProcessorROCm::ProcessMagnitudesToGPU

# ── Библиотека ─────────────────────────────────────────────────────
add_library(DspStats STATIC)
add_library(DspStats::DspStats ALIAS DspStats)

# PIC нужен для Python bindings (.so линкует static DspStats)
set_target_properties(DspStats PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_sources(DspStats PRIVATE
  src/statistics/src/statistics_processor.cpp
  src/statistics/src/statistics_sort_gpu.hip
)

target_include_directories(DspStats
  PUBLIC
    # Стандарт AMD: #include <stats/operations/...>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/generated>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    kernels/
    src
)
target_link_libraries(DspStats PUBLIC
  DspCore::DspCore
  DspSpectrum::DspSpectrum   # SNR_05: snr_estimator_op.hpp — публичный header,
                             # клиенты транзитивно нуждаются в spectrum
  roc::rocprim)
target_compile_features(DspStats PUBLIC cxx_std_17)

# Прокинуть ENABLE_ROCM=1 — для #if ENABLE_ROCM блоков в коде.
# PUBLIC чтобы downstream (radar) тоже видел.
if(ENABLE_ROCM)
  target_compile_definitions(DspStats PUBLIC ENABLE_ROCM=1)
endif()

# ── Git Version ────────────────────────────────────────────────────
find_package(Git QUIET)
if(NOT GIT_EXECUTABLE)
    set(GIT_EXECUTABLE git)
endif()

add_custom_target(git_version_DspStats ALL
    COMMAND ${CMAKE_COMMAND}
        -D SRC_DIR=${CMAKE_CURRENT_SOURCE_DIR}
        -D BIN_DIR=${CMAKE_CURRENT_BINARY_DIR}
        -D GIT_EXECUTABLE=${GIT_EXECUTABLE}
        -D MODULE_PREFIX=DSPSTATS
        -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/version.cmake
    COMMENT "[DspStats] Checking git version..."
    VERBATIM
)
add_dependencies(DspStats git_version_DspStats)

# ── Install + Export ────────────────────────────────────────────────
include(GNUInstallDirs)
install(TARGETS DspStats EXPORT DspStatsTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/generated/version.h"
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/DspStats
)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/generated/version.json"
  DESTINATION ${CMAKE_INSTALL_DATADIR}/DspStats
)

include(CMakePackageConfigHelpers)
configure_package_config_file(cmake/DspStatsConfig.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/DspStatsConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/DspStats
)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/DspStatsConfigVersion.cmake"
  VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion
)
install(EXPORT DspStatsTargets NAMESPACE DspStats::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/DspStats
)
install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/DspStatsConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/DspStatsConfigVersion.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/DspStats
)

if(DSP_STATS_BUILD_TESTS)
  add_subdirectory(tests)
endif()
if(DSP_STATS_BUILD_PYTHON)
  add_subdirectory(python)
endif()
