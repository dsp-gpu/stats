cmake_minimum_required(VERSION 3.25)
project(DspStats VERSION 0.1.0 LANGUAGES CXX HIP)

# ── Опции ──────────────────────────────────────────────────────────
option(DSP_STATS_BUILD_TESTS  "Build tests"            ON)
option(DSP_STATS_BUILD_PYTHON "Build Python bindings"  OFF)

# ── Зависимости ────────────────────────────────────────────────────
find_package(hip     REQUIRED)   # lowercase — Linux case-sensitive!
find_package(rocprim REQUIRED)

include(cmake/fetch_deps.cmake)
fetch_dsp_core()

# ── Библиотека ─────────────────────────────────────────────────────
add_library(DspStats STATIC)
add_library(DspStats::DspStats ALIAS DspStats)

target_sources(DspStats PRIVATE
  src/statistics/src/statistics_processor.cpp
  src/statistics/src/statistics_sort_gpu.hip
)

target_include_directories(DspStats
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    src/statistics/include
    kernels/
    src
)
target_link_libraries(DspStats PUBLIC DspCore::DspCore roc::rocprim)
target_compile_features(DspStats PUBLIC cxx_std_17)

# ── Install + Export ────────────────────────────────────────────────
include(GNUInstallDirs)
install(TARGETS DspStats EXPORT DspStatsTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

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
