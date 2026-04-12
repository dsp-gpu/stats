# ── DspStats tests ─────────────────────────────────────────────────
add_executable(test_stats_main main.cpp)
target_link_libraries(test_stats_main PRIVATE DspStats::DspStats DspCore::TestUtils)
target_include_directories(test_stats_main PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(test_stats_main PRIVATE
  $<$<BOOL:${ENABLE_ROCM}>:ENABLE_ROCM=1>)
