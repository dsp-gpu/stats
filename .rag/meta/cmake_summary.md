<!-- type:meta_cmake_specific repo:stats inherits:dsp_gpu__root__meta_cmake_common__v1 -->

# CMake Specific — stats

```yaml
inherits: dsp_gpu__root__meta_cmake_common__v1
specific_only: true
target: DspStats
description: "GPU statistics: Welford, medians, radix sort"
adds_find_package: [hip, rocprim]
adds_links: [DspCore::DspCore, DspSpectrum::DspSpectrum, roc::rocprim]
```

## Project

- **Target**: `DspStats`
- **Описание**: GPU statistics: Welford, medians, radix sort

## Уникальные find_package

```cmake
find_package(hip REQUIRED)
find_package(rocprim REQUIRED)
```

## Линкуемые библиотеки

```cmake
target_link_libraries(DspStats PUBLIC
  DspCore::DspCore
  DspSpectrum::DspSpectrum
  roc::rocprim
)
```

## Исходники (2 файлов)

```cmake
target_sources(DspStats PRIVATE
  src/statistics/statistics_processor.cpp
  src/statistics/statistics_sort_gpu.hip
)
```

## Прочие специфичные строки (9)

```cmake
<TARGET>::<TARGET>
DESCRIPTION "GPU statistics: Welford, medians, radix sort"
fetch_dsp_spectrum()
find_package(hip     REQUIRED)
find_package(rocprim REQUIRED)
roc::rocprim)
src/statistics/statistics_processor.cpp
src/statistics/statistics_sort_gpu.hip
target_link_libraries(<TARGET> PUBLIC
```

