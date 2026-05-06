<!-- type:meta_targets repo:stats source:stats/CMakeLists.txt -->

# Build Targets — stats

## Targets

- **`DspStats`** (library)
  - PUBLIC: `DspCore::DspCore`, `DspSpectrum::DspSpectrum`, `roc::rocprim`

## BUILD-флаги (option)

- `DSP_STATS_BUILD_TESTS` (default `ON`) — Build tests
- `DSP_STATS_BUILD_PYTHON` (default `OFF`) — Build Python bindings

## Зависимости от DSP репо

- `core` — через `fetch_dsp_core()`
- `spectrum` — через `fetch_dsp_spectrum()`

## External find_package

- `hip` (required)
- `rocprim` (required)
