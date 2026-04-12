# DspStats

> Part of [dsp-gpu](https://github.com/dsp-gpu) organization.

Statistical DSP library: Welford mean/variance, median estimation, radix sort.

## Dependencies

- DspCore (core)
- ROCm: rocprim

## Build

```bash
cmake -S . -B build --preset local-dev
cmake --build build
```

## Contents

- `statistics/` — Welford fused mean/variance, median histograms, radix sort
