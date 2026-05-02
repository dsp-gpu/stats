#pragma once

// ============================================================================
// test_helpers_rocm — хелперы для ROCm-тестов (hipMallocManaged)
//
// ЧТО:    Утилиты для работы с unified memory (hipMallocManaged):
//         CPU заполняет данные, GPU читает без hipMemcpy.
// ЗАЧЕМ:  Упрощает setup маленьких тестовых датасетов — без явного H2D copy.
// ПОЧЕМУ: Только для малых датасетов (unified memory медленнее hipMalloc+Memcpy).
//         Caller владеет указателем — обязан вызвать hipFree().
//
// История: Создан: 2026-04-12
// ============================================================================

/**
 * @file test_helpers_rocm.hpp
 * @brief Хелперы для ROCm-тестов на базе hipMallocManaged (unified memory).
 * @note Test fixture, не публичный API. ROCm-only.
 *       Unified memory: CPU заполняет, GPU читает без hipMemcpy. Только для малых датасетов.
 *       Caller владеет указателем — обязан вызвать hipFree() после использования.
 */

#if ENABLE_ROCM

#include <core/interface/input_data.hpp>

#include <hip/hip_runtime.h>

#include <complex>
#include <stdexcept>
#include <cstdint>
#include <cstddef>

namespace test_helpers_rocm {

/// Allocate unified memory (CPU fill + GPU read without hipMemcpy). Caller must hipFree.
inline void* AllocateManagedForTest(size_t bytes) {
    void* ptr = nullptr;
    hipError_t e = hipMallocManaged(&ptr, bytes);
    if (e != hipSuccess) {
        throw std::runtime_error("AllocateManagedForTest: hipMallocManaged failed: " +
                                  std::string(hipGetErrorString(e)));
    }
    return ptr;
}

/// Build InputData<void*> for complex<float> managed buffer (beam_count * n_point complex)
inline drv_gpu_lib::InputData<void*> MakeManagedInput(
    void* ptr, uint32_t beam_count, uint32_t n_point)
{
    drv_gpu_lib::InputData<void*> out;
    out.data = ptr;
    out.antenna_count = beam_count;
    out.n_point = n_point;
    out.gpu_memory_bytes = static_cast<size_t>(beam_count) * n_point * sizeof(std::complex<float>);
    return out;
}

/// Build InputData<void*> for float magnitudes managed buffer (beam_count * n_point float)
inline drv_gpu_lib::InputData<void*> MakeManagedMagnitudeInput(
    void* ptr, uint32_t beam_count, uint32_t n_point)
{
    drv_gpu_lib::InputData<void*> out;
    out.data = ptr;
    out.antenna_count = beam_count;
    out.n_point = n_point;
    out.gpu_memory_bytes = static_cast<size_t>(beam_count) * n_point * sizeof(float);
    return out;
}

}  // namespace test_helpers_rocm

#endif  // ENABLE_ROCM
