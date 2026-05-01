#pragma once

/**
 * @file gather_decimated_kernel.hpp
 * @brief HIP kernel-source: gather_decimated (вырезка подвыборки для SNR-estimator).
 *
 * @note Тип B (technical header): R"HIP(...)HIP" source для hiprtc.
 *       Kernel `gather_decimated` (часть SNR_03 pipeline):
 *         - вход: complex<float> матрица [n_antennas × n_samples] (row-major)
 *         - выход: [n_ant_out × n_actual] с шагами step_antennas / step_samples
 *         - launch: grid(ceil(n_ant_out/64), 1, 1), block(64, 1, 1)
 *         - thread mapping: 1 thread = 1 output antenna, sequential loop по samples
 * @note ПОЧЕМУ thread-per-antenna (а не thread-per-element):
 *       при step_samples > 8 (stride > 64 байт = cache line) соседние потоки
 *       варпа читают из разных cache line'ов → ×32 amplification memory txns.
 *       Sequential loop внутри потока → L2 prefetcher идёт вдоль строки.
 * @note Source конкатенируется со statistics-main source (`GetStatisticsKernelSource()`),
 *       который уже определяет `struct float2_t` — здесь НЕ переопределяем.
 *
 * История:
 *   - Создан:  2026-04-09 (SNR_03)
 *   - Изменён: 2026-05-01 (унификация формата шапки под dsp-asst RAG-индексер)
 */

#if ENABLE_ROCM

namespace statistics {
namespace kernels {

/**
 * @brief HIP kernel source: gather_decimated
 *
 * Launch config (см. SnrEstimatorOp::ExecuteGather):
 *   grid(ceil(n_ant_out / 64), 1, 1)
 *   block(64, 1, 1)
 *
 * Функция `gather_decimated` (без суффикса `_kernel` — имя совпадает с
 * `kernel("gather_decimated")` в Op).
 *
 * @note NOT standalone — concatenates with statistics main source which
 *       already defines struct float2_t и BLOCK_SIZE. Мы используем явное
 *       `__launch_bounds__(64)` чтобы не зависеть от BLOCK_SIZE из main source.
 */
inline const char* GetGatherDecimatedKernelSource() {
    return R"HIP(

// ═══════════════════════════════════════════════════════════════
// Kernel: gather_decimated      (SNR_03)
// Gather subset [n_ant_out × n_samp_out] from [n_antennas × n_samples]
// with step_antennas / step_samples strides.
// Thread mapping: 1 thread = 1 antenna, sequential loop over samples.
// ═══════════════════════════════════════════════════════════════
__launch_bounds__(64)
extern "C" __global__ void gather_decimated(
    const float2_t* __restrict__ src,   // [n_antennas × n_samples], row-major
    float2_t*       __restrict__ dst,   // [n_ant_out × n_samp_out], row-major
    unsigned int n_samples,             // ширина исходной матрицы
    unsigned int n_samp_out,            // ширина выходной матрицы
    unsigned int step_antennas,         // шаг по строкам (antennas)
    unsigned int step_samples,          // шаг по столбцам (samples)
    unsigned int n_ant_out)             // число выходных антенн
{
    unsigned int ant = blockIdx.x * blockDim.x + threadIdx.x;
    if (ant >= n_ant_out) return;

    // size_t cast для защиты от переполнения при n_antennas * n_samples > 4G.
    const float2_t* src_row =
        src + (size_t)ant * (size_t)step_antennas * (size_t)n_samples;
    float2_t* dst_row =
        dst + (size_t)ant * (size_t)n_samp_out;

    // Sequential loop — L2 prefetcher видит линейный паттерн (stride const).
    for (unsigned int s = 0; s < n_samp_out; ++s) {
        dst_row[s] = src_row[(size_t)s * (size_t)step_samples];
    }
}

)HIP";
}

}  // namespace kernels
}  // namespace statistics

#endif  // ENABLE_ROCM
