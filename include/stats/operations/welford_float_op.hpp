#pragma once

// ============================================================================
// WelfordFloatOp — Welford-статистика по уже-вычисленным float magnitudes
//                  (Layer 5 Ref03)
//
// ЧТО:    Concrete Op (наследник GpuKernelOp): считает mean(|z|), variance(|z|),
//         std(|z|) по float-входу (магнитуды уже вычислены другим этапом).
//         3 LDS-массива × (kBlockSize+1) floats: mean_mag, M2, count.
//         Результат — те же 5 floats per beam, что у WelfordFusedOp,
//         но mean_re=mean_im=0 (комплексной части нет).
//
// ЗАЧЕМ:  Используется когда вход — уже подготовленные магнитуды (например,
//         после ComputeMagnitudesOp в spectrum, или напрямую из Python через
//         vector<float>). Путь ComputeStatisticsFloat / ComputeAllFloat
//         в StatisticsProcessor. Также — стадия post-CFAR обработки в SNR
//         pipeline (где |X|² уже посчитано).
//
// ПОЧЕМУ: - Layer 5 Ref03: отдельный Op а не флаг в WelfordFusedOp — SRP
//           (разный input layout, разный kernel, разная shared memory).
//         - LDS +1 padding (P3-B) — устранение bank conflicts при tree
//           reduction. 3 массива (vs 5 у fused) — нет complex sum_re/sum_im.
//         - Один блок на beam (`bc` × 1 × 1 grid) — та же модель что у
//           WelfordFusedOp; верхняя оценка n_point ≤ 64K.
//         - BufferSet<0> — Op stateless, all temp memory в LDS.
//
// Использование:
//   statistics::WelfordFloatOp wf;
//   wf.Initialize(ctx);
//   // kMagnitudes уже заполнен (магнитуды compute или Python upload)
//   wf.Execute(beam_count, n_point);
//   // kResult: beam_count × {0, 0, mean_mag, var, std}
//
// История:
//   - Создан:  2026-03-14 (Ref03 Layer 5, путь ComputeStatisticsFloat)
//   - Изменён: 2026-05-01 (унификация формата шапки под dsp-asst RAG-индексер)
// ============================================================================

#if ENABLE_ROCM

#include <core/services/gpu_kernel_op.hpp>
#include <core/services/buffer_set.hpp>
#include <core/interface/gpu_context.hpp>
#include <stats/statistics_types.hpp>

#include <hip/hip_runtime.h>
#include <stdexcept>
#include <string>

namespace statistics {

/**
 * @class WelfordFloatOp
 * @brief Layer 5 Ref03 Op: Welford по уже-вычисленным float magnitudes.
 *
 * @note Stateless (BufferSet<0>, all temp в LDS).
 * @note Требует #if ENABLE_ROCM. Зависит от kernel `welford_float`.
 * @note mean_re/mean_im в выходе ВСЕГДА 0 (вход не комплексный).
 * @see statistics::WelfordFusedOp — аналог по complex-входу (single-pass).
 */
class WelfordFloatOp : public drv_gpu_lib::GpuKernelOp {
public:
  /**
   * @brief Возвращает имя Op'а для логирования и профилирования.
   *
   * @return C-строка "WelfordFloat" (статический литерал).
   *   @test_check std::string(result) == "WelfordFloat"
   */
  const char* Name() const override { return "WelfordFloat"; }

  /**
   * @brief Выполнить Welford по float-магнитудам.
   * @param beam_count Число beam'ов.
   *   @test { range=[1..50000], value=128, unit="лучей/каналов" }
   * @param n_point    Сэмплов на beam.
   *   @test { range=[100..1300000], value=6000 }
   *
   * Читает kMagnitudes (float), пишет kResult (5 floats per beam:
   * mean_re=0, mean_im=0, mean_mag, variance, std_dev).
   * @throws std::runtime_error при сбое hipModuleLaunchKernel("welford_float").
   *   @test_check throws on hipModuleLaunchKernel != hipSuccess
   */
  void Execute(size_t beam_count, size_t n_point) {
    unsigned int bc = static_cast<unsigned int>(beam_count);
    unsigned int np = static_cast<unsigned int>(n_point);

    void* magnitudes_buf = ctx_->GetShared(shared_buf::kMagnitudes);
    void* result_buf = ctx_->RequireShared(
        shared_buf::kResult,
        beam_count * 5 * sizeof(float));

    // 3 shared arrays × (kBlockSize+1) floats: mean_mag, M2, count
    size_t shared_mem = 3 * (kBlockSize + 1) * sizeof(float);

    void* args[] = { &magnitudes_buf, &result_buf, &bc, &np };

    hipError_t err = hipModuleLaunchKernel(
        kernel("welford_float"),
        bc, 1, 1,
        kBlockSize, 1, 1,
        shared_mem, stream(),
        args, nullptr);

    if (err != hipSuccess) {
      throw std::runtime_error("WelfordFloatOp: " +
                                std::string(hipGetErrorString(err)));
    }
  }

private:
  static constexpr unsigned int kBlockSize = 256;
  drv_gpu_lib::BufferSet<0> bufs_;
};

}  // namespace statistics

#endif  // ENABLE_ROCM
