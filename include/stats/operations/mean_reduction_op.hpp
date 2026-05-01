#pragma once

// ============================================================================
// MeanReductionOp — иерархический complex mean per-beam (Layer 5 Ref03)
//
// ЧТО:    Concrete Op (наследник GpuKernelOp): два-фазная reduce-сумма
//         complex<float> по beam'у с делением на n. Phase 1 — block-level
//         partial sums (2D grid: blocks_per_beam × beam_count, double-load),
//         Phase 2 — финальная reduce одного блока на beam (warp shuffle,
//         деление на n_point). Результат: beam_count × float2 (re, im).
//
// ЗАЧЕМ:  Это первый Op фасадов ComputeMean / ComputeAll. Mean нужен:
//           - сам по себе (ComputeMean → MeanResult per beam),
//           - как промежуточная величина для CFAR-noise-floor (через Welford).
//         Иерархическая reduce — единственный быстрый путь для beam_count×
//         n_point > 256 (типично 256×4096): naive single-block упрётся в
//         shared memory; multi-block требует sync через global.
//
// ПОЧЕМУ: - Layer 5 Ref03 (одна Op = один логический «считай mean»),
//           Phase 1/2 — деталь реализации, не публичный API.
//         - Double-load (kDoubleLoadElements = 2×kBlockSize) — каждый поток
//           читает 2 элемента → ×2 occupancy memory pipeline + вдвое меньше
//           блоков → меньше overhead запуска.
//         - 2D grid (blocks_per_beam × beam_count): blockIdx.y = beam_id
//           убирает div/mod в kernel (P3-A optimization) — чистый
//           memory-bound throughput.
//         - Private BufferSet<1> (reduce_buf) — partial sums per (block, beam).
//           Размер ленив: только когда меняется beam_count × blocks_per_beam.
//         - Shared kResult выделяется размером WelfordResult (5 floats per
//           beam) — чтобы тот же буфер переиспользовать в WelfordFusedOp без
//           реаллокации (фасад вызывает Mean → потом Welford на kResult).
//
// Использование:
//   statistics::MeanReductionOp mean_op;
//   mean_op.Initialize(ctx);              // один раз
//   ctx.RequireShared(shared_buf::kInput, beam_count*n_point*sizeof(float)*2);
//   // upload в kInput
//   mean_op.Execute(beam_count, n_point); // out → kResult [bc × float2]
//
// История:
//   - Создан:  2026-03-14 (Ref03 Layer 5, выделено из StatisticsProcessor)
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
 * @class MeanReductionOp
 * @brief Layer 5 Ref03 Op: иерархический complex mean per-beam (Phase 1 + Phase 2).
 *
 * @note Stateless по семантике (private reduce_buf — кэш аллокации).
 * @note Требует #if ENABLE_ROCM. Зависит от kernels mean_reduce_phase1/_final.
 * @see drv_gpu_lib::GpuKernelOp — базовый Layer 3.
 * @see statistics::WelfordFusedOp — single-pass mean+var+std (без отдельного MeanOp).
 */
class MeanReductionOp : public drv_gpu_lib::GpuKernelOp {
public:
  const char* Name() const override { return "MeanReduction"; }

  /**
   * @brief Выполнить иерархическую complex mean reduce по всем beam'ам.
   * @param beam_count Число beam'ов.
   * @param n_point    Сэмплов на beam.
   *
   * Читает ctx_->GetShared(kInput), пишет ctx_->GetShared(kResult).
   * Результат: beam_count × float2 (re, im) в kResult.
   */
  void Execute(size_t beam_count, size_t n_point) {
    unsigned int bc = static_cast<unsigned int>(beam_count);
    unsigned int np = static_cast<unsigned int>(n_point);

    // Double-load: each block covers 2 × kBlockSize elements
    unsigned int blocks_per_beam = (np + kDoubleLoadElements - 1) / kDoubleLoadElements;

    // Allocate private reduce buffer
    size_t reduce_count = beam_count * blocks_per_beam;
    bufs_.Require(kReduce, reduce_count * 2 * sizeof(float));  // float2

    void* input_buf  = ctx_->GetShared(shared_buf::kInput);
    void* reduce_buf = bufs_.Get(kReduce);
    void* result_buf = ctx_->RequireShared(
        shared_buf::kResult,
        beam_count * 5 * sizeof(float));  // max: WelfordResult (5 floats)

    // Phase 1: block-level reduction (2D grid: blocks_per_beam × beam_count)
    void* args1[] = { &input_buf, &reduce_buf, &bc, &np };

    hipError_t err = hipModuleLaunchKernel(
        kernel("mean_reduce_phase1"),
        blocks_per_beam, bc, 1,
        kBlockSize, 1, 1,
        0, stream(),
        args1, nullptr);
    if (err != hipSuccess) {
      throw std::runtime_error("MeanReductionOp phase1: " +
                                std::string(hipGetErrorString(err)));
    }

    // Phase 2: final reduction (one block per beam)
    unsigned int final_block = 1;
    while (final_block < blocks_per_beam && final_block < kBlockSize) {
      final_block *= 2;
    }
    if (final_block > kBlockSize) final_block = kBlockSize;

    void* args2[] = { &reduce_buf, &result_buf, &bc, &blocks_per_beam, &np };

    err = hipModuleLaunchKernel(
        kernel("mean_reduce_final"),
        bc, 1, 1,
        final_block, 1, 1,
        0, stream(),
        args2, nullptr);
    if (err != hipSuccess) {
      throw std::runtime_error("MeanReductionOp final: " +
                                std::string(hipGetErrorString(err)));
    }
  }

protected:
  void OnRelease() override { bufs_.ReleaseAll(); }

private:
  static constexpr unsigned int kBlockSize = 256;
  static constexpr unsigned int kDoubleLoadElements = kBlockSize * 2;

  enum Buf { kReduce, kBufCount };
  drv_gpu_lib::BufferSet<kBufCount> bufs_;
};

}  // namespace statistics

#endif  // ENABLE_ROCM
