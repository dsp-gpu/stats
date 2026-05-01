#pragma once

// ============================================================================
// MedianHistogramOp — точная медиана 4-проходным byte-histogram (float input)
//                     (Layer 5 Ref03)
//
// ЧТО:    Concrete Op (наследник GpuKernelOp): считает median(|z|) per beam
//         через 4 прохода byte-wise гистограммы по IEEE-754 float-битам.
//         На каждом проходе: histogram (256 bins) → find_median_bucket
//         сужает диапазон по одному байту (MSB → LSB). После 4 проходов
//         target_value содержит точный uint32-bitcast медианы; ConvertResultToFloat
//         делает unxor sign-bit и memcpy → float.
//
// ЗАЧЕМ:  Для крупных beam'ов (n_point > 100K) sort O(n log n) проигрывает
//         histogram O(4n). MedianRadixSortOp на 1M элементов выделяет ~8MB
//         temp + sort всех элементов; histogram выделяет beam×256×4 байт
//         (256 KB на beam_count=256) и читает данные 4 раза без перестановок.
//         В StatisticsProcessor auto-select: n_point > kHistogramThreshold
//         (=100'000) → MedianHistogramOp, иначе MedianRadixSortOp.
//
// ПОЧЕМУ: - Layer 5 Ref03 (один Op = одна стратегия median, SRP).
//         - 4-pass byte histogram — известный алгоритм для exact median по
//           IEEE-754 float (sign-magnitude → ordered by byte). Работает
//           точно (без приближений), без перестановки данных.
//         - 1024 blocks_per_beam cap — баланс между параллелизмом и
//           overhead atomics на 256-bin global histogram.
//         - find_median_bucket: 1 thread per beam (1 block per beam) —
//           последовательно ищет bucket где cumulative count проходит
//           median_rank. Не bottleneck — beam_count типично ≤ 256.
//         - Sign-bit XOR (`u ^ 0x80000000`) — IEEE-754 хитрость:
//           перед битовым sort'ом флипаем sign, после — обратно. Делается
//           на host (beam_count небольшой) — проще чем kernel для post-process.
//         - BufferSet<3>: hist_buf (beam×256×u32), target_prefix, target_value
//           (по beam_count×u32). Lazy alloc через AllocatePrivateBuffers.
//
// Использование:
//   statistics::MedianHistogramOp mh;
//   mh.Initialize(ctx);
//   // kMagnitudes уже заполнен (compute_magnitudes или float Python upload)
//   mh.Execute(beam_count, n_point);
//   // kMediansCompact: float[beam_count] точные медианы
//
// История:
//   - Создан:  2026-03-14 (Ref03 Layer 5, путь histogram median для больших n)
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
#include <vector>
#include <cstring>
#include <algorithm>

namespace statistics {

/**
 * @class MedianHistogramOp
 * @brief Layer 5 Ref03 Op: точная медиана через 4-pass byte-histogram (float input).
 *
 * @note Lazy-alloc private buffers (BufferSet<3>: hist, prefix, value).
 * @note Требует #if ENABLE_ROCM. Зависит от kernels histogram_median_pass + find_median_bucket.
 * @note Эффективен для n_point > kHistogramThreshold (≈100K). Для меньших — MedianRadixSortOp.
 * @see statistics::MedianHistogramComplexOp — то же на complex-input (без отдельного |z|).
 * @see statistics::MedianRadixSortOp — альтернатива через rocPRIM sort (быстрее на малых n).
 */
class MedianHistogramOp : public drv_gpu_lib::GpuKernelOp {
public:
  const char* Name() const override { return "MedianHistogram"; }

  /**
   * @brief Выполнить histogram-based median на float-магнитудах.
   * @param beam_count Число beam'ов.
   * @param n_point    Сэмплов на beam.
   *
   * Читает kMagnitudes (float), пишет kMediansCompact (float[beam_count]).
   */
  void Execute(size_t beam_count, size_t n_point) {
    AllocatePrivateBuffers(beam_count);

    unsigned int bc = static_cast<unsigned int>(beam_count);
    unsigned int np = static_cast<unsigned int>(n_point);
    unsigned int median_rank = np / 2;

    unsigned int blocks_per_beam = std::min(
        static_cast<unsigned int>((n_point + kBlockSize - 1) / kBlockSize),
        1024u);

    void* data_ptr = ctx_->GetShared(shared_buf::kMagnitudes);
    void* hist_buf = bufs_.Get(kHist);
    void* prefix   = bufs_.Get(kPrefix);
    void* value    = bufs_.Get(kValue);

    // Zero out target buffers
    hipMemsetAsync(prefix, 0, beam_count * sizeof(unsigned int), stream());
    hipMemsetAsync(value,  0, beam_count * sizeof(unsigned int), stream());

    hipFunction_t hist_kernel = kernel("histogram_median_pass");
    hipFunction_t bucket_kernel = kernel("find_median_bucket");

    // 4 passes: byte 0 (MSB) through byte 3 (LSB)
    for (unsigned int pass = 0; pass < 4; ++pass) {
      // Clear histogram bins
      hipMemsetAsync(hist_buf, 0,
                     beam_count * 256 * sizeof(unsigned int), stream());

      // Histogram kernel: 2D grid (blocks_per_beam × beam_count)
      void* hist_args[] = { &data_ptr, &hist_buf, &np, &bc, &pass, &value };
      hipError_t err = hipModuleLaunchKernel(
          hist_kernel,
          blocks_per_beam, bc, 1,
          kBlockSize, 1, 1,
          0, stream(),
          hist_args, nullptr);
      if (err != hipSuccess) {
        throw std::runtime_error("MedianHistogramOp histogram pass " +
                                  std::to_string(pass) + ": " +
                                  hipGetErrorString(err));
      }

      // Find median bucket: 1 block per beam, 1 thread
      void* bucket_args[] = { &hist_buf, &prefix, &value, &median_rank, &pass };
      err = hipModuleLaunchKernel(
          bucket_kernel,
          bc, 1, 1, 1, 1, 1,
          0, stream(),
          bucket_args, nullptr);
      if (err != hipSuccess) {
        throw std::runtime_error("MedianHistogramOp bucket pass " +
                                  std::to_string(pass) + ": " +
                                  hipGetErrorString(err));
      }
    }

    // Convert uint32 → float on host (beam_count is small, typically ≤256)
    ConvertResultToFloat(beam_count);
  }

protected:
  void OnRelease() override { bufs_.ReleaseAll(); }

private:
  static constexpr unsigned int kBlockSize = 256;

  enum Buf { kHist, kPrefix, kValue, kBufCount };
  drv_gpu_lib::BufferSet<kBufCount> bufs_;

  void AllocatePrivateBuffers(size_t beam_count) {
    bufs_.Require(kHist,   beam_count * 256 * sizeof(unsigned int));
    bufs_.Require(kPrefix, beam_count * sizeof(unsigned int));
    bufs_.Require(kValue,  beam_count * sizeof(unsigned int));

    ctx_->RequireShared(shared_buf::kMediansCompact,
                        beam_count * sizeof(float));
  }

  void ConvertResultToFloat(size_t beam_count) {
    std::vector<unsigned int> target_host(beam_count);
    hipError_t err = hipMemcpyDtoH(target_host.data(), bufs_.Get(kValue),
                                    beam_count * sizeof(unsigned int));
    if (err != hipSuccess) {
      throw std::runtime_error("MedianHistogramOp: DtoH target_value failed");
    }

    std::vector<float> medians_host(beam_count);
    for (size_t b = 0; b < beam_count; ++b) {
      unsigned int u = target_host[b] ^ 0x80000000u;
      float val;
      std::memcpy(&val, &u, sizeof(float));
      medians_host[b] = val;
    }

    err = hipMemcpyHtoD(
        ctx_->GetShared(shared_buf::kMediansCompact),
        medians_host.data(),
        beam_count * sizeof(float));
    if (err != hipSuccess) {
      throw std::runtime_error("MedianHistogramOp: HtoD medians failed");
    }
  }
};

}  // namespace statistics

#endif  // ENABLE_ROCM
