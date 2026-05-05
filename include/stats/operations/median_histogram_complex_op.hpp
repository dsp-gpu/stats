#pragma once

// ============================================================================
// MedianHistogramComplexOp — точная медиана |z| через 4-pass histogram
//                            прямо из complex-входа (Layer 5 Ref03)
//
// ЧТО:    Concrete Op (наследник GpuKernelOp): то же что MedianHistogramOp,
//         но kernel `histogram_median_pass_complex` считает |z|=√(re²+im²)
//         on-the-fly внутри гистограммы. Промежуточный buffer магнитуд
//         (beam_count × n_point × float) НЕ выделяется.
//
// ЗАЧЕМ:  Когда вход — complex<float> и магнитуды НЕ нужны после median
//         (только сама медиана), эта Op экономит memory + одно kernel-launch
//         (compute_magnitudes пропускается). Используется в фасадном пути
//         ComputeMedian (без ComputeAll), где нет повторного использования
//         |z| для Welford. Для n_point > kHistogramThreshold (=100K).
//
// ПОЧЕМУ: - Layer 5 Ref03: SRP — отдельный Op для complex input, чтобы не
//           тащить условный switch is_complex внутри одного класса
//           (исходно MedianHistogramOp::ExecuteHistogramMedian с флагом).
//         - Тот же 4-pass byte-histogram алгоритм, что у MedianHistogramOp,
//           отличается только kernel'ом первого прохода (хелпер с complex).
//         - find_median_bucket — общий kernel (работает по uint32 bitcast,
//           не зависит от того, был ли вход real или complex).
//         - BufferSet<3> идентичен MedianHistogramOp: hist, prefix, value.
//
// Использование:
//   statistics::MedianHistogramComplexOp mhc;
//   mhc.Initialize(ctx);
//   // kInput содержит complex<float>[beam_count × n_point]
//   mhc.Execute(beam_count, n_point);
//   // kMediansCompact: float[beam_count] точные медианы |z|
//
// История:
//   - Создан:  2026-03-14 (Ref03 Layer 5, путь histogram median без |z| buffer)
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
 * @class MedianHistogramComplexOp
 * @brief Layer 5 Ref03 Op: точная медиана |z| через 4-pass histogram прямо из complex.
 *
 * @note Не аллоцирует промежуточный |z|-буфер — экономия памяти vs Magnitudes+Histogram.
 * @note Lazy-alloc private buffers (BufferSet<3>: hist, prefix, value).
 * @note Требует #if ENABLE_ROCM. Kernels: histogram_median_pass_complex + find_median_bucket.
 * @see statistics::MedianHistogramOp — аналог по уже-вычисленным float magnitudes.
 * @see statistics::MedianRadixSortOp — альтернатива через rocPRIM sort.
 */
class MedianHistogramComplexOp : public drv_gpu_lib::GpuKernelOp {
public:
  /**
   * @brief Возвращает имя Op'а для логирования и профилирования.
   *
   * @return C-строка "MedianHistogramComplex" (статический литерал).
   *   @test_check std::string(result) == "MedianHistogramComplex"
   */
  const char* Name() const override { return "MedianHistogramComplex"; }

  /**
   * @brief Выполнить histogram-based median на complex-входе (|z| on-the-fly).
   * @param beam_count Число beam'ов.
   *   @test { range=[1..50000], value=128, unit="лучей/каналов", error_values=[-1, 100000, 3.14] }
   * @param n_point    Сэмплов на beam.
   *   @test { range=[100..1300000], value=6000, error_values=[-1, 3000000, 3.14] }
   *
   * Читает kInput (complex<float>), пишет kMediansCompact (float[beam_count]).
   * |z| вычисляется внутри kernel'а гистограммы (отдельного буфера нет).
   * @throws std::runtime_error при сбое hipModuleLaunchKernel или DtoH/HtoD копирования.
   *   @test_check throws on hipModuleLaunchKernel != hipSuccess || hipMemcpy failure
   */
  void Execute(size_t beam_count, size_t n_point) {
    AllocatePrivateBuffers(beam_count);

    unsigned int bc = static_cast<unsigned int>(beam_count);
    unsigned int np = static_cast<unsigned int>(n_point);
    unsigned int median_rank = np / 2;

    unsigned int blocks_per_beam = std::min(
        static_cast<unsigned int>((n_point + kBlockSize - 1) / kBlockSize),
        1024u);

    void* data_ptr = ctx_->GetShared(shared_buf::kInput);
    void* hist_buf = bufs_.Get(kHist);
    void* prefix   = bufs_.Get(kPrefix);
    void* value    = bufs_.Get(kValue);

    // Zero out target buffers
    hipMemsetAsync(prefix, 0, beam_count * sizeof(unsigned int), stream());
    hipMemsetAsync(value,  0, beam_count * sizeof(unsigned int), stream());

    hipFunction_t hist_kernel = kernel("histogram_median_pass_complex");
    hipFunction_t bucket_kernel = kernel("find_median_bucket");

    for (unsigned int pass = 0; pass < 4; ++pass) {
      hipMemsetAsync(hist_buf, 0,
                     beam_count * 256 * sizeof(unsigned int), stream());

      void* hist_args[] = { &data_ptr, &hist_buf, &np, &bc, &pass, &value };
      hipError_t err = hipModuleLaunchKernel(
          hist_kernel,
          blocks_per_beam, bc, 1,
          kBlockSize, 1, 1,
          0, stream(),
          hist_args, nullptr);
      if (err != hipSuccess) {
        throw std::runtime_error("MedianHistogramComplexOp histogram pass " +
                                  std::to_string(pass) + ": " +
                                  hipGetErrorString(err));
      }

      void* bucket_args[] = { &hist_buf, &prefix, &value, &median_rank, &pass };
      err = hipModuleLaunchKernel(
          bucket_kernel,
          bc, 1, 1, 1, 1, 1,
          0, stream(),
          bucket_args, nullptr);
      if (err != hipSuccess) {
        throw std::runtime_error("MedianHistogramComplexOp bucket pass " +
                                  std::to_string(pass) + ": " +
                                  hipGetErrorString(err));
      }
    }

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
      throw std::runtime_error("MedianHistogramComplexOp: DtoH target_value failed");
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
      throw std::runtime_error("MedianHistogramComplexOp: HtoD medians failed");
    }
  }
};

}  // namespace statistics

#endif  // ENABLE_ROCM
