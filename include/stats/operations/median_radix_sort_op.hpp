#pragma once

// ============================================================================
// MedianRadixSortOp — медиана |z| через rocPRIM segmented radix sort
//                     (Layer 5 Ref03)
//
// ЧТО:    Concrete Op: универсальная медиана через полную segmented sort
//         всех элементов. Two execution paths:
//           - Execute(beam, n_point)      — complex input: magnitudes + sort + extract
//           - ExecuteFloat(beam, n_point) — float magnitudes готовы: только sort + extract
//         Pipeline: compute_magnitudes → gpu_sort::ExecuteSort (rocPRIM) →
//         extract_medians (берёт средний элемент каждого сегмента).
//
// ЗАЧЕМ:  Универсальная стратегия median, работает для любых n_point.
//         StatisticsProcessor использует её для n_point ≤ kHistogramThreshold
//         (=100K) — на малых данных rocPRIM segmented_radix_sort быстрее
//         чем 4 прохода histogram (меньше overhead, лучше cache locality).
//         Также используется внутри SnrEstimatorOp для медианы SNR по
//         антеннам после CFAR (n_ant_out обычно ≤ 50, sort оптимален).
//
// ПОЧЕМУ: - Layer 5 Ref03: третья стратегия median, отдельный Op (SRP).
//           Все 3 стратегии (Histogram / HistogramComplex / RadixSort) имеют
//           одинаковый contract — пишут kMediansCompact, фасад выбирает
//           без if'ов в hot-path.
//         - rocPRIM (а не CUB / thrust) — единственный разрешённый sort
//           под ROCm (см. rule 09-rocm-only). Segmented variant сортирует
//           все beam'ы за ОДИН device-call.
//         - Two-step alloc (Query → Allocate temp) — обязательная семантика
//           rocPRIM: размер temp storage runtime-зависим.
//         - Buffer reuse: AllocatePrivateBuffers рано-возвращается, если
//           beam_count×n_point не изменились (Hot-loop без re-alloc).
//         - extract_medians — отдельный kernel вместо host-side: даже один
//           hipMemcpyDtoH per beam'у дороже, чем launch одного kernel
//           (P0-B optimization из statistics_kernels_rocm.hpp).
//         - Offsets uploaded async — простой stride n_point, можно было
//           использовать iterator, но явные offsets гибче для будущих
//           variable-length сегментов.
//
// Использование:
//   statistics::MedianRadixSortOp ms;
//   ms.Initialize(ctx);
//   // Complex путь:
//   ms.Execute(beam_count, n_point);
//   // Float путь (kMagnitudes уже заполнен):
//   ms.ExecuteFloat(beam_count, n_point);
//
// История:
//   - Создан:  2026-03-14 (Ref03 Layer 5, объединил 3 helper'а в один Op)
//   - Изменён: 2026-05-01 (унификация формата шапки под dsp-asst RAG-индексер)
// ============================================================================

#if ENABLE_ROCM

#include <core/services/gpu_kernel_op.hpp>
#include <core/services/buffer_set.hpp>
#include <core/interface/gpu_context.hpp>
#include <stats/statistics_types.hpp>
#include <stats/statistics_sort_gpu.hpp>

#include <hip/hip_runtime.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace statistics {

/**
 * @class MedianRadixSortOp
 * @brief Layer 5 Ref03 Op: медиана |z| через rocPRIM segmented radix sort (универсальный путь).
 *
 * @note Lazy + cached buffer allocation (BufferSet<3>: sort_buf, sort_temp, offsets).
 * @note Требует #if ENABLE_ROCM. Зависит от rocPRIM (через gpu_sort::ExecuteSort).
 * @note Эффективен для n_point ≤ kHistogramThreshold (≈100K). Для больших — Histogram-варианты.
 * @note Two paths: Execute (complex input) + ExecuteFloat (магнитуды готовы).
 * @see statistics::gpu_sort::ExecuteSort — rocPRIM-обёртка (компилируется hipcc).
 * @see statistics::MedianHistogramOp / MedianHistogramComplexOp — альтернатива для больших n.
 */
class MedianRadixSortOp : public drv_gpu_lib::GpuKernelOp {
public:
  const char* Name() const override { return "MedianRadixSort"; }

  /**
   * @brief Выполнить radix-sort медиану (complex input — full pipeline).
   * @param beam_count Число beam'ов.
   * @param n_point    Сэмплов на beam.
   *
   * Pipeline: magnitudes → sort → extract_medians.
   * Читает kInput, пишет kMediansCompact (float[beam_count]).
   */
  void Execute(size_t beam_count, size_t n_point) {
    AllocatePrivateBuffers(beam_count, n_point);

    size_t total = beam_count * n_point;
    ExecuteMagnitudes(total);
    ExecuteSort(beam_count, n_point, total);
    ExecuteExtractMedians(beam_count, n_point);
  }

  /**
   * @brief Выполнить radix-sort медиану по уже-вычисленным float magnitudes.
   * @param beam_count Число beam'ов.
   * @param n_point    Сэмплов на beam.
   *
   * Без compute_magnitudes. Читает kMagnitudes, пишет kMediansCompact.
   */
  void ExecuteFloat(size_t beam_count, size_t n_point) {
    AllocatePrivateBuffers(beam_count, n_point);

    size_t total = beam_count * n_point;
    ExecuteSort(beam_count, n_point, total);
    ExecuteExtractMedians(beam_count, n_point);
  }

protected:
  void OnRelease() override {
    bufs_.ReleaseAll();
    sort_temp_size_ = 0;
    current_beams_ = 0;
    current_n_point_ = 0;
  }

private:
  static constexpr unsigned int kBlockSize = 256;

  enum Buf { kSortBuf, kSortTemp, kOffsets, kBufCount };
  drv_gpu_lib::BufferSet<kBufCount> bufs_;

  size_t sort_temp_size_ = 0;
  size_t current_beams_ = 0;
  size_t current_n_point_ = 0;

  void AllocatePrivateBuffers(size_t beam_count, size_t n_point) {
    if (beam_count == current_beams_ && n_point == current_n_point_
        && bufs_.Get(kSortBuf)) {
      return;  // already allocated
    }

    size_t total = beam_count * n_point;

    // Sort output buffer
    bufs_.Require(kSortBuf, total * sizeof(float));

    // Offsets buffer + async upload
    bufs_.Require(kOffsets, (beam_count + 1) * sizeof(unsigned int));
    {
      std::vector<unsigned int> host_offsets(beam_count + 1);
      for (size_t i = 0; i <= beam_count; ++i) {
        host_offsets[i] = static_cast<unsigned int>(i * n_point);
      }
      hipError_t err = hipMemcpyAsync(
          bufs_.Get(kOffsets), host_offsets.data(),
          (beam_count + 1) * sizeof(unsigned int),
          hipMemcpyHostToDevice, stream());
      if (err != hipSuccess) {
        throw std::runtime_error("MedianRadixSortOp: offsets upload failed");
      }
    }

    // Query sort temp size + allocate
    {
      auto* d_offsets = static_cast<const unsigned int*>(bufs_.Get(kOffsets));
      hipError_t err = gpu_sort::QuerySortTempSize(
          sort_temp_size_, d_offsets, d_offsets + 1,
          static_cast<unsigned int>(total),
          static_cast<unsigned int>(beam_count),
          stream());
      if (err != hipSuccess) {
        throw std::runtime_error("MedianRadixSortOp: QuerySortTempSize failed");
      }
      if (sort_temp_size_ > 0) {
        bufs_.Require(kSortTemp, sort_temp_size_);
      }
    }

    // Ensure shared magnitudes buffer exists
    ctx_->RequireShared(shared_buf::kMagnitudes,
                        total * sizeof(float));
    // Ensure shared medians_compact buffer exists
    ctx_->RequireShared(shared_buf::kMediansCompact,
                        beam_count * sizeof(float));

    current_beams_ = beam_count;
    current_n_point_ = n_point;
  }

  void ExecuteMagnitudes(size_t total_elements) {
    unsigned int total = static_cast<unsigned int>(total_elements);
    unsigned int grid = (total + kBlockSize * 2 - 1) / (kBlockSize * 2);

    void* input_buf = ctx_->GetShared(shared_buf::kInput);
    void* mag_buf   = ctx_->GetShared(shared_buf::kMagnitudes);

    void* args[] = { &input_buf, &mag_buf, &total };

    hipError_t err = hipModuleLaunchKernel(
        kernel("compute_magnitudes"),
        grid, 1, 1, kBlockSize, 1, 1,
        0, stream(), args, nullptr);
    if (err != hipSuccess) {
      throw std::runtime_error("MedianRadixSortOp magnitudes: " +
                                std::string(hipGetErrorString(err)));
    }
  }

  void ExecuteSort(size_t beam_count, size_t n_point, size_t total) {
    auto* d_offsets = static_cast<const unsigned int*>(bufs_.Get(kOffsets));
    auto* mag_buf   = static_cast<const float*>(
        ctx_->GetShared(shared_buf::kMagnitudes));
    auto* sort_buf  = static_cast<float*>(bufs_.Get(kSortBuf));

    hipError_t err = gpu_sort::ExecuteSort(
        bufs_.Get(kSortTemp), sort_temp_size_,
        mag_buf, sort_buf,
        d_offsets, d_offsets + 1,
        static_cast<unsigned int>(total),
        static_cast<unsigned int>(beam_count),
        stream());
    if (err != hipSuccess) {
      throw std::runtime_error("MedianRadixSortOp sort: " +
                                std::string(hipGetErrorString(err)));
    }
  }

  void ExecuteExtractMedians(size_t beam_count, size_t n_point) {
    unsigned int bc = static_cast<unsigned int>(beam_count);
    unsigned int np = static_cast<unsigned int>(n_point);
    unsigned int grid = (bc + kBlockSize - 1) / kBlockSize;

    void* sort_buf    = bufs_.Get(kSortBuf);
    void* medians_buf = ctx_->GetShared(shared_buf::kMediansCompact);

    void* args[] = { &sort_buf, &medians_buf, &np, &bc };

    hipError_t err = hipModuleLaunchKernel(
        kernel("extract_medians"),
        grid, 1, 1, kBlockSize, 1, 1,
        0, stream(), args, nullptr);
    if (err != hipSuccess) {
      throw std::runtime_error("MedianRadixSortOp extract: " +
                                std::string(hipGetErrorString(err)));
    }
  }
};

}  // namespace statistics

#endif  // ENABLE_ROCM
