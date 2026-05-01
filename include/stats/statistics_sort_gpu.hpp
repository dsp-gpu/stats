#pragma once

// ============================================================================
// statistics::gpu_sort — обёртка над rocPRIM segmented radix sort
//
// ЧТO:    Две функции namespace-уровня:
//           - QuerySortTempSize  — узнать сколько байт temp storage требует sort
//           - ExecuteSort        — выполнить параллельный per-segment sort
//         Сегмент = beam (один beam_count = num_segments). Все 256 beam'ов
//         сортируются за ОДИН GPU-вызов (rocprim::segmented_radix_sort_keys).
//         Реализация — в statistics_sort_gpu.hip (внутри hipcc unit).
//
// ЗАЧЕМ:  Радикс-сортировка нужна `MedianRadixSortOp` для медианы по магнитудам
//         (брать средний элемент после sort'а сегмента). rocPRIM требует hipcc-
//         компиляцию (из-за rocprim/rocprim.hpp), поэтому объявления вынесены
//         в обычный header — `.cpp` (g++) видит только сигнатуры, реализация
//         компилируется hipcc отдельно. Это разрывает rocPRIM-зависимость для
//         основной части `statistics`.
//
// ПОЧЕМУ: - Two-step API (Query → Allocate → Execute) — стандарт rocPRIM:
//           размер temp storage зависит от числа сегментов и алгоритма,
//           узнать его можно только runtime-вызовом с null storage.
//         - Сортировка float (а не uint32 + bit-reinterpret) — rocPRIM
//           корректно обрабатывает float ordering (sign-bit flip встроен).
//         - segmented (а не batched) — beam'ы могут иметь разные смещения
//           (хотя сейчас все равны n_point); offsets-массив универсальнее.
//
// Использование:
//   size_t tmp_bytes = 0;
//   gpu_sort::QuerySortTempSize(tmp_bytes, d_begin, d_end, total, beams, str);
//   void* tmp = ...; // выделить tmp_bytes на device
//   gpu_sort::ExecuteSort(tmp, tmp_bytes, mag_in, mag_out,
//                          d_begin, d_end, total, beams, str);
//
// История:
//   - Создан:  2026-02-23
//   - Изменён: 2026-05-01 (унификация формата шапки под dsp-asst RAG-индексер)
// ============================================================================

#if ENABLE_ROCM

#include <hip/hip_runtime.h>
#include <cstddef>

namespace statistics {
namespace gpu_sort {

/**
 * @brief Узнать размер temp storage для segmented radix sort (Query-фаза rocPRIM).
 *
 * @param[out] temp_size    Требуемый размер temp storage (байт).
 * @param d_begin_offsets   Device ptr: начальные смещения per-segment [beam_count].
 * @param d_end_offsets     Device ptr: конечные смещения per-segment   [beam_count].
 * @param total_elements    beam_count × n_point.
 * @param num_segments      beam_count.
 * @param stream            HIP stream.
 */
hipError_t QuerySortTempSize(
    size_t&               temp_size,
    const unsigned int*   d_begin_offsets,
    const unsigned int*   d_end_offsets,
    unsigned int          total_elements,
    unsigned int          num_segments,
    hipStream_t           stream);

/**
 * @brief Выполнить segmented radix sort (по возрастанию) на float-магнитудах.
 *
 * Каждый сегмент (beam) сортируется независимо и параллельно.
 * Все beam'ы (типично 256) — за ОДИН GPU-вызов rocPRIM.
 *
 * @param temp_storage      Pre-allocated temp buffer (получен через QuerySortTempSize).
 * @param temp_size         Размер temp_storage (байт).
 * @param keys_in           Device ptr: исходные float-магнитуды [total_elements].
 * @param keys_out          Device ptr: отсортированный вывод    [total_elements].
 * @param d_begin_offsets   Device ptr: начальные смещения [num_segments].
 * @param d_end_offsets     Device ptr: конечные смещения   [num_segments].
 * @param total_elements    beam_count × n_point.
 * @param num_segments      beam_count.
 * @param stream            HIP stream.
 */
hipError_t ExecuteSort(
    void*                 temp_storage,
    size_t                temp_size,
    const float*          keys_in,
    float*                keys_out,
    const unsigned int*   d_begin_offsets,
    const unsigned int*   d_end_offsets,
    unsigned int          total_elements,
    unsigned int          num_segments,
    hipStream_t           stream);

}  // namespace gpu_sort
}  // namespace statistics

#endif  // ENABLE_ROCM
