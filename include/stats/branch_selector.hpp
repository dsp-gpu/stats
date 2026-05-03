#pragma once

// ============================================================================
// BranchSelector — stateful hysteresis-классификатор SNR (Low / Mid / High)
//
// ЧТО:    Получает на вход измеренный SNR в dB (от StatisticsProcessor::
//         ComputeSnrDb) и возвращает категорию BranchType (Low/Mid/High).
//         Хранит текущее состояние между вызовами; переход в новую ветку
//         происходит ТОЛЬКО когда измерение пересекает порог + hysteresis_db.
//         NaN/Inf — не меняют state (один плохой фрейм не ломает каскад).
//
// ЗАЧЕМ:  Caller обрабатывает radar-фреймы потоком; SNR около границы
//         (например, ровно −15 dB) при каждом фрейме чуть-чуть колеблется
//         из-за шума. Без гистерезиса pipeline бы дёргался Low ↔ Mid каждый
//         фрейм → разные ветки processing'а, разрывы корреляции, потеря
//         трекинга. Hysteresis ±2 dB фиксирует состояние, переключение
//         только при заметном движении SNR.
//
// ПОЧЕМУ: - Отдельный класс (SOLID/SRP): facade `StatisticsProcessor`
//           остаётся stateless (ему всё равно как caller использует SNR);
//           state живёт здесь, по одному экземпляру на pipeline-поток.
//         - NOT thread-safe — намеренно: один экземпляр = один pipeline,
//           синхронизация на caller. Атомики/мьютексы — оверкилл для
//           одной enum-переменной в hot-loop'е.
//         - Header-only inline (методы короткие, switch + if-каскад):
//           компилятор инлайнит, нет cpp/линкера для одного класса.
//         - Защита от NaN/Inf через `std::isfinite` (не `>=`/`<=`):
//           NaN-сравнения не работают как ожидалось, а Inf может «протолкнуть»
//           ветку вверх через все пороги за один вызов.
//
// Использование:
//   statistics::BranchSelector sel;
//   while (running) {
//     auto r = proc.ComputeSnrDb(iq, n_ant, n_samp, cfg);
//     auto branch = sel.Select(r.snr_db_global, cfg.thresholds);
//     switch (branch) {
//       case BranchType::Low:  process_low(iq);  break;
//       case BranchType::Mid:  process_mid(iq);  break;
//       case BranchType::High: process_high(iq); break;
//     }
//   }
//
// История:
//   - Создан:  2026-04-09 (SNR_05: вынесено из StatisticsProcessor в SOLID-класс)
//   - Изменён: 2026-05-01 (унификация формата шапки под dsp-asst RAG-индексер)
// ============================================================================

#include <stats/statistics_types.hpp>

#include <cmath>  // std::isfinite

namespace statistics {

/**
 * @class BranchSelector
 * @brief Stateful hysteresis-классификатор SNR → Low/Mid/High с защитой от дребезга.
 *
 * @note NOT thread-safe — один экземпляр на поток/pipeline.
 * @note Header-only (inline методы); состояние = одна enum-переменная.
 * @note NaN/Inf оставляют ветку прежней (плохой фрейм не ломает каскад).
 * @see statistics::SnrEstimationResult — источник snr_db_global.
 * @see statistics::BranchThresholds — пороги (калибровано Python Эксп.5).
 *
 * Переходы (с защитой через hysteresis h):
 *   Low  → Mid:  snr > low_to_mid + h
 *   Mid  → Low:  snr < low_to_mid - h
 *   Mid  → High: snr > mid_to_high + h
 *   High → Mid:  snr < mid_to_high - h
 */
class BranchSelector {
public:
  BranchSelector() = default;

  /**
   * @brief Выбрать ветку с гистерезисом и обновить state.
   * @param snr_db  Измеренный SNR_db (от StatisticsProcessor::ComputeSnrDb).
   * @param thr     Пороги (из SnrEstimationConfig::thresholds).
   * @return Новая (или прежняя, если не пересекли границу) ветка.
   *   @test_check result == BranchType::{Low|Mid|High}; NaN/Inf оставляет prev state
   */
  BranchType Select(float snr_db, const BranchThresholds& thr) {
    // Защита от NaN/Inf: невалидное измерение → оставляем текущую ветку
    // (ломать переключение одним плохим фреймом нельзя).
    if (!std::isfinite(snr_db)) {
      return current_;
    }
    const float h = thr.hysteresis_db;
    switch (current_) {
      case BranchType::Low:
        if (snr_db > thr.low_to_mid_db + h) {
          current_ = BranchType::Mid;
        }
        break;
      case BranchType::Mid:
        if (snr_db > thr.mid_to_high_db + h) {
          current_ = BranchType::High;
        } else if (snr_db < thr.low_to_mid_db - h) {
          current_ = BranchType::Low;
        }
        break;
      case BranchType::High:
        if (snr_db < thr.mid_to_high_db - h) {
          current_ = BranchType::Mid;
        }
        break;
    }
    return current_;
  }

  /**
   * @brief Возвращает текущую ветку SNR без изменения внутреннего состояния.
   *
   * @return Текущее значение `current_` (Low/Mid/High).
   *   @test_check result == BranchType::{Low|Mid|High}
   */
  BranchType Current() const { return current_; }

  /**
   * @brief Сбрасывает state к указанной ветке (по умолчанию Low).
   *
   * @param to Целевая ветка для инициализации state.
   *   @test { values=[BranchType::Low, BranchType::Mid, BranchType::High] }
   */
  void Reset(BranchType to = BranchType::Low) { current_ = to; }

private:
  BranchType current_ = BranchType::Low;
};

}  // namespace statistics
