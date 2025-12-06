import numpy as np
from execution_core import simulate_core_logic

def make_synthetic_data(n: int = 300):
    """Генерируем тестовые ряды без настоящего рынка.

    Важно только одно: первые `cut_idx` баров должны оставаться
    идентичными при любых манипуляциях с будущим.
    """
    rng = np.random.default_rng(42)

    # Базовый тренд + шум
    closes = 1.0 + np.cumsum(rng.normal(0, 0.001, size=n))
    opens = closes + rng.normal(0, 0.0003, size=n)

    # High / Low вокруг open/close
    highs = np.maximum(opens, closes) + np.abs(rng.normal(0, 0.0005, size=n))
    lows = np.minimum(opens, closes) - np.abs(rng.normal(0, 0.0005, size=n))

    # ATR условно постоянный
    atrs = np.full(n, 0.005, dtype=np.float64)

    # Идентификаторы дней (по 48 баров в день)
    day_ids = (np.arange(n) // 48).astype(np.int64)

    # Простая логика "сигнала" на основе локального тренда
    delta = np.diff(closes, prepend=closes[0])
    p_longs = (delta > 0).astype(np.float64) * 0.9
    p_shorts = (delta < 0).astype(np.float64) * 0.9

    # Один режим для простоты
    regimes = np.zeros(n, dtype=np.int64)

    return opens, highs, lows, closes, atrs, day_ids, p_longs, p_shorts, regimes


def run_core(opens, highs, lows, closes, atrs, day_ids, p_longs, p_shorts, regimes):
    """Обертка вокруг simulate_core_logic с фиксированными параметрами."""
    sl_mult = 2.0
    tp_mult = 4.0
    conf_threshold = 0.60
    vol_exit_mult = 4.0
    trail_on = 0.0
    trail_act_mult = 1.5
    trail_off_mult = 0.5
    max_hold_bars = 96.0
    pullback_mult = 0.5
    fill_wait_bars = 4
    abort_threshold = 0.8
    mode_sniper = 0
    commission = 0.0004
    deposit = 10_000.0
    risk_per_trade = 0.01

    equity, trades = simulate_core_logic(
        opens, highs, lows, closes, atrs, day_ids,
        p_longs, p_shorts, regimes,
        sl_mult, tp_mult, conf_threshold, vol_exit_mult,
        trail_on, trail_act_mult, trail_off_mult,
        max_hold_bars,
        pullback_mult, fill_wait_bars, abort_threshold,
        mode_sniper, commission, deposit, risk_per_trade
    )
    return equity, trades


def check_no_lookahead(cut_ratio: float = 0.5, atol: float = 1e-12):
    """Проверяет, зависит ли поведение ядра ДО cut_idx от будущего.

    Идея:
    1) Считаем траекторию equities / trades на исходных данных.
    2) Жестко перемешиваем ВСЕ бары после cut_idx (это "будущее").
    3) Снова прогоняем ядро.
    4) Если equity[0:cut_idx] и точки входа по сделкам до cut_idx совпадают,
       значит ядро не заглядывает за текущий бар.
    """
    # 1. Базовый прогон
    base = make_synthetic_data()
    equity_1, trades_1 = run_core(*base)

    n = len(equity_1)
    cut_idx = int(n * cut_ratio)

    # 2. Копия данных + перемешивание будущего
    pert = [arr.copy() for arr in base]
    (
        opens, highs, lows, closes,
        atrs, day_ids, p_longs, p_shorts, regimes
    ) = pert

    future_idx = np.arange(cut_idx + 1, n)
    shuffled = future_idx.copy()
    if len(shuffled) > 0:
        rng = np.random.default_rng(123)
        rng.shuffle(shuffled)

        # Перемешиваем все, что может влиять на сделки в будущем
        for arr in [opens, highs, lows, closes, atrs, day_ids, p_longs, p_shorts, regimes]:
            arr[future_idx] = arr[shuffled]

    equity_2, trades_2 = run_core(
        opens, highs, lows, closes, atrs, day_ids,
        p_longs, p_shorts, regimes
    )

    # 3. Сравнение equity до cut_idx
    if not np.allclose(equity_1[:cut_idx+1], equity_2[:cut_idx+1], atol=atol, rtol=0):
        raise AssertionError("Equity до cut_idx изменился после изменения будущего — возможен look-ahead bias в ядре.")

    # 4. Сравнение точек входа для сделок, открытых до cut_idx
    def entries(trades):
        out = []
        for row in trades:
            entry_i = int(row[0])
            if entry_i <= cut_idx:
                # округляем цену входа, т.к. это float
                entry_price = float(row[2])
                out.append((entry_i, round(entry_price, 8)))
        return sorted(out)

    e1 = entries(trades_1)
    e2 = entries(trades_2)

    if e1 != e2:
        raise AssertionError("Набор входов по сделкам до cut_idx изменился — ядро опирается на будущее при принятии решения.")

    print(f"✅ check_no_lookahead PASSED: ядро ведет себя одинаково до бара {cut_idx} при любых изменениях будущего.")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🧪 TEST: simulate_core_logic — проверка на заглядывание в будущее")
    print("="*60)
    try:
        check_no_lookahead()
        print("\n🎉 РЕЗУЛЬТАТ: look-ahead bias в ядре НЕ обнаружен.\n")
    except AssertionError as e:
        print("\n🛑 РЕЗУЛЬТАТ: возможна утечка будущего!\n")
        print(str(e))
