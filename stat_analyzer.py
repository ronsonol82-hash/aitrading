# stat_analyzer.py
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime, timedelta

from data_loader import DataLoader
from indicators import FeatureEngineer
from config import Config


def calculate_probabilities(df, sl_range, tp_range, look_ahead=100):
    """
    Математический расчет вероятности касания TP раньше SL
    на основе чистой волатильности (ATR).
    """
    print("🧮 Calculating Probability Surface...")

    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    atrs = df["atr"].values

    results = {}

    total_iterations = len(sl_range) * len(tp_range)
    with tqdm(total=total_iterations) as pbar:
        for sl_mult in sl_range:
            for tp_mult in tp_range:
                wins = 0
                losses = 0

                # шаг 10 баров, чтобы не умереть по времени
                for i in range(100, len(closes) - look_ahead, 10):
                    entry = closes[i]
                    vol = atrs[i]
                    if vol == 0:
                        continue

                    sl_price_long = entry - (vol * sl_mult)
                    tp_price_long = entry + (vol * tp_mult)

                    outcome = 0  # 0=TimeOut, 1=TP, -1=SL
                    for j in range(1, look_ahead):
                        idx = i + j
                        if idx >= len(closes):
                            break
                        if lows[idx] <= sl_price_long:
                            outcome = -1
                            break
                        if highs[idx] >= tp_price_long:
                            outcome = 1
                            break

                    if outcome == 1:
                        wins += 1
                    elif outcome == -1:
                        losses += 1

                total = wins + losses
                if total > 0:
                    win_rate = wins / total
                    rr_ratio = tp_mult / sl_mult
                    expected_value = (win_rate * rr_ratio) - (1 - win_rate)

                    results[(sl_mult, tp_mult)] = {
                        "win_rate": win_rate,
                        "ev": expected_value,
                        "rr": rr_ratio,
                    }
                pbar.update(1)

    return results


def _infer_lookahead(timeframe: str, days: int = 10) -> int:
    """
    Подбираем горизонт прогноза по таймфрейму:
    примерно `days` календарных дней.
    """
    per_day = {
        "15m": 96,
        "30m": 48,
        "1h": 24,
        "2h": 12,
        "4h": 6,
        "1d": 1,
    }.get(timeframe, 24)
    return max(20, per_day * days)


def run_analysis():
    # 1. Загрузка данных по ТЕКУЩЕМУ конфигу
    print("📥 Loading Data for Analysis (Config-aware)...")

    symbol = Config.LEADER_SYMBOL
    tf = Config.TIMEFRAME_LTF

    end = datetime.utcnow()
    start = end - timedelta(days=365)  # год истории для статистики

    df = DataLoader.get_binance_data(symbol, pd.to_datetime(start), pd.to_datetime(end), tf)
    if df is None or df.empty:
        print("❌ Не удалось загрузить данные. Проверь соединение/DataLoader.")
        return

    # Добавляем фичи, чтобы atr считался так же, как в основном пайплайне
    df = FeatureEngineer.add_features(df)
    if "atr" not in df.columns:
        print("❌ В df нет ATR даже после FeatureEngineer. Проверь indicators/features_lib.")
        return

    print(f"   Symbol: {symbol}, TF: {tf}, Period: {start.date()} → {end.date()} (bars={len(df)})")

    # 2. Сетка SL/TP вокруг текущей стратегии
    strat = Config.get_strategy_params()
    base_sl = float(strat.get("sl", Config.DEFAULT_STRATEGY.get("sl", 2.0)))
    base_tp = float(strat.get("tp", Config.DEFAULT_STRATEGY.get("tp", 4.0)))

    sl_min = max(0.5, base_sl * 0.75)
    sl_max = base_sl * 2.0
    tp_min = max(base_sl * 1.1, base_tp * 0.6)
    tp_max = base_tp * 2.2

    sl_grid = np.round(np.arange(sl_min, sl_max + 1e-9, 0.25), 2)
    tp_grid = np.round(np.arange(tp_min, tp_max + 1e-9, 0.5), 2)

    look_ahead = _infer_lookahead(tf, days=10)

    print("\n🧩 SAMPLING GRID:")
    print(f"   Base SL/TP from Config:  SL={base_sl:.2f} ATR | TP={base_tp:.2f} ATR")
    print(f"   SL grid: {sl_grid[0]:.2f} .. {sl_grid[-1]:.2f} (step {sl_grid[1]-sl_grid[0]:.2f})")
    print(f"   TP grid: {tp_grid[0]:.2f} .. {tp_grid[-1]:.2f} (step {tp_grid[1]-tp_grid[0]:.2f})")
    print(f"   Look-ahead horizon: {look_ahead} bars.\n")

    # 3. Расчет статистики
    stats = calculate_probabilities(df, sl_grid, tp_grid, look_ahead=look_ahead)
    if not stats:
        print("❌ Недостаточно данных для статистики.")
        return

    # 4. Аггрегация
    best_ev = -999.0
    best_ev_key = None
    best_wr = 0.0
    best_wr_key = None

    sorted_stats = sorted(stats.items(), key=lambda x: x[1]["ev"], reverse=True)

    print("📊 STATISTICAL RESULTS (Top 10 Configs):")
    print(f"{'SL (ATR)':<9} | {'TP (ATR)':<9} | {'WinRate':<10} | {'R:R':<7} | {'Exp.Value':<10}")
    print("-" * 60)

    for (sl, tp), m in sorted_stats[:10]:
        if m["ev"] > best_ev:
            best_ev = m["ev"]
            best_ev_key = (sl, tp)
        if m["win_rate"] > best_wr:
            best_wr = m["win_rate"]
            best_wr_key = (sl, tp)

        print(
            f"{sl:<9.2f} | {tp:<9.2f} | {m['win_rate']*100:<9.2f}% | "
            f"{m['rr']:<7.2f} | {m['ev']:<10.4f}"
        )

    print("-" * 60)
    print("🏁 SUMMARY:")
    print(
        f"   Best by Expected Value: SL={best_ev_key[0]:.2f}, "
        f"TP={best_ev_key[1]:.2f}, EV={best_ev:.4f}"
    )
    print(
        f"   Best by WinRate:        SL={best_wr_key[0]:.2f}, "
        f"TP={best_wr_key[1]:.2f}, WR={best_wr*100:.2f}%"
    )

    print("\n✅ RECOMMENDATION FOR OPTIMIZER / Config:")
    print(f"   • Центрируй диапазон SL вокруг ≈ {best_ev_key[0]:.2f} ATR (сейчас: {base_sl:.2f}).")
    print(f"   • Центрируй диапазон TP вокруг ≈ {best_ev_key[1]:.2f} ATR (сейчас: {base_tp:.2f}).")
    print("   • При любых будущих изменениях стратегии скрипт сам перестроит поверхность.")

if __name__ == "__main__":
    run_analysis()
