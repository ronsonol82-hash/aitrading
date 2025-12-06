# debug_core.py
import numpy as np
from optimizer import WFOptimizer
from execution_core import simulate_core_logic
from config import Config


def _get_live_strategy():
    """
    Берем текущую стратегию:
    - DEFAULT_STRATEGY как база
    - поверх накатываем значения из файла стратегии (Config.get_strategy_params())
    """
    base = Config.DEFAULT_STRATEGY.copy()
    live = Config.get_strategy_params()
    base.update(live)
    return base


def run_debug():
    print("\n🕵️‍♂️ DEBUG CORE PROBE (Config-aware)\n")

    # === 0. Снимок текущих настроек ===
    strat = _get_live_strategy()
    mode_name = strat.get("mode", "classic")
    mode_flag = 1 if mode_name.lower() == "sniper" else 0

    print("📋 CURRENT CONFIG SNAPSHOT")
    print(f"   Leader:       {Config.LEADER_SYMBOL}")
    print(f"   Assets:       {Config.ASSETS}")
    print(f"   TF LTF / HTF: {Config.TIMEFRAME_LTF} / {Config.TIMEFRAME_HTF}")
    print(f"   LOOK_AHEAD:   {Config.LOOK_AHEAD}")
    print(f"   RR_RATIO:     {Config.RR_RATIO}")
    print(f"   Deposit:      {Config.DEPOSIT}")
    print(f"   Risk/trade:   {Config.RISK_PER_TRADE}")
    print(f"   Commission:   {Config.COMMISSION}")
    print(f"   Mode:         {mode_name.upper()}")
    print("   Strategy params:")
    for k in ["sl", "tp", "conf", "pullback", "vol_exit",
              "trail_on", "trail_act", "trail_off",
              "max_hold", "fill_wait", "abort"]:
        if k in strat:
            print(f"      {k:10s} = {strat[k]}")

    # === 1. Загружаем данные из оптимизатора ===
    opt = WFOptimizer()
    if not getattr(opt, "data_store", None):
        print("\n❌ WFOptimizer.data_store пуст. Сначала запусти signal_generator.py и оптимизатор.")
        return

    asset = Config.LEADER_SYMBOL
    if asset not in opt.data_store:
        # fallback — первый доступный актив
        asset = list(opt.data_store.keys())[0]
        print(f"\n⚠️ {Config.LEADER_SYMBOL} не найден в data_store. Использую {asset}.")
    else:
        print(f"\n📉 Используем актив: {asset}")

    d = opt.data_store[asset]
    n = len(d["close"])
    print(f"   Bars in store: {n}")

    # === 2. Ищем «живой» сигнал под текущий порог conf ===
    conf_thr = float(strat.get("conf", 0.6))
    target_idx = -1

    search_start = max(100, n - 500)
    for i in range(search_start, n):
        if (d["probs_long"][i] > conf_thr or d["probs_short"][i] > conf_thr) and d["regimes"][i] != 0:
            target_idx = i
            break

    if target_idx == -1:
        print("\n⚠️ Сильных сигналов под текущий conf не нашли.")
        print("   Беру хвост истории просто для smoke-test ядра.")
        target_idx = max(100, n - 200)
    else:
        side = "LONG" if d["probs_long"][target_idx] > d["probs_short"][target_idx] else "SHORT"
        print(f"\n🎯 Найден сигнал:")
        print(f"   Index:   {target_idx}")
        print(f"   Side:    {side}")
        print(f"   Close:   {d['close'][target_idx]:.4f}")
        print(f"   p_long:  {d['probs_long'][target_idx]:.3f}")
        print(f"   p_short: {d['probs_short'][target_idx]:.3f}")
        print(f"   ATR:     {d['atr'][target_idx]:.4f}")
        print(f"   Regime:  {d['regimes'][target_idx]}")

    # === 3. Готовим окно для ядра ===
    start = max(0, target_idx - 100)
    end = min(n, target_idx + 400)
    window_len = end - start

    print(f"\n🧱 CORE INPUT WINDOW: [{start}:{end}]  (len={window_len})")

    params = {
        "sl": float(strat.get("sl", 2.0)),
        "tp": float(strat.get("tp", 4.0)),
        "conf": float(strat.get("conf", 0.6)),
        "vol_exit": float(strat.get("vol_exit", 10.0)),
        "trail_on": float(strat.get("trail_on", 1.0)),
        "trail_act": float(strat.get("trail_act", 2.0)),
        "trail_off": float(strat.get("trail_off", 0.5)),
        "max_hold": int(strat.get("max_hold", 48)),
        "pullback": float(strat.get("pullback", 0.01)),
        "fill_wait": int(strat.get("fill_wait", 5)),
        "abort": float(strat.get("abort", 0.8)),
    }

    print("\n⚙️ EFFECTIVE STRATEGY PARAMS (что реально идет в ядро):")
    for k, v in params.items():
        print(f"   {k:10s} = {v}")
    print(f"   mode_flag  = {mode_flag} ({'SNIPER' if mode_flag else 'CLASSIC'})")

    # === 4. Запуск ядра ===
    try:
        print("\n🚀 RUNNING NUMBA CORE...")
        equity, trades = simulate_core_logic(
            d["open"][start:end],
            d["high"][start:end],
            d["low"][start:end],
            d["close"][start:end],
            d["atr"][start:end],
            d["day_ids"][start:end],
            d["probs_long"][start:end],
            d["probs_short"][start:end],
            d["regimes"][start:end],
            params["sl"],
            params["tp"],
            params["conf"],
            params["vol_exit"],
            params["trail_on"],
            params["trail_act"],
            params["trail_off"],
            params["max_hold"],
            params["pullback"],
            params["fill_wait"],
            params["abort"],
            mode_flag,
            Config.COMMISSION,
            float(Config.DEPOSIT),
            float(Config.RISK_PER_TRADE),
        )

        print("\n✅ CORE FINISHED SUCCESSFULLY!")
        print(f"   Final Equity: {equity[-1]:.2f}")
        print(f"   Trades Made:  {len(trades)}")

        if len(trades) > 0:
            last = trades[-1]
            # формат трейда такой же, как раньше
            print("\n   Last Trade:")
            print(f"      Type:  {last[4]}")   # 'LONG' / 'SHORT'
            print(f"      Entry: {last[2]:.4f}")
            print(f"      Exit:  {last[3]:.4f}")
            print(f"      PnL:   {last[5]:.4f}")
            print(f"      Reason:{last[6]}")
        else:
            print("\n   ⚠️ No trades triggered. Возможна слишком высокая conf / жесткие фильтры.")

    except Exception as e:
        print(f"\n💥 CRASH IN CORE: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_debug()
