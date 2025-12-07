# optimizer.py
import numpy as np
import pandas as pd
import random
import json
import os
import pickle
import warnings
import argparse
import sys
from tqdm import tqdm
from numba import njit
from config import Config
from typing import Dict, Any

try:
    from execution_core import simulate_core_logic
except ImportError:
    print("❌ Critical Error: execution_core.py not found!")
    sys.exit(1)

warnings.filterwarnings("ignore")

# --- КОНФИГУРАЦИЯ ---
CURRENT_MODE = "sniper" 
POP_SIZE = 60
GENERATIONS = 30
SURVIVORS = 15
TRAIN_WINDOW = 800    
TEST_WINDOW = 400      

SIGNAL_FILE = "data_cache/production_signals_v1.pkl"
RESULT_FILE = "best_strategy_params.json"
WFO_REPORT_FILE = "wfo_optimization_report.csv"
SETTINGS_FILE = "optimizer_settings.json"

# --- ДЕФОЛТНЫЕ ДИАПАЗОНЫ ДЛЯ ОПТИМИЗАТОРА ---

# Крипта: более широкие тренды, большие TP, длиннее удержание
CRYPTO_DEFAULT_RANGES = {
    "sl_min": 1.5, "sl_max": 2.5,
    "tp_min": 3.0, "tp_max": 6.0,
    "conf_min": 0.65, "conf_max": 0.85,
    "pullback_min": 0.0, "pullback_max": 0.15,
    "trail_act_min": 1.2, "trail_act_max": 2.0,
    "max_hold_min": 24, "max_hold_max": 72,
    "train_window": 800,
    "test_window": 400,
}

# Акции (Тинькофф / MOEX): движения мягче, держим меньше, цели ближе
STOCKS_DEFAULT_RANGES = {
    "sl_min": 1.0, "sl_max": 2.0,
    "tp_min": 1.5, "tp_max": 4.0,
    "conf_min": 0.55, "conf_max": 0.80,
    "pullback_min": 0.0, "pullback_max": 0.20,
    "trail_act_min": 0.8, "trail_act_max": 1.8,
    "max_hold_min": 16, "max_hold_max": 64,
    "train_window": 600,
    "test_window": 300,
}

DEFAULT_RANGES = CRYPTO_DEFAULT_RANGES  # для обратной совместимости

def load_settings() -> Dict[str, Any]:
    """
    Загружает диапазоны параметров для генетического оптимизатора.

    Приоритет профиля:
    1) ENV OPTIMIZER_PROFILE: crypto / stocks / both / auto
    2) Config.UNIVERSE_MODE, если OPTIMIZER_PROFILE=auto
    """
    # --- 1. Читаем профиль из ENV ---
    env_profile = os.getenv("OPTIMIZER_PROFILE", "auto").lower()

    # --- 2. Профиль по умолчанию — из UNIVERSE_MODE ---
    try:
        mode = getattr(Config, "UNIVERSE_MODE", None)
        universe_profile = getattr(mode, "value", "both") if mode is not None else "both"
    except Exception:
        universe_profile = "both"

    # --- 3. Итоговый ключ профиля ---
    if env_profile in ("crypto", "stocks", "both"):
        profile_key = env_profile
    else:
        profile_key = universe_profile

    # --- 4. Базовые диапазоны по профилю ---
    base_map: Dict[str, Any] = {
        "crypto": CRYPTO_DEFAULT_RANGES,
        "stocks": STOCKS_DEFAULT_RANGES,
        "both":   CRYPTO_DEFAULT_RANGES,  # both по умолчанию crypto
    }
    base_settings: Dict[str, Any] = dict(base_map.get(profile_key, CRYPTO_DEFAULT_RANGES))

    # --- 5. Подмешиваем overrides из optimizer_settings.json, если есть ---
    final_settings: Dict[str, Any] = dict(base_settings)

    try:
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)

            overrides: Dict[str, Any] = {}

            if isinstance(data, dict):
                # Вариант 1: будущий формат с "profiles" внутри
                if "profiles" in data and isinstance(data["profiles"], dict):
                    profiles = data["profiles"]
                    prof = profiles.get(profile_key)
                    if isinstance(prof, dict):
                        overrides = prof
                else:
                    # Вариант 2: текущий формат GUI — профили на верхнем уровне
                    if profile_key in data and isinstance(data[profile_key], dict):
                        overrides = data[profile_key]
                    # Вариант 3: старый плоский формат: sl_min/tp_min в корне
                    elif ("sl_min" in data) or ("tp_min" in data):
                        overrides = data

            for param_name, bounds in overrides.items():
                final_settings[param_name] = bounds

    except Exception as e:
        print(f"⚠️ load_settings: error while reading {SETTINGS_FILE}: {e}")

    print(f"[OPTIMIZER] Active profile: {profile_key} (env={env_profile}, universe={universe_profile})")
    print(f"[OPTIMIZER] Parameters loaded: {len(final_settings)}")

    return final_settings

# Глобальная переменная для настроек, чтобы не читать файл миллион раз
GENE_RANGES = load_settings()

@njit(fastmath=True)
def calculate_sortino(pnl_array, expected_trades):
    n_trades = len(pnl_array)
    if n_trades < expected_trades * 0.5: return -20.0 
    
    mean_ret = np.mean(pnl_array)
    if mean_ret <= 0: return -10.0 
    
    downside = pnl_array[pnl_array < 0]
    if len(downside) == 0: downside_std = 0.0001
    else: downside_std = np.std(downside)
    if downside_std < 1e-6: downside_std = 1e-6
    
    sortino = mean_ret / downside_std
    return sortino

def init_genome():
    g = {}
    r = GENE_RANGES # Short alias
    
    g['sl'] = random.uniform(r.get('sl_min', 1.5), r.get('sl_max', 2.5))
    g['tp'] = random.uniform(r.get('tp_min', 3.0), r.get('tp_max', 6.0))
    g['conf'] = random.uniform(r.get('conf_min', 0.65), r.get('conf_max', 0.85))
    
    g['trail_on'] = 1.0 
    g['trail_act'] = random.uniform(r.get('trail_act_min', 1.2), r.get('trail_act_max', 2.0))
    g['trail_off'] = random.uniform(0.2, 0.5) # Пока оставим жестко, или добавь в JSON
    
    g['vol_exit'] = random.uniform(6.0, 10.0)
    g['max_hold'] = float(random.randint(int(r.get('max_hold_min', 24)), int(r.get('max_hold_max', 72))))
    
    # Динамический pullback из настроек GUI
    g['pullback'] = random.uniform(r.get('pullback_min', 0.0), r.get('pullback_max', 0.15))
    
    g['fill_wait'] = int(random.randint(2, 8))
    g['abort'] = random.uniform(0.65, 0.85)
    return g

def mutate(g):
    new_g = g.copy()
    keys = list(g.keys())
    key = random.choice(keys)
    power = random.uniform(0.9, 1.1) 
    val = new_g[key]
    r = GENE_RANGES
    
    if key == 'sl': 
        new_g['sl'] = np.clip(val * power, r.get('sl_min', 1.5), r.get('sl_max', 2.5)) 
        new_g['tp'] = max(new_g['tp'], new_g['sl'] * 2.0)
        
    if key == 'tp': 
        new_g['tp'] = np.clip(val * power, r.get('tp_min', 2.0), r.get('tp_max', 8.0))
        
    elif key == 'conf': 
        new_g['conf'] = np.clip(val + random.uniform(-0.05, 0.05), r.get('conf_min', 0.6), r.get('conf_max', 0.9))

    elif key == 'trail_act':
        new_g['trail_act'] = np.clip(val * power, r.get('trail_act_min', 1.1), r.get('trail_act_max', 2.5))
        
    elif key == 'pullback': 
        # Мутация не должна выбивать нас из заданного диапазона
        new_g['pullback'] = np.clip(val * power, r.get('pullback_min', 0.0), r.get('pullback_max', 0.2))
        
    elif key == 'max_hold': 
        new_g['max_hold'] = int(val * power)
        
    return new_g

def crossover(p1, p2):
    child = {}
    for key in p1.keys():
        if random.random() < 0.5: child[key] = p1[key] 
        else: child[key] = p2[key] 
    return child

class WFOptimizer:
    def __init__(self):
        self.data_store = self._load_data()
        
    def _load_data(self):
        """
        Загружает production-сигналы и готовит структуры под Numba-ядро.

        - day_ids: чистый ndarray int64 (Numba-friendly)
        - atr: гарантированно > 0, чтобы избежать деления на ноль в ядре
        """
        if not os.path.exists(SIGNAL_FILE):
            print(f"❌ Нет файла {SIGNAL_FILE}. Запустите Signal Factory.")
            sys.exit(1)

        with open(SIGNAL_FILE, "rb") as f:
            raw_data = pickle.load(f)

        numba_data: dict[str, dict] = {}

        for sym, df in raw_data.items():
            # На всякий случай упорядочим по времени
            df = df.sort_index()

            # Индексы → timestamps → day_ids (Numba любит ndarray, а не Index)
            timestamps = df.index.astype(np.int64) // 10**9
            day_ids = (timestamps // 86400).astype(np.int64).values

            # Базовые массивы цен
            open_arr  = df["open"].values.astype(np.float64)
            high_arr  = df["high"].values.astype(np.float64)
            low_arr   = df["low"].values.astype(np.float64)
            close_arr = df["close"].values.astype(np.float64)

            # ATR: берём из df, заполняем NaN нулями
            if "atr" in df.columns:
                atr_arr = df["atr"].fillna(0.0).values.astype(np.float64)
            else:
                # Фоллбэк: если по каким-то причинам atr не найден
                atr_arr = np.zeros_like(close_arr, dtype=np.float64)

            # 🔒 Подстраховка: ATR не должен быть <= 0, иначе в ядре словим деление на ноль
            for k in range(len(atr_arr)):
                if atr_arr[k] <= 0.0:
                    base = abs(close_arr[k]) if not np.isnan(close_arr[k]) else 1.0
                    # 1% от цены или минимальный epsilon
                    atr_arr[k] = max(base * 0.01, 1e-8)

            numba_data[sym] = {
                "open":        open_arr,
                "high":        high_arr,
                "low":         low_arr,
                "close":       close_arr,
                "atr":         atr_arr,
                "regimes":     df["regime"].values.astype(np.int64),
                "day_ids":     day_ids,
                "probs_long":  df["p_long"].values.astype(np.float64),
                "probs_short": df["p_short"].values.astype(np.float64),
            }

        print(f"✅ Данные загружены. Активов: {len(numba_data)}")
        return numba_data

    def _run_simulation_wrapper(self, genome, start_idx, end_idx):
        all_pnls = []
        mode_int = 1 if CURRENT_MODE == 'sniper' else 0
        for sym, d in self.data_store.items():
            total_len = len(d['close'])
            if start_idx >= total_len: continue
            curr_end = min(end_idx, total_len)
            
            trail_on = genome['trail_on']
            if not getattr(Config, 'USE_TRAILING', True): trail_on = 0.0
            
            p_pullback = genome.get('pullback', 0.01)
            p_fill_wait = int(genome.get('fill_wait', 2))
            p_abort = genome.get('abort', 0.8)

            _, trades = simulate_core_logic(
                d['open'][start_idx:curr_end], d['high'][start_idx:curr_end],
                d['low'][start_idx:curr_end], d['close'][start_idx:curr_end],
                d['atr'][start_idx:curr_end], d['day_ids'][start_idx:curr_end],
                d['probs_long'][start_idx:curr_end], d['probs_short'][start_idx:curr_end],
                d['regimes'][start_idx:curr_end],
                genome['sl'], genome['tp'], genome['conf'], genome['vol_exit'],
                trail_on, genome['trail_act'], genome['trail_off'],
                genome['max_hold'],
                p_pullback, p_fill_wait, p_abort,
                mode_int, Config.COMMISSION, 
                1000.0, Config.RISK_PER_TRADE
            )
            
            if len(trades) > 0:
                entry_prices = trades[:, 2]; exit_prices = trades[:, 3]; directions = trades[:, 4] 
                raw_pnls = (exit_prices - entry_prices) / entry_prices
                adj_pnls = raw_pnls * directions
                net_pnls = adj_pnls - (Config.COMMISSION * 2.0) 
                all_pnls.extend(net_pnls)
        return np.array(all_pnls)
    
    def _run_equity_wrapper(self, genome, start_idx, end_idx):
        mode_int = 1 if CURRENT_MODE == 'sniper' else 0
        total_equity_start = 0.0; total_equity_end = 0.0; any_trades = False
        
        for sym, d in self.data_store.items():
            total_len = len(d['close'])
            if start_idx >= total_len: continue
            curr_end = min(end_idx, total_len)
            deposit = 1000.0
            trail_on = genome['trail_on']
            if not getattr(Config, 'USE_TRAILING', True): trail_on = 0.0
            
            max_hold = genome.get('max_hold', 96.0)
            p_pullback = genome.get('pullback', 0.1)
            p_fill_wait = int(genome.get('fill_wait', 2))
            p_abort = genome.get('abort', 0.8)

            equity, trades = simulate_core_logic(
                d['open'][start_idx:curr_end], d['high'][start_idx:curr_end],
                d['low'][start_idx:curr_end], d['close'][start_idx:curr_end],
                d['atr'][start_idx:curr_end], d['day_ids'][start_idx:curr_end],
                d['probs_long'][start_idx:curr_end], d['probs_short'][start_idx:curr_end],
                d['regimes'][start_idx:curr_end],
                genome['sl'], genome['tp'], genome['conf'], genome['vol_exit'],
                trail_on, genome['trail_act'], genome['trail_off'],
                max_hold, p_pullback, p_fill_wait, p_abort,
                mode_int, Config.COMMISSION, 
                deposit, Config.RISK_PER_TRADE
            )
            total_equity_start += deposit
            if len(equity) > 0: total_equity_end += float(equity[-1])
            else: total_equity_end += deposit
            if len(trades) > 0: any_trades = True

        if (not any_trades) or total_equity_start == 0.0: return 0.0
        return (total_equity_end / total_equity_start) - 1.0    

    def optimize_block(self, start_idx, end_idx):
        population = [init_genome() for _ in range(POP_SIZE)]
        best_genome = None; best_score = -9999
        num_days = (end_idx - start_idx) / 96 
        if CURRENT_MODE == 'sniper': expected_trades = num_days * 0.2 * len(self.data_store)
        else: expected_trades = num_days * 1.5 * len(self.data_store)
        
        for gen in range(GENERATIONS):
            scores = []
            for genome in population:
                pnl_array = self._run_simulation_wrapper(genome, start_idx, end_idx)
                score = calculate_sortino(pnl_array, expected_trades)
                scores.append(score)
            indices = np.argsort(scores)[::-1] 
            top_idx = indices[:SURVIVORS]
            if scores[top_idx[0]] > best_score:
                best_score = scores[top_idx[0]]; best_genome = population[top_idx[0]]
            new_pop = [population[i] for i in top_idx]
            while len(new_pop) < POP_SIZE:
                p1, p2 = random.sample(new_pop[:SURVIVORS], 2)
                new_pop.append(mutate(crossover(p1, p2)))
            population = new_pop
        return best_genome, best_score

    def run_walk_forward(self):
        print(f"\n🎮 MODE ACTIVATED: {CURRENT_MODE.upper()}")
        print(f"🚀 [WFO] Start Unified Optimization. Train: {TRAIN_WINDOW} | Test: {TEST_WINDOW}")
        if len(self.data_store) == 0:
            return

        # --- 0. Длина доступной истории ---
        first_asset = list(self.data_store.keys())[0]
        total_len = len(self.data_store[first_asset]['close'])

        # --- 1. Локальные окна WF с авто-адаптацией ---
        train_window = TRAIN_WINDOW
        test_window = TEST_WINDOW

        if train_window + test_window >= total_len:
            # Авто-сплит: ~70% на обучение, остаток на тест
            new_train = int(total_len * 0.7)
            new_test = max(1, total_len - new_train - 1)

            print(
                f"⚠️ [WFO] Конфиг окна ({TRAIN_WINDOW} + {TEST_WINDOW}) "
                f"не помещается в историю ({total_len} баров). "
                f"Используем адаптированные окна: train={new_train}, test={new_test}."
            )

            train_window = new_train
            test_window = new_test

        current_idx = 0
        wfo_results = []
        final_best_genome = init_genome()

        # total_len - train_window >= 1 гарантируем выше
        pbar = tqdm(total=max(1, total_len - train_window))

        # --- 2. Walk-Forward цикл с локальными окнами ---
        while current_idx + train_window + test_window < total_len:
            train_start = current_idx
            train_end = current_idx + train_window

            best_genome, train_score = self.optimize_block(train_start, train_end)

            test_start = train_end
            test_end = train_end + test_window

            oos_pnl_arr = self._run_simulation_wrapper(best_genome, test_start, test_end)
            oos_equity_ret = self._run_equity_wrapper(best_genome, test_start, test_end)

            num_test_days = (test_window / 96.0)
            exp_trades_test = (
                num_test_days
                * (0.2 if CURRENT_MODE == 'sniper' else 1.5)
                * len(self.data_store)
            )
            oos_sortino = calculate_sortino(oos_pnl_arr, exp_trades_test)
            total_profit_pct = np.sum(oos_pnl_arr) if len(oos_pnl_arr) > 0 else 0.0

            wfo_results.append(
                {
                    "train_start": train_start,
                    "test_start": test_start,
                    "oos_score": oos_sortino,
                    "oos_profit_pct": total_profit_pct,
                    "oos_equity_ret": oos_equity_ret,
                    "trades_count": len(oos_pnl_arr),
                    "params": best_genome,
                }
            )
            final_best_genome = best_genome

            current_idx += test_window
            pbar.update(test_window)

        pbar.close()

        if not wfo_results:
            print("\n⚠️ [WARNING] Оптимизация не состоялась: слишком мало данных под заданные окна.")
            return

        df_res = pd.DataFrame(wfo_results)
        if "oos_equity_ret" in df_res.columns:
            total_oos_ret = (1 + df_res["oos_equity_ret"]).prod() - 1.0
        else:
            total_oos_ret = 0.0

        df_res.to_csv(WFO_REPORT_FILE, index=False)

        # DEBUG: финальный геном на всей истории
        full_equity_ret = self._run_equity_wrapper(final_best_genome, 0, total_len)
        print(f"\n[DEBUG] Full-period equity return (final genome): {full_equity_ret*100:.2f}%")

        print(f"\n🏆 [{CURRENT_MODE.upper()} WFO REPORT]")
        start_deposit = 1000.0
        net_profit_usd = start_deposit * total_oos_ret
        end_balance = start_deposit + net_profit_usd
        total_trades = df_res["trades_count"].sum() if not df_res.empty else 0

        print(f"   💰 Start Deposit:   ${start_deposit:.2f}")
        print(f"   🏁 Final Balance:   ${end_balance:.2f}")
        print(f"   💵 Net Profit:      ${net_profit_usd:.2f} ({total_oos_ret*100:.2f}%)")
        print(f"   📊 Total Trades:    {total_trades}")

        final_best_genome["mode"] = CURRENT_MODE
        with open(RESULT_FILE, "w") as f:
            json.dump(final_best_genome, f, indent=4)
        print(f"   💾 Saved to {RESULT_FILE}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Strategy Optimizer (Unified Core)")
    parser.add_argument("--mode", type=str, choices=['sniper', 'scalper'], default='sniper')
    args = parser.parse_args()
    CURRENT_MODE = args.mode
    
    # Reload ranges (включая train/test окна из GUI)
    settings = load_settings()
    GENE_RANGES = settings

    # Позволяем GUI переопределять окна WALK-FORWARD
    TRAIN_WINDOW = int(settings.get("train_window", TRAIN_WINDOW))
    TEST_WINDOW = int(settings.get("test_window", TEST_WINDOW))
    
    opt = WFOptimizer()
    opt.run_walk_forward()