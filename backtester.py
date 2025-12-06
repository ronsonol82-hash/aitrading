# backtester.py
import pandas as pd
import numpy as np
import json
import os
from typing import Optional

from config import Config
from execution_core import simulate_core_logic

# Брокерный слой и роутер
from brokers import OrderRequest
from execution_router import ExecutionRouter, GlobalAccountState

class PortfolioManager:
    """
    Управляет деньгами. Жесткий бухгалтер.
    """
    def __init__(self):
        self.balance = Config.DEPOSIT
        self.peak_balance = Config.DEPOSIT
        self.equity_history = [] 
        self.positions = {} 
        self.closed_trades = []
        self.daily_pnl = {}

    def check_risk_limits(self, date_str):
        # Лимит дневной просадки (от пика депозита)
        # Если просадка (отрицательная) меньше лимита (тоже отрицательного), стоп.
        # Пример: PnL -40, Limit -30. -40 < -30 -> True (Stop)
        limit = -(self.peak_balance * Config.MAX_DAILY_DRAWDOWN)
        current_daily_pnl = self.daily_pnl.get(date_str, 0)
        
        if current_daily_pnl < limit:
            return False # Stop trading for today
        return True
    
    def update_pnl(self, date_str, pnl):
        self.daily_pnl.setdefault(date_str, 0)
        self.daily_pnl[date_str] += pnl
        
    def get_exposure(self):
        return sum([pos['size'] * pos['entry'] for pos in self.positions.values()])

class PortfolioBacktester:
    def __init__(self, portfolio_data, model_engine_class, feature_cols, router: Optional[ExecutionRouter] = None):
        self.data = portfolio_data 
        self.symbols = list(portfolio_data.keys())
        self.model_class = model_engine_class
        # Инициализируем модели (заглушка, т.к. в симуляции Numba они не нужны)
        self.models = {sym: None for sym in self.symbols}
        
        # --- ExecutionRouter интеграция ---
        # Если роутер не передан явно — поднимаем свой с Config.ASSET_ROUTING.
        self.router: ExecutionRouter = router or ExecutionRouter()
        
        self.pm = PortfolioManager()
        self.feature_cols = feature_cols
        
        # Синхронизируем начальный баланс с глобальным состоянием брокеров (если оно реализовано)
        try:
            snapshot: GlobalAccountState = self.router.get_global_account_state()
            # Если брокеры уже знают свой баланс — берем его, иначе оставляем, как инициализировал PortfolioManager
            if snapshot.balance > 0:
                self.pm.balance = snapshot.balance
            if snapshot.equity > 0:
                self.pm.peak_balance = snapshot.equity
        except Exception:
            # Если кто-то из брокеров пока без account_state — не ломаемся
            pass
        
        # Загружаем параметры
        self.params = Config.get_strategy_params()
        print(f"🧬 [BACKTEST] Active Strategy: {self.params.get('mode', 'unknown').upper()}")
        print(f"   SL: {self.params.get('sl')} | TP: {self.params.get('tp')} | Conf: {self.params.get('conf')}")
        
        self.last_trade_date = {sym: None for sym in self.symbols}

    def run_simulation(self, progress_callback=None):
        """
        Ускоренная версия через execution_core (Numba) с корректной
        агрегацией equity по инструментам разной длины.
        """
        if not self.symbols:
            return {}

        print("🚀 [BACKTEST] Запуск Numba-Optimized симуляции...")

        # --- 1. ПАРАМЕТРЫ СТРАТЕГИИ ---
        sl = float(self.params.get("sl", 2.0))
        tp = float(self.params.get("tp", 4.0))
        conf = float(self.params.get("conf", 0.60))
        vol_exit = float(self.params.get("vol_exit", 4.0))
        max_hold = float(self.params.get("max_hold", 96.0))
        trail_on = float(self.params.get("trail_on", 0.0))
        trail_act = float(self.params.get("trail_act", 99.0))
        trail_off = float(self.params.get("trail_off", 99.0))

        pullback = float(self.params.get("pullback", 0.01))
        fill_wait = int(self.params.get("fill_wait", 6))
        abort = float(self.params.get("abort", 0.8))

        mode_sniper = 1 if self.params.get("mode", "sniper") == "sniper" else 0

        deposit_per_symbol = Config.DEPOSIT / len(self.symbols)

        # здесь вместо одного numpy-массива храним per-symbol серии дельт
        equity_components = []   # список pd.Series (eq - deposit_per_symbol)
        all_trades = []

        for sym in self.symbols:
            if sym not in self.data:
                continue

            df = self.data[sym]
            if df.empty:
                continue
            if "p_long" not in df.columns or "p_short" not in df.columns:
                continue

            # --- 2. ПОДГОТОВКА МАССИВОВ ДЛЯ NUMBA ---
            opens = df["open"].values.astype(np.float64)
            highs = df["high"].values.astype(np.float64)
            lows = df["low"].values.astype(np.float64)
            closes = df["close"].values.astype(np.float64)

            if "atr" in df.columns:
                atrs = df["atr"].fillna(0).values.astype(np.float64)
            else:
                atrs = (closes * 0.01).astype(np.float64)

            # используем чистый numpy, без pandas.Index
            # Преобразуем DatetimeIndex -> ndarray[int64] (секунды с эпохи)
            timestamps = df.index.to_numpy(dtype="int64") // 10**9
            day_ids = (timestamps // 86400).astype(np.int64)

            p_longs = df["p_long"].values.astype(np.float64)
            p_shorts = df["p_short"].values.astype(np.float64)
            regimes = df["regime"].values.astype(np.int64)

            # --- 3. ВЫЗОВ ЯДРА ---
            eq, trades = simulate_core_logic(
                opens, highs, lows,
                closes, atrs, day_ids,
                p_longs, p_shorts, regimes,
                sl, tp, conf, vol_exit, trail_on,
                trail_act, trail_off, max_hold,
                pullback, fill_wait,
                abort, mode_sniper,
                Config.COMMISSION,
                deposit_per_symbol,
                Config.RISK_PER_TRADE,
            )

            eq = np.asarray(eq, dtype=np.float64)
            if eq.size == 0:
                continue

            # на всякий случай синхронизируем длину с df
            n = min(len(df), len(eq))
            if n != len(eq):
                eq = eq[:n]
            if n != len(df):
                df_local = df.iloc[:n]
            else:
                df_local = df

            # --- 4. СОХРАНЯЕМ ДЕЛЬТУ EQUITY ДЛЯ ЭТОГО ИНСТРУМЕНТА ---
            delta_series = pd.Series(
                eq - deposit_per_symbol, index=df_local.index, name=sym
            )
            equity_components.append(delta_series)

            # --- 5. ДЕКОДИРУЕМ СДЕЛКИ ---
            for t in trades:
                entry_idx = int(t[0])
                exit_idx = int(t[1])
                if entry_idx >= n or exit_idx >= n:
                    continue

                row_data = {
                    "symbol": sym,
                    "entry_date": df_local.index[entry_idx],
                    "exit_date": df_local.index[exit_idx],
                    "entry_price": t[2],
                    "exit_price": t[3],
                    "type": "LONG" if t[4] == 1 else "SHORT",
                    "pnl": t[5],
                    "reason": ["SL", "TP", "PANIC", "TIME", "SMART_CUT", "TRAIL"][
                        int(t[6])
                    ],
                }
                all_trades.append(row_data)

        # --- 6. СБОРКА PORTFOLIO EQUITY ---
        if not equity_components:
            # На всякий случай: если вообще ничего не наторговали
            df_equity = pd.DataFrame(
                {"balance": np.array([Config.DEPOSIT], dtype=np.float64)}
            )
            trades_df = pd.DataFrame(all_trades)
            return {
                "equity": df_equity,
                "closed_trades": trades_df,
                "total_return": 0.0,
                "max_drawdown": 0.0,
            }

        # выравниваем все компоненты по времени
        delta_df = pd.concat(equity_components, axis=1).sort_index()
        delta_df = delta_df.ffill().fillna(0.0)

        portfolio_balance = Config.DEPOSIT + delta_df.sum(axis=1)
        df_equity = pd.DataFrame(index=delta_df.index)
        df_equity["balance"] = portfolio_balance

        trades_df = pd.DataFrame(all_trades)
        if not trades_df.empty:
            trades_df = trades_df.sort_values("entry_date")
            total_return = (df_equity["balance"].iloc[-1] / Config.DEPOSIT) - 1.0
            roll_max = df_equity["balance"].cummax()
            dd = (df_equity["balance"] - roll_max) / roll_max
            max_dd = dd.min()
        else:
            total_return = 0.0
            max_dd = 0.0

        return {
            "equity": df_equity,
            "closed_trades": trades_df,
            "total_return": total_return,
            "max_drawdown": abs(max_dd),
        }
    
    # --- MANUAL TRADING LOGIC (Python Loop) ---
    # Используется для GUI War Room (пошаговая) или глубокой отладки
    def manage_positions(self, timestamp, date_str):
        closed_ids = []
        
        for sym, pos in self.pm.positions.items():
            try:
                row = self.data[sym].loc[timestamp]
            except: continue 
                
            high, low, close, atr = row['high'], row['low'], row['close'], row['atr']
            entry_atr = pos.get('entry_atr', atr)
            is_long = (pos['type'] == 'LONG')
            exit_price = close 
            reason = None
            
            # Параметры из единого конфига
            sl_param = self.params['sl']
            tp_param = self.params['tp']
            
            current_pnl_atr = (close - pos['entry']) / entry_atr if is_long else (pos['entry'] - close) / entry_atr
            
            # --- 1. SMART TRAILING ---
            if self.params.get('trail_on', 0) > 0.5:
                act = self.params['trail_act']
                off = self.params['trail_off']
                
                if current_pnl_atr > act:
                    dist = entry_atr * off
                    if is_long:
                        new_sl = close - dist
                        if new_sl > pos['sl']: pos['sl'] = new_sl
                    else:
                        new_sl = close + dist
                        if new_sl < pos['sl']: pos['sl'] = new_sl

            # --- 2. EXITS ---
            # Panic
            if atr > entry_atr * self.params['vol_exit']:
                exit_price = close; reason = 'PANIC_VOL'
            
            # Hard SL/TP
            elif is_long:
                if low <= pos['sl']: exit_price = pos['sl'] * (1 - Config.SLIPPAGE); reason = 'SL'
                elif high >= pos['tp']: exit_price = pos['tp'] * (1 - Config.SLIPPAGE); reason = 'TP'
            else: # Short
                if high >= pos['sl']: exit_price = pos['sl'] * (1 + Config.SLIPPAGE); reason = 'SL'
                elif low <= pos['tp']: exit_price = pos['tp'] * (1 + Config.SLIPPAGE); reason = 'TP'
            
            # --- 3. CLOSE ---
            if reason:
                pnl = (exit_price - pos['entry']) * pos['size'] if is_long else (pos['entry'] - exit_price) * pos['size']
                fee = (pos['entry'] + exit_price) * pos['size'] * Config.COMMISSION
                net_pnl = pnl - fee

                # 3.1. Закрываем позицию через ExecutionRouter (обратный ордер)
                if self.router is not None:
                    close_side = "sell" if is_long else "buy"
                    try:
                        client_id = f"close-{sym}-{int(timestamp)}-{reason}"
                        self.router.execute_order(
                            symbol=sym,
                            side=close_side,
                            quantity=pos['size'],
                            order_type="market",
                            price=float(exit_price),  # опять же фиксируем историческую цену
                            client_id=client_id,
                        )
                    except Exception as e:
                        print(f"[WARN] Router close order failed for {sym}: {e}")
                
                # 3.2. Обновляем локальную бухгалтерию
                self.pm.balance += net_pnl
                self.pm.update_pnl(date_str, net_pnl)
                
                self.pm.closed_trades.append({
                    'symbol': sym,
                    'entry': pos['entry'],
                    'exit': exit_price,
                    'type': pos['type'],
                    'pnl': net_pnl,
                    'reason': reason,
                    'date': timestamp,
                })
                closed_ids.append(sym)
                self.last_trade_date[sym] = date_str
        
        for sym in closed_ids:
            del self.pm.positions[sym]

    def scan_for_entries(self, timestamp, date_str, block_predictions):
        if not self.pm.check_risk_limits(date_str): return
        if len(self.pm.positions) >= Config.MAX_OPEN_POSITIONS: return

        for sym in self.symbols:
            if sym in self.pm.positions: continue
            
            # Sniper logic
            if self.params.get('mode') == 'sniper':
                if self.last_trade_date[sym] == date_str: continue

            if sym not in block_predictions: continue
            try:
                pred_row = block_predictions[sym].loc[timestamp] 
            except KeyError: continue
            
            p_long = pred_row['l']
            p_short = pred_row['s']
            regime = pred_row['regime']
            
            # Panic Filter logic
            if regime == 2:
                req_conf = 0.90 if self.params.get('mode') == 'sniper' else 0.75
                if max(p_long, p_short) < req_conf: continue

            conf_threshold = self.params['conf']
            signal = None; confidence = 0.0
            
            if p_long > conf_threshold:
                signal = 'LONG'; confidence = p_long
            elif p_short > conf_threshold:
                signal = 'SHORT'; confidence = p_short
                
            if signal:
                try:
                    market_row = self.data[sym].loc[timestamp]
                    self.open_smart_position(sym, market_row, signal, confidence, timestamp, date_str)
                    
                    if self.params.get('mode') == 'sniper':
                         self.last_trade_date[sym] = date_str
                except KeyError: pass

    def open_smart_position(self, sym, row, pos_type, confidence, timestamp, date_str):
        # Kelly-like money management
        base_risk = 0.01
        max_risk = 0.03
        threshold = self.params['conf']
        
        scale = (confidence - threshold) / (1.0 - threshold + 1e-6)
        risk_per_trade = base_risk + (max_risk - base_risk) * scale
        risk_per_trade = min(max_risk, max(base_risk, risk_per_trade))
        
        risk_amount = self.pm.balance * risk_per_trade
        atr = row['atr']
        if atr == 0:
            return

        sl_dist = atr * self.params['sl']
        tp_dist = atr * self.params['tp']
        
        entry_price = row['close'] * (1 + Config.SLIPPAGE) if pos_type == 'LONG' else row['close'] * (1 - Config.SLIPPAGE)
        
        # Если ATR большой -> dist большой -> size маленький.
        if sl_dist == 0:
            return
        size = risk_amount / sl_dist
        
        if pos_type == 'LONG':
            sl_price = entry_price - sl_dist
            tp_price = entry_price + tp_dist
        else:
            sl_price = entry_price + sl_dist
            tp_price = entry_price - tp_dist
        
        # --- 1) Отправляем сигнал через ExecutionRouter ---
        if self.router is not None:
            try:
                client_id = f"open-{sym}-{int(timestamp)}"
                self.router.execute_signal(
                    symbol=sym,
                    pos_type=pos_type,
                    size=size,
                    price=float(entry_price),  # в бэктесте явно фиксируем цену из истории
                    client_id=client_id,
                )
            except Exception as e:
                print(f"[WARN] Router open signal failed for {sym}: {e}")
        
        # --- 2) Регистрируем позицию в локальном портфельном менеджере ---
        self.pm.positions[sym] = {
            'type': pos_type,
            'entry': entry_price,
            'sl': sl_price,
            'tp': tp_price,
            'size': size,
            'entry_atr': atr,
            'open_ts': timestamp,
            'conf': confidence,
        }