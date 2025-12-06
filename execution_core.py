# execution_core.py
import numpy as np
from numba import njit

@njit(fastmath=True)
def simulate_core_logic(
    opens, highs, lows, closes, atrs, day_ids,
    p_longs, p_shorts, regimes,
    sl_mult, tp_mult, conf_threshold, vol_exit_mult,
    trail_on, trail_act_mult, trail_off_mult, 
    max_hold_bars,
    pullback_mult, fill_wait_bars, abort_threshold,
    mode_sniper, commission, deposit, risk_per_trade
):
    n = len(closes)
    equity = np.zeros(n)
    
    in_position = False; pos_type = 0; entry_price = 0.0; entry_idx = 0; pos_size = 0.0   
    sl_price = 0.0; tp_price = 0.0
    
    # Moon Mode State
    is_moon_active = False 
    
    pending_type = 0; pending_price = 0.0; pending_sl_dist = 0.0; pending_tp_dist = 0.0; pending_start_idx = 0
    current_balance = deposit
    out_trades = np.zeros((10000, 7), dtype=np.float64); t_ptr = 0
    
    for i in range(1, n):
        equity[i] = current_balance
        op = opens[i]; hi = highs[i]; lo = lows[i]; cl = closes[i]; atr = atrs[i]

        # --- 1. PENDING ORDER EXPIRATION (FIXED BUG) ---
        # Если ордер висит слишком долго — отменяем его
        if pending_type != 0:
            if (i - pending_start_idx) > fill_wait_bars:
                pending_type = 0  # Сброс зомби-ордера
                # Мы не делаем continue, чтобы дать шанс найти новый сигнал прямо на этом баре
        
        # --- 2. PENDING ORDER FILL LOGIC ---
        is_filled = False
        if pending_type == 1:
            if lo <= pending_price:
                in_position = True
                pos_type = 1
                entry_price = pending_price
                if op < pending_price: entry_price = op  # Gap protection

                sl_price = entry_price - pending_sl_dist
                tp_price = entry_price + pending_tp_dist
                entry_idx = i

                risk_amt = current_balance * risk_per_trade
                dist_to_sl = pending_sl_dist if pending_sl_dist > 0.0 else atr

                if dist_to_sl <= 0.0:
                    in_position = False; pos_type = 0; is_filled = False
                else:
                    pos_size = risk_amt / dist_to_sl
                    current_balance -= (pos_size * entry_price * commission)
                    is_filled = True

        elif pending_type == -1:
            if hi >= pending_price:
                in_position = True
                pos_type = -1
                entry_price = pending_price
                if op > pending_price: entry_price = op  # Gap protection

                sl_price = entry_price + pending_sl_dist
                tp_price = entry_price - pending_tp_dist
                entry_idx = i

                risk_amt = current_balance * risk_per_trade
                dist_to_sl = pending_sl_dist if pending_sl_dist > 0.0 else atr

                if dist_to_sl <= 0.0:
                    in_position = False; pos_type = 0; is_filled = False
                else:
                    pos_size = risk_amt / dist_to_sl
                    current_balance -= (pos_size * entry_price * commission)
                    is_filled = True

        if is_filled:
            pending_type = 0
            is_moon_active = False
            continue

        # --- POSITION MANAGEMENT ---
        if in_position:
            exit_signal = False; exit_price = 0.0; reason = 0 
            
            # --- ЛОГИКА "РАКЕТЫ" (BREAKOUT TRIGGER) 🚀 ---
            # Триггер: Цена закрылась за пределами канала?
            # Или просто прошла далеко от входа (> 3 ATR)
            dist_from_entry = 0.0
            if pos_type == 1: dist_from_entry = hi - entry_price
            else: dist_from_entry = entry_price - lo
            
            # Если мы уже в хорошем плюсе (> 2.5 ATR), активируем режим Ракеты.
            # Это значит - мы не хотим тейк, мы хотим лететь.
            if dist_from_entry > (atr * 2.5):
                is_moon_active = True
            
            current_tp_target = tp_price
            if is_moon_active:
                if pos_type == 1: current_tp_target = entry_price + (atr * 100)
                else: current_tp_target = entry_price - (atr * 100)

            # 1. Check Hard SL/TP
            if pos_type == 1:
                if lo <= sl_price: 
                    exit_signal = True; exit_price = sl_price; reason = 0
                    if op < sl_price: exit_price = op 
                elif hi >= current_tp_target:
                    exit_signal = True; exit_price = current_tp_target; reason = 1
                    if op > current_tp_target: exit_price = op
            else:
                if hi >= sl_price:
                    exit_signal = True; exit_price = sl_price; reason = 0
                    if op > sl_price: exit_price = op
                elif lo <= current_tp_target:
                    exit_signal = True; exit_price = current_tp_target; reason = 1
                    if op < current_tp_target: exit_price = op
            
            # 2. Dynamic Trailing
            if not exit_signal and trail_on > 0.5:
                trail_activation_dist = atr * trail_act_mult
                base_trail_offset = atr * trail_off_mult
                
                if pos_type == 1:
                    dist_from_entry = hi - entry_price
                    if dist_from_entry > trail_activation_dist:
                        
                        # Если РАКЕТА: Трал широкий (1.5x от базы), даем дышать
                        if is_moon_active:
                            moon_offset = base_trail_offset * 1.5 
                            new_sl = hi - moon_offset
                        else:
                            # Если КАНАЛ: Сжатие (Squeeze)
                            dist_remain = tp_price - hi
                            total_run = tp_price - entry_price
                            squeeze_factor = 0.0
                            if total_run > 0: squeeze_factor = dist_remain / total_run
                            if squeeze_factor < 0: squeeze_factor = 0
                            if squeeze_factor > 1: squeeze_factor = 1
                            dynamic_offset = base_trail_offset * squeeze_factor
                            if dynamic_offset < (base_trail_offset * 0.1): dynamic_offset = base_trail_offset * 0.1
                            new_sl = hi - dynamic_offset

                        if new_sl > sl_price: sl_price = new_sl

                elif pos_type == -1:
                    dist_from_entry = entry_price - lo
                    if dist_from_entry > trail_activation_dist:
                        
                        if is_moon_active:
                            moon_offset = base_trail_offset * 1.5
                            new_sl = lo + moon_offset
                        else:
                            dist_remain = lo - tp_price
                            total_run = entry_price - tp_price
                            squeeze_factor = 0.0
                            if total_run > 0: squeeze_factor = dist_remain / total_run
                            if squeeze_factor < 0: squeeze_factor = 0
                            if squeeze_factor > 1: squeeze_factor = 1
                            dynamic_offset = base_trail_offset * squeeze_factor
                            if dynamic_offset < (base_trail_offset * 0.1): dynamic_offset = base_trail_offset * 0.1
                            new_sl = lo + dynamic_offset

                        if new_sl < sl_price: sl_price = new_sl

            # 3. Time Exit & 4. Smart Cut & 5. Volatility Panic Exit
            if not exit_signal and (i - entry_idx) > max_hold_bars:
                exit_signal = True; exit_price = cl; reason = 3
                
            abort_threshold_dynamic = abort_threshold
            if is_moon_active: abort_threshold_dynamic = 0.98 # В ракете терпим почти всё

            if not exit_signal:
                p_l_curr = p_longs[i]; p_s_curr = p_shorts[i]
                if pos_type == 1 and p_s_curr > abort_threshold_dynamic:
                    exit_signal = True; exit_price = cl; reason = 4
                elif pos_type == -1 and p_l_curr > abort_threshold_dynamic:
                    exit_signal = True; exit_price = cl; reason = 4
                    
            if not exit_signal:
                bar_size = hi - lo
                if bar_size > (atr * vol_exit_mult):
                    if pos_type == 1 and cl < op: exit_signal = True; exit_price = cl; reason = 2
                    elif pos_type == -1 and cl > op: exit_signal = True; exit_price = cl; reason = 2

            if exit_signal:
                pnl = 0.0
                if pos_type == 1: pnl = (exit_price - entry_price) / entry_price
                else: pnl = (entry_price - exit_price) / entry_price
                
                current_balance -= (pos_size * exit_price * commission)
                profit_abs = pos_size * (exit_price - entry_price) if pos_type == 1 else pos_size * (entry_price - exit_price)
                current_balance += profit_abs
                
                if t_ptr < 10000:
                    out_trades[t_ptr, 0] = entry_idx; out_trades[t_ptr, 1] = i
                    out_trades[t_ptr, 2] = entry_price; out_trades[t_ptr, 3] = exit_price
                    out_trades[t_ptr, 4] = pos_type; out_trades[t_ptr, 5] = pnl
                    final_reason = reason
                    if is_moon_active and reason == 0: final_reason = 5 
                    out_trades[t_ptr, 6] = final_reason
                    t_ptr += 1
                in_position = False; pos_type = 0; pending_type = 0; continue 

        # --- ENTRY LOGIC ---
        if not in_position and pending_type == 0:
            p_long = p_longs[i]; p_short = p_shorts[i]
            valid_signal = False; new_type = 0
            
            if p_long > conf_threshold: new_type = 1; valid_signal = True
            elif p_short > conf_threshold: new_type = -1; valid_signal = True
                
            if valid_signal:
                pullback_dist = atr * pullback_mult
                if new_type == 1:
                    pending_price = cl - pullback_dist 
                    if pending_price > hi: pending_price = cl 
                else: 
                    pending_price = cl + pullback_dist 
                    if pending_price < lo: pending_price = cl
                
                pending_type = new_type; pending_start_idx = i
                pending_sl_dist = atr * sl_mult; pending_tp_dist = atr * tp_mult

    return equity, out_trades[:t_ptr]