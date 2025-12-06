# features_lib.py
import pandas as pd
import numpy as np
from numba import njit
import warnings
from config import Config

warnings.filterwarnings("ignore")

# ==========================================
# 🧠 NUMBA KERNELS (Ring Buffer Edition)
# ==========================================

@njit(cache=True)
def calc_psychological_strength_series(
    level_price,
    closes, highs, lows, atrs,
    alpha=0.95,
    tol_mult=0.3
):
    """
    Возвращает массив strength[i] — вес уровня на каждом баре.
    Честный расчет: не заглядывает в будущее.
    """
    n = len(closes)
    out = np.zeros(n, dtype=np.float64)
    strength = 0.0

    if n < 2: return out

    for i in range(n - 1):
        # 1. Decay (Забывание старых заслуг)
        strength *= alpha

        atr_val = atrs[i]
        if atr_val <= 0: atr_val = max(closes[i] * 0.01, 1e-8)
        
        tol = atr_val * tol_mult
        
        # 2. Проверка касания
        touched = False
        if abs(highs[i] - level_price) <= tol: touched = True
        elif abs(lows[i] - level_price) <= tol: touched = True
        elif abs(closes[i] - level_price) <= tol: touched = True

        if touched:
            # 3. Реакция (Отскок)
            # Смотрим i+1, так как мы "проживаем" этот бар
            rebound_dist = abs(closes[i+1] - closes[i])
            rebound_score = rebound_dist / max(atr_val, 1e-8)
            
            # Клиппинг
            if rebound_score > 3.0: rebound_score = 3.0
            
            strength += (1.0 + rebound_score)

        out[i+1] = strength 

    return out

@njit(cache=True)
def find_quality_levels_numba(highs, lows, closes, atrs, window=10):
    """
    Оптимизированный поиск уровней с кольцевым буфером и Series-силой.
    """
    n = len(closes)
    
    # Результирующие массивы
    sup_price = np.full(n, np.nan, dtype=np.float64)
    sup_strength = np.zeros(n, dtype=np.float64)
    res_price = np.full(n, np.nan, dtype=np.float64)
    res_strength = np.zeros(n, dtype=np.float64)
    
    # --- КОЛЬЦЕВЫЕ БУФЕРЫ (Вместо list.pop) ---
    max_levels = 15
    fractal_sups = np.full(max_levels, np.nan, dtype=np.float64)
    fractal_ress = np.full(max_levels, np.nan, dtype=np.float64)
    sup_ptr = 0 # Указатель куда писать следующий уровень
    res_ptr = 0
    
    # Параметры расчета силы "на лету"
    lookback = 100
    alpha = 0.95
    
    for i in range(window * 2, n):
        # 1. Фрактальный поиск (High/Low)
        center_idx = i - window
        
        # Check Resistance
        is_res = True
        for k in range(1, window + 1):
            if highs[center_idx] < highs[center_idx - k] or highs[center_idx] < highs[center_idx + k]: 
                is_res = False; break
        if is_res:
            fractal_ress[res_ptr % max_levels] = highs[center_idx]
            res_ptr += 1

        # Check Support
        is_sup = True
        for k in range(1, window + 1):
            if lows[center_idx] > lows[center_idx - k] or lows[center_idx] > lows[center_idx + k]: 
                is_sup = False; break
        if is_sup:
            fractal_sups[sup_ptr % max_levels] = lows[center_idx]
            sup_ptr += 1
            
        cur_close = closes[i]
        cur_atr = atrs[i] if atrs[i] > 0 else max(cur_close * 0.01, 1e-8)
        
        # 2. Поиск лучшей ПОДДЕРЖКИ (ближайшей снизу)
        best_s_price = np.nan
        min_dist_s = 1e9
        
        # Перебор кольцевого буфера
        # Нам не важен порядок, нам важна цена
        limit_s = min(sup_ptr, max_levels) # Сколько реально заполнено (или макс)
        # Если буфер переполнился, мы читаем все max_levels.
        # Если нет - читаем от 0 до sup_ptr.
        # Важно: если переполнился, sup_ptr > max_levels, поэтому limit = max_levels.
        # Но чтобы читать данные, надо пройти весь массив.
        
        iter_limit_s = max_levels if sup_ptr >= max_levels else sup_ptr
        
        for j in range(iter_limit_s):
            lvl = fractal_sups[j]
            if np.isnan(lvl): continue
            
            if lvl < cur_close:
                dist = cur_close - lvl
                if dist < min_dist_s:
                    min_dist_s = dist
                    best_s_price = lvl
        
        # 3. Расчет силы Поддержки "на лету" (без вызова внешней функции для скорости)
        s_score = 0.0
        if not np.isnan(best_s_price):
            start_lb = max(0, i - lookback)
            curr_str = 0.0
            
            for k in range(start_lb, i):
                curr_str *= alpha
                
                c_atr = atrs[k]
                if c_atr <= 0: c_atr = max(closes[k]*0.01, 1e-8)
                
                tol = c_atr * 0.3
                touched = False
                
                if abs(lows[k] - best_s_price) <= tol: touched = True
                elif abs(highs[k] - best_s_price) <= tol: touched = True
                
                # k < i-1 гарантирует, что мы смотрим отскок k->k+1, который уже случился
                if touched and k < i - 1:
                    reb = abs(closes[k+1] - closes[k]) / c_atr
                    if reb > 3.0: reb = 3.0
                    curr_str += (1.0 + reb)
            
            s_score = curr_str

        # 4. Поиск лучшего СОПРОТИВЛЕНИЯ (ближайшего сверху)
        best_r_price = np.nan
        min_dist_r = 1e9
        
        iter_limit_r = max_levels if res_ptr >= max_levels else res_ptr
        
        for j in range(iter_limit_r):
            lvl = fractal_ress[j]
            if np.isnan(lvl): continue
            
            if lvl > cur_close:
                dist = lvl - cur_close
                if dist < min_dist_r:
                    min_dist_r = dist
                    best_r_price = lvl
        
        r_score = 0.0
        if not np.isnan(best_r_price):
            start_lb = max(0, i - lookback)
            curr_str = 0.0
            for k in range(start_lb, i):
                curr_str *= alpha
                
                c_atr = atrs[k]
                if c_atr <= 0: c_atr = max(closes[k]*0.01, 1e-8)
                
                tol = c_atr * 0.3
                touched = False
                
                if abs(highs[k] - best_r_price) <= tol: touched = True
                elif abs(lows[k] - best_r_price) <= tol: touched = True
                
                if touched and k < i - 1:
                    reb = abs(closes[k+1] - closes[k]) / c_atr
                    if reb > 3.0: reb = 3.0
                    curr_str += (1.0 + reb)
            r_score = curr_str

        # Сохранение результатов
        sup_price[i] = best_s_price
        sup_strength[i] = s_score
        res_price[i] = best_r_price
        res_strength[i] = r_score

    return sup_price, sup_strength, res_price, res_strength

@njit(cache=True)
def calc_impulse_fib_numba(highs, lows, closes, period=100):
    n = len(closes)
    on_fib_382 = np.zeros(n); on_fib_618 = np.zeros(n)
    for i in range(period, n):
        win_h = highs[i-period:i]; win_l = lows[i-period:i]
        glob_max = np.max(win_h); glob_min = np.min(win_l)
        rng = glob_max - glob_min
        if rng == 0: continue
        f382 = glob_min + rng * 0.382; f618 = glob_min + rng * 0.618
        f382_inv = glob_max - rng * 0.382; f618_inv = glob_max - rng * 0.618
        cur = closes[i]; tol = rng * 0.03
        if abs(cur - f382) < tol or abs(cur - f382_inv) < tol: on_fib_382[i] = 1.0
        if abs(cur - f618) < tol or abs(cur - f618_inv) < tol: on_fib_618[i] = 1.0
    return on_fib_382, on_fib_618

class StructureFeatures:
    @staticmethod
    def calc_linreg_channel(series, window=96):
        center_line = series.ewm(span=window).mean() 
        std = series.rolling(window).std()
        upper = center_line + (2.0 * std)
        lower = center_line - (2.0 * std)
        rng = (upper - lower).replace(0, 1)
        position = (series - lower) / rng
        slope = center_line.diff(3)
        return upper, lower, position, slope

    @staticmethod
    def process_all(df):
        if df is None or df.empty: return df

        # 1. Basics
        high = df['high']; low = df['low']; close = df['close']; open_ = df['open']
        prev_close = close.shift()
        tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean().fillna(method='bfill')
        df['returns'] = close.pct_change()
        df['volatility'] = df['returns'].rolling(24).std().fillna(0)
        df['ema_trend'] = close.ewm(span=200).mean()
        df['ema_slope'] = df['ema_trend'].diff(3)
        
        # 2. Indicators for Wedge Detection
        # BBW (Bollinger Band Width) - мера сжатия пружины
        bb_mid = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        df['bb_width'] = ( (bb_mid + 2*bb_std) - (bb_mid - 2*bb_std) ) / bb_mid
        # Относительное сжатие (текущая ширина к средней за 100 свечей)
        df['squeeze_factor'] = df['bb_width'] / df['bb_width'].rolling(100).mean()

        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, 1e-9)
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi'] = df['rsi'].fillna(50)

        vol_ma = df['volume'].rolling(20).mean()
        df['rvol'] = df['volume'] / vol_ma.replace(0, 1).fillna(1.0)
        df['adx'] = ((high - low) / close).rolling(14).mean() * 100
        df['adx'] = df['adx'].fillna(0)
        
        candle_color = np.where(close >= open_, 1, -1)
        df['streak_3'] = pd.Series(candle_color).rolling(3).sum().fillna(0).values
        body_size = (close - open_).abs()
        df['bar_size_atr'] = body_size / df['atr'].replace(0, 1)

        highs = high.values; lows = low.values; closes = close.values; atr = df['atr'].values 
        s_price, s_str, r_price, r_str = find_quality_levels_numba(highs, lows, closes, atr, window=10)
        df['sup_dist_atr'] = (closes - s_price) / pd.Series(atr).replace(0, 1)
        df['res_dist_atr'] = (r_price - closes) / pd.Series(atr).replace(0, 1)
        df['sup_strength'] = s_str; df['res_strength'] = r_str
        df['sup_dist_atr'] = df['sup_dist_atr'].fillna(999)
        df['res_dist_atr'] = df['res_dist_atr'].fillna(999)

        fib382, fib618 = calc_impulse_fib_numba(highs, lows, closes, period=96) 
        df['fib_382'] = fib382; df['fib_618'] = fib618
        
        ch_up, ch_lo, ch_pos, ch_slope = StructureFeatures.calc_linreg_channel(close, window=96)
        df['channel_pos'] = ch_pos; df['channel_slope'] = ch_slope
        
        df = StructureFeatures._calculate_pyramid_score(df)
        if 'regime' not in df.columns: df['regime'] = 0 
        if 'leader_close' not in df.columns: df['leader_close'] = df['close'] 
        return df.fillna(0)

    @staticmethod
    def _calculate_pyramid_score(df):
        score = np.zeros(len(df))
        w_channel = 60.0; w_level = 20.0     

        body = (df['close'] - df['open']).abs()
        upper_wick = df['high'] - np.maximum(df['close'], df['open'])
        lower_wick = np.minimum(df['close'], df['open']) - df['low']
        
        is_bullish_pinbar = (lower_wick > body * 1.0) & (lower_wick > upper_wick)
        is_bearish_pinbar = (upper_wick > body * 1.0) & (upper_wick > lower_wick)
        is_bullish_engulfing = (df['close'] > df['open']) & (body > df['atr'] * 0.4)
        is_bearish_engulfing = (df['close'] < df['open']) & (body > df['atr'] * 0.4)

        valid_long_trigger = is_bullish_pinbar | is_bullish_engulfing
        valid_short_trigger = is_bearish_pinbar | is_bearish_engulfing

        # --- TREND FILTERS ---
        # Жесткий фильтр: наклон EMA (Angle), а не просто положение
        # Если EMA смотрит ВВЕРХ, шортить ЗАПРЕЩЕНО, даже если цена упала под нее
        ema_slope_up = df['ema_slope'] > 0
        ema_slope_down = df['ema_slope'] < 0
        
        global_bullish = (df['close'] > df['ema_trend']) | ema_slope_up
        global_bearish = (df['close'] < df['ema_trend']) | ema_slope_down
        
        slope_pct = df['channel_slope'] / df['close'] 
        is_uptrend_local = slope_pct > 0.0003  
        is_downtrend_local = slope_pct < -0.0003
        at_bottom = df['channel_pos'] < 0.25; at_top = df['channel_pos'] > 0.75    

        # --- DETECTOR: FALLING WEDGE (НИСХОДЯЩИЙ КЛИН) ---
        # 1. Канал смотрит ВНИЗ (is_downtrend_local)
        # 2. Цена на ДНЕ (at_bottom)
        # 3. Волатильность СЖАЛАСЬ (squeeze_factor < 0.8) - это главное отличие клина от канала!
        # 4. RSI не падает (Дивергенция) - тут упрощенно берем RSI > 30 (не перепродан в хлам)
        is_falling_wedge = is_downtrend_local & at_bottom & (df['squeeze_factor'] < 0.85) & (df['rsi'] > 30)
        
        # === LONG LOGIC ===
        # Разрешаем, если: Глобально Бык ИЛИ Клин!
        # Если Клин - это мощнейший разворотный сигнал, берем даже против тренда
        allow_long = global_bullish | is_uptrend_local | is_falling_wedge
        
        long_signal = at_bottom & allow_long & valid_long_trigger
        
        score[long_signal] += w_channel
        score[long_signal & is_uptrend_local] += 30.0 
        score[long_signal & is_falling_wedge] += 50.0 # БОНУС ЗА КЛИН!
        
        # === SHORT LOGIC ===
        allow_short = global_bearish | is_downtrend_local
        
        # Запрет шортить аптренд (даже если цена на хаю канала)
        # Если EMA смотрит вверх - шорты запрещены категорически
        veto_short = ema_slope_up 
        
        short_signal = at_top & allow_short & valid_short_trigger & (~veto_short)
        
        score[short_signal] -= w_channel
        score[short_signal & is_downtrend_local] -= 30.0

        on_sup = df['sup_dist_atr'] < 1.0; on_res = df['res_dist_atr'] < 1.0
        score[long_signal & on_sup] += w_level
        score[short_signal & on_res] -= w_level
        
        panic_bar = df['bar_size_atr'] > 5.0
        score[panic_bar] = 0 
        score[df['rsi'] > 85] -= 1000.0; score[df['rsi'] < 15] += 1000.0

        df['confluence_score'] = score
        df.loc[on_sup, 'level_quality'] = df.loc[on_sup, 'sup_strength']
        df.loc[on_res, 'level_quality'] = df.loc[on_res, 'res_strength']
        return df