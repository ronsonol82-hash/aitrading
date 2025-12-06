# indicators.py
import pandas as pd
import numpy as np
from numba import njit
from features_lib import StructureFeatures  # <--- ВАЖНО: Импорт новой библиотеки

# --- NUMBA ДЛЯ РАЗМЕТКИ (Triple Barrier) ---
# Оставляем старый Labeling только для целей обучения (Target),
# но сами фичи (Features) теперь считаются в features_lib.

@njit
def triple_barrier_numba(closes, highs, lows, atrs, look_ahead, rr_ratio):
    n = len(closes)
    labels = np.zeros(n, dtype=np.int32) 
    
    for i in range(n - look_ahead):
        entry = closes[i]
        vol = atrs[i]
        if vol == 0: continue

        # Динамические цели
        tp_dist = vol * rr_ratio
        sl_dist = vol * 1.0 
        
        tp_long = entry + tp_dist
        sl_long = entry - sl_dist
        tp_short = entry - tp_dist
        sl_short = entry + sl_dist
        
        long_res = 0
        short_res = 0
        
        for j in range(1, look_ahead + 1):
            idx = i + j
            if idx >= n: break
            
            # Логика Long
            if long_res == 0:
                hit_sl = lows[idx] <= sl_long
                hit_tp = highs[idx] >= tp_long
                if hit_sl and hit_tp: long_res = -1 
                elif hit_sl: long_res = -1
                elif hit_tp: long_res = 1
            
            # Логика Short
            if short_res == 0:
                hit_sl = highs[idx] >= sl_short
                hit_tp = lows[idx] <= tp_short
                if hit_sl and hit_tp: short_res = -1
                elif hit_sl: short_res = -1
                elif hit_tp: short_res = 1
            
            if long_res != 0 and short_res != 0: break
            
        if long_res == 1 and short_res != 1:
            labels[i] = 1 # LONG
        elif short_res == 1 and long_res != 1:
            labels[i] = 2 # SHORT
            
    return labels

# --- КЛАСС-АДАПТЕР ---
    
class FeatureEngineer:
    
    @staticmethod
    def add_features(df):
        """
        ПЕРЕХВАТЧИК: Вместо старых индикаторов вызываем StructureFeatures.
        """
        if df is None or df.empty:
            # Нечего считать — сразу выходим.
            return df

        print("   🏗️ [STRUCTURE] Calculating Confluence Scores...") 
        # Вызываем логику из features_lib.py
        return StructureFeatures.process_all(df)

    @staticmethod
    def add_htf_features(df_ltf, df_htf):
        """
        Мердж старшего ТФ: забираем только "Психологию" и "Структуру".
        """
        if df_htf is None or df_htf.empty: return df_ltf

        # 1. Считаем фичи на старшем ТФ (пока он целостный)
        # Важно делать copy(), чтобы не замусорить кэш
        df_htf_feat = StructureFeatures.process_all(df_htf.copy())
        
        # 2. Выбираем только "Абстрактные" фичи (Без цен!)
        cols_to_keep = [
            'volatility',    # Общая температура больницы
            'sup_strength',  # Сила поддержки 4H (число 0..100)
            'res_strength',  # Сила сопротивления 4H
            'sup_dist_atr',  # Дистанция в ATR (относительная величина!)
            'res_dist_atr', 
            'channel_pos',   # Позиция в канале 0..1
            'squeeze_factor' # Сжатие
        ]
        
        # Фильтруем, если вдруг какой-то колонки нет
        cols = [c for c in cols_to_keep if c in df_htf_feat.columns]
        
        # Добавляем префикс htf_
        df_ready = df_htf_feat[cols].add_prefix('htf_')
        
        # 3. Мерджим (Backward direction - без заглядывания в будущее)
        merged = pd.merge_asof(
            df_ltf.sort_index(),
            df_ready.sort_index(),
            left_index=True,
            right_index=True,
            direction='backward'
        )
        
        # Заполняем пропуски (HTF данные редкие, они "тянутся" вперед)
        return merged.fillna(method='ffill').fillna(0)

    @staticmethod
    def add_levels_distance(df, window=3):
        """
        Совместимость с leak_test.py.
        Честно считает dist_to_max/dist_to_min ТОЛЬКО по прошлым барам.
        """
        if df is None or df.empty:
            return df

        highs = df['high'].values.astype(np.float64)
        lows = df['low'].values.astype(np.float64)
        closes = df['close'].values.astype(np.float64)

        if 'atr' in df.columns:
            atrs = df['atr'].values.astype(np.float64)
        else:
            # простой fallback, чтобы не падать
            atrs = np.ones_like(closes, dtype=np.float64)

        n = len(df)
        dist_to_max = np.zeros(n, dtype=np.float64)
        dist_to_min = np.zeros(n, dtype=np.float64)

        for i in range(n):
            if i < window:
                # истории мало — считаем расстояние нулевым
                dist_to_max[i] = 0.0
                dist_to_min[i] = 0.0
                continue

            prev_high = np.max(highs[i-window:i])
            prev_low = np.min(lows[i-window:i])
            atr_val = atrs[i] if atrs[i] > 0 else 1.0

            # ВАЖНО: только прошедшие бары, без заглядывания вперёд
            dist_to_max[i] = (prev_high - closes[i]) / atr_val
            dist_to_min[i] = (prev_low - closes[i]) / atr_val

        df_out = df.copy()
        df_out['dist_to_max'] = dist_to_max
        df_out['dist_to_min'] = dist_to_min
        return df_out

    @staticmethod
    def label_data(df, look_ahead, rr_ratio):
        """
        Разметка данных для учителя (Supervised Learning).
        """
        closes = df['close'].values.astype(np.float64)
        highs = df['high'].values.astype(np.float64)
        lows = df['low'].values.astype(np.float64)
        atrs = df['atr'].fillna(0).values.astype(np.float64)
        
        labels = triple_barrier_numba(closes, highs, lows, atrs, int(look_ahead), float(rr_ratio))
        
        df_labeled = df.copy()
        df_labeled['target'] = labels
        return df_labeled.iloc[:-int(look_ahead)]