import pandas as pd
import pickle
from collections import Counter

def check():
    try:
        with open("data_cache/production_signals_v1.pkl", "rb") as f:
            data = pickle.load(f)
            
        print("📊 CLASSS BALANCE CHECK (Post-Fix):")
        total_bars = 0
        total_longs = 0
        total_shorts = 0
        
        for sym, df in data.items():
            # Смотрим на regime (это предсказанный класс: 0=Flat, 1=Long, 2=Short, если маппинг верный)
            # Но лучше посмотреть на исходные метки, если они сохранились. 
            # В production_signals у нас уже предикты. 
            
            # Давай оценим агрессивность модели:
            high_prob_long = df[df['p_long'] > 0.60]
            high_prob_short = df[df['p_short'] > 0.60]
            
            print(f"   🔹 {sym}: Longs>0.6: {len(high_prob_long)} | Shorts>0.6: {len(high_prob_short)} | Total: {len(df)}")
            
            total_longs += len(high_prob_long)
            total_shorts += len(high_prob_short)
            total_bars += len(df)
            
        print(f"\n📢 TOTAL SIGNAL DENSITY: {((total_longs + total_shorts) / total_bars * 100):.2f}%")
        if (total_longs + total_shorts) == 0:
            print("❌ МОДЕЛЬ МЕРТВА. Она вообще не выдает уверенных сигналов.")
        elif ((total_longs + total_shorts) / total_bars) < 0.01:
            print("⚠️ ОЧЕНЬ МАЛО СИГНАЛОВ. Модель боится входить.")
        else:
            print("✅ Сигналов достаточно для торговли.")

    except Exception as e:
        print(f"Ошибка: {e}")

if __name__ == "__main__":
    check()