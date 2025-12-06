# inspect_probs.py
import pandas as pd
import pickle
import numpy as np

def inspect():
    try:
        print("📂 Loading cache...")
        with open("data_cache/production_signals_v1.pkl", "rb") as f:
            data = pickle.load(f)
            
        print(f"\n📊 PROBABILITY AUDIT ({len(data)} assets)")
        print("-" * 60)
        print(f"{'Asset':<10} | {'Max Long':<10} | {'Max Short':<10} | {'Mean Reg':<10}")
        print("-" * 60)
        
        all_max = 0
        
        for sym, df in data.items():
            max_l = df['p_long'].max()
            max_s = df['p_short'].max()
            mean_reg = df['regime'].mean()
            
            all_max = max(all_max, max_l, max_s)
            
            print(f"{sym:<10} | {max_l:.4f}     | {max_s:.4f}      | {mean_reg:.2f}")
            
            # Distribution check
            high_conf = df[(df['p_long'] > 0.6) | (df['p_short'] > 0.6)]
            if not high_conf.empty:
                 print(f"   ✅ Found {len(high_conf)} bars with prob > 0.60")
            else:
                 print(f"   ⚠️ NO SIGNALS > 0.60 found!")

        print("-" * 60)
        print(f"🚀 GLOBAL MAX CONFIDENCE: {all_max:.4f}")
        
        if all_max < 0.6:
            print("\n❌ ДИАГНОЗ: Модель 'не уверена'. Снижайте порог conf до 0.51 - 0.55.")
        elif all_max == 0:
            print("\n❌ ДИАГНОЗ: Сигналы пустые (нули). Запустите Factory заново.")
        else:
            print("\n✅ ДИАГНОЗ: Сигналы есть. Нужно просто опустить планку в Optimizer.")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect()