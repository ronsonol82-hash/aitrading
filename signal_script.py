import pandas as pd
import pickle

file_path = "data_cache/production_signals_v1.pkl"
try:
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    
    print(f"Активов в файле: {len(data)}")
    for sym, df in list(data.items())[:3]: # Проверим первые 3 актива
        max_long = df['p_long'].max()
        max_short = df['p_short'].max()
        print(f"Asset: {sym} | Max P_Long: {max_long:.4f} | Max P_Short: {max_short:.4f}")
        
        if max_long == 0 and max_short == 0:
            print("⚠️ ВНИМАНИЕ: Модель вернула нули! (Скорее всего, сработал ConstantModel)")
except Exception as e:
    print(f"Ошибка чтения файла: {e}")