# test_gpu.py
import sys
import time

print("🔍 STARTING GPU DIAGNOSTICS...")
print("------------------------------------------------")

# 1. Проверка PyTorch (для FinBERT)
try:
    import torch
    print(f"📚 PyTorch Version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"✅ PyTorch CUDA Available: YES")
        print(f"   Device Name: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM Total:  {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("⚠️ PyTorch CUDA: NO (FinBERT будет работать на CPU, это медленно)")
except ImportError:
    print("❌ PyTorch не установлен.")

print("------------------------------------------------")

# 2. Проверка XGBoost (для торговли)
try:
    import xgboost as xgb
    import numpy as np
    
    print(f"📚 XGBoost Version: {xgb.__version__}")
    print("🧪 Запуск тестового обучения на GPU...")
    
    # Создаем микро-датасет
    X = np.random.rand(1000, 10)
    y = np.random.randint(0, 2, 1000)
    
    # Пытаемся обучить с tree_method='hist' и device='cuda'
    clf = xgb.XGBClassifier(
        tree_method='hist', 
        device='cuda', 
        n_estimators=10,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    start = time.time()
    clf.fit(X, y)
    end = time.time()
    
    print(f"✅ XGBoost GPU Test Passed! (Time: {end - start:.4f}s)")
    
except xgb.core.XGBoostError as e:
    print(f"❌ XGBoost Error: {e}")
    print("   Скорее всего, проблема с драйверами или версией CUDA.")
except Exception as e:
    print(f"❌ General Error: {e}")

print("------------------------------------------------")
print("🏁 DIAGNOSTICS FINISHED.")