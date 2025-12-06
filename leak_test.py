import pandas as pd
import numpy as np
import xgboost as xgb
import sys
import os
import inspect

# --- НАСТРОЙКА ПУТЕЙ ---
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# --- ЗАГРУЗКА МОДУЛЕЙ ---
try:
    from data_loader import DataLoader
    import features_lib
    import signal_generator
except ImportError as e:
    print(f"❌ Ошибка импорта основных модулей: {e}")
    sys.exit(1)

def auto_load_data(loader, symbol):
    """
    Загружает данные для одного инструмента тем же способом, что и фабрики сигналов:
    через DataLoader.get_portfolio_data.
    """
    from datetime import datetime, timedelta
    from config import Config

    print("🕵️‍♂️ Ищу метод загрузки данных...")

    try:
        end = datetime.now()
        # Берём тот же горизонт, что и в SignalFactory/UniversalSignalFactory
        start = end - timedelta(days=2500)

        print("🔎 Использую DataLoader.get_portfolio_data (как в производстве сигналов).")
        data_dict = DataLoader.get_portfolio_data(
            [symbol],
            Config.LEADER_SYMBOL,
            start,
            end,
            Config.TIMEFRAME_LTF,
            Config.TIMEFRAME_HTF,
        )

        if not isinstance(data_dict, dict) or symbol not in data_dict:
            print("❌ get_portfolio_data не вернул данные для символа.")
            return None

        df = data_dict[symbol]
        if df is None or df.empty:
            print("❌ Получен пустой DataFrame после get_portfolio_data.")
            return None

        print(f"✅ Данные получены через get_portfolio_data: {len(df)} строк.")
        return df

    except Exception as e:
        print(f"❌ Ошибка при загрузке через get_portfolio_data: {e}")
        return None

def auto_generate_features(df):
    """
    Генерирует фичи тем же способом, что и в боевом пайплайне:
    через indicators.FeatureEngineer.add_features.
    """
    print("🕵️‍♂️ Ищу генератор фичей...")

    try:
        from indicators import FeatureEngineer

        print("✅ Использую indicators.FeatureEngineer.add_features.")
        df_feat = FeatureEngineer.add_features(df.copy())
        return df_feat
    except Exception as e:
        print(f"❌ Не удалось применить FeatureEngineer.add_features: {e}")
        return None

def analyze_leak():
    print("\n" + "="*50)
    print("🕵️‍♂️ DETECTIVE V4: ROBUST MODE")
    print("="*50)

    # 1. Load Data
    loader = DataLoader()
    symbol = 'BTCUSDT'
    
    df = auto_load_data(loader, symbol)

    if df is None or df.empty:
        print("❌ CRITICAL: Не удалось загрузить данные.")
        return
    print(f"📊 Загружено {len(df)} свечей.")

    # 2. Generate Features
    close_prices = df['close'].copy()
    df_features = auto_generate_features(df.copy())
    
    if df_features is None:
        print("❌ CRITICAL: Фичи не созданы.")
        return

    # Проверка на добавление колонок
    if len(df_features.columns) == len(df.columns):
        print("⚠️ WARNING: Количество колонок не изменилось. Возможно, генератор не сработал.")

    # 3. Подготовка X и y (Shift(-1) = Future)
    target = (close_prices.shift(-1) / close_prices) - 1.0
    
    valid_idx = target.dropna().index.intersection(df_features.index)
    X = df_features.loc[valid_idx].copy()
    y = target.loc[valid_idx]
    
    # Чистка
    drop_cols = ['open', 'high', 'low', 'close', 'volume', 'date', 'timestamp', 'symbol', 'target', 'TARGET', 'open_time', 'close_time']
    X = X.drop(columns=[c for c in drop_cols if c in X.columns], errors='ignore')
    X = X.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).fillna(0)

    print(f"\n🔍 АНАЛИЗ {len(X.columns)} ФИЧЕЙ...")

    # 4. Correlation Check
    corrs = []
    for col in X.columns:
        if X[col].nunique() > 1:
            c = X[col].corr(y)
            if not np.isnan(c): corrs.append((col, abs(c)))
            
    corrs.sort(key=lambda x: x[1], reverse=True)
    
    print("\n📊 ТОП-10 КОРРЕЛЯЦИЙ С БУДУЩИМ:")
    for name, val in corrs[:10]:
        status = "🟢"
        if val > 0.15: status = "⚠️"
        if val > 0.8: status = "🚨 LEAK!"
        print(f"{status} {name:<30} : {val:.4f}")

    # 5. XGBoost Check
    if len(X.columns) > 0:
        print("\n🌲 XGBoost Check...")
        model = xgb.XGBRegressor(n_estimators=50, max_depth=3, n_jobs=-1, random_state=42)
        model.fit(X, y)
        imps = sorted(list(zip(X.columns, model.feature_importances_)), key=lambda x: x[1], reverse=True)
        
        print("\n🏆 ТОП ВАЖНОСТИ (XGBoost):")
        for name, val in imps[:5]:
             print(f"{name:<30} : {val:.4f} {'🚨' if val > 0.9 else ''}")
             
        # Вердикт
        top_corr = corrs[0][1] if corrs else 0
        top_imp = imps[0][1] if imps else 0
        
        print("\n" + "="*40)
        if top_corr > 0.9 or top_imp > 0.95:
            print("🛑 ОБНАРУЖЕНА УТЕЧКА! Бот знает будущее.")
        elif top_corr > 0.2:
            print("⚠️ ЕСТЬ ПОДОЗРИТЕЛЬНЫЕ ФИЧИ (>0.2). Проверь их.")
        else:
            print("✅ УТЕЧЕК НЕТ. Результаты теста чистые.")
        print("="*40)

if __name__ == "__main__":
    analyze_leak()