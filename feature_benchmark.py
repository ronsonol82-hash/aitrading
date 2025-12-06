# feature_benchmark.py
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss

from config import Config
from data_loader import DataLoader
from indicators import FeatureEngineer
from model_engine import MLEngine


def evaluate_feature_set(name: str, feature_cols: list[str], df_source: pd.DataFrame):
    print("\n" + "=" * 60)
    print(f"🧪 FEATURE SET: {name}")
    print("=" * 60)

    # 1) Оставляем только реально существующие фичи
    cols = [c for c in feature_cols if c in df_source.columns]
    missing = [c for c in feature_cols if c not in df_source.columns]
    if missing:
        print(f"   ⚠️ Пропущены колонки (нет в df): {missing}")
    if len(cols) < 3:
        print("   ❌ Слишком мало фич после фильтрации, пропуск.")
        return None

    print(f"   ✅ Будем использовать {len(cols)} фич: {cols}")

    # 2) Чистим датафрейм под этот набор
    needed_cols = cols + ["target", "fwd_return"]
    df = df_source.dropna(subset=needed_cols).copy()
    if len(df) < 500:
        print(f"   ❌ Недостаточно строк после очистки: {len(df)}")
        return None

    # 3) Обучаем модель (временной сплит внутри MLEngine)
    engine = MLEngine(model_dir=None, regime_preset="auto")
    engine.train(df, cols, target_col="target")

    # 4) Предсказания по всей выборке и выделение OOS-хвоста
    probs, regimes = engine.predict_batch(df, cols)
    if probs is None:
        print("   ❌ predict_batch вернул None.")
        return None

    n = len(df)
    test_start = int(n * 0.85)
    y_true_full = df["target"].values
    fwd_full = df["fwd_return"].values

    y_test = y_true_full[test_start:]
    p_test = probs[test_start:]
    fwd_test = fwd_full[test_start:]

    # 5) Фильтр валидных строк
    valid_mask = (
        (y_test >= 0)
        & (y_test <= 2)
        & np.isfinite(p_test).all(axis=1)
        & np.isfinite(fwd_test)
    )

    y_val = y_test[valid_mask]
    p_val = p_test[valid_mask]
    fwd_val = fwd_test[valid_mask]

    if len(y_val) < 100:
        print(f"   ❌ Мало точек в тесте после фильтрации: {len(y_val)}")
        return None

    # Нормируем вероятности построчно
    row_sums = p_val.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    p_val = p_val / row_sums

    # 6) AUC / logloss
    classes_present = np.unique(y_val)
    if len(classes_present) < 2:
        auc = np.nan
        print("   ⚠️ В тесте только один класс — AUC не определяется.")
    else:
        try:
            auc = roc_auc_score(y_val, p_val, multi_class="ovr")
        except Exception as e:
            print(f"   ⚠️ Ошибка при расчёте AUC: {e}")
            auc = np.nan

    try:
        ll = log_loss(y_val, p_val, labels=[0, 1, 2])
    except Exception as e:
        print(f"   ⚠️ Ошибка при расчёте logloss: {e}")
        ll = np.nan

    # 7) Простейший Sharpe по сигналам на 1-барном горизонте
    p_neutral = p_val[:, 0]
    p_long = p_val[:, 1]
    p_short = p_val[:, 2]

    min_edge = getattr(Config, "MIN_EDGE", 0.15)
    edge_long = p_long - p_neutral
    edge_short = p_short - p_neutral

    long_mask = (p_long > p_short) & (edge_long > min_edge)
    short_mask = (p_short > p_long) & (edge_short > min_edge)

    direction = np.zeros_like(p_long)
    direction[long_mask] = 1.0
    direction[short_mask] = -1.0

    # Доходность на 1 бар вперёд * направление
    signal_ret = fwd_val * direction
    signal_ret = signal_ret[direction != 0]

    trades_count = len(signal_ret)
    if trades_count > 1 and signal_ret.std() > 1e-8:
        # Условно-годовой Sharpe, просто для сравнения между наборами
        sharpe = (signal_ret.mean() / signal_ret.std()) * np.sqrt(365)
    else:
        sharpe = 0.0

    print(f"   🧾 OOS samples: {len(y_val)} | Signals: {trades_count}")
    print(f"   🎯 AUC:      {auc:.4f}" if not np.isnan(auc) else "   🎯 AUC:      n/a")
    print(f"   📉 LogLoss:  {ll:.4f}" if not np.isnan(ll) else "   📉 LogLoss:  n/a")
    print(f"   ♠️ Sharpe:   {sharpe:.3f}")

    return {
        "name": name,
        "n_features": len(cols),
        "n_oos": int(len(y_val)),
        "signals": int(trades_count),
        "auc": float(auc) if not np.isnan(auc) else np.nan,
        "logloss": float(ll) if not np.isnan(ll) else np.nan,
        "sharpe": float(sharpe),
    }


def main():
    print("🧪 FEATURE LAB v1.0 — сравнение наборов фич на одном датасете")
    print(f"   Лидер: {Config.LEADER_SYMBOL}, ТФ: {Config.TIMEFRAME_LTF}")

    end = datetime.now()
    start = end - timedelta(days=1800)

    print(f"   Период данных: {start} -> {end}")

    data = DataLoader.get_portfolio_data(
        [Config.LEADER_SYMBOL],
        Config.LEADER_SYMBOL,
        start,
        end,
        Config.TIMEFRAME_LTF,
        Config.TIMEFRAME_HTF,
    )

    if not data or Config.LEADER_SYMBOL not in data:
        print("❌ DataLoader не вернул данных по лидеру, выход.")
        sys.exit(1)

    df_raw = data[Config.LEADER_SYMBOL]
    if df_raw is None or df_raw.empty:
        print("❌ Пустой DataFrame по лидеру.")
        sys.exit(1)

    # Фичи + таргет
    print("   🛠 Расчёт фичей и таргета...")
    df_feat = FeatureEngineer.add_features(df_raw.copy())
    df_lbl = FeatureEngineer.label_data(df_feat, Config.LOOK_AHEAD, Config.RR_RATIO)

    # Простая 1-барная доходность вперёд, чтобы оценить качество сигналов
    df_lbl["fwd_return"] = df_lbl["close"].shift(-1) / df_lbl["close"] - 1.0

    # Чистим и фиксируем индекс
    df_lbl = df_lbl.dropna(subset=["target", "fwd_return"]).reset_index(drop=True)

    # 3–4 сценария фич (градиент от простого к полному)
    feature_sets = {
        "LEVELS_ONLY": [
            "confluence_score",
            "sup_dist_atr",
            "res_dist_atr",
            "sup_strength",
            "res_strength",
            "level_quality",
        ],
        "LEVELS_PLUS_MOMENTUM": [
            "confluence_score",
            "sup_dist_atr",
            "res_dist_atr",
            "sup_strength",
            "res_strength",
            "level_quality",
            "rvol",
            "rsi",
            "adx",
            "volatility",
        ],
        "STANDARD_FEATURE_COLS": Config.FEATURE_COLS,
        "UNIVERSAL_FEATURE_COLS": Config.UNIVERSAL_FEATURE_COLS,
    }

    results = []
    for name, cols in feature_sets.items():
        res = evaluate_feature_set(name, cols, df_lbl)
        if res is not None:
            results.append(res)

    if not results:
        print("\n❌ Не удалось посчитать ни один сценарий.")
        return

    print("\n" + "=" * 60)
    print("📊 SUMMARY (sorted by Sharpe)")
    print("=" * 60)

    df_res = pd.DataFrame(results)
    df_res = df_res.sort_values("sharpe", ascending=False)
    with pd.option_context("display.max_columns", 10, "display.width", 120):
        print(df_res.to_string(index=False, float_format=lambda x: f"{x:0.4f}"))


if __name__ == "__main__":
    main()
