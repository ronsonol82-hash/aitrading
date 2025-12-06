# validation_report.py

"""
Валидатор стратегии SNIPER.

Ключевые отличия исправленной версии:
1. Жёсткая нормализация дат и колонок.
2. PnL по ЦЕНАМ считается отдельно (price_pnl_pct) только для диагностики.
3. Основной PnL для отчёта берётся из equity_after:
   real_pnl_pct = equity_after / equity_before - 1.
4. Кривая equity строится через компаундинг real_pnl_pct (PnL счёта).
"""

import os
import json
from dataclasses import dataclass, asdict
from typing import Dict

import numpy as np
import pandas as pd

try:
    from config import Config
except ImportError:
    # Фолбек, если config.py не найден или сломан
    class Config:
        DEPOSIT = 1000.0
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# --- НАСТРОЙКИ ВАЛИДАЦИИ ---
TRADES_CSV = "debug_trades.csv"
INITIAL_DEPOSIT = Config.DEPOSIT
REPORT_JSON = "validation_report.json"


@dataclass
class EquityStats:
    label: str
    start_date: str
    end_date: str
    start_balance: float
    final_balance: float
    total_return_pct: float
    max_drawdown_pct: float
    total_trades: int
    win_rate_pct: float
    avg_win_pct: float
    avg_loss_pct: float
    profit_factor: float
    expectancy_pct_per_trade: float
    sharpe_like: float

    def to_dict(self) -> Dict:
        return asdict(self)


def load_trades(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Файл сделок не найден: {csv_path}")

    df = pd.read_csv(csv_path)

    if df.empty:
        raise ValueError("Файл debug_trades.csv пуст.")

    # 1. Чистим названия колонок от пробелов
    df.columns = [c.strip() for c in df.columns]

    # 2. Конвертируем даты
    if "exit_date" not in df.columns:
        raise KeyError("Нет колонки 'exit_date'!")
    
    df["exit_date"] = pd.to_datetime(df["exit_date"], errors="coerce")
    if df["exit_date"].isnull().any():
        print(f"⚠️ Внимание: удалено {df['exit_date'].isnull().sum()} строк с некорректной датой выхода.")
        df = df.dropna(subset=["exit_date"])

    df = df.sort_values("exit_date").reset_index(drop=True)

    # 3. equity_after (если есть) приведём к float
    if "equity_after" in df.columns:
        df["equity_after"] = pd.to_numeric(df["equity_after"], errors="coerce")
        if df["equity_after"].isnull().any():
            print(f"⚠️ Внимание: удалено {df['equity_after'].isnull().sum()} строк с некорректным equity_after.")
            df = df.dropna(subset=["equity_after"])

    # 4. PnL по ценам – для диагностики
    if {"entry_price", "exit_price", "type"}.issubset(df.columns):
        print("🔍 Считаю price PnL (для диагностики)...")
        is_short = df["type"].astype(str).str.upper().isin(["SHORT", "SELL", "-1"])
        direction = np.where(is_short, -1.0, 1.0)
        df["price_pnl_pct"] = ((df["exit_price"] - df["entry_price"]) / df["entry_price"]) * direction
    elif "pnl" in df.columns:
        print("ℹ️ Нет цен, используем колонку 'pnl' как оценку price PnL.")
        df["price_pnl_pct"] = pd.to_numeric(df["pnl"], errors="coerce")
    else:
        raise KeyError("Не нашёл ни цен входа/выхода, ни колонки 'pnl'.")

    return df


def attach_real_pnl_from_equity(df_trades: pd.DataFrame) -> pd.DataFrame:
    """
    Главный фикс:
    - Если есть колонка equity_after — считаем PnL в терминах СЧЁТА.
    - Если её нет — используем price_pnl_pct как грубую оценку доходности счёта.
    """
    df = df_trades.copy()

    if "equity_after" in df.columns:
        df = df.sort_values("exit_date").reset_index(drop=True)

        equity = df["equity_after"].astype(float).values
        prev_equity = np.concatenate([[INITIAL_DEPOSIT], equity[:-1]])

        real_pnl = equity / prev_equity - 1.0
        df["real_pnl_pct"] = real_pnl

        print("✅ real_pnl_pct рассчитан из equity_after (PnL счёта).")
    else:
        print("⚠️ В логах нет 'equity_after'. "
              "Использую price_pnl_pct как оценку доходности счёта.")
        df["real_pnl_pct"] = df["price_pnl_pct"]

    return df

def build_global_equity(df_trades: pd.DataFrame, initial_deposit: float) -> pd.DataFrame:
    """
    Строим единую глобальную кривую депозита для ВСЕЙ истории
    на основе real_pnl_pct. Потом все срезы (годы, полугодия)
    читают старт/финиш и локальную просадку именно с этой кривой,
    а не пересчитывают депозит с нуля.
    """
    df = df_trades.copy().sort_values("exit_date").reset_index(drop=True)

    balances_before = []
    balances_after = []
    balance = float(initial_deposit)

    for _, row in df.iterrows():
        balances_before.append(balance)
        balance *= (1.0 + float(row["real_pnl_pct"]))
        if balance < 0:
            balance = 0.0
        balances_after.append(balance)

    df["equity_before_global"] = balances_before
    df["equity_after_global"] = balances_after

    return df

def simulate_equity(
    df_trades: pd.DataFrame,
    initial_deposit: float,
) -> pd.DataFrame:
    """
    Здесь real_pnl_pct трактуется как доходность СЧЁТА на сделку.
    Просто компаундинг: balance *= (1 + real_pnl_pct).
    """
    if df_trades.empty:
        raise ValueError("simulate_equity вызван с пустым набором сделок.")

    balance = initial_deposit
    balances = [balance]
    dates = [df_trades["exit_date"].iloc[0]]

    for _, row in df_trades.iterrows():
        pnl_pct = float(row["real_pnl_pct"])
        balance *= (1.0 + pnl_pct)
        if balance < 0:
            balance = 0.0
        balances.append(balance)
        dates.append(row["exit_date"])

    df_eq = pd.DataFrame({"date": dates, "balance": balances})
    df_eq["peak"] = df_eq["balance"].cummax()
    df_eq["drawdown"] = np.where(
        df_eq["peak"] > 1e-9,
        (df_eq["balance"] - df_eq["peak"]) / df_eq["peak"],
        0.0,
    )

    return df_eq


def compute_stats(label: str, df_trades: pd.DataFrame, use_global_equity: bool = True) -> EquityStats:
    if df_trades.empty:
        return EquityStats(label, "N/A", "N/A",
                           INITIAL_DEPOSIT, INITIAL_DEPOSIT,
                           0, 0, 0, 0, 0, 0, 0, 0, 0)

    # --- 1) Источник кривой депозит / просадка ---
    if use_global_equity and {"equity_before_global", "equity_after_global"}.issubset(df_trades.columns):
        # Берём кусок ИЗ общей глобальной equity-кривой
        eq_before = float(df_trades["equity_before_global"].iloc[0])
        eq_after_values = df_trades["equity_after_global"].astype(float).values

        balances = np.concatenate([[eq_before], eq_after_values])
        dates = np.concatenate([[df_trades["exit_date"].iloc[0]], df_trades["exit_date"].values])

        df_eq = pd.DataFrame({"date": dates, "balance": balances})
        df_eq["peak"] = df_eq["balance"].cummax()
        df_eq["drawdown"] = np.where(
            df_eq["peak"] > 1e-9,
            (df_eq["balance"] - df_eq["peak"]) / df_eq["peak"],
            0.0,
        )
    else:
        # Fallback: старое поведение – депозит всегда стартует с INITIAL_DEPOSIT
        df_eq = simulate_equity(df_trades, INITIAL_DEPOSIT)

    # --- 2) Доходность и просадка для этого среза ---
    start_bal = float(df_eq["balance"].iloc[0])
    final_bal = float(df_eq["balance"].iloc[-1])
    
    total_ret = (final_bal / start_bal - 1.0) * 100.0 if start_bal > 0 else 0.0
    max_dd_pct = float(df_eq["drawdown"].min() * 100.0)

    # --- 3) Статистика по сделкам (как и раньше, по real_pnl_pct) ---
    pnl_series = df_trades["real_pnl_pct"]
    wins = pnl_series[pnl_series > 0]
    losses = pnl_series[pnl_series <= 0]

    total_trades = len(pnl_series)
    win_rate = (len(wins) / total_trades * 100.0) if total_trades > 0 else 0.0
    
    avg_win = wins.mean() * 100.0 if not wins.empty else 0.0
    avg_loss = losses.mean() * 100.0 if not losses.empty else 0.0

    profit_factor = (wins.sum() / abs(losses.sum())) if losses.sum() != 0 else 999.0

    sharpe = (pnl_series.mean() / pnl_series.std() * np.sqrt(total_trades)) if pnl_series.std() > 0 else 0.0

    return EquityStats(
        label=label,
        start_date=str(df_trades["exit_date"].min()),
        end_date=str(df_trades["exit_date"].max()),
        start_balance=float(start_bal),
        final_balance=float(final_bal),
        total_return_pct=float(total_ret),
        max_drawdown_pct=float(max_dd_pct),
        total_trades=total_trades,
        win_rate_pct=float(win_rate),
        avg_win_pct=float(avg_win),
        avg_loss_pct=float(avg_loss),
        profit_factor=float(profit_factor),
        expectancy_pct_per_trade=float(pnl_series.mean() * 100.0),
        sharpe_like=float(sharpe),
    )

def split_by_years(df_trades: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    slices: Dict[str, pd.DataFrame] = {}

    df_trades = df_trades.copy()
    df_trades["year"] = df_trades["exit_date"].dt.year
    df_trades["month"] = df_trades["exit_date"].dt.month

    for year in sorted(df_trades["year"].unique()):
        df_year = df_trades[df_trades["year"] == year]
        if len(df_year) == 0:
            continue

        slices[f"{year}_full"] = df_year
        slices[f"{year}_H1"] = df_year[df_year["month"] <= 6]
        slices[f"{year}_H2"] = df_year[df_year["month"] >= 7]

    return slices


def main():
    base_dir = Config.BASE_DIR
    csv_path = os.path.join(base_dir, TRADES_CSV)

    print(f"📥 Загрузка: {csv_path}")
    
    try:
        df_trades = load_trades(csv_path)
        df_trades = attach_real_pnl_from_equity(df_trades)
        # НОВОЕ: строим единую глобальную equity-кривую
        df_trades = build_global_equity(df_trades, INITIAL_DEPOSIT)
    except Exception as e:
        print(f"❌ Критическая ошибка при загрузке/подготовке: {e}")
        return

    # Отчет по всей истории (по глобальной кривой)
    global_stats = compute_stats("FULL_HISTORY", df_trades, use_global_equity=True)
    
    # Отчет по годам / полугодиям – тоже по глобальной кривой
    slices = split_by_years(df_trades)
    all_stats = [global_stats]
    
    print("\n" + "="*60)
    print("🚀 РЕЗУЛЬТАТЫ ВАЛИДАЦИИ (Пересчитанные)".center(60))
    print("="*60)

    for label, df_slice in slices.items():
        if len(df_slice) > 0:
            stats = compute_stats(label, df_slice, use_global_equity=True)
            all_stats.append(stats)
            
            print(f"\n📅 {stats.label: <12} | Bal: {stats.start_balance:.0f} -> {stats.final_balance:.0f}")
            print(f"   Return: {stats.total_return_pct:.1f}% | DD: {stats.max_drawdown_pct:.1f}%")
            print(f"   Trades: {stats.total_trades} | WR: {stats.win_rate_pct:.1f}% | PF: {stats.profit_factor:.2f}")

    s = global_stats
    print("\n" + "#"*60)
    print(f"🌍 TOTAL HISTORY ({s.start_date} -> {s.end_date})")
    print(f"💰 {s.start_balance:.2f} -> {s.final_balance:,.2f}")
    print(f"📈 Return: {s.total_return_pct:,.2f}%")
    print(f"📉 Max Drawdown: {s.max_drawdown_pct:.2f}%")
    print(f"🎲 Trades: {s.total_trades} (WinRate: {s.win_rate_pct:.1f}%)")
    print("#"*60 + "\n")

    out_path = os.path.join(base_dir, REPORT_JSON)
    with open(out_path, "w", encoding="utf-8") as f:
        data = {x.label: x.to_dict() for x in all_stats}
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"💾 Отчет сохранен: {out_path}")


if __name__ == "__main__":
    main()
