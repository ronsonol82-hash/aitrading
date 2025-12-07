# debug_replayer.py
import pandas as pd
import numpy as np
import pickle
import os
import sys
from datetime import datetime, timedelta
import argparse

# Импортируем наши модули
from config import Config
from data_loader import DataLoader
from backtester import PortfolioBacktester
from model_engine import MLEngine
from visualizer import TradeVisualizer
from indicators import FeatureEngineer 

def load_signals():
    path = "data_cache/production_signals_v1.pkl"
    if not os.path.exists(path):
        print(f"❌ Файл сигналов {path} не найден! Сначала запусти signal_generator.py")
        sys.exit(1)
    
    with open(path, "rb") as f:
        return pickle.load(f)

def run_debug(oos_start_str=None, enable_plots: bool = False, asset_class: str = "all"):
    print("🐞 [DEBUG] Запуск реплеера сделок...")

    # 1. Загружаем сигналы (чтобы знать диапазон дат)
    signals_map = load_signals()
    first_sym = list(signals_map.keys())[0]
    
    start_date = signals_map[first_sym].index[0]
    end_date = signals_map[first_sym].index[-1]
    
    print(f"   📅 Период: {start_date} -> {end_date}")

    # Опциональная граница OOS
    oos_start = None
    if oos_start_str:
        try:
            oos_start = pd.to_datetime(oos_start_str)
            print(f"   🚧 OOS-граница: {oos_start}")
        except Exception:
            print(f"⚠️ Не получилось распарсить oos_start={oos_start_str}")
    # ⬇️ Загрузка рынка — ВСЕГДА, вне if
    print("   📥 Загрузка рыночных данных...")

    leader_map = {sym: Config.get_leader_for_symbol(sym) for sym in Config.ASSETS}

    market_data = DataLoader.get_portfolio_data(
        Config.ASSETS,
        leader_map,
        start_date - timedelta(days=20),
        end_date + timedelta(days=1),
        Config.TIMEFRAME_LTF,
        Config.TIMEFRAME_HTF,
    )
    
    print("   🛠 Calculating indicators...")
    for sym in market_data:
        market_data[sym] = FeatureEngineer.add_features(market_data[sym])

    # 3. Подмешиваем сигналы
    clean_data = {}
    for sym, df in market_data.items():
        if sym in signals_map:
            sig_df = signals_map[sym]
            common_idx = df.index.intersection(sig_df.index)
            if common_idx.empty: continue
            
            df_slice = df.loc[common_idx].copy()
            sig_slice = sig_df.loc[common_idx]
            
            df_slice['p_long'] = sig_slice['p_long']
            df_slice['p_short'] = sig_slice['p_short']
            df_slice['regime'] = sig_slice['regime']
            
            clean_data[sym] = df_slice
    
    if not clean_data:
        print("❌ Ошибка: Нет пересечения дат.")
        return

    # 4. Запускаем Бэктест
    print("   🚀 Запуск симуляции...")
    backtester = PortfolioBacktester(clean_data, MLEngine, Config.FEATURE_COLS)
    
    results = backtester.run_simulation()
    
    # После бэктеста
    trades = results["closed_trades"]
    equity = results["equity"]

    # --- NEW: фильтрация по классу актива ---
    trades = filter_trades_by_asset_class(trades, asset_class)
    print(f"🔎 [DEBUG] Asset class filter: {asset_class}, trades after filter: {len(trades)}")
    
    # === [CRITICAL UPDATE] СИНХРОНИЗАЦИЯ БАЛАНСА ===
    # Добавляем в таблицу сделок реальный баланс на момент закрытия.
    # Это нужно для plot_equity.py
    if not trades.empty and not equity.empty:
        # Мапим дату выхода на баланс этого дня/свечи
        # Используем .map по индексу equity
        trades['equity_after'] = trades['exit_date'].map(equity['balance'])
        
        # Если вдруг даты не совпали (редко, но бывает при ресемплинге), заполняем ffill
        if trades['equity_after'].isnull().any():
            trades['equity_after'] = trades['equity_after'].fillna(method='ffill')
    # ===============================================

    # 5. РАСШИРЕННАЯ СТАТИСТИКА (TRACE OUTPUT)
    print("\n" + "="*50)
    print("📊 EXTENDED PERFORMANCE REPORT")
    print("="*50)
    
    if not trades.empty:
        # PnL Analysis (по всему портфелю)
        wins = trades[trades['pnl'] > 0]
        losses = trades[trades['pnl'] <= 0]
        win_rate = len(wins) / len(trades)
        
        avg_win = wins['pnl'].mean() * 100 if not wins.empty else 0.0
        avg_loss = losses['pnl'].mean() * 100 if not losses.empty else 0.0
        
        # Sharpe Calculation (Annualized) по дневным изменениям equity
        equity['returns'] = equity['balance'].pct_change()
        returns = equity['returns'].dropna()
        sharpe = 0.0
        if len(returns) > 1:
            std = returns.std()
            if std > 1e-8:
                # *6 — примерно 6 четырёхчасовых свечей в день
                sharpe = (returns.mean() / std) * np.sqrt(365 * 6)
        
        print(f"💰 Final Balance:   ${equity['balance'].iloc[-1]:.2f}")
        print(f"📈 Total Return:    {results['total_return']*100:.2f}%")
        print(f"📉 Max Drawdown:    {results['max_drawdown']*100:.2f}%")
        print(f"🛒 Total Trades:    {len(trades)}")
        print(f"✅ Win Rate:        {win_rate*100:.2f}% ({len(wins)} W / {len(losses)} L)")
        print(f"⚖️ Avg Win/Loss:    +{avg_win:.2f}% / {avg_loss:.2f}%")
        print(f"♠️ Sharpe Ratio:    {sharpe:.2f}")
        
        # --- НОВОЕ: пер-символьная статистика ---
        print("\n📈 PER-SYMBOL STATS:")
        print("-" * 80)
        for sym in sorted(trades['symbol'].unique()):
            sym_trades = trades[trades['symbol'] == sym]
            n = len(sym_trades)
            if n == 0:
                continue
            
            sym_wins = sym_trades[sym_trades['pnl'] > 0]
            sym_losses = sym_trades[sym_trades['pnl'] <= 0]
            sym_wr = len(sym_wins) / n if n > 0 else 0.0
            sym_total_pnl = sym_trades['pnl'].sum() * 100
            sym_avg_pnl = sym_trades['pnl'].mean() * 100
            
            print(
                f"{sym:8} | Trades: {n:4d} | "
                f"WinRate: {sym_wr*100:5.1f}% | "
                f"TotalPnL: {sym_total_pnl:7.2f}% | "
                f"AvgPnL: {sym_avg_pnl:6.2f}%"
            )
        
        print("\n📜 LAST 20 TRADES:")
        print("-" * 80)
        # Форматированный вывод последних сделок
        last_trades = trades.tail(20).copy()
        last_trades['pnl_usd'] = last_trades['pnl'] * 1000  # Примерно от депозита
        print(
            last_trades[
                ['entry_date', 'symbol', 'type', 'entry_price', 'exit_price', 'pnl', 'reason']
            ].to_string()
        )
        print("-" * 80)
        
        # Сохранение
        save_path = "debug_trades.csv"
        trades.to_csv(save_path, index=False)
        print(f"\n💾 Full trade log saved to: {save_path}")

    else:
        print("🤷‍♂️ Сделок не было.")

    # 6. ВИЗУАЛИЗАЦИЯ (по запросу)
    if enable_plots and not trades.empty:
        print("\n🎨 [VISUALIZER] Генерируем графики (по флагу enable_plots=True).")
        vis = TradeVisualizer()
        active_symbols = trades['symbol'].unique()
        for sym in active_symbols:
            print(f"   Opening chart for {sym}.")
            df_vis = clean_data[sym]
            vis.plot_trades(
                symbol=sym,
                df=df_vis,
                trades=trades,
                title_suffix="[DEBUG REPLAY]",
                oos_start=oos_start,
            )

def classify_symbol(symbol: str) -> str:
    """
    Грубая эвристика:
    - если тикер заканчивается на USDT/USDC/BTC/ETH → считаем криптой;
    - иначе → считаем стоком (MOEX/FX/прочее).
    """
    if not isinstance(symbol, str):
        return "stocks"
    s = symbol.upper()
    if s.endswith(("USDT", "USDC", "BTC", "ETH")):
        return "crypto"
    return "stocks"


def filter_trades_by_asset_class(trades_df: pd.DataFrame, asset_class: str) -> pd.DataFrame:
    """
    trades_df: pandas.DataFrame с колонкой 'symbol'
    asset_class: 'all' | 'crypto' | 'stocks'
    """
    if asset_class == "all":
        return trades_df

    if "symbol" not in trades_df.columns:
        # ничего не знаем → не фильтруем, но можно вывести warning
        print("⚠️ No 'symbol' column in trades data, cannot filter by asset class.")
        return trades_df

    mask = trades_df["symbol"].apply(classify_symbol)
    return trades_df[mask == asset_class].copy()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug Replayer")
    parser.add_argument(
        "--oos_start",
        type=str,
        default=None,
        help="Дата (YYYY-MM-DD или YYYY-MM-DD HH:MM:SS), с которой считать OOS-период",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Рисовать графики эквити/просадки и сделок.",
    )
    # --- NEW: выбор класса актива ---
    parser.add_argument(
        "--asset_class",
        type=str,
        default="all",
        choices=["all", "crypto", "stocks"],
        help="Фильтр сделок по классу актива: all / crypto / stocks",
    )

    args = parser.parse_args()

    try:
        run_debug(
            oos_start_str=args.oos_start,
            enable_plots=args.plot,
            asset_class=args.asset_class,
        )
    except KeyboardInterrupt:
        print("\n🛑 Прервано пользователем.")
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()