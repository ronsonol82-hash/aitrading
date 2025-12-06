# main.py
from datetime import datetime, timedelta
from config import Config
from data_loader import DataLoader
from indicators import FeatureEngineer
from model_engine import MLEngine
from backtester import PortfolioBacktester

def main():
    print("🚀 Запуск AI Hedge Fund System v4.0...")
    print(f"Портфель: {Config.ASSETS}")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=500) 
    
    try:
        # Качаем портфель
        portfolio_data = DataLoader.get_portfolio_data(
            Config.ASSETS, Config.LEADER_SYMBOL, 
            start_date, end_date, 
            Config.TIMEFRAME_LTF, Config.TIMEFRAME_HTF
        )
        
        print("🛠 Расчет индикаторов...")
        for sym, df in portfolio_data.items():
            df = FeatureEngineer.add_channel(df)
            df = FeatureEngineer.add_features(df)
            df = FeatureEngineer.label_data(df, Config.LOOK_AHEAD, Config.RR_RATIO)
            portfolio_data[sym] = df
            
        feature_cols = [
            'channel_pos', 'channel_slope', 'atr_rel', 'rsi', 'vol_ratio', 'trend_global', 'adx', 'dist_ema',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'is_weekend',
            'ret_1', 'ret_3', 'ret_5', 'bb_width', 'sentiment', 'sentiment_ema', 
            'macd', 'macd_signal', 'macd_hist', 'cci', 'corr_leader', 'rel_strength'
        ]
        
        print("🧠 Инициализация ИИ...")
        backtester = PortfolioBacktester(portfolio_data, MLEngine, feature_cols)
        
        print("⚔️ Старт симуляции...")
        metrics = backtester.run_simulation()
        
        print("\n" + "="*40)
        print(f"🏁 ИТОГОВЫЙ РЕЗУЛЬТАТ ПОРТФЕЛЯ")
        print("="*40)
        print(f"💰 Чистая прибыль: ${metrics['profit']:.2f}")
        print(f"📈 Sharpe Ratio:   {metrics['sharpe']:.2f}")
        print(f"🎲 Всего сделок:   {metrics['trades']}")
        print(f"⚖️ Profit Factor:  {metrics['pf']:.2f}")
        
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()