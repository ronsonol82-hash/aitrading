# noise_radar.py
import numpy as np
import pandas as pd
from data_loader import DataLoader

def get_hurst_exponent(time_series, max_lag=20):
    """
    Вычисляет показатель Хёрста.
    H < 0.5 - Mean Reverting (Шум/Флэт)
    H ~ 0.5 - Random Walk (Полный хаос)
    H > 0.5 - Trending (Трендовый, можно торговать)
    """
    lags = range(2, max_lag)
    tau = [np.sqrt(np.std(np.subtract(time_series[lag:], time_series[:-lag]))) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0

def analyze_market_noise():
    print("📡 SCANNIG MARKET NOISE STRUCTURE...")
    df = DataLoader.get_binance_data("BTCUSDT", pd.to_datetime("2024-01-01"), pd.to_datetime("2024-05-01"), "1h")
    
    closes = df['close'].values
    
    # Скользящее окно Хёрста (например, за неделю - 168 часов)
    window = 168 
    hurst_values = []
    
    print(f"   Analysing rolling Hurst exponent (Window: {window} bars)...")
    for i in range(window, len(closes)):
        segment = closes[i-window:i]
        h = get_hurst_exponent(segment)
        hurst_values.append(h)
        
    avg_hurst = np.mean(hurst_values)
    min_hurst = np.min(hurst_values)
    max_hurst = np.max(hurst_values)
    
    print("\n📉 NOISE REPORT:")
    print(f"   Avg Hurst: {avg_hurst:.4f}")
    print(f"   Min Hurst: {min_hurst:.4f} (Chaos/MeanReversion)")
    print(f"   Max Hurst: {max_hurst:.4f} (Strong Trend)")
    
    valid_trading_time = sum(1 for h in hurst_values if h > 0.5) / len(hurst_values)
    print(f"   Tradable Time (H > 0.5): {valid_trading_time*100:.1f}%")
    
    if avg_hurst < 0.45:
        print("\n❌ VERDICT: MARKET IS NOISY. Increase timeframe or look for other assets.")
    elif avg_hurst > 0.55:
        print("\n✅ VERDICT: MARKET IS TRENDING. Good for breakout strategies.")
    else:
        print("\n⚠️ VERDICT: RANDOM WALK. Be careful with breakouts.")

if __name__ == "__main__":
    analyze_market_noise()