# visualizer.py
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime

# Импортируем "мозги", чтобы визуализатор рисовал то же самое, что видит бот
try:
    from features_lib import StructureFeatures, find_quality_levels_numba
except ImportError:
    print("⚠️ Ошибка импорта features_lib. Убедитесь, что файл существует.")

class TradeVisualizer:
    def __init__(self):
        pass

    def plot_trades(self, symbol, df, trades, title_suffix="", oos_start=None):
        """
        Рисует свечной график с наложением:
        1. Сделок (треугольники)
        2. Каналов (Голубые линии)
        3. Уровней (Красные/Зеленые линии)
        4. Эквити (снизу)
        """
        if df.empty:
            print(f"⚠️ Нет данных для {symbol}")
            return

        # --- 1. ПОДГОТОВКА ДАННЫХ ДЛЯ ОТРИСОВКИ ---
        
        # Пересчитываем Канал (Голубые линии), чтобы быть уверенными
        try:
            ch_up, ch_lo, _, _ = StructureFeatures.calc_linreg_channel(df['close'], window=96)
        except:
            ch_up = df['close'] * 1.05; ch_lo = df['close'] * 0.95

        # Пересчитываем Уровни (Красные линии)
        try:
            # Для Numba нужны numpy массивы
            highs = df['high'].values.astype(np.float64)
            lows = df['low'].values.astype(np.float64)
            closes = df['close'].values.astype(np.float64)
            atr = df['atr'].fillna(0).values.astype(np.float64)
            
            sup_price, _, res_price, _ = find_quality_levels_numba(highs, lows, closes, atr, window=10)
        except:
            sup_price = np.full(len(df), np.nan)
            res_price = np.full(len(df), np.nan)

        # Создаем полотно (2 окна: График цены и Эквити)
        fig = make_subplots(
            rows=2, cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.05, 
            row_heights=[0.75, 0.25],
            subplot_titles=(f"{symbol} Chart {title_suffix}", "Equity Curve")
        )

        # --- 2. СВЕЧИ ---
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['open'], high=df['high'], low=df['low'], close=df['close'],
            name='Price'
        ), row=1, col=1)

        # --- 3. ТЕХНИЧЕСКИЕ ЛИНИИ (ТВОЯ СТРАТЕГИЯ) ---
        
        # Голубой Канал (Верх)
        fig.add_trace(go.Scatter(
            x=df.index, y=ch_up,
            mode='lines',
            line=dict(color='rgba(0, 191, 255, 0.8)', width=1, dash='dash'), # DeepSkyBlue
            name='Channel Top'
        ), row=1, col=1)

        # Голубой Канал (Низ)
        fig.add_trace(go.Scatter(
            x=df.index, y=ch_lo,
            mode='lines',
            line=dict(color='rgba(0, 191, 255, 0.8)', width=1, dash='dash'),
            fill='tonexty', # Заливка между линиями (Туннель)
            fillcolor='rgba(0, 191, 255, 0.05)',
            name='Channel Bottom'
        ), row=1, col=1)

        # Уровни Поддержки (Зеленые ступеньки)
        # Используем shape='hv' (Horizontal-Vertical), чтобы рисовать полки
        fig.add_trace(go.Scatter(
            x=df.index, y=sup_price,
            mode='lines',
            line=dict(color='rgba(50, 205, 50, 0.6)', width=1, shape='hv'), # LimeGreen
            name='Support Level'
        ), row=1, col=1)

        # Уровни Сопротивления (Красные ступеньки)
        fig.add_trace(go.Scatter(
            x=df.index, y=res_price,
            mode='lines',
            line=dict(color='rgba(255, 69, 0, 0.6)', width=1, shape='hv'), # OrangeRed
            name='Resistance Level'
        ), row=1, col=1)
        
        # EMA Trend (Желтая ватерлиния)
        if 'ema_trend' in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df['ema_trend'],
                line=dict(color='rgba(255, 215, 0, 0.5)', width=2),
                name='EMA 200'
            ), row=1, col=1)

        # --- 4. СДЕЛКИ ---
        # Фильтруем сделки только для этого символа
        sym_trades = trades[trades['symbol'] == symbol].copy()
        
        if not sym_trades.empty:
            # Разделяем на Long и Short
            longs = sym_trades[sym_trades['type'] == 'LONG']
            shorts = sym_trades[sym_trades['type'] == 'SHORT']

            # Маркеры входа LONG (Зеленый треугольник вверх)
            fig.add_trace(go.Scatter(
                x=longs['entry_date'], y=longs['entry_price'],
                mode='markers',
                marker=dict(symbol='triangle-up', size=10, color='lime'),
                name='Long Entry',
                customdata=longs[['reason', 'pnl']],
                hovertemplate="Long Entry<br>Reason: %{customdata[0]}<br>PnL: %{customdata[1]:.2f}"
            ), row=1, col=1)
            
            # Маркеры выхода LONG (Крестик)
            fig.add_trace(go.Scatter(
                x=longs['exit_date'], y=longs['exit_price'],
                mode='markers',
                marker=dict(symbol='x', size=8, color='white', line=dict(width=2, color='lime')),
                name='Long Exit'
            ), row=1, col=1)

            # Маркеры входа SHORT (Красный треугольник вниз)
            fig.add_trace(go.Scatter(
                x=shorts['entry_date'], y=shorts['entry_price'],
                mode='markers',
                marker=dict(symbol='triangle-down', size=10, color='red'),
                name='Short Entry',
                customdata=shorts[['reason', 'pnl']],
                hovertemplate="Short Entry<br>Reason: %{customdata[0]}<br>PnL: %{customdata[1]:.2f}"
            ), row=1, col=1)
            
            # Маркеры выхода SHORT
            fig.add_trace(go.Scatter(
                x=shorts['exit_date'], y=shorts['exit_price'],
                mode='markers',
                marker=dict(symbol='x', size=8, color='white', line=dict(width=2, color='red')),
                name='Short Exit'
            ), row=1, col=1)
            
            # Линии сделок (Соединяем вход и выход)
            for _, t in sym_trades.iterrows():
                color = 'green' if t['pnl'] > 0 else 'red'
                fig.add_trace(go.Scatter(
                    x=[t['entry_date'], t['exit_date']],
                    y=[t['entry_price'], t['exit_price']],
                    mode='lines',
                    line=dict(color=color, width=1, dash='dot'),
                    showlegend=False
                ), row=1, col=1)

        # --- 5. ЭКВИТИ ---
        # Если передан DataFrame с эквити
        if 'balance' in df.columns:
             fig.add_trace(go.Scatter(
                x=df.index, y=df['balance'],
                line=dict(color='#00ff00', width=2),
                fill='tozeroy',
                fillcolor='rgba(0, 255, 0, 0.1)',
                name='Equity'
            ), row=2, col=1)

        # Вертикальная линия OOS start, если задана
        if oos_start is not None:
            try:
                dt_oos = pd.to_datetime(oos_start)

                # Линия на верхнем графике (цена)
                fig.add_vline(
                    x=dt_oos,
                    line_dash="dash",
                    line_width=1.5,
                    line_color="red",
                    row=1,
                    col=1,
                )

                # Линия на нижнем графике (equity), чтобы глаз видел одну границу
                fig.add_vline(
                    x=dt_oos,
                    line_dash="dash",
                    line_width=1.0,
                    line_color="red",
                    row=2,
                    col=1,
                )

                # Небольшая подпись сверху
                fig.add_annotation(
                    x=dt_oos,
                    y=1.02,
                    xref="x",
                    yref="paper",
                    text="OOS",
                    showarrow=False,
                    font=dict(color="red", size=10),
                )
            except Exception:
                # Если дата кривая — просто тихо игнорируем
                pass

        # --- НАСТРОЙКИ ---
        fig.update_layout(
            template='plotly_dark',
            height=900,
            title_text=f"Analysis: {symbol}",
            xaxis_rangeslider_visible=False,
            hovermode='x unified'
        )
        
        # Показываем график
        fig.show()

# Для быстрого тестирования (если запускаем файл напрямую)
if __name__ == "__main__":
    print("Запустите backtester.py или debug_replayer.py для использования визуализатора.")