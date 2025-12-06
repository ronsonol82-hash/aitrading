import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

# --- НАСТРОЙКИ ---
FILENAME = 'debug_trades.csv'

def print_drawdown_details(df):
    """
    Анализирует просадку по каждой сделке и находит самую глубокую яму.
    """
    equity = df['balance'].values
    dates = df['date'].values
    
    # 1. High Water Mark (Тонкая красная линия)
    # cummax() бежит по массиву и запоминает "самое большое число, которое я видел до сих пор"
    peaks = np.maximum.accumulate(equity)
    
    # 2. Просадка в процентах для каждой точки
    drawdowns = (equity - peaks) / peaks
    
    # 3. Поиск глобального дна
    max_dd_idx = np.argmin(drawdowns)
    max_dd_val = drawdowns[max_dd_idx]
    
    # 4. Поиск пика, с которого началось это конкретное падение
    peak_val = peaks[max_dd_idx]
    # Ищем последнюю точку перед дном, где баланс был равен пику
    peak_idx = np.where(equity[:max_dd_idx] == peak_val)[0][-1]
    
    print("\n" + "="*45)
    print("🩸 АУДИТ БОЛИ (Max Drawdown Analysis)")
    print("="*45)
    print(f"📉 Макс. просадка (Depth): {max_dd_val * 100:.2f}%")
    print(f"🏔  Пик перед падением:     ${peak_val:.2f} (Дата: {pd.to_datetime(dates[peak_idx]).date()})")
    print(f"🕳  Дно просадки:           ${equity[max_dd_idx]:.2f} (Дата: {pd.to_datetime(dates[max_dd_idx]).date()})")
    print(f"💸 Потеряно от пика:       ${(peak_val - equity[max_dd_idx]):.2f}")
    
    duration = pd.to_datetime(dates[max_dd_idx]) - pd.to_datetime(dates[peak_idx])
    print(f"⏳ Время падения на дно:    {duration.days} дней")
    
    # Recovery (восстановление)
    # Пытаемся найти, когда мы снова пробили этот пик
    recovery_slice = equity[max_dd_idx:]
    recovery_dates = dates[max_dd_idx:]
    recovered_idx = np.where(recovery_slice >= peak_val)[0]
    
    if len(recovered_idx) > 0:
        rec_date = pd.to_datetime(recovery_dates[recovered_idx[0]])
        full_duration = rec_date - pd.to_datetime(dates[peak_idx])
        print(f"✅ Восстановление заняло:   {full_duration.days} дней (Дата: {rec_date.date()})")
    else:
        print(f"⚠️ Восстановление:          ЕЩЕ НЕ ВОССТАНОВИЛСЯ (Drawdown Active)")
        
    print("="*45 + "\n")

def plot_equity_curve():
    # 1. Загрузка
    try:
        df = pd.read_csv(FILENAME)
    except FileNotFoundError:
        print(f"❌ Файл {FILENAME} не найден! Сначала запустите run_debug.")
        return

    if df.empty or 'equity_after' not in df.columns:
        print("❌ ОШИБКА: Нет данных или колонки equity_after.")
        return

    # 2. Подготовка
    df['exit_date'] = pd.to_datetime(df['exit_date'])
    df = df.sort_values(by='exit_date')

    # Формируем красивую линию времени с точкой старта
    start_date = df['exit_date'].iloc[0] - pd.Timedelta(days=1)
    
    # Предполагаем старт с 1000 (или берем equity первой сделки минус профит)
    # Для простоты: начнем график с 1000.
    start_balance = 1000.0
    
    dates = [start_date] + df['exit_date'].tolist()
    equity = [start_balance] + df['equity_after'].tolist()

    df_equity = pd.DataFrame({'date': dates, 'balance': equity})
    
    # --- ГЛАВНАЯ МАТЕМАТИКА ---
    # High Water Mark (Та самая красная линия)
    df_equity['peak'] = df_equity['balance'].cummax()
    # Drawdown Curve (Кривая боли)
    df_equity['drawdown'] = (df_equity['balance'] - df_equity['peak']) / df_equity['peak']

    # Вывод статистики в консоль
    final_balance = df_equity['balance'].iloc[-1]
    total_return = ((final_balance / start_balance) - 1) * 100
    print_drawdown_details(df_equity)

    print(f"📊 ИТОГ: ${start_balance} -> ${final_balance:.2f} ({total_return:.2f}%)")

    # 3. ВИЗУАЛИЗАЦИЯ
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    
    # === Верхний график: Equity + High Water Mark ===
    
    # 1. Зеленая линия: Реальный баланс
    ax1.plot(df_equity['date'], df_equity['balance'], color='#00ff00', linewidth=1.5, label='Equity (Real)')
    
    # 2. Красная линия: High Water Mark (Идеал, к которому стремимся)
    ax1.plot(df_equity['date'], df_equity['peak'], color='#ff3333', linewidth=1.0, linestyle='--', alpha=0.8, label='High Water Mark (Max Balance)')
    
    # 3. Заливка между ними (визуализация упущенных денег)
    ax1.fill_between(df_equity['date'], df_equity['balance'], df_equity['peak'], color='red', alpha=0.15, label='Drawdown Area')
    
    # Точки сделок
    if len(df) < 800: # Рисуем точки только если не слишком тесно
        # Для точек нам нужно сопоставить даты сделок
        # Берем данные из исходного df сделок
        winners = df[df['pnl'] > 0]
        losers = df[df['pnl'] <= 0]
        ax1.scatter(winners['exit_date'], winners['equity_after'], color='lime', s=15, alpha=0.6, zorder=3)
        ax1.scatter(losers['exit_date'], losers['equity_after'], color='red', s=15, alpha=0.6, zorder=3)

    ax1.set_title(f'Equity Curve vs High Water Mark (Net Profit: {total_return:.1f}%)', fontsize=14, color='white', fontweight='bold')
    ax1.set_ylabel('Balance ($)', fontsize=12)
    ax1.grid(True, alpha=0.15, linestyle=':')
    ax1.legend(loc='upper left', fontsize=10)

    # === Нижний график: Underwater Chart (Только просадка) ===
    # Это "Подводный график" - показывает насколько процентов мы под водой
    
    dd_pct = df_equity['drawdown'] * 100
    ax2.plot(df_equity['date'], dd_pct, color='#ff4444', linewidth=1.2)
    ax2.fill_between(df_equity['date'], dd_pct, 0, color='#ff4444', alpha=0.3)
    
    # Линии боли
    ax2.axhline(0, color='gray', linewidth=0.5)
    ax2.axhline(-10, color='yellow', linestyle=':', alpha=0.4, label='-10%')
    ax2.axhline(-20, color='orange', linestyle=':', alpha=0.4, label='-20%')
    
    # Если была просадка ниже 30%, подсветим
    min_dd = dd_pct.min()
    if min_dd < -30:
        ax2.axhline(min_dd, color='red', linestyle='--', alpha=0.5, label=f'Max DD {min_dd:.1f}%')

    ax2.set_ylabel('Drawdown (%)', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.grid(True, alpha=0.15, linestyle=':')
    ax2.legend(loc='lower right', fontsize=8)

    # Форматирование
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gcf().autofmt_xdate()

    plt.tight_layout()
    save_name = 'equity_hwm_chart.png'
    plt.savefig(save_name, dpi=150)
    print(f"💾 График с High Water Mark сохранен: {save_name}")
    plt.show()

if __name__ == "__main__":
    plot_equity_curve()