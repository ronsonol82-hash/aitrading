# test_full_cycle.py
import asyncio
import pickle
from datetime import datetime, timedelta
from config import Config
from execution_router import ExecutionRouter


async def test_full_cycle():
    """
    Тест полного цикла: загрузка сигналов + исполнение через роутер.
    """
    print("🧪 Тест полного цикла генерация+исполнение...")
    
    # Инициализируем роутер
    router = ExecutionRouter()
    await router.initialize()
    
    try:
        # 1. Получаем глобальное состояние
        global_state = await router.get_global_account_state()
        print(f"🌍 Начальное состояние: equity={global_state.equity:.2f}")
        
        # 2. Загружаем существующие сигналы
        signals_file = "data_cache/production_signals_v1.pkl"
        try:
            with open(signals_file, "rb") as f:
                signals = pickle.load(f)
            
            print(f"📊 Загружено сигналов для {len(signals)} активов")
            
            # 3. Берем последние сигналы для теста
            for symbol, df in list(signals.items())[:3]:  # Только 3 актива для теста
                if not df.empty:
                    last_signal = df.iloc[-1]
                    p_long = last_signal.get('p_long', 0)
                    p_short = last_signal.get('p_short', 0)
                    
                    print(f"   {symbol}: p_long={p_long:.3f}, p_short={p_short:.3f}")
                    
                    # 4. Тестовое исполнение (только для симуляционного режима)
                    if Config.EXECUTION_MODE in ["backtest", "paper"]:
                        # Определяем направление по вероятностям
                        if p_long > 0.6:
                            print(f"   🟢 Сигнал LONG для {symbol} (вероятность: {p_long:.2%})")
                            # Рассчитываем размер позиции (упрощенно)
                            price = await router.get_broker_for_symbol(symbol).get_current_price(symbol)
                            balance = global_state.balance / len(signals)  # Упрощенное распределение
                            size = (balance * 0.1) / price  # 10% от выделенного капитала
                            
                            if size > 0:
                                try:
                                    result = await router.execute_signal(
                                        symbol=symbol,
                                        pos_type="LONG",
                                        size=size
                                    )
                                    print(f"     📈 Ордер исполнен: {result.order_id}")
                                except NotImplementedError as e:
                                    print(f"     ⚠️  Брокер не поддерживает торговлю: {e}")
                                    
                        elif p_short > 0.6:
                            print(f"   🔴 Сигнал SHORT для {symbol} (вероятность: {p_short:.2%})")
                            # Аналогично для SHORT
                        
        except FileNotFoundError:
            print(f"⚠️  Файл сигналов не найден: {signals_file}")
            print("   Сначала запустите signal_generator.py")
            
        # 5. Проверяем финальное состояние
        final_state = await router.get_global_account_state()
        print(f"🌍 Финальное состояние: equity={final_state.equity:.2f}")
        
        # 6. Получаем открытые позиции
        positions = await router.list_all_positions()
        print(f"📊 Открытые позиции: {len(positions)}")
        for pos in positions:
            print(f"   {pos.symbol}: {pos.quantity:.6f} по {pos.avg_price:.2f}")
            
    except Exception as e:
        print(f"❌ Ошибка в тесте: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        await router.close()
        print("🔒 Роутер закрыт")


if __name__ == "__main__":
    asyncio.run(test_full_cycle())