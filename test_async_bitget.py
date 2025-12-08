# test_async_bitget.py
import asyncio
from datetime import datetime, timedelta
from config import Config

async def test_bitget_async():
    """
    Тест асинхронной работы с Bitget API.
    """
    from brokers import get_broker
    
    print("🧪 Тест асинхронного Bitget клиента...")
    
    # Создаем брокера
    broker = get_broker("bitget")
    
    try:
        # Инициализируем
        await broker.initialize()
        print("✅ Брокер инициализирован")
        
        # Получаем текущую цену
        price = await broker.get_current_price("BTCUSDT")
        print(f"💰 Текущая цена BTCUSDT: {price}")
        
        # Получаем исторические свечи
        end = datetime.now()
        start = end - timedelta(days=1)
        
        candles = await broker.get_historical_klines(
            symbol="BTCUSDT",
            interval="1h",
            start=start,
            end=end
        )
        print(f"📊 Получено свечей: {len(candles)}")
        if not candles.empty:
            print(f"   Первая свеча: {candles.index[0]}")
            print(f"   Последняя свеча: {candles.index[-1]}")
        
        # Тест состояния аккаунта (требует API ключи)
        try:
            account_state = await broker.get_account_state()
            print(f"🏦 Состояние аккаунта: equity={account_state.equity}, balance={account_state.balance}")
        except Exception as e:
            print(f"⚠️  Состояние аккаунта не доступно: {e}")
        
        # Тест получения позиций
        try:
            positions = await broker.list_open_positions()
            print(f"📊 Открытые позиции: {len(positions)}")
        except Exception as e:
            print(f"⚠️  Позиции не доступны: {e}")
            
    except Exception as e:
        print(f"❌ Ошибка теста: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Корректно закрываем
        await broker.close()
        print("🔒 Брокер закрыт")


async def test_execution_router():
    """
    Тест асинхронного ExecutionRouter.
    """
    from execution_router import ExecutionRouter
    
    print("\n🧪 Тест ExecutionRouter...")
    
    router = ExecutionRouter()
    
    try:
        await router.initialize()
        print("✅ Роутер инициализирован")
        
        # Получаем глобальное состояние аккаунта
        try:
            global_state = await router.get_global_account_state()
            print(f"🌍 Глобальное состояние: equity={global_state.equity}, balance={global_state.balance}")
        except Exception as e:
            print(f"⚠️  Глобальное состояние не доступно: {e}")
            
    except Exception as e:
        print(f"❌ Ошибка теста роутера: {e}")
        
    finally:
        await router.close()
        print("🔒 Роутер закрыт")


async def main():
    """Основная асинхронная функция."""
    await test_bitget_async()
    await test_execution_router()


if __name__ == "__main__":
    asyncio.run(main())