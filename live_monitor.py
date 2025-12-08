# live_monitor.py
import asyncio
import time
from datetime import datetime
from execution_router import ExecutionRouter
from config import Config


class LiveMonitor:
    """
    Монитор состояния в реальном времени.
    Показывает состояние счета, позиции и цены.
    """
    
    def __init__(self, update_interval: int = 10):
        self.update_interval = update_interval
        self.router = ExecutionRouter()
        self.running = False
        
    async def initialize(self):
        await self.router.initialize()
        
    async def monitor_loop(self):
        """Основной цикл мониторинга"""
        self.running = True
        print("🚀 Запуск монитора в реальном времени")
        print("   Нажмите Ctrl+C для остановки")
        print("-" * 50)
        
        try:
            while self.running:
                await self.update_display()
                await asyncio.sleep(self.update_interval)
                
        except KeyboardInterrupt:
            print("\n🛑 Остановка монитора...")
        finally:
            self.running = False
            
    async def update_display(self):
        """Обновление отображения"""
        # Очищаем экран (работает в большинстве терминалов)
        print("\033[H\033[J", end="")
        
        print(f"📈 МОНИТОР В РЕАЛЬНОМ ВРЕМЕНИ")
        print(f"   Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Режим: {Config.EXECUTION_MODE.value}")
        print(f"   Юниверс: {Config.UNIVERSE_MODE.value}")
        print("-" * 50)
        
        try:
            # Получаем состояние счета
            account_state = await self.router.get_global_account_state()
            print(f"🏦 СОСТОЯНИЕ СЧЕТА:")
            print(f"   Equity: ${account_state.equity:,.2f}")
            print(f"   Balance: ${account_state.balance:,.2f}")
            
            # Детали по брокерам
            for broker_name, state in account_state.details.items():
                print(f"   {broker_name}: ${state.equity:,.2f}")
                
            print("-" * 50)
            
            # Получаем позиции
            positions = await self.router.list_all_positions()
            print(f"📊 ОТКРЫТЫЕ ПОЗИЦИИ ({len(positions)}):")
            
            if positions:
                total_unrealized = 0
                for pos in positions:
                    unrealized = pos.unrealized_pnl or 0
                    total_unrealized += unrealized
                    
                    pnl_percent = (unrealized / (pos.avg_price * abs(pos.quantity))) * 100 if pos.avg_price > 0 else 0
                    pnl_sign = "+" if unrealized >= 0 else ""
                    
                    print(f"   {pos.symbol}:")
                    print(f"     Направление: {'LONG' if pos.quantity > 0 else 'SHORT'}")
                    print(f"     Количество: {abs(pos.quantity):.6f}")
                    print(f"     Средняя цена: ${pos.avg_price:,.2f}")
                    print(f"     PnL: {pnl_sign}${unrealized:,.2f} ({pnl_sign}{pnl_percent:.2f}%)")
                    print(f"     Брокер: {pos.broker}")
                    
                print(f"   📊 Суммарный PnL: {total_unrealized:+,.2f}")
            else:
                print("   Нет открытых позиций")
                
            print("-" * 50)
            
            # Быстрая проверка цен для ключевых активов
            key_assets = ["BTCUSDT", "ETHUSDT", "SBER", "MOEX"][:3]
            print(f"💰 КЛЮЧЕВЫЕ ЦЕНЫ:")
            
            for asset in key_assets:
                try:
                    broker = await self.router.get_broker_for_symbol(asset)
                    price = await broker.get_current_price(asset)
                    print(f"   {asset}: ${price:,.2f}" if "USDT" in asset else f"   {asset}: {price:,.2f} RUB")
                except Exception:
                    continue
                    
        except Exception as e:
            print(f"❌ Ошибка обновления: {e}")
            
    async def close(self):
        self.running = False
        await self.router.close()


async def main():
    monitor = LiveMonitor(update_interval=5)  # Обновление каждые 5 секунд
    
    try:
        await monitor.initialize()
        await monitor.monitor_loop()
    finally:
        await monitor.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Завершение работы")