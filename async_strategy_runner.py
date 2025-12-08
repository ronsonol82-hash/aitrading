# async_strategy_runner.py
import asyncio
import pickle
from datetime import datetime
from typing import Dict
import pandas as pd
from execution_router import ExecutionRouter
from config import Config


class AsyncStrategyRunner:
    """
    Асинхронный раннер стратегий.
    Загружает сигналы и исполняет их через ExecutionRouter.
    """
    
    def __init__(self, signals_file: str = "data_cache/production_signals_v1.pkl"):
        self.signals_file = signals_file
        self.signals: Dict[str, pd.DataFrame] = {}
        self.router = ExecutionRouter()
        
    async def initialize(self):
        """Инициализация роутера и загрузка сигналов"""
        await self.router.initialize()
        self.load_signals()
        
    def load_signals(self):
        """Загрузка сигналов из файла"""
        try:
            with open(self.signals_file, "rb") as f:
                self.signals = pickle.load(f)
            print(f"📊 Загружено сигналов для {len(self.signals)} активов")
        except FileNotFoundError:
            print(f"⚠️  Файл сигналов не найден: {self.signals_file}")
            self.signals = {}
            
    async def get_current_signals(self, symbol: str):
        """Получение текущих сигналов для символа"""
        if symbol not in self.signals:
            return None
            
        df = self.signals[symbol]
        if df.empty:
            return None
            
        # Возвращаем последний сигнал
        return df.iloc[-1]
    
    async def run_strategy(self, risk_per_trade: float = 0.02):
        """
        Запуск стратегии на основе загруженных сигналов.
        """
        if not self.signals:
            print("❌ Нет сигналов для торговли")
            return
            
        # Получаем состояние счета
        account_state = await self.router.get_global_account_state()
        total_equity = account_state.equity
        
        print(f"🏦 Начальный капитал: {total_equity:.2f}")
        
        # Проходим по всем активам
        trades_executed = 0
        
        for symbol, df in self.signals.items():
            if df.empty:
                continue
                
            # Получаем последний сигнал
            last_signal = df.iloc[-1]
            p_long = last_signal.get('p_long', 0)
            p_short = last_signal.get('p_short', 0)
            
            # Определяем направление
            if p_long > 0.65:  # Порог для LONG
                await self.execute_trade(
                    symbol=symbol,
                    side="buy",
                    probability=p_long,
                    equity=total_equity,
                    risk_per_trade=risk_per_trade,
                    signal_data=last_signal
                )
                trades_executed += 1
                
            elif p_short > 0.65:  # Порог для SHORT
                await self.execute_trade(
                    symbol=symbol,
                    side="sell",
                    probability=p_short,
                    equity=total_equity,
                    risk_per_trade=risk_per_trade,
                    signal_data=last_signal
                )
                trades_executed += 1
                
        print(f"✅ Исполнено ордеров: {trades_executed}")
        
    async def execute_trade(self, symbol: str, side: str, probability: float, 
                           equity: float, risk_per_trade: float, signal_data: pd.Series):
        """Исполнение одной сделки"""
        try:
            # Получаем текущую цену
            broker = await self.router.get_broker_for_symbol(symbol)
            current_price = await broker.get_current_price(symbol)
            
            # Рассчитываем размер позиции
            risk_amount = equity * risk_per_trade
            position_size = risk_amount / current_price
            
            # Исполняем ордер
            print(f"📈 Исполнение {side.upper()} для {symbol}:")
            print(f"   Цена: {current_price:.2f}, Вероятность: {probability:.2%}")
            print(f"   Размер: {position_size:.6f}")
            
            result = await self.router.execute_order(
                symbol=symbol,
                side=side,
                quantity=position_size,
                order_type="market"
            )
            
            print(f"   ✅ Ордер исполнен: {result.order_id}")
            
        except NotImplementedError:
            print(f"⚠️  Брокер для {symbol} не поддерживает торговлю")
        except Exception as e:
            print(f"❌ Ошибка исполнения для {symbol}: {e}")
            
    async def close(self):
        """Закрытие ресурсов"""
        await self.router.close()


async def main():
    """Основная асинхронная функция"""
    runner = AsyncStrategyRunner()
    
    try:
        await runner.initialize()
        await runner.run_strategy(risk_per_trade=0.01)  # 1% риска на сделку
        
        # Показываем итоговое состояние
        account_state = await runner.router.get_global_account_state()
        print(f"\n🏦 Итоговый капитал: {account_state.equity:.2f}")
        
        positions = await runner.router.list_all_positions()
        print(f"📊 Открытых позиций: {len(positions)}")
        
    finally:
        await runner.close()


if __name__ == "__main__":
    asyncio.run(main())