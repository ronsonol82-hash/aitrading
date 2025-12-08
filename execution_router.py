# execution_router.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from config import Config
from brokers import (
    get_broker,
    BrokerAPI,
    OrderRequest,
    OrderResult,
    AccountState,
    Position,
)


@dataclass
class GlobalAccountState:
    """
    Сводное состояние портфеля по всем брокерам.
    """
    equity: float
    balance: float
    details: Dict[str, AccountState]


class ExecutionRouter:
    """
    Асинхронный роутер исполнения ордеров и сигналов.
    
    Задачи:
      - знает, какой символ к какому брокеру относится (Config.ASSET_ROUTING);
      - лениво поднимает брокеров через get_broker(name);
      - агрегирует состояние счёта и позиции по всем брокерам;
      - даёт простые async методы execute_order / execute_signal.
    """

    def __init__(
        self,
        asset_routing: Optional[Dict[str, str]] = None,
        default_broker: Optional[str] = None,
    ):
        # Если ASSET_ROUTING/DEFAULT_BROKER ещё не заведены в Config,
        # подтянем безопасные значения по умолчанию.
        self.asset_routing: Dict[str, str] = asset_routing or getattr(
            Config, "ASSET_ROUTING", {}
        )
        self.default_broker: str = default_broker or getattr(
            Config, "DEFAULT_BROKER", "bitget"
        )

        # Локальный кеш брокеров: "bitget" -> BrokerAPI
        self._brokers: Dict[str, BrokerAPI] = {}

    # ---------- Lifecycle ----------
    
    async def initialize(self) -> None:
        """
        Асинхронная инициализация всех брокеров.
        Вызывается перед первым использованием роутера.
        """
        broker_names = set(self.asset_routing.values())
        broker_names.add(self.default_broker)

        for name in broker_names:
            try:
                broker = get_broker(name)
                await broker.initialize()
                self._brokers[name] = broker
            except Exception as e:
                print(f"[WARN] ExecutionRouter: failed to init broker '{name}': {e}")

    async def close(self) -> None:
        """
        Корректное закрытие всех брокеров.
        """
        for name, broker in list(self._brokers.items()):
            try:
                await broker.close()
            except Exception as e:
                print(f"[WARN] ExecutionRouter: failed to close broker '{name}': {e}")
            finally:
                self._brokers.pop(name, None)

    # ---------- Вспомогательные методы ----------

    def get_broker_name_for_symbol(self, symbol: str) -> str:
        """
        Вернуть имя брокера для данного тикера.
        Если тикер не прописан явно — используем default_broker.
        """
        return self.asset_routing.get(symbol, self.default_broker)

    async def get_broker_for_symbol(self, symbol: str) -> BrokerAPI:
        """
        Получить инстанс брокера для тикера (ленивая инициализация).
        """
        name = self.get_broker_name_for_symbol(symbol)
        if name not in self._brokers:
            try:
                broker = get_broker(name)
                await broker.initialize()
                self._brokers[name] = broker
            except Exception as e:
                raise RuntimeError(f"Failed to initialize broker '{name}' for symbol '{symbol}': {e}")
        return self._brokers[name]

    # ---------- Высокоуровневые операции ----------

    async def execute_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str = "market",
        price: Optional[float] = None,
        client_id: Optional[str] = None,
    ) -> OrderResult:
        """
        Универсальное исполнение ордера:
          - выбирает брокера по symbol;
          - формирует OrderRequest;
          - отправляет в broker.place_order.
        """
        broker = await self.get_broker_for_symbol(symbol)
        order = OrderRequest(
            symbol=symbol,
            side=side,  # "buy" / "sell"
            quantity=quantity,
            order_type=order_type,
            price=price,
            client_id=client_id,
        )
        return await broker.place_order(order)

    async def execute_signal(
        self,
        symbol: str,
        pos_type: str,
        size: float,
        price: Optional[float] = None,
        client_id: Optional[str] = None,
    ) -> OrderResult:
        """
        Обертка над execute_order для сигналов стратегии:
          pos_type: "LONG" / "SHORT"
        """
        side = "buy" if pos_type.upper() == "LONG" else "sell"
        return await self.execute_order(
            symbol=symbol,
            side=side,
            quantity=size,
            order_type="market",
            price=price,
            client_id=client_id,
        )

    async def get_global_account_state(self) -> GlobalAccountState:
        """
        Агрегирует состояние по всем брокерам.
        """
        total_equity = 0.0
        total_balance = 0.0
        details: Dict[str, AccountState] = {}

        # Используем только уже инициализированных брокеров
        for name, broker in self._brokers.items():
            try:
                state = await broker.get_account_state()
            except NotImplementedError:
                continue
            except Exception as e:
                print(f"[WARN] ExecutionRouter: get_account_state failed for {name}: {e}")
                continue

            total_equity += state.equity
            total_balance += state.balance
            details[name] = state

        return GlobalAccountState(
            equity=total_equity,
            balance=total_balance,
            details=details,
        )

    async def list_all_positions(self) -> List[Position]:
        """
        Собирает все открытые позиции по всем брокерам.
        """
        positions: List[Position] = []

        for name, broker in self._brokers.items():
            try:
                broker_positions = await broker.list_open_positions()
            except NotImplementedError:
                continue
            except Exception as e:
                print(f"[WARN] ExecutionRouter: list_open_positions failed for {name}: {e}")
                continue

            for p in broker_positions:
                # Если брокер не проставил имя сам — проставим здесь
                if not getattr(p, "broker", None):
                    p.broker = name
                positions.append(p)

        return positions