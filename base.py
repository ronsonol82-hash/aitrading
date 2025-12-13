# brokers/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, Optional, List
from datetime import datetime

import pandas as pd


Side = Literal["buy", "sell"]
OrderType = Literal["market", "limit"]


@dataclass
class OrderRequest:
    """
    Запрос на выставление ордера.
    """
    symbol: str
    side: Side
    quantity: float
    order_type: OrderType = "market"
    price: Optional[float] = None
    client_id: Optional[str] = None  # для идемпотентности / трекинга


@dataclass
class OrderResult:
    """
    Результат размещения ордера на стороне брокера.
    """
    order_id: str
    symbol: str
    side: Side
    quantity: float
    price: float
    status: str           # например: "new", "filled", "partially_filled", "canceled"
    broker: str           # имя брокера: "bitget", "tinkoff", "bitget_sim", "tinkoff_sim", ...


@dataclass
class Position:
    """
    Открытая позиция по инструменту.

    symbol        — человекочитаемый ключ (тикер/пара), используется в стратегии и UI
    instrument_id — истинный идентификатор инструмента у брокера (например FIGI для Tinkoff)
    """
    symbol: str
    instrument_id: Optional[str] = None  # P0.5++: FIGI для Tinkoff, None для crypto/симулятора
    quantity: float = 0.0
    avg_price: float = 0.0
    unrealized_pnl: Optional[float] = None
    broker: Optional[str] = None


@dataclass
class AccountState:
    """
    Сводное состояние счёта у брокера.
    """
    equity: float
    balance: float
    currency: str = "USDT"
    margin_used: float = 0.0
    broker: Optional[str] = None


class BrokerAPI(ABC):
    """
    Базовый контракт для любых брокеров (Bitget, Tinkoff, симулятор).

    ВАЖНО:
      - Все сетевые/торговые операции объявлены как async;
      - Вызывающий код обязан их вызывать через `await`.
    """

    name: str

    # --- Жизненный цикл брокера ---

    @abstractmethod
    async def initialize(self) -> None:
        """
        Асинхронная инициализация (создание HTTP-сессий, подготовка ресурсов).
        Может быть пустой, если брокеру ничего не нужно.
        """
        ...

    @abstractmethod
    async def close(self) -> None:
        """
        Освобождение ресурсов (закрытие HTTP-сессий и т.п.).
        """
        ...

    # --- Маркет-дата ---

    @abstractmethod
    async def get_historical_klines(
        self,
        symbol: str,
        interval: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """
        Вернуть DataFrame со свечами:
        index=datetime, колонки: open, high, low, close, volume, ...
        """
        ...

    @abstractmethod
    async def get_current_price(self, symbol: str) -> float:
        """
        Текущая цена инструмента.
        """
        ...

    # --- Аккаунт / портфель ---

    @abstractmethod
    async def get_account_state(self) -> AccountState:
        """
        Текущее состояние аккаунта у брокера.
        """
        ...

    @abstractmethod
    async def list_open_positions(self) -> List[Position]:
        """
        Список открытых позиций.
        """
        ...

    # --- Торговля ---

    @abstractmethod
    async def place_order(self, order: OrderRequest) -> OrderResult:
        """
        Разместить ордер у брокера.
        """
        ...

    @abstractmethod
    async def cancel_order(self, order_id: str) -> None:
        """
        Отменить ордер по его ID.
        """
        ...

    @abstractmethod
    async def get_open_orders(self, symbol: str) -> List[OrderResult]:
        """
        Список активных ордеров по инструменту.
        В симуляторе на первом этапе может возвращать пустой список.
        """
        ...
    # --- P0 extensions (optional) ---

    async def close_position(self, symbol: str, reason: str = "") -> None:
        """
        Опционально: закрыть позицию по symbol.
        По умолчанию не реализовано.
        """
        raise NotImplementedError(f"{self.__class__.__name__}.close_position not implemented")

    def normalize_qty(self, symbol: str, qty: float, price: float | None = None) -> float:
        """
        Опционально: привести количество к шагу/лотности/минимумам.
        По умолчанию возвращаем как есть.
        """
        return float(qty)

# ================================
# Unified Broker Interface Mixin
# (оставляем как есть, на будущее)
# ================================
class UnifiedBrokerMixin:
    """
    Даёт брокеру единый интерфейс, который понимают:
    - ExecutionRouter
    - Backtester
    - ModelEngine (через DataLoader)
    """

    def get_candles_unified(self, symbol, interval, limit=500):
        raise NotImplementedError

    def place_order_unified(self, symbol, side, size):
        raise NotImplementedError

    def get_balance_unified(self):
        raise NotImplementedError

    def get_positions_unified(self):
        return []  # необязательно для тинькоффа в режиме spot
