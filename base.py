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
    broker: str           # имя брокера: "bitget", "tinkoff", "simulated"


@dataclass
class Position:
    """
    Открытая позиция по инструменту.
    """
    symbol: str
    quantity: float
    avg_price: float
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
    Все реальные клиенты будут реализовывать этот интерфейс.
    """

    name: str

    # --- Маркет-дата ---

    @abstractmethod
    def get_historical_klines(
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
    def get_current_price(self, symbol: str) -> float:
        """
        Текущая цена инструмента.
        """
        ...

    # --- Аккаунт / портфель ---

    @abstractmethod
    def get_account_state(self) -> AccountState:
        """
        Текущее состояние аккаунта у брокера.
        """
        ...

    @abstractmethod
    def list_open_positions(self) -> List[Position]:
        """
        Список открытых позиций.
        """
        ...

    # --- Торговля ---

    @abstractmethod
    def place_order(self, order: OrderRequest) -> OrderResult:
        """
        Разместить ордер у брокера.
        """
        ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> None:
        """
        Отменить ордер по его ID.
        """
        ...
# ================================
# Unified Broker Interface Mixin
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