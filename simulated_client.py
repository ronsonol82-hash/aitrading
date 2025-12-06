# brokers/simulated_client.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from .base import BrokerAPI, OrderRequest, OrderResult, Position, AccountState


@dataclass
class _SimPositionState:
    """
    Внутреннее состояние позиции для симулятора.
    """
    quantity: float = 0.0
    avg_price: float = 0.0


class SimulatedBroker(BrokerAPI):
    """
    Универсальный симулятор счёта.

    Идея:
      - для market-data используем реальный брокер (Bitget/Tinkoff),
        переданный как data_broker;
      - ордера, позиции и PnL считаем локально;
      - таким образом, один и тот же движок (стратегия) может работать
        как с криптой, так и с акциями, не зная, что это симулятор.
    """

    def __init__(
        self,
        name: str,
        data_broker: BrokerAPI,
        starting_equity: float,
        currency: str = "USDT",
    ):
        # "логическое" имя: bitget_sim / tinkoff_sim
        self.name = f"{name}_sim"
        self._underlying = data_broker
        self._currency = currency

        # Начальный капитал, реализованный PnL и позиции
        self._starting_equity = float(starting_equity)
        self._realized_pnl = 0.0
        self._positions: Dict[str, _SimPositionState] = {}

        # Простая генерация ID ордеров
        self._order_seq = 0

    # ---------- MARKET DATA (делегируем реальному брокеру) ----------

    def get_historical_klines(
        self,
        symbol: str,
        interval: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        return self._underlying.get_historical_klines(
            symbol=symbol,
            interval=interval,
            start=start,
            end=end,
        )

    def get_current_price(self, symbol: str) -> float:
        return self._underlying.get_current_price(symbol)

    # ---------- ВСПОМОГАТЕЛЬНОЕ: переоценка позиций ----------

    def _revalue_positions(self) -> Dict[str, Position]:
        """
        Пересчитываем нереализованный PnL по всем инструментам.
        Возвращаем snapshot позиций в виде Position.
        """
        result: Dict[str, Position] = {}

        for symbol, state in self._positions.items():
            if state.quantity == 0:
                continue

            last_price = self.get_current_price(symbol)
            qty = state.quantity
            avg = state.avg_price

            if qty > 0:
                # long
                unrealized = (last_price - avg) * qty
            else:
                # short
                unrealized = (avg - last_price) * abs(qty)

            result[symbol] = Position(
                symbol=symbol,
                quantity=qty,
                avg_price=avg,
                unrealized_pnl=unrealized,
                broker=self.name,
            )

        return result

    # ---------- ACCOUNT / PORTFOLIO ----------

    def get_account_state(self) -> AccountState:
        """
        Возвращаем сводное состояние счёта:
          - equity = стартовый капитал + реализованный PnL + суммарный нереализованный PnL
          - balance = стартовый капитал + реализованный PnL (без учёта плавающего)
        """
        positions = self._revalue_positions()
        total_unrealized = sum(
            p.unrealized_pnl or 0.0 for p in positions.values()
        )

        balance = self._starting_equity + self._realized_pnl
        equity = balance + total_unrealized

        return AccountState(
            equity=equity,
            balance=balance,
            currency=self._currency,
            margin_used=0.0,
            broker=self.name,
        )

    def list_open_positions(self) -> List[Position]:
        return list(self._revalue_positions().values())

    # ---------- TRADING LOGIC (упрощённый каркас) ----------

    def _next_order_id(self) -> str:
        self._order_seq += 1
        return f"{self.name}-ord-{self._order_seq}"

    def place_order(self, order: OrderRequest) -> OrderResult:
        """
        Упрощённая логика:
          - ордер всегда исполняется "мгновенно" по текущей цене
            (или по переданному order.price, если он указан)
          - PnL считаем только реализованный; маржу/комиссии и риск не трогаем (пока)
        """
        symbol = order.symbol
        side = order.side
        qty = float(order.quantity)

        if qty <= 0:
            raise ValueError("SimulatedBroker: quantity must be > 0")

        # "Исполняем" ордер
        trade_price = float(order.price) if order.price is not None else float(
            self.get_current_price(symbol)
        )

        # Обновляем состояние позиции
        state = self._positions.get(symbol, _SimPositionState())

        signed_qty = qty if side == "buy" else -qty

        if state.quantity == 0:
            # Открытие новой позиции
            state.quantity = signed_qty
            state.avg_price = trade_price

        else:
            # Есть существующая позиция
            old_qty = state.quantity
            old_avg = state.avg_price

            # Тот же direction (усиление позиции)
            if (old_qty > 0 and signed_qty > 0) or (old_qty < 0 and signed_qty < 0):
                new_qty = old_qty + signed_qty
                if new_qty == 0:
                    state.quantity = 0.0
                    state.avg_price = 0.0
                else:
                    state.avg_price = (old_avg * old_qty + trade_price * signed_qty) / new_qty
                    state.quantity = new_qty

            else:
                # Частичное/полное закрытие или разворот
                closing_qty = min(abs(old_qty), abs(signed_qty))

                if old_qty > 0:
                    # закрываем long
                    realized = (trade_price - old_avg) * closing_qty
                else:
                    # закрываем short
                    realized = (old_avg - trade_price) * closing_qty

                self._realized_pnl += realized

                new_qty = old_qty + signed_qty
                if new_qty == 0:
                    state.quantity = 0.0
                    state.avg_price = 0.0
                else:
                    # Для простоты считаем, что "обратный хвост" открыт по текущей цене
                    state.quantity = new_qty
                    state.avg_price = trade_price

        self._positions[symbol] = state

        order_id = self._next_order_id()

        return OrderResult(
            order_id=order_id,
            symbol=symbol,
            side=order.side,
            quantity=qty,
            price=trade_price,
            status="filled",
            broker=self.name,
        )

    def cancel_order(self, order_id: str) -> None:
        """
        В текущем каркасе ордеры исполняются мгновенно, так что отмена —
        no-op. Позже можно будет хранить pending-ордера и реально их отменять.
        """
        return None
