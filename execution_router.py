# execution_router.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional
from risk_utils import calc_position_size
from config import Config, ExecutionMode
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
        self._daily_anchor_date: str | None = None
        self._daily_anchor_equity: float | None = None

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

    async def _ensure_daily_anchor(self) -> None:
        """Фиксируем equity на начало текущего дня (для MAX_DAILY_DRAWDOWN)."""
        today = date.today().isoformat()
        if self._daily_anchor_date != today:
            snap = await self.get_global_account_state()
            self._daily_anchor_date = today
            self._daily_anchor_equity = float(snap.equity or 0.0)

    async def _check_daily_drawdown_guard(self) -> None:
        """
        В LIVE запрещает новые ордера при превышении MAX_DAILY_DRAWDOWN
        (в процентах от утреннего equity).
        """
        mode_obj = getattr(Config, "EXECUTION_MODE", ExecutionMode.BACKTEST)
        mode = mode_obj.value if isinstance(mode_obj, ExecutionMode) else str(mode_obj).lower()

        if mode != "live":
            return

        max_dd = float(getattr(Config, "MAX_DAILY_DRAWDOWN", 0.0) or 0.0)
        if max_dd <= 0:
            return

        await self._ensure_daily_anchor()
        anchor = float(self._daily_anchor_equity or 0.0)
        if anchor <= 0:
            return

        snap = await self.get_global_account_state()
        equity = float(snap.equity or 0.0)

        dd = (anchor - equity) / anchor
        if dd >= max_dd:
            raise RuntimeError(
                f"[RISK] MAX_DAILY_DRAWDOWN reached: {dd:.2%} >= {max_dd:.2%}. "
                f"New orders blocked until next day."
            )

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
        price: float | None = None,
        client_id: str | None = None,
    ):
        """
        Централизованная точка исполнения ордеров.
        Здесь добавляем жёсткие risk-check'и, которые будут работать
        и для PAPER, и для LIVE режимов.
        """
        broker = await self.get_broker_for_symbol(symbol)

        # --- P0: daily drawdown guard (правильный async + halt) ---
        if not hasattr(self, "_trading_halted"):
            self._trading_halted = False

        if self._trading_halted:
            raise RuntimeError("TRADING HALTED: previously triggered")

        try:
            await self._check_daily_drawdown_guard()
        except RuntimeError as e:
            # Жёстко тормозим торговлю до следующего дня
            self._trading_halted = True
            # Опционально: закрыть всё (лучше async)
            await self.close_all_positions(reason="daily_drawdown_guard")
            raise

        # --- RISK-GUARD: определяем режим ---
        mode_obj = getattr(Config, "EXECUTION_MODE", ExecutionMode.BACKTEST)
        if isinstance(mode_obj, ExecutionMode):
            mode = mode_obj.value
        else:
            mode = str(mode_obj).lower()

        # --- Ограничение количества одновременных позиций ---
        max_positions = getattr(Config, "MAX_OPEN_POSITIONS", None)
        if max_positions is not None and mode == "live":
            # Используем уже существующий метод, который собирает позиции
            positions = await self.list_all_positions()
            if len(positions) >= max_positions:
                raise RuntimeError(
                    f"[RISK] Max open positions {max_positions} reached, "
                    f"order for {symbol} is blocked."
                )

        # --- Ограничение нотионала позиции ---
        max_pos_notional = getattr(Config, "MAX_POSITION_NOTIONAL", None)
        if max_pos_notional is not None and mode == "live":
            max_pos_notional = float(max_pos_notional or 0.0)
            if max_pos_notional > 0:
                # приоритет: если price передали в execute_order — используем его
                if price is not None and float(price) > 0:
                    last_price = float(price)
                else:
                    last_price = float(await broker.get_current_price(symbol))

                notional = last_price * float(quantity)
                if notional > max_pos_notional:
                    raise RuntimeError(
                        f"[RISK] MAX_POSITION_NOTIONAL exceeded: {notional:.2f} > {max_pos_notional:.2f} "
                        f"({symbol}). Order blocked."
                    )

        # --- P0: normalize qty to broker rules (step/min/lot) ---
        try:
            last_price = float(price) if (price is not None and float(price) > 0) else float(await broker.get_current_price(symbol))
        except Exception:
            last_price = float(price or 0.0)

        quantity = float(broker.normalize_qty(symbol, float(quantity), price=last_price))
        if quantity <= 0:
            raise RuntimeError(f"[RISK] normalized quantity is 0 for {symbol}. Order blocked.")

        # --- Сборка и отправка ордера ---
        order = OrderRequest(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=order_type,
            price=price,
            client_id=client_id,
        )
        return await broker.place_order(order)

    async def close_all_positions(self, reason: str = "") -> None:
        brokers = self.get_active_brokers()
        for br in brokers:
            try:
                positions = await br.list_open_positions()
                for p in positions:
                    # P0: close_position может быть не реализован у некоторых брокеров
                    if hasattr(br, "close_position"):
                        await br.close_position(p.symbol, reason=reason)
            except Exception as e:
                print(f"[KILL] failed for {getattr(br, 'name', 'unknown')}: {e}")

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