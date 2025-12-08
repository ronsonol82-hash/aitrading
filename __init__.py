# brokers/__init__.py
from __future__ import annotations

from typing import Dict

from .base import BrokerAPI, OrderRequest, OrderResult, Position, AccountState

__all__ = [
    "BrokerAPI",
    "OrderRequest",
    "OrderResult",
    "Position",
    "AccountState",
    "get_broker",
]

# Простой кеш, чтобы не создавать клиентов по тысяче раз
_BROKER_CACHE: Dict[str, BrokerAPI] = {}


def _create_real_broker(uname: str) -> BrokerAPI:
    """
    Создаём "реальный" брокер без учёта режима исполнения.
    """
    from config import Config

    if uname == "bitget":
        from .bitget_client import BitgetBroker
        cfg = Config.BROKERS.get("bitget", {})
        return BitgetBroker(cfg)

    if uname == "tinkoff":
        # Новый клиент на Invest API v2 (REST gateway)
        from .tinkoff_client import TinkoffV2Broker
        cfg = Config.BROKERS.get("tinkoff", {})
        return TinkoffV2Broker(cfg)

    raise ValueError(f"Unknown real broker: {uname}")


def get_broker(name: str) -> BrokerAPI:
    """
    Фабрика брокеров.

    Логика:
      - читаем Config.EXECUTION_MODE (backtest/paper/live)
      - для bitget/tinkoff:
          * в режимах BACKTEST/PAPER возвращаем SimulatedBroker,
            обёрнутый вокруг реального брокера (для market-data)
          * в режиме LIVE возвращаем реальный брокер
      - симулятор создаём отдельно для каждого брокера:
          "sim_bitget", "sim_tinkoff"
    """
    from config import Config, ExecutionMode  # локальный импорт, чтобы избежать циклов

    # Нормализуем имя брокера
    uname = name.lower()

    # Определяем режим
    mode_obj = getattr(Config, "EXECUTION_MODE", ExecutionMode.BACKTEST)
    mode = mode_obj.value if isinstance(mode_obj, ExecutionMode) else str(mode_obj).lower()

    # --- Режимы BACKTEST / PAPER: используем SimulatedBroker ---
    if uname in ("bitget", "tinkoff") and mode in ("backtest", "paper"):
        cache_key = f"sim_{uname}"
        if cache_key in _BROKER_CACHE:
            return _BROKER_CACHE[cache_key]

        from .simulated_client import SimulatedBroker

        # реальный брокер только для данных
        real_broker = _create_real_broker(uname)
        starting_equity = getattr(Config, "DEPOSIT", 10_000)

        sim = SimulatedBroker(
            name=uname,
            data_broker=real_broker,
            starting_equity=starting_equity,
            currency="USDT",
        )
        _BROKER_CACHE[cache_key] = sim
        return sim

    # --- Режим LIVE (или нестандартный) — возвращаем реальных брокеров ---
    if uname in _BROKER_CACHE:
        return _BROKER_CACHE[uname]

    broker = _create_real_broker(uname)
    _BROKER_CACHE[uname] = broker
    return broker
