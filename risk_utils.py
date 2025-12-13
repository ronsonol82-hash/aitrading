# risk_utils.py
"""
Утилиты для риск-менеджмента и расчета размера позиции.
Формула классическая:
    risk_amount = equity * risk_per_trade
    stop_distance = ATR * sl_mult
    size = risk_amount / stop_distance
+ опциональное ограничение по notional через MAX_POSITION_NOTIONAL.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from config import Config


@dataclass
class PositionSizeResult:
    size: float                 # размер позиции (кол-во юнитов)
    notional: float             # нотионал (size * price)
    risk_amount: float          # риск в деньгах
    atr: float                  # использованный ATR
    sl_mult: float              # множитель стопа
    stop_distance: float        # расстояние стопа в ценах


def calc_position_size(
    equity: float,
    risk_per_trade: float,
    atr: float,
    sl_mult: Optional[float] = None,
    price: Optional[float] = None,
    max_notional: Optional[float] = None,
) -> PositionSizeResult:
    """
    Расчет размера позиции по ATR.

    equity         - текущий капитал счета
    risk_per_trade - доля риска на сделку (0.01 = 1%)
    atr            - ATR в тех же единицах, что и цена
    sl_mult        - множитель стопа в ATR (если None - берем из Config.DEFAULT_STRATEGY['sl'])
    price          - текущая цена инструмента (если хотим считать notional)
    max_notional   - лимит по нотионалу (например, Config.MAX_POSITION_NOTIONAL)

    Возвращает PositionSizeResult; если не удалось посчитать — size=0.
    """
    # Базовые проверки
    if equity <= 0 or risk_per_trade <= 0 or atr is None or atr <= 0:
        return PositionSizeResult(
            size=0.0,
            notional=0.0 if price else 0.0,
            risk_amount=0.0,
            atr=float(atr or 0.0),
            sl_mult=float(sl_mult or Config.DEFAULT_STRATEGY.get("sl", 2.0)),
            stop_distance=0.0,
        )

    if sl_mult is None:
        sl_mult = Config.DEFAULT_STRATEGY.get("sl", 2.0)

    stop_distance = atr * sl_mult
    if stop_distance <= 0:
        return PositionSizeResult(
            size=0.0,
            notional=0.0 if price else 0.0,
            risk_amount=0.0,
            atr=float(atr),
            sl_mult=float(sl_mult),
            stop_distance=0.0,
        )

    risk_amount = equity * risk_per_trade
    if risk_amount <= 0:
        return PositionSizeResult(
            size=0.0,
            notional=0.0 if price else 0.0,
            risk_amount=0.0,
            atr=float(atr),
            sl_mult=float(sl_mult),
            stop_distance=stop_distance,
        )

    # Базовый размер позиции в юнитах
    size = risk_amount / stop_distance

    notional = 0.0
    if price is not None and price > 0:
        notional = size * price

        # Ограничение по нотионалу, если задано
        if max_notional is not None and max_notional > 0:
            if notional > max_notional:
                size = max_notional / price
                notional = max_notional

    # Защита от отрицательных и микроскопических значений
    if size <= 0:
        size = 0.0
        notional = 0.0

    return PositionSizeResult(
        size=float(size),
        notional=float(notional),
        risk_amount=float(risk_amount),
        atr=float(atr),
        sl_mult=float(sl_mult),
        stop_distance=float(stop_distance),
    )
