# brokers/bitget_client.py
from __future__ import annotations

from datetime import datetime
from typing import List, Dict, Any

import pandas as pd
import requests

from .base import BrokerAPI, OrderRequest, OrderResult, Position, AccountState


class BitgetBroker(BrokerAPI):
    """
    Брокер для Bitget (SPOT, публичные эндпоинты).

    На этом этапе реализуем только:
      - get_historical_klines
      - get_current_price

    Остальные методы пока поднимают NotImplementedError.
    """

    name = "bitget"

    def __init__(self, config: Dict[str, Any]):
        # V2 API (spot)
        self.base_url = config.get("base_url", "https://api.bitget.com")
        self.session = requests.Session()

    # ---------- Вспомогательные мапперы ----------

    @staticmethod
    def _to_bitget_symbol(symbol: str) -> str:
        """
        Внутренний BTCUSDT -> биржевой тикер Bitget.

        Для spot V2 Bitget использует просто 'BTCUSDT' (без _SPBL),
        см. /api/v2/spot/market/candles и /api/v2/spot/market/tickers.
        """
        # На будущее: сюда можно добавить более сложное сопоставление
        return symbol

    @staticmethod
    def _interval_to_granularity(interval: str) -> str:
        """
        Маппинг внутренних таймфреймов на granularity Bitget.

        Поддерживаем типовые варианты, остальное пробрасываем как есть.
        """
        normalized = interval.lower()

        mapping = {
            "1m": "1min",
            "3m": "3min",
            "5m": "5min",
            "15m": "15min",
            "30m": "30min",
            "1h": "1h",
            "4h": "4h",
            "6h": "6h",
            "12h": "12h",
            "1d": "1day",
            "1day": "1day",
            "3d": "3day",
            "1w": "1week",
            "1wk": "1week",
            "1week": "1week",
            "1mo": "1M",
            "1mth": "1M",
        }

        return mapping.get(normalized, normalized)

    # ---------- Реализация BrokerAPI (market data) ----------

    def get_historical_klines(
        self,
        symbol: str,
        interval: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """
        Исторические свечи с Bitget SPOT V2.

        Эндпоинт:
          GET /api/v2/spot/market/candles
          params: symbol, granularity, startTime, endTime, limit

        Ограничения:
          - limit <= 1000, поэтому для ОЧЕНЬ длинных диапазонов
            будет урезание по количеству свечей (TODO: реализовать пэйджинг).
        """
        bg_symbol = self._to_bitget_symbol(symbol)
        granularity = self._interval_to_granularity(interval)

        start_ms = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)

        url = f"{self.base_url}/api/v2/spot/market/candles"
        params = {
            "symbol": bg_symbol,
            "granularity": granularity,
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": 1000,
        }

        resp = self.session.get(url, params=params, timeout=10)
        resp.raise_for_status()
        payload = resp.json()

        if payload.get("code") != "00000":
            raise RuntimeError(
                f"Bitget candles error: code={payload.get('code')} msg={payload.get('msg')}"
            )

        data = payload.get("data", [])
        if not data:
            return pd.DataFrame(
                columns=[
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "taker_buy_base",
                    "funding_rate",
                    "imbalance",
                ]
            )

        rows = []
        for item in data:
            # Поддерживаем и dict, и list, и более "грязные" форматы
            if isinstance(item, dict):
                # ts может прийти строкой с плавающей точкой — страхуемся
                ts_raw = item.get("ts")
                ts_ms = int(float(ts_raw))
                open_ = float(item["open"])
                high = float(item["high"])
                low = float(item["low"])
                close = float(item["close"])
                base_vol = float(item.get("baseVol", item.get("baseVolume", 0.0)))

            else:
                # Ожидаемый формат V2:
                # [ts, open, high, low, close, baseVol, quoteVol, usdtVol]
                if len(item) < 6:
                    raise RuntimeError(f"Bitget candles: unexpected array length={len(item)}: {item}")

                try:
                    ts_ms = int(float(item[0]))  # ts в миллисекундах, но может быть строкой/float
                except Exception as e:
                    raise RuntimeError(f"Bitget candles: cannot parse ts from item[0]={item[0]}: {e}")

                open_ = float(item[1])
                high = float(item[2])
                low = float(item[3])
                close = float(item[4])
                base_vol = float(item[5])

            ts = datetime.utcfromtimestamp(ts_ms / 1000.0)

            rows.append(
                {
                    "open_time": ts,
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": base_vol,
                    "taker_buy_base": 0.0,
                    "funding_rate": 0.0,
                    "imbalance": 0.0,
                }
            )

        df = pd.DataFrame(rows)
        df.sort_values("open_time", inplace=True)
        df.set_index("open_time", inplace=True)

        # Возвращаем тот же набор колонок, который ожидал старый get_binance_data
        return df[
            [
                "open",
                "high",
                "low",
                "close",
                "volume",
                "taker_buy_base",
                "funding_rate",
                "imbalance",
            ]
        ]

    def get_current_price(self, symbol: str) -> float:
        """
        Текущая цена спота.

        Эндпоинт:
          GET /api/v2/spot/market/tickers?symbol=BTCUSDT

        Берём поле lastPr из первого элемента data.
        """
        bg_symbol = self._to_bitget_symbol(symbol)

        url = f"{self.base_url}/api/v2/spot/market/tickers"
        params = {"symbol": bg_symbol}

        resp = self.session.get(url, params=params, timeout=5)
        resp.raise_for_status()
        payload = resp.json()

        if payload.get("code") != "00000":
            raise RuntimeError(
                f"Bitget ticker error: code={payload.get('code')} msg={payload.get('msg')}"
            )

        data = payload.get("data", [])
        if not data:
            raise RuntimeError(f"Bitget ticker empty for symbol={bg_symbol}")

        ticker = data[0]
        # В V2 тикерах поле lastPr — последняя цена
        last_pr = float(ticker.get("lastPr"))
        return last_pr

    # ============================
    # Unified OHLCV
    # ============================
    #def get_candles_unified(self, symbol, interval, limit=500):
    #    raw = self.get_candles(symbol, interval, limit)
    #    out = []
    #    for c in raw:
    #        out.append({
    #            "timestamp": int(c[0]),
    #            "open": float(c[1]),
    #            "high": float(c[2]),
    #            "low": float(c[3]),
    #            "close": float(c[4]),
    #            "volume": float(c[5])
    #        })
    #    return out

    # ============================
    # Unified ORDER
    # ============================
    #def place_order_unified(self, symbol, side, size):
    #    r = self.place_order(symbol, side, size)
    #    if r.get("code") != "00000":
    #        return {
    #            "order_id": None,
    #            "status": "rejected",
    #            "filled_size": 0,
    #            "avg_price": 0,
    #            "side": side
    #        }

    #    data = r.get("data", {})
    #    return {
    #        "order_id": data.get("orderId", ""),
    #        "status": "submitted",
    #        "filled_size": float(data.get("size", 0)),
    #        "avg_price": float(data.get("priceAvg", 0)),
    #        "side": side
    #    }

    # ============================
    # Unified BALANCE
    # ============================
    #def get_balance_unified(self):
    #    bal = self.get_balance()
    #    return {
    #        "equity": float(bal.get("equity", 0)),
    #        "available": float(bal.get("available", 0)),
    #    }

    # ---------- Остальные методы пока не реализуем ----------

    def get_account_state(self) -> AccountState:
        raise NotImplementedError("Bitget account state is not implemented yet.")

    def list_open_positions(self) -> List[Position]:
        raise NotImplementedError("Bitget positions are not implemented yet.")

    def place_order(self, order: OrderRequest) -> OrderResult:
        raise NotImplementedError("Bitget trading is not implemented yet.")

    def cancel_order(self, order_id: str) -> None:
        raise NotImplementedError("Bitget cancel_order is not implemented yet.")
