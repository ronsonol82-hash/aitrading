# brokers/bitget_client.py
from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Literal, Union

import aiohttp
import pandas as pd

from .base import BrokerAPI, OrderRequest, OrderResult, Position, AccountState

logger = logging.getLogger(__name__)


class BitgetBroker(BrokerAPI):
    """
    Async-брокер для Bitget (SPOT, V2 API).

    Реализовано:
      - get_historical_klines (через /api/v2/spot/market/candles)
      - get_current_price   (через /api/v2/spot/market/tickers)
      - get_account_state   (через /api/v2/spot/account/assets)
      - place_order         (через /api/v2/spot/trade/place-order)
      - cancel_order        (через /api/v2/spot/trade/cancel-order)
      - get_open_orders     (через /api/v2/spot/trade/unfilled-orders)
      - list_open_positions (по балансу спотовых монет)

    В проде всё это ещё придётся полировать под твои конкретные настройки.
    """

    name = "bitget"

    def __init__(self, config: Dict[str, Any]):
        self.api_key: str = config.get("api_key", "")
        self.api_secret: str = config.get("api_secret", "")
        self.passphrase: str = config.get("passphrase", "")
        self.base_url: str = config.get("base_url", "https://api.bitget.com")

        self.session: Optional[aiohttp.ClientSession] = None
        self.timeout = aiohttp.ClientTimeout(total=10)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        if self.session is None:
            self.session = aiohttp.ClientSession(timeout=self.timeout)
            logger.info("BitgetBroker: async session initialized.")

    async def close(self) -> None:
        if self.session is not None:
            await self.session.close()
            self.session = None
            logger.info("BitgetBroker: async session closed.")

    # ------------------------------------------------------------------
    # Вспомогательные мапперы и подпись
    # ------------------------------------------------------------------

    @staticmethod
    def _to_bitget_symbol(symbol: str) -> str:
        """
        Внутренний BTCUSDT -> биржевой тикер Bitget.
        Для spot V2 Bitget использует просто 'BTCUSDT'.
        """
        return symbol

    @staticmethod
    def _interval_to_granularity(interval: str) -> str:
        """
        Маппинг внутренних таймфреймов на granularity Bitget.
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

    def _generate_signature(
        self,
        method: str,
        path: str,
        query: str,
        body: str,
        timestamp: str,
    ) -> str:
        """
        Bitget V2 signature:
          sign = base64( HMAC_SHA256( secret, ts + method + requestPath + body ) )
        где requestPath = path + '?' + query (если есть query).
        """
        request_path = path
        if query:
            request_path = f"{path}?{query}"

        message = f"{timestamp}{method.upper()}{request_path}{body}"
        mac = hmac.new(
            self.api_secret.encode("utf-8"),
            message.encode("utf-8"),
            digestmod=hashlib.sha256,
        )
        return base64.b64encode(mac.digest()).decode("utf-8")

    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        signed: bool = False,
    ) -> Dict[str, Any]:
        if self.session is None:
            await self.initialize()

        assert self.session is not None

        url = f"{self.base_url}{endpoint}"
        ts = str(int(time.time() * 1000))

        headers = {
            "Content-Type": "application/json",
            "Locale": "en-US",
        }

        body_str = ""
        query_str = ""

        if params:
            query_parts = [f"{k}={v}" for k, v in params.items()]
            query_str = "&".join(query_parts)

        if data:
            # фиксируем сериализацию для подписи
            body_str = json.dumps(data, separators=(",", ":"), sort_keys=True)

        if signed:
            if not (self.api_key and self.api_secret and self.passphrase):
                raise ValueError("BitgetBroker: API credentials missing for signed request.")
            signature = self._generate_signature(method, endpoint, query_str, body_str, ts)
            headers.update(
                {
                    "ACCESS-KEY": self.api_key,
                    "ACCESS-SIGN": signature,
                    "ACCESS-TIMESTAMP": ts,
                    "ACCESS-PASSPHRASE": self.passphrase,
                }
            )

        try:
            async with self.session.request(
                method.upper(),
                url,
                params=params if method.upper() == "GET" else None,
                data=body_str if method.upper() in {"POST", "DELETE"} else None,
                headers=headers,
            ) as resp:
                text = await resp.text()

                if resp.status != 200:
                    logger.error(f"Bitget HTTP {resp.status}: {text}")
                    raise RuntimeError(f"Bitget HTTP {resp.status}")

                payload = json.loads(text)
        except Exception as e:
            logger.exception(f"Bitget request failed: {method} {endpoint}: {e}")
            raise

        if payload.get("code") != "00000":
            code = payload.get("code")
            msg = payload.get("msg", "Unknown error")
            logger.error(f"Bitget API error {code}: {msg}")
            raise RuntimeError(f"Bitget API error {code}: {msg}")

        return payload.get("data", {})

    # ------------------------------------------------------------------
    # BrokerAPI: MARKET DATA
    # ------------------------------------------------------------------

    async def get_historical_klines(
        self,
        symbol: str,
        interval: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        bg_symbol = self._to_bitget_symbol(symbol)
        granularity = self._interval_to_granularity(interval)

        start_ms = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)

        params = {
            "symbol": bg_symbol,
            "granularity": granularity,
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": 1000,
        }

        data = await self._request(
            "GET",
            "/api/v2/spot/market/candles",
            params=params,
            signed=False,
        )

        if not data or (isinstance(data, list) and len(data) == 0):
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
            if isinstance(item, dict):
                ts_raw = item.get("ts")
                ts_ms = int(float(ts_raw))
                open_ = float(item["open"])
                high = float(item["high"])
                low = float(item["low"])
                close = float(item["close"])
                base_vol = float(item.get("baseVol", item.get("baseVolume", 0.0)))
            else:
                # ожидаемый формат: [ts, open, high, low, close, baseVol, quoteVol, usdtVol]
                if len(item) < 6:
                    logger.warning(f"Bitget candles: unexpected array length={len(item)}: {item}")
                    continue
                ts_ms = int(float(item[0]))
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

        if not rows:
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

        df = pd.DataFrame(rows)
        df.sort_values("open_time", inplace=True)
        df.set_index("open_time", inplace=True)

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

    async def get_current_price(self, symbol: str) -> float:
        bg_symbol = self._to_bitget_symbol(symbol)
        params = {"symbol": bg_symbol}

        data = await self._request(
            "GET",
            "/api/v2/spot/market/tickers",
            params=params,
            signed=False,
        )

        if not data or (isinstance(data, list) and len(data) == 0):
            raise RuntimeError(f"Bitget ticker empty for symbol={bg_symbol}")

        ticker = data[0]
        last_pr = ticker.get("lastPr", "0")
        if not last_pr:
            raise RuntimeError(f"Bitget ticker missing lastPr for symbol={bg_symbol}")
        
        return float(last_pr)

    # ------------------------------------------------------------------
    # BrokerAPI: ACCOUNT / PORTFOLIO
    # ------------------------------------------------------------------

    async def get_account_state(self) -> AccountState:
        """
        Упрощённо считаем equity по USDT.
        Остальные монеты сейчас можно считать как "дополнительно сверху".
        """
        data = await self._request(
            "GET",
            "/api/v2/spot/account/assets",
            signed=True,
        )

        if not data or (isinstance(data, list) and len(data) == 0):
            logger.warning("Bitget account assets returned empty data")
            return AccountState(
                equity=0.0,
                balance=0.0,
                currency="USDT",
                margin_used=0.0,
                broker=self.name,
            )

        details: Dict[str, Dict[str, float]] = {}
        usdt_total = 0.0

        for asset in data:
            coin = asset.get("coin")
            if not coin:
                continue
                
            available = float(asset.get("available", 0))
            frozen = float(asset.get("frozen", 0))
            total = available + frozen

            details[coin] = {
                "available": available,
                "total": total,
            }

            if coin == "USDT":
                usdt_total = total

        return AccountState(
            equity=usdt_total,
            balance=usdt_total,
            currency="USDT",
            margin_used=0.0,
            broker=self.name,
        )

    async def list_open_positions(self) -> List[Position]:
        """
        Для спота считаем "позициями" любые монеты с total > 0, кроме USDT.
        Пока без средней цены входа (avg_price=0) и unrealized_pnl=0.
        """
        data = await self._request(
            "GET",
            "/api/v2/spot/account/assets",
            signed=True,
        )

        if not data or (isinstance(data, list) and len(data) == 0):
            return []

        positions: List[Position] = []
        for asset in data:
            coin = asset.get("coin")
            if not coin or coin == "USDT":
                continue
                
            available = float(asset.get("available", 0))
            frozen = float(asset.get("frozen", 0))
            total = available + frozen
            
            if total <= 0:
                continue

            positions.append(
                Position(
                    symbol=f"{coin}USDT",
                    quantity=total,
                    avg_price=0.0,
                    unrealized_pnl=0.0,
                    broker=self.name,
                )
            )

        return positions

    # ------------------------------------------------------------------
    # BrokerAPI: TRADING
    # ------------------------------------------------------------------

    async def place_order(self, order: OrderRequest) -> OrderResult:
        endpoint = "/api/v2/spot/trade/place-order"

        side_map: Dict[str, str] = {"buy": "buy", "sell": "sell"}
        type_map: Dict[str, str] = {"limit": "limit", "market": "market"}

        payload: Dict[str, Any] = {
            "symbol": self._to_bitget_symbol(order.symbol),
            "side": side_map.get(order.side, "buy"),
            "orderType": type_map.get(order.order_type, "limit"),
            "force": "normal",
            "size": str(order.quantity),
        }

        if order.order_type == "limit":
            if order.price is None:
                raise ValueError("BitgetBroker.place_order: price required for limit order.")
            payload["price"] = str(order.price)

        if order.client_id:
            payload["clientOid"] = order.client_id

        data = await self._request(
            "POST",
            endpoint,
            data=payload,
            signed=True,
        )

        order_id = data.get("orderId", "")
        return OrderResult(
            order_id=order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=order.price or 0.0,
            status="new",
            broker=self.name,
        )

    async def cancel_order(self, order_id: str, symbol: str) -> None:
        """
        Отмена ордера на Bitget (требует symbol).
        """
        endpoint = "/api/v2/spot/trade/cancel-order"
        payload = {
            "symbol": self._to_bitget_symbol(symbol),
            "orderId": order_id,
        }
        
        try:
            await self._request("POST", endpoint, data=payload, signed=True)
            logger.info(f"Order {order_id} for {symbol} cancelled successfully")
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id} for {symbol}: {e}")
            raise

    async def get_open_orders(self, symbol: str) -> List[OrderResult]:
        params = {"symbol": self._to_bitget_symbol(symbol)}
        data = await self._request(
            "GET",
            "/api/v2/spot/trade/unfilled-orders",
            params=params,
            signed=True,
        )

        if not data or (isinstance(data, list) and len(data) == 0):
            return []

        results: List[OrderResult] = []
        for item in data:
            try:
                ord_id = item.get("orderId", "")
                side_str = item.get("side", "buy").lower()
                side: Literal["buy", "sell"] = "buy" if side_str == "buy" else "sell"

                qty = float(item.get("size", 0))
                price = float(item.get("priceAvg", item.get("price", 0)))
                status = item.get("status", "open")

                # Приводим статус Bitget к внутреннему представлению
                status_map = {
                    "new": "new",
                    "partially_filled": "partial",
                    "filled": "filled",
                    "cancelled": "canceled",
                }
                internal_status = status_map.get(status, "open")

                # Получаем timestamp создания ордера
                ts_ms = int(item.get("cTime", 0))
                create_time = datetime.fromtimestamp(ts_ms / 1000.0) if ts_ms > 0 else None

                results.append(
                    OrderResult(
                        order_id=ord_id,
                        symbol=symbol,
                        side=side,
                        quantity=qty,
                        price=price,
                        status=internal_status,
                        broker=self.name,
                        create_time=create_time,
                    )
                )
            except Exception as e:
                logger.warning(f"BitgetBroker.get_open_orders: skip bad item {item}: {e}")
                continue

        return results
    
    def normalize_qty(self, symbol: str, qty: float, price: float | None = None) -> float:
        # P0: грубая нормализация под spot:
        # - округляем до 6 знаков
        # - отсекаем микроскопические значения
        q = float(qty)
        if q <= 0:
            return 0.0
        q = round(q, 6)
        if q <= 0:
            return 0.0
        return q

    async def close_position(self, symbol: str, reason: str = "") -> None:
        # P0: spot close = SELL доступного количества монеты (base asset)
        # Берём позиции через list_open_positions (у тебя это по балансу монет)
        positions = await self.list_open_positions()
        pos = next((p for p in positions if p.symbol == symbol and float(p.quantity or 0.0) > 0), None)
        if pos is None:
            return

        order = OrderRequest(symbol=symbol, side="sell", quantity=float(pos.quantity), order_type="market")
        await self.place_order(order)