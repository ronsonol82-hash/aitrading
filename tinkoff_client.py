# brokers/tinkoff_client.py
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any
import asyncio
import json
import time

import pandas as pd
import requests

from .base import BrokerAPI, OrderRequest, OrderResult, Position, AccountState
# см. brokers/base.py для описания интерфейса BrokerAPI


class TinkoffV2Broker(BrokerAPI):
    """
    Клиент для Tinkoff Invest API v2 через REST-gateway.

    Сейчас:
      - get_historical_klines  -> async через asyncio.to_thread
      - get_current_price      -> async через asyncio.to_thread
      - остальное (торговля, аккаунт) — NotImplemented
    """

    name = "tinkoff"  # важно: совпадает с uname="tinkoff" в brokers.__init__

    def __init__(self, config: Dict[str, Any]):
        """
        Ожидаем в config:
          {
            "token": "...",            # обязательный
            "sandbox": bool            # опционально, на будущее
          }
        """
        token = config.get("token") or ""
        if not token:
            # даём понятную ошибку, чтобы не было "тихих" зависаний
            raise RuntimeError(
                "TinkoffV2Broker: не задан токен. "
                "Укажи его в Config.BROKERS['tinkoff']['token'] "
                "или в переменной окружения TINKOFF_API_TOKEN."
            )

        self.sandbox = bool(config.get("sandbox", False))

        # базовый REST-адрес V2
        self.base_url: str = config.get(
            "base_url", "https://invest-public-api.tinkoff.ru/rest"
        )

        # HTTP-сессия с заголовками авторизации
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {token}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            }
        )

        # --- глобальный rate-limit для исторических свечей ---
        self._history_min_interval = 60.0 / 25.0  # ~2.4с
        self._last_history_call_ts: float = 0.0

    # =====================================================================
    # Lifecycle
    # =====================================================================

    async def initialize(self) -> None:
        # здесь ничего особенного, но для единообразия интерфейса пусть будет
        return None

    async def close(self) -> None:
        try:
            self.session.close()
        except Exception:
            pass

    # =====================================================================
    # ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ (sync)
    # =====================================================================

    @staticmethod
    def _to_rfc3339(dt: datetime) -> str:
        """
        Инвест-API ожидает time в RFC3339, с таймзоной.
        Если tz не указана — считаем UTC.
        """
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat()

    @staticmethod
    def _interval_to_v2_enum(interval: str) -> str:
        """
        Маппинг внутренних интервалов на CandleInterval v2.
        """
        norm = interval.lower()

        if norm in ("1m", "1min"):
            return "CANDLE_INTERVAL_1_MIN"
        if norm in ("5m", "5min"):
            return "CANDLE_INTERVAL_5_MIN"
        if norm in ("15m", "15min"):
            return "CANDLE_INTERVAL_15_MIN"
        if norm in ("30m", "30min"):
            return "CANDLE_INTERVAL_30_MIN"
        if norm in ("1h", "60m", "hour"):
            return "CANDLE_INTERVAL_HOUR"
        if norm in ("1d", "1day", "day", "24h"):
            return "CANDLE_INTERVAL_DAY"

        # Fallback — час, чтобы не ломать код
        return "CANDLE_INTERVAL_HOUR"

    @staticmethod
    def _max_delta_for_interval(interval: str) -> timedelta:
        """
        Максимальный допустимый период для одного запроса GetCandles.
        """
        norm = interval.lower()

        # 1–15 минут → до 1 дня
        if norm in (
            "1m", "1min",
            "2m", "2min",
            "3m", "3min",
            "5m", "5min",
            "10m", "10min",
            "15m", "15min",
        ):
            return timedelta(days=1)

        # 30 минут → до 2 дней
        if norm in ("30m", "30min"):
            return timedelta(days=2)

        # 1 час → до 1 недели
        if norm in ("1h", "60m", "hour"):
            return timedelta(days=7)

        # 1 день → до 1 года
        if norm in ("1d", "1day", "day", "24h"):
            return timedelta(days=365)

        # всё остальное — консервативный дефолт: 31 день
        return timedelta(days=31)

    @staticmethod
    def _q_to_float(q: Dict[str, Any]) -> float:
        """
        Quotation {units, nano} -> float.
        """
        if not q:
            return 0.0
        units = float(q.get("units", 0))
        nano = float(q.get("nano", 0)) / 1e9
        return units + nano

    @staticmethod
    def _resolve_figi(symbol: str) -> str:
        """
        Маппинг тикера (SBER, GAZP...) -> FIGI.

        Берём из Config.TINKOFF_FIGI_MAP.
        """
        from config import Config  # локальный импорт, чтобы не ловить циклы

        figi_map = getattr(Config, "TINKOFF_FIGI_MAP", {})
        figi = figi_map.get(symbol)
        if not figi:
            raise KeyError(
                f"TinkoffV2Broker: не найден FIGI для символа '{symbol}'. "
                f"Добавь его в Config.TINKOFF_FIGI_MAP."
            )
        return figi
    
    @staticmethod
    def _resolve_ticker(figi: str) -> str:
        """
        Reverse-map FIGI -> TICKER через Config.TINKOFF_FIGI_MAP.
        Если FIGI не найден — возвращаем исходный figi (fallback).
        """
        from config import Config  # локальный импорт, чтобы не ловить циклы

        figi_map = getattr(Config, "TINKOFF_FIGI_MAP", {}) or {}
        # P0.5+: строим обратный мап на лету (в P0 достаточно; позже можно кэшировать)
        reverse = {v: k for k, v in figi_map.items() if v}
        return reverse.get(figi, figi)

    def _post_with_backoff(
        self,
        url: str,
        payload: Dict[str, Any],
        timeout: int = 10,
        is_history: bool = False,
    ) -> requests.Response:
        """
        POST-запрос с:
          - экспоненциальным backoff'ом на 429;
          - учётом глобального лимита для исторических свечей (is_history=True).
        """
        max_attempts = 5
        backoff = 0.5  # стартовая задержка

        for attempt in range(1, max_attempts + 1):
            # глобальный rate-limit для GetCandles
            if is_history:
                now = time.time()
                elapsed = now - self._last_history_call_ts
                wait = self._history_min_interval - elapsed
                if wait > 0:
                    time.sleep(wait)
                self._last_history_call_ts = time.time()

            try:
                resp = self.session.post(
                    url, data=json.dumps(payload), timeout=timeout
                )
                resp.raise_for_status()
                return resp

            except requests.HTTPError as e:
                status = e.response.status_code if e.response is not None else None
                # 429 — ждём и повторяем
                if status == 429 and attempt < max_attempts:
                    print(
                        f"[TINKOFF V2] 429 Too Many Requests "
                        f"(попытка {attempt}/{max_attempts}), "
                        f"ждём {backoff:.1f} c..."
                    )
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                # другие статусы — пробрасываем
                raise

            except requests.RequestException as e:
                # сетевые ошибки/таймауты — тоже с бэкоффом
                if attempt < max_attempts:
                    print(
                        f"[TINKOFF V2] сетевой сбой/таймаут: {e} "
                        f"(попытка {attempt}/{max_attempts}), "
                        f"ждём {backoff:.1f} c..."
                    )
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                raise

        raise RuntimeError("TinkoffV2Broker: исчерпаны попытки запроса к API.")

    # ---------------------------------------------------------------------
    # Sync-хелпер для свечей
    # ---------------------------------------------------------------------

    def _get_historical_klines_sync(
        self,
        symbol: str,
        interval: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        # --- Нормализуем время в UTC ---
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        else:
            start = start.astimezone(timezone.utc)

        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)
        else:
            end = end.astimezone(timezone.utc)

        if end <= start:
            # Пустой диапазон — сразу отдаём пустой DF в ожидаемом формате
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

        figi = self._resolve_figi(symbol)
        interval_enum = self._interval_to_v2_enum(interval)
        max_delta = self._max_delta_for_interval(interval)

        url = (
            f"{self.base_url}/"
            "tinkoff.public.invest.api.contract.v1.MarketDataService/GetCandles"
        )

        rows: list[dict] = []
        cur_from = start

        chunk_idx = 0
        max_chunks = 1000  # защита от безумия на очень больших диапазонах

        while cur_from < end and chunk_idx < max_chunks:
            chunk_idx += 1
            cur_to = min(cur_from + max_delta, end)

            payload = {
                "figi": figi,
                "from": self._to_rfc3339(cur_from),
                "to": self._to_rfc3339(cur_to),
                "interval": interval_enum,
            }

            try:
                resp = self._post_with_backoff(
                    url, payload, timeout=10, is_history=True
                )
                raw = resp.json()
            except Exception as e:
                print(
                    f"[TINKOFF V2] Candles request failed for {symbol} "
                    f"({interval}, {cur_from}..{cur_to}): {e}"
                )
                break

            candles = raw.get("candles", [])
            if not candles:
                cur_from = cur_to
                continue

            for c in candles:
                try:
                    ts = datetime.fromisoformat(
                        c["time"].replace("Z", "+00:00")
                    ).astimezone(timezone.utc).replace(tzinfo=None)

                    rows.append(
                        {
                            "open_time": ts,
                            "open": self._q_to_float(c.get("open", {})),
                            "high": self._q_to_float(c.get("high", {})),
                            "low": self._q_to_float(c.get("low", {})),
                            "close": self._q_to_float(c.get("close", {})),
                            "volume": float(c.get("volume", 0)),
                            "taker_buy_base": 0.0,
                            "funding_rate": 0.0,
                            "imbalance": 0.0,
                        }
                    )
                except Exception as e:
                    print(f"[TINKOFF V2] skip bad candle: {e}")
                    continue

            cur_from = cur_to

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
        df.drop_duplicates(subset=["open_time"], keep="last", inplace=True)
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

    # ---------------------------------------------------------------------
    # Sync-хелпер для цены
    # ---------------------------------------------------------------------

    def _get_current_price_sync(self, symbol: str) -> float:
        figi = self._resolve_figi(symbol)

        url = (
            f"{self.base_url}/"
            "tinkoff.public.invest.api.contract.v1.MarketDataService/GetLastPrices"
        )
        payload = {"instrumentId": [figi]}

        try:
            resp = self.session.post(
                url, data=json.dumps(payload), timeout=5
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            raise RuntimeError(f"TinkoffV2Broker: price request failed for {symbol}: {e}")

        prices = data.get("lastPrices", [])
        if not prices:
            raise RuntimeError(
                f"TinkoffV2Broker: no price data for FIGI={figi} (symbol={symbol})"
            )

        price_obj = prices[0].get("price", {})
        return self._q_to_float(price_obj)

    def _get_accounts_sync(self) -> list[dict]:
        url = f"{self.base_url}/tinkoff.public.invest.api.contract.v1.UsersService/GetAccounts"
        resp = self._post_with_backoff(url, payload={}, timeout=10, is_history=False)
        return resp.json().get("accounts", []) or []

    def _get_portfolio_sync(self, account_id: str) -> dict:
        url = f"{self.base_url}/tinkoff.public.invest.api.contract.v1.OperationsService/GetPortfolio"
        payload = {"accountId": account_id}
        resp = self._post_with_backoff(url, payload=payload, timeout=10, is_history=False)
        return resp.json() or {}

    # =====================================================================
    # BrokerAPI: MARKET DATA (async-обёртки)
    # =====================================================================

    async def get_historical_klines(
        self,
        symbol: str,
        interval: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        return await asyncio.to_thread(
            self._get_historical_klines_sync,
            symbol,
            interval,
            start,
            end,
        )

    async def get_current_price(self, symbol: str) -> float:
        return await asyncio.to_thread(
            self._get_current_price_sync,
            symbol,
        )

    # =====================================================================
    # BrokerAPI: ACCOUNT / TRADING (пока заглушки)
    # =====================================================================

    async def get_account_state(self) -> AccountState:
        def _sync():
            accounts = self._get_accounts_sync()
            if not accounts:
                return AccountState(equity=0.0, balance=0.0, currency="RUB", margin_used=0.0, broker=self.name)

            # P0: берём первый аккаунт
            account_id = accounts[0].get("id") or accounts[0].get("accountId")
            if not account_id:
                return AccountState(equity=0.0, balance=0.0, currency="RUB", margin_used=0.0, broker=self.name)

            pf = self._get_portfolio_sync(account_id)
            total = self._q_to_float(pf.get("totalAmountPortfolio", {}))
            cash = self._q_to_float(pf.get("totalAmountCurrencies", {}))

            # Валюта у total/cash может быть в других полях, но для P0 фиксируем RUB
            return AccountState(equity=total, balance=cash, currency="RUB", margin_used=0.0, broker=self.name)

        return await asyncio.to_thread(_sync)
    
    async def list_open_positions(self) -> List[Position]:
        def _sync():
            accounts = self._get_accounts_sync()
            if not accounts:
                return []

            account_id = accounts[0].get("id") or accounts[0].get("accountId")
            if not account_id:
                return []

            pf = self._get_portfolio_sync(account_id)
            raw_pos = pf.get("positions", []) or []

            result: list[Position] = []
            for p in raw_pos:
                figi = p.get("figi")
                qty = self._q_to_float(p.get("quantity", {}))
                avg = self._q_to_float(p.get("averagePositionPrice", {}))

                if qty == 0:
                    continue

                # P0.5+: унифицируем symbol как тикер через Config.TINKOFF_FIGI_MAP
                symbol = self._resolve_ticker(figi or "")
                result.append(
                    Position(
                        symbol=symbol,
                        quantity=qty,
                        avg_price=avg,
                        unrealized_pnl=None,
                        broker=self.name,
                    )
                )
            return result

        return await asyncio.to_thread(_sync)

    async def place_order(self, order: OrderRequest) -> OrderResult:
        def _sync():
            accounts = self._get_accounts_sync()
            if not accounts:
                raise RuntimeError("TinkoffV2Broker: no accounts available")

            account_id = accounts[0].get("id") or accounts[0].get("accountId")
            if not account_id:
                raise RuntimeError("TinkoffV2Broker: accountId missing")

            # symbol ожидаем как тикер → FIGI
            figi = self._resolve_figi(order.symbol)

            url = f"{self.base_url}/tinkoff.public.invest.api.contract.v1.OrdersService/PostOrder"
            direction = "ORDER_DIRECTION_BUY" if order.side == "buy" else "ORDER_DIRECTION_SELL"
            order_type = "ORDER_TYPE_MARKET" if order.order_type == "market" else "ORDER_TYPE_LIMIT"

            payload = {
                "accountId": account_id,
                "figi": figi,
                "quantity": int(round(float(order.quantity))),  # P0: лотность грубо
                "direction": direction,
                "orderType": order_type,
                "orderId": order.client_id or f"bot-{int(time.time()*1000)}",
            }

            # limit price, если нужен
            if order.order_type == "limit":
                # price -> quotation
                pr = float(order.price or 0.0)
                units = int(pr)
                nano = int(round((pr - units) * 1e9))
                payload["price"] = {"units": str(units), "nano": nano}

            resp = self._post_with_backoff(url, payload=payload, timeout=10, is_history=False)
            data = resp.json() or {}

            oid = data.get("orderId") or payload["orderId"]
            exec_price = self._q_to_float(data.get("executedOrderPrice", {})) or float(order.price or 0.0)

            return OrderResult(
                order_id=str(oid),
                symbol=order.symbol,
                side=order.side,
                quantity=float(order.quantity),
                price=float(exec_price),
                status="filled",  # P0: считаем market как filled
                broker=self.name,
            )

        return await asyncio.to_thread(_sync)

    async def cancel_order(self, order_id: str) -> None:
        def _sync():
            accounts = self._get_accounts_sync()
            if not accounts:
                return
            account_id = accounts[0].get("id") or accounts[0].get("accountId")
            if not account_id:
                return

            url = f"{self.base_url}/tinkoff.public.invest.api.contract.v1.OrdersService/CancelOrder"
            payload = {"accountId": account_id, "orderId": order_id}
            self._post_with_backoff(url, payload=payload, timeout=10, is_history=False)

        await asyncio.to_thread(_sync)

    async def get_open_orders(self, symbol: str) -> List[OrderResult]:
        def _sync():
            accounts = self._get_accounts_sync()
            if not accounts:
                return []
            account_id = accounts[0].get("id") or accounts[0].get("accountId")
            if not account_id:
                return []

            url = f"{self.base_url}/tinkoff.public.invest.api.contract.v1.OrdersService/GetOrders"
            payload = {"accountId": account_id}
            resp = self._post_with_backoff(url, payload=payload, timeout=10, is_history=False)
            raw = resp.json() or {}
            orders = raw.get("orders", []) or []

            result: list[OrderResult] = []
            for o in orders:
                result.append(
                    OrderResult(
                        order_id=str(o.get("orderId", "")),
                        symbol=symbol,
                        side="buy" if o.get("direction") == "ORDER_DIRECTION_BUY" else "sell",
                        quantity=float(o.get("lotsRequested", 0) or 0),
                        price=self._q_to_float(o.get("initialSecurityPrice", {})),
                        status=str(o.get("executionReportStatus", "new")),
                        broker=self.name,
                    )
                )
            return result

        return await asyncio.to_thread(_sync)
