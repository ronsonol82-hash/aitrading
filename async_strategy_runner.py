# async_strategy_runner.py
import asyncio
import pickle
from datetime import datetime
from typing import Dict
import pandas as pd
from execution_router import ExecutionRouter
from config import Config
from risk_utils import calc_position_size


class AsyncStrategyRunner:
    """
    Асинхронный раннер стратегий.
    Загружает сигналы и исполняет их через ExecutionRouter.
    """
    
    def __init__(self, signals_file: str = "data_cache/production_signals_v1.pkl"):
        self.signals_file = signals_file
        self.signals: Dict[str, pd.DataFrame] = {}
        self.router = ExecutionRouter()

        # --- NEW ---
        self.assets_filter: list[str] | None = None
        self._stop: bool = False
        self._protections: dict[str, dict] = {}
        
    async def initialize(self):
        """Инициализация роутера и загрузка сигналов"""
        await self.router.initialize()
        self.load_signals()

    # --- NEW ---
    def set_assets(self, assets: list[str]):
        """
        Ограничивает торговлю указанным списком тикеров.
        Если None/пусто — торгуем всем, что есть в signals.
        """
        if assets:
            self.assets_filter = list(assets)
        else:
            self.assets_filter = None

    def request_stop(self):
        """
        Просим бесконечный цикл run_forever() мягко остановиться.
        """
        self._stop = True

    @staticmethod
    def _compute_risk_per_trade(confidence: float,
                                base_risk: float,
                                max_risk: float,
                                threshold: float) -> float:
        """
        Kelly-like money management:
        - при confidence = threshold → base_risk
        - при confidence → 1.0 → max_risk
        """
        if confidence is None:
            return base_risk

        # Нормализуем [threshold..1] → [0..1]
        scale = (confidence - threshold) / (1.0 - threshold + 1e-6)
        scale = max(0.0, min(1.0, scale))

        risk = base_risk + (max_risk - base_risk) * scale
        # Защита от выхода за пределы
        return max(base_risk, min(max_risk, risk))
        
    def load_signals(self):
        """Загрузка сигналов из файла"""
        try:
            with open(self.signals_file, "rb") as f:
                self.signals = pickle.load(f)
            print(f"📊 Загружено сигналов для {len(self.signals)} активов")
        except FileNotFoundError:
            print(f"⚠️  Файл сигналов не найден: {self.signals_file}")
            self.signals = {}
            
    async def get_current_signals(self, symbol: str):
        """Получение текущих сигналов для символа"""
        if symbol not in self.signals:
            return None
            
        df = self.signals[symbol]
        if df.empty:
            return None
            
        # Возвращаем последний сигнал
        return df.iloc[-1]
    
    async def run_strategy(self, risk_per_trade: float | None = None):
        """
        Запуск стратегии на основе загруженных сигналов.

        ВАЖНО:
        - базовый риск берём из Config.RISK_PER_TRADE (или из аргумента),
        - далее для каждой сделки масштабируем его по Kelly-подобной формуле
        как в Backtester.open_smart_position().
        """
        await self._check_protective_exits()
        if not self.signals:
            print("❌ Нет сигналов для торговли")
            return

        # 1) Состояние счёта
        account_state = await self.router.get_global_account_state()
        total_equity = account_state.equity

        # 2) Параметры стратегии (порог confidence и т.п.)
        try:
            params = Config.get_strategy_params()
        except AttributeError:
            params = getattr(Config, "DEFAULT_STRATEGY", {})

        threshold = float(params.get("conf", 0.6))      # как в бэктесте
        base_risk = risk_per_trade if risk_per_trade is not None else 0.01
        max_risk = getattr(Config, "MAX_RISK_PER_TRADE", 0.03) or 0.03
        
        print(f"🏦 Начальный капитал: {total_equity:.2f}")
        
        # Проходим по всем активам
        trades_executed = 0
        
        for symbol, df in self.signals.items():
            if df.empty:
                continue

            # Фильтр по выбранным активам
            if self.assets_filter and symbol not in self.assets_filter:
                continue

            # Последняя строка с сигналами
            last_signal = df.iloc[-1]
            p_long = float(last_signal.get("p_long", 0.0) or 0.0)
            p_short = float(last_signal.get("p_short", 0.0) or 0.0)

            # Берём максимум как "confidence" — как в Backtester
            confidence = max(p_long, p_short)

            # Динамический риск для этой сделки
            risk_this_trade = self._compute_risk_per_trade(
                confidence=confidence,
                base_risk=base_risk,
                max_risk=max_risk,
                threshold=threshold,
            )

            # P0: позиции, чтобы понимать можно ли SELL
            positions = await self.router.list_all_positions()
            pos_map = {p.symbol: p for p in positions}
            pos = pos_map.get(symbol)

            if p_long > 0.65:
                # long entry допускаем только если позиции нет
                if pos is None or float(pos.quantity or 0.0) <= 0:
                    await self.execute_trade(
                        symbol=symbol,
                        side="buy",
                        probability=p_long,
                        equity=total_equity,
                        risk_per_trade=risk_this_trade,
                        signal_data=last_signal,
                    )
                    trades_executed += 1
                else:
                    # уже есть long — на P0 не пирамидим
                    continue

            elif p_short > 0.65:
                # P0: SHORT на SPOT запрещён. SELL — только закрытие long.
                if pos is not None and float(pos.quantity or 0.0) > 0:
                    await self.execute_trade(
                        symbol=symbol,
                        side="sell",  # close long
                        probability=p_short,
                        equity=total_equity,
                        risk_per_trade=risk_this_trade,
                        signal_data=last_signal,
                    )
                    trades_executed += 1
                else:
                    print(f"⛔ [P0] SHORT blocked on SPOT for {symbol}. No long to close → skip.")
                    continue
                
        print(f"✅ Исполнено ордеров: {trades_executed}")

    async def run_forever(
        self,
        risk_per_trade: float | None = None,
        interval_sec: float = 60.0,
    ):
        """
        Бесконечный цикл исполнения стратегии.

        - каждые interval_sec перезагружает сигналы и один раз пробегает run_strategy();
        - завершение через request_stop().
        """
        if risk_per_trade is None:
            risk_per_trade = getattr(Config, "RISK_PER_TRADE", 0.02)

        self._stop = False

        while not self._stop:
            # На всякий случай перезагрузим сигналы (вдруг файл обновился)
            self.load_signals()

            if not self.signals:
                print("❌ Нет сигналов для торговли (run_forever)")
            else:
                await self.run_strategy(risk_per_trade=risk_per_trade)

            # Если попросили остановиться — выходим без лишнего sleep
            if self._stop:
                break

            try:
                await asyncio.sleep(interval_sec)
            except asyncio.CancelledError:
                break
        
    async def execute_trade(
        self,
        symbol: str,
        side: str,
        probability: float,
        equity: float,
        risk_per_trade: float,
        signal_data: pd.Series,
    ):
        """Исполнение одной сделки с расчетом размера по ATR."""
        try:
            # 1) Текущая цена
            broker = await self.router.get_broker_for_symbol(symbol)
            current_price = await broker.get_current_price(symbol)

            # 2) ATR и параметры риска
            atr_value = float(signal_data.get("atr", 0.0))
            sl_mult = Config.DEFAULT_STRATEGY.get("sl", 2.0)
            max_notional = getattr(Config, "MAX_POSITION_NOTIONAL", None)

            ps = calc_position_size(
                equity=equity,
                risk_per_trade=risk_per_trade,
                atr=atr_value,
                sl_mult=sl_mult,
                price=current_price,
                max_notional=max_notional,
            )

            position_size = ps.size

            if position_size <= 0:
                print(
                    f"⚠️  Пропуск {symbol}: размер позиции <= 0 "
                    f"(ATR={atr_value:.6f}, equity={equity:.2f})"
                )
                return

            # 3) Исполняем ордер
            print(f"📈 Исполнение {side.upper()} для {symbol}:")
            print(f"   Цена: {current_price:.4f}, Вероятность: {probability:.2%}")
            print(
                f"   ATR={atr_value:.6f}, SLxATR={sl_mult}, "
                f"StopDist={ps.stop_distance:.6f}"
            )
            print(
                f"   Риск: {ps.risk_amount:.2f}, "
                f"Нотионал: {ps.notional:.2f}, "
                f"Размер: {position_size:.6f}"
            )

            result = await self.router.execute_order(
                symbol=symbol,
                side=side,
                quantity=position_size,
                order_type="market",
            )

            # --- P0.5+: reconciliation after trade ---
            # Идея: после сделки проверяем, что позиция стала такой, какой мы ожидаем.
            # Если нет — HALT (чтобы не наращивать хаос в LIVE).

            try:
                # небольшая пауза на обновление состояния у брокера/роутера
                await asyncio.sleep(0.4)

                positions_after = await self.router.list_all_positions()

                # P0.5++: сначала пытаемся найти позицию по instrument_id (FIGI), потом по symbol.
                pos_after = None

                # 1) FIGI-match (только если есть в конфиге)
                figi = None
                try:
                    from config import Config
                    figi_map = getattr(Config, "TINKOFF_FIGI_MAP", {}) or {}
                    figi = figi_map.get(symbol)  # symbol тут тикер типа SBER
                except Exception:
                    figi = None

                if figi:
                    for p in positions_after:
                        if getattr(p, "instrument_id", None) == figi:
                            pos_after = p
                            break

                # 2) Fallback: symbol-match
                if pos_after is None:
                    for p in positions_after:
                        if p.symbol == symbol:
                            pos_after = p
                            break

                qty_after = float(pos_after.quantity or 0.0) if pos_after is not None else 0.0
                qty_sent = float(position_size)

                if side == "buy":
                    # ожидаем, что позиция появилась (qty_after > 0)
                    if qty_after <= 0:
                        # жёстко тормозим — дальше торговать опасно
                        if hasattr(self.router, "_trading_halted"):
                            self.router._trading_halted = True
                        raise RuntimeError(f"[RECON] BUY failed: expected position >0 for {symbol}, got {qty_after}")

                    # Optional: sanity-check на слишком сильное расхождение
                    # (на SPOT обычно qty_after ~ qty_sent, но может отличаться из-за нормализации/комиссий)
                    if qty_sent > 0 and qty_after < qty_sent * 0.5:
                        if hasattr(self.router, "_trading_halted"):
                            self.router._trading_halted = True
                        raise RuntimeError(
                            f"[RECON] BUY suspicious fill: {symbol} qty_after={qty_after}, qty_sent={qty_sent}"
                        )

                elif side == "sell":
                    # sell у нас P0 = закрытие long, значит ожидаем qty_after == 0
                    if qty_after > 0:
                        if hasattr(self.router, "_trading_halted"):
                            self.router._trading_halted = True
                        raise RuntimeError(f"[RECON] SELL failed: expected position 0 for {symbol}, got {qty_after}")

            except Exception as e:
                print(str(e))
                # Пробрасываем выше, чтобы цикл стратегии не продолжал торговлю
                raise

            print(f"   ✅ Ордер исполнен: {result.order_id}")

        except NotImplementedError:
            print(f"⚠️  Брокер для {symbol} не поддерживает торговлю")
        except Exception as e:
            print(f"❌ Ошибка исполнения для {symbol}: {e}")

    async def _check_protective_exits(self) -> None:
        # Снимок позиций
        positions = await self.router.list_all_positions()
        pos_map = {p.symbol: p for p in positions}

        for symbol, prot in list(self._protections.items()):
            p = pos_map.get(symbol)
            if p is None or float(p.quantity or 0.0) <= 0:
                # позиции нет — защита не нужна
                self._protections.pop(symbol, None)
                continue

            broker = await self.router.get_broker_for_symbol(symbol)
            last = await broker.get_current_price(symbol)

            sl = float(prot.get("sl", 0.0) or 0.0)
            tp = float(prot.get("tp", 0.0) or 0.0)

            # P0: только long
            if sl > 0 and last <= sl:
                print(f"🛑 [SL] {symbol} last={last:.6f} <= sl={sl:.6f} → CLOSE")
                await self.router.execute_order(symbol=symbol, side="sell", quantity=float(p.quantity), order_type="market")
                self._protections.pop(symbol, None)
                continue

            if tp > 0 and last >= tp:
                print(f"🎯 [TP] {symbol} last={last:.6f} >= tp={tp:.6f} → CLOSE")
                await self.router.execute_order(symbol=symbol, side="sell", quantity=float(p.quantity), order_type="market")
                self._protections.pop(symbol, None)
                continue 

    async def close(self):
        """Закрытие ресурсов"""
        await self.router.close()


async def main():
    """Основная асинхронная функция"""
    runner = AsyncStrategyRunner()
    
    try:
        await runner.initialize()
        await runner.run_strategy(risk_per_trade=0.01)  # 1% риска на сделку
        
        # Показываем итоговое состояние
        account_state = await runner.router.get_global_account_state()
        print(f"\n🏦 Итоговый капитал: {account_state.equity:.2f}")
        
        positions = await runner.router.list_all_positions()
        print(f"📊 Открытых позиций: {len(positions)}")
        
    finally:
        await runner.close()


if __name__ == "__main__":
    asyncio.run(main())