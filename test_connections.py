import sys
import os
import time
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from config import Config
from brokers import get_broker


# ====== Цвета для консоли (UI) ======
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_status(prefix, status, msg=""):
    """
    Красивый статус-логгер: OK / FAIL.
    """
    if status == "OK":
        print(f"[{prefix}] ... {Colors.OKGREEN}ONLINE ✅{Colors.ENDC} | {msg}")
    elif status == "WARN":
        print(f"[{prefix}] ... {Colors.WARNING}WARN ⚠️{Colors.ENDC} | {msg}")
    else:
        print(f"[{prefix}] ... {Colors.FAIL}FAILED ❌{Colors.ENDC} | {msg}")


# ====== Вспомогательная обёртка: запуск с таймаутом ======
def run_with_timeout(func, timeout_sec, *args, **kwargs):
    """
    Запустить функцию в отдельном потоке и ждать не дольше timeout_sec.
    Возвращает (result, error):

      - result: то, что вернула функция, если успела
      - error: None, либо:
          - "timeout"  -> если превысили лимит времени
          - Exception  -> если внутри был raise
    """
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            result = future.result(timeout=timeout_sec)
            return result, None
        except FuturesTimeoutError:
            return None, "timeout"
        except Exception as e:
            return None, e


# ====== TEST: BITGET ======
def test_bitget():
    print(f"\n{Colors.HEADER}--- TESTING BITGET (CRYPTO) ---{Colors.ENDC}")
    symbol = "BTCUSDT"

    try:
        broker = get_broker("bitget")

        # 1) Тест цены
        t0 = time.perf_counter()
        price = broker.get_current_price(symbol)
        dt = time.perf_counter() - t0
        print_status("BITGET DATA", "OK", f"{symbol} Price: ${price:,.2f} (t={dt:.2f}s)")

        # 2) Тест исторических свечей (последние 24 часа, 1h)
        end = datetime.now()
        start = end - timedelta(days=1)

        print(f"{Colors.OKBLUE}    → Requesting candles "
              f"{symbol}, interval=1h, window=1d [{start}..{end}]{Colors.ENDC}")
        t0 = time.perf_counter()
        klines = broker.get_historical_klines(symbol, "1h", start, end)
        dt = time.perf_counter() - t0

        if klines is not None and not klines.empty:
            first_ts = klines.index[0]
            last_ts = klines.index[-1]
            print_status(
                "BITGET HISTORY",
                "OK",
                f"Loaded {len(klines)} candles in {dt:.2f}s "
                f"(from {first_ts} to {last_ts})"
            )
        else:
            print_status(
                "BITGET HISTORY",
                "FAIL",
                "Empty DataFrame returned (no candles for last 1d)"
            )

    except Exception as e:
        print_status("BITGET", "FAIL", f"{type(e).__name__}: {e}")
        print(f"{Colors.WARNING}Check .env keys for BITGET!{Colors.ENDC}")


# ====== TEST: TINKOFF ======
def _load_tinkoff_candles(broker, symbol, interval, days):
    """
    Чистая функция: грузит свечи Tinkoff, чтобы можно было
    оборачивать её в run_with_timeout.
    """
    end = datetime.now()
    start = end - timedelta(days=days)
    return broker.get_historical_klines(symbol, interval, start, end), (start, end)


def test_tinkoff():
    print(f"\n{Colors.HEADER}--- TESTING TINKOFF (STOCKS) ---{Colors.ENDC}")
    symbol = "SBER"

    # 0) Проверяем FIGI в конфиге
    figi = Config.TINKOFF_FIGI_MAP.get(symbol)
    if not figi:
        print_status("CONFIG", "FAIL", f"No FIGI found for {symbol} in Config.TINKOFF_FIGI_MAP")
        return

    try:
        broker = get_broker("tinkoff")

        # 1) Тест цены
        t0 = time.perf_counter()
        price = broker.get_current_price(symbol)
        dt = time.perf_counter() - t0
        print_status("TINKOFF DATA", "OK", f"{symbol} (FIGI: {figi}) Price: {price} RUB (t={dt:.2f}s)")

        # 2) Последовательно тестируем несколько окон истории
        tests = [
            ("1h", 1, 10),    # 1 день по часу, timeout 10s
            ("1h", 7, 15),    # 7 дней по часу, timeout 15s
            ("4h", 30, 20),   # 30 дней по 4h, timeout 20s
        ]

        for interval, days, timeout_sec in tests:
            print(
                f"{Colors.OKBLUE}    → Requesting candles {symbol}, "
                f"interval={interval}, window={days}d, timeout={timeout_sec}s{Colors.ENDC}"
            )
            t0 = time.perf_counter()
            (klines, (start, end)), err = run_with_timeout(
                _load_tinkoff_candles, timeout_sec, broker, symbol, interval, days
            )
            dt = time.perf_counter() - t0

            step_name = f"TINKOFF {interval} {days}d"

            if err is None:
                if klines is not None and not klines.empty:
                    first_ts = klines.index[0]
                    last_ts = klines.index[-1]
                    print_status(
                        step_name,
                        "OK",
                        f"Loaded {len(klines)} candles in {dt:.2f}s "
                        f"(from {first_ts} to {last_ts})"
                    )
                else:
                    print_status(
                        step_name,
                        "WARN",
                        f"Empty DataFrame (no candles in range {start}..{end})"
                    )

            elif err == "timeout":
                print_status(
                    step_name,
                    "FAIL",
                    f"Timeout after {dt:.2f}s — get_historical_klines "
                    f"did not return (возможно, rate-limit или блокировка API)"
                )
                # Дополнительно: не мучаем API дальше
                break

            else:
                print_status(
                    step_name,
                    "FAIL",
                    f"{type(err).__name__}: {err}"
                )
                break

    except Exception as e:
        print_status("TINKOFF", "FAIL", f"{type(e).__name__}: {e}")
        print(f"{Colors.WARNING}Check .env keys for TINKOFF!{Colors.ENDC}")


# ====== MAIN ======
if __name__ == "__main__":
    print(f"{Colors.BOLD}Current Execution Mode: {Config.EXECUTION_MODE.value}{Colors.ENDC}")

    test_bitget()
    test_tinkoff()

    print("\nDone.")
