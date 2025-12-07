# config.py
from enum import Enum
from dotenv import load_dotenv
import os, json

load_dotenv()

class UniverseMode(str, Enum):
    CRYPTO = "crypto"
    STOCKS = "stocks"
    BOTH = "both"

# текущий режим юниверса по умолчанию — читаем из ENV, fallback BOTH
_env_universe = os.getenv("UNIVERSE_MODE", "both").lower()
try:
    UNIVERSE_MODE: UniverseMode = UniverseMode(_env_universe)
except ValueError:
    UNIVERSE_MODE = UniverseMode.BOTH

# --- Глобальный список крипто-активов (как объекты с метаданными) ---
CRYPTO_UNIVERSE = [
    # --- The Originals (2017) ---
    {"symbol": "BTCUSDT", "group": "major", "quote": "USDT"}, # Listing: Aug 2017
    {"symbol": "ETHUSDT", "group": "major", "quote": "USDT"}, # Listing: Aug 2017
    {"symbol": "BNBUSDT", "group": "major", "quote": "USDT"}, # Listing: Jul 2017 (Native)
    {"symbol": "LTCUSDT", "group": "major", "quote": "USDT"}, # Listing: 2017
    {"symbol": "NEOUSDT", "group": "old",   "quote": "USDT"}, # Listing: 2017
    {"symbol": "IOTAUSDT","group": "old",   "quote": "USDT"}, # Listing: 2017
    {"symbol": "ETCUSDT", "group": "old",   "quote": "USDT"}, # Listing: 2017

    # --- Class of late 2017 / early 2018 ---
    {"symbol": "XRPUSDT", "group": "major", "quote": "USDT"}, # Listing: 2018 (Spot) / Older on others
    {"symbol": "XLMUSDT", "group": "old",   "quote": "USDT"}, # Listing: 2018
    {"symbol": "ADAUSDT", "group": "major", "quote": "USDT"}, # Listing: Apr 2018
    {"symbol": "TRXUSDT", "group": "major", "quote": "USDT"}, # Listing: Jun 2018
    {"symbol": "EOSUSDT", "group": "old",   "quote": "USDT"}, # Listing: May 2018
    {"symbol": "QTUMUSDT","group": "old",   "quote": "USDT"}, # Listing: 2018
    {"symbol": "ICXUSDT", "group": "old",   "quote": "USDT"}, # Listing: 2018
    {"symbol": "VETUSDT", "group": "old",   "quote": "USDT"}, # Listing: 2018 (Rebrand from VEN)
    {"symbol": "LINKUSDT","group": "defi",  "quote": "USDT"}, # Listing: Jan 2019 (But old reliable data)
    
    # --- Old & Liquid (выжившие) ---
    {"symbol": "ZECUSDT", "group": "old",   "quote": "USDT"},
    {"symbol": "DASHUSDT","group": "old",   "quote": "USDT"},
    {"symbol": "BATUSDT", "group": "old",   "quote": "USDT"},
    {"symbol": "ZRXUSDT", "group": "old",   "quote": "USDT"},
    {"symbol": "IOSTUSDT","group": "old",   "quote": "USDT"},
]

# --- Вселенная акций (Тинькофф / МОЕХ) ---
EQUITY_UNIVERSE = [
"SBER", "GAZP", "LKOH", "GMKN", "TATN", "NVTK", "SNGS", "ROSN",
"PLZL", "MGNT", "NLMK", "CHMF", "ALRS", "MOEX", "T", "SNGSP",
"CNYRUB_TOM", "HKDRUB_TOM", "TRYRUB_TOM", "KZTRUB_TOM", "GLDRUB_TOM",
]

def get_assets_for_universe(mode: UniverseMode | None = None) -> list[str]:
    """
    Возвращает список тикеров в зависимости от режима:
    - только крипта
    - только акции/фьючи биржи
    - совместно
    """
    if mode is None:
        mode = UNIVERSE_MODE

    if mode == UniverseMode.CRYPTO:
        return [x["symbol"] for x in CRYPTO_UNIVERSE]
    elif mode == UniverseMode.STOCKS:
        return list(EQUITY_UNIVERSE)
    else:
        return [x["symbol"] for x in CRYPTO_UNIVERSE] + list(EQUITY_UNIVERSE)

class ExecutionMode(str, Enum):
    """
    Режим исполнения стратегий:
      - BACKTEST: чистый симулятор, ордера идут только в SimulatedBroker
      - PAPER: "бумажная" торговля (пока можем вести как тот же симулятор)
      - LIVE: реальные брокеры (Bitget/Tinkoff), когда до них дойдём
    """
    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"


class Config:
    # --- Режим юниверса (крипта/биржа/оба), синхронизируется с GUI ---
    UNIVERSE_MODE: UniverseMode = UNIVERSE_MODE

    # --- ПУТИ ---
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data_cache")
    MODEL_DIR = "models_checkpoints"
    STRATEGY_FILE = "best_strategy_params.json"
    # --- РЕЖИМ ИСПОЛНЕНИЯ ---
    # Можно переключать через переменную окружения EXECUTION_MODE:
    #   backtest / paper / live
    try:
        EXECUTION_MODE = ExecutionMode(
            os.getenv("EXECUTION_MODE", "backtest").lower()
        )
    except ValueError:
        # На случай опечатки в переменной окружения
        EXECUTION_MODE = ExecutionMode.BACKTEST

    # --- Лидеры рынка ---
    # Для обратной совместимости оставляем LEADER_SYMBOL как "дефолт" (BTC)
    LEADER_SYMBOL_CRYPTO = "BTCUSDT"
    LEADER_SYMBOL_EQUITY = "MOEX"   # <-- сюда поставишь тикер индекса МОЕХ
    LEADER_SYMBOL = LEADER_SYMBOL_CRYPTO



    # Режим юниверса, синхронизированный с глобальным/ENV
    UNIVERSE_MODE: UniverseMode = UNIVERSE_MODE

    # Ссылаемся на глобальные списки, чтобы не дублировать
    CRYPTO_UNIVERSE = CRYPTO_UNIVERSE
    EQUITY_UNIVERSE = EQUITY_UNIVERSE

    # --- Общий универсум активов для фабрик/вар-рума ---
    # Всегда строим список через helper с учётом UNIVERSE_MODE
    ASSETS = get_assets_for_universe(UNIVERSE_MODE)

    @classmethod
    def crypto_symbols(cls) -> list[str]:
        return [x["symbol"] for x in cls.CRYPTO_UNIVERSE]

    # 1h показал Hurst 0.44 (Шум). Нам нужно уйти выше, где больше трендов.
    TIMEFRAME_LTF = "4h"   # Было "1h". На 4H шума меньше.
    TIMEFRAME_HTF = "1d"   # Было "4h". Старший ТФ - Дневка.
    
    # --- MONEY MANAGEMENT ---
    DEPOSIT = 1000          
    RISK_PER_TRADE = 0.02   
    MAX_OPEN_POSITIONS = 4  
    
    # --- КОМИССИИ ---
    COMMISSION = 0.00075      
    SLIPPAGE = 0.0005        
    MAX_DAILY_DRAWDOWN = 0.05 

    # --- ГЛОБАЛЬНЫЕ ПЕРЕКЛЮЧАТЕЛИ (FIXED) ---
    USE_TRAILING = True     # Включаем трейлинг-стоп
    USE_TIME_DECAY = True   # Выход по времени всё ещё глобальный

    # --- Профили настроек под разные юниверсы ---
    CRYPTO_MM = {
        "TIMEFRAME_LTF": "4h",   # как и было — 4H крипта
        "TIMEFRAME_HTF": "1d",
        "DEPOSIT": 1000,
        "RISK_PER_TRADE": 0.02,
        "MAX_OPEN_POSITIONS": 4,
        "COMMISSION": 0.00075,
        "SLIPPAGE": 0.0005,
        "MAX_DAILY_DRAWDOWN": 0.05,
        "STRATEGY": {
            "sl": 2.0,
            "tp": 3.5,       # таргет 3.5 ATR под криптовый тренд
            "conf": 0.60,
            "vol_exit": 6.0,
            "trail_on": 1.0,
            "trail_act": 1.2,
            "trail_off": 0.3,
            "max_hold": 48,  # 48 баров 4h ≈ 8 суток
            "mode": "sniper",
            "abort": 0.75,
            "pullback": 0.20,
            "fill_wait": 8,
        },
    }

    STOCKS_MM = {
        "TIMEFRAME_LTF": "1h",   # базовый ТФ для акций (ниже агрегацию сделаем в data_loader)
        "TIMEFRAME_HTF": "1d",
        "DEPOSIT": 1000,
        "RISK_PER_TRADE": 0.01,  # аккуратнее с риском
        "MAX_OPEN_POSITIONS": 2,
        "COMMISSION": 0.00025,   # ближе к реалиям МОЕХ
        "SLIPPAGE": 0.0008,
        "MAX_DAILY_DRAWDOWN": 0.03,
        "STRATEGY": {
            "sl": 1.5,           # стоп ближе — вола меньше
            "tp": 2.5,           # тейк тоже ближе
            "conf": 0.55,        # чуть мягче порог уверенности
            "vol_exit": 4.0,     # быстрее выходим по всплеску волы
            "trail_on": 1.0,
            "trail_act": 0.8,    # трейлинг включаем раньше
            "trail_off": 0.4,    # даём немного больше воздуха
            "max_hold": 32,      # 32 бара 1h ≈ до 4 торговых дней
            "mode": "sniper",
            "abort": 0.75,
            "pullback": 0.15,    # поменьше откат для входа
            "fill_wait": 12,     # чуть больше времени на исполнение
        },
    }

    # --- Применяем профиль к публичным атрибутам ---
    if UNIVERSE_MODE == UniverseMode.CRYPTO:
        _mm = CRYPTO_MM
    elif UNIVERSE_MODE == UniverseMode.STOCKS:
        _mm = STOCKS_MM
    else:
        # BOTH: по умолчанию работаем как крипто-профиль,
        # а акции будем подстраивать отдельной логикой
        _mm = CRYPTO_MM

    # Таймфреймы
    TIMEFRAME_LTF = _mm["TIMEFRAME_LTF"]
    TIMEFRAME_HTF = _mm["TIMEFRAME_HTF"]

    # Money management
    DEPOSIT = _mm["DEPOSIT"]
    RISK_PER_TRADE = _mm["RISK_PER_TRADE"]
    MAX_OPEN_POSITIONS = _mm["MAX_OPEN_POSITIONS"]

    # Комиссии и риски по дню
    COMMISSION = _mm["COMMISSION"]
    SLIPPAGE = _mm["SLIPPAGE"]
    MAX_DAILY_DRAWDOWN = _mm["MAX_DAILY_DRAWDOWN"]

    # Параметры стратегии по умолчанию
    DEFAULT_STRATEGY = _mm["STRATEGY"]

    # --- ML CONSTANTS ---
    LOOK_AHEAD = 12         
    RR_RATIO = 1.5
    LABEL_THRESHOLD = 0.0015 # Требуем движение 0.15% (было меньше)          
    MIN_EDGE = 0.15         
    
    # --- FEATURES ---
    # Набор фич для стандартного walk-forward (оставляем как есть, если всё работало)
    FEATURE_COLS = [
        'confluence_score',
        'sup_dist_atr', 'res_dist_atr',
        'level_quality',
        'sup_strength', 'res_strength',
        'fib_382', 'fib_618',
        'rvol', 'rsi', 'adx', 'volatility',
        'regime', 'funding_rate', 'leader_close',
    ]

    # Набор фич для универсальной модели (БЕЗ абсолютных цен)
    UNIVERSAL_FEATURE_COLS = [
        # интегральный скор
        'confluence_score',

        # локальная структура и уровни
        'sup_dist_atr', 'res_dist_atr',
        'sup_strength', 'res_strength',
        'level_quality',

        # фибо-зоны
        'fib_382', 'fib_618',

        # моментум / волатильность / объём
        'rvol', 'rsi', 'adx', 'volatility',

        # режим и фандинг (относительные величины)
        'regime', 'funding_rate',

        # контекст старшего ТФ (если уже прокинут через add_htf_features)
        'htf_sup_strength', 'htf_res_strength',
        'htf_sup_dist_atr', 'htf_res_dist_atr',
        'htf_channel_pos', 'htf_squeeze_factor',
    ]
    
    # --- PYRAMID SCORING (НОВЫЕ ВЕСА) ---
    # Примечание: Сейчас основные веса жестко заданы в features_lib.py
    SCORING = {
        'channel_weight': 40.0,   # Самый важный фактор - мы у границы?
        'level_weight': 30.0,     # Совпадение с красной линией
        'rsi_weight': 20.0,       # Перегрев/Перепроданность
        'trend_bonus': 20.0       # Бонус за торговлю по наклону канала
    }

    # --- XGBOOST ---
    MODEL_MAX_DEPTH = 10
    MODEL_N_ESTIMATORS = 800
    MODEL_LEARNING_RATE = 0.01
    USE_CALIBRATION = True
    
    # --- SYSTEM ---
    WALK_FORWARD_WINDOW = 800 
    USE_REDIS = True 
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
    
    # --- БРОКЕРЫ / БИРЖИ ---
    # Основной поставщик данных и торговли для крипты по умолчанию — Bitget
    DEFAULT_BROKER = "bitget"

    BROKERS = {
        "bitget": {
            "type": "crypto",
            "name": "Bitget",
            "api_key": os.getenv("BITGET_API_KEY", ""),
            "api_secret": os.getenv("BITGET_API_SECRET", ""),
            "passphrase": os.getenv("BITGET_API_PASSPHRASE", ""),
            "base_url": os.getenv("BITGET_BASE_URL", "https://api.bitget.com"),
        },
        "tinkoff": {
            "type": "broker",
            "name": "Tinkoff",
            "token": os.getenv("TINKOFF_API_TOKEN", ""),
            # По умолчанию считаем, что работаем в песочнице (для ордеров),
            # чтобы случайно не торгануть живыми деньгами.
            "sandbox": os.getenv("TINKOFF_SANDBOX", "true").lower() == "true",
        },
    }

    # Куда какие тикеры маршрутизируем (будем дополнять по ходу)
    ASSET_ROUTING = {
        # Криптовалюта — Bitget
        "BTCUSDT": "bitget",
        "ETHUSDT": "bitget",

    # Акции МОЕХ — через Тинькофф
    # --- MOEX / ВАЛЮТЫ → TINKOFF ---
    "SBER": "tinkoff",
    "GAZP": "tinkoff",
    "LKOH": "tinkoff",
    "GMKN": "tinkoff",
    "ROSN": "tinkoff",
    "NVTK": "tinkoff",
    "PLZL": "tinkoff",
    "CHMF": "tinkoff",
    "NLMK": "tinkoff",
    "SNGS": "tinkoff",
    "SNGSP": "tinkoff",
    "ALRS": "tinkoff",
    "TATN": "tinkoff",
    "MGNT": "tinkoff",
    "MOEX": "tinkoff",
    "T": "tinkoff",
    # Валютные инструменты
    "GLDRUB_TOM": "tinkoff",
    "HKDRUB_TOM": "tinkoff",
    "KZTRUB_TOM": "tinkoff",
    "CNYRUB_TOM": "tinkoff",
    "TRYRUB_TOM": "tinkoff",
}

    # FIGI-идентификаторы для Тинькофф (заполнишь под себя)
    # Важно: FIGI в Тинькофф могут отличаться от "официального" FIGI,
    # поэтому лучше ориентироваться на то, что возвращает API /market/stocks
    # или инструменты v2.
    TINKOFF_FIGI_MAP = {
        # FIGI - Тинькофф инвестиции. Ищутся скриптом
        "SBER": "BBG004730N88",
        "LKOH": "BBG004731032",
        "GAZP": "BBG004730RP0",
        "GMKN": "BBG004731489",
        "TATN": "BBG004RVFFC0",
        "NVTK": "BBG00475KKY8",
        "SNGS": "BBG0047315D0",
        "ROSN": "BBG004731354",
        "PLZL": "BBG000R607Y3",
        "MGNT": "BBG004RVFCY3",
        "NLMK": "BBG004S681B4",
        "CHMF": "BBG00475K6C3",
        "ALRS": "BBG004S68B31",
        "MOEX": "BBG004730JJ5",
        "T": "TCS80A107UL4",
        "SNGSP": "BBG004S681M2",
        "CNYRUB_TOM": "BBG0013HRTL0",
        "HKDRUB_TOM": "BBG0013HSW87",
        "TRYRUB_TOM": "BBG0013J12N1",
        "KZTRUB_TOM": "BBG0013HG026",
        "GLDRUB_TOM": "BBG000VJ5YR4",
    }

    TG_API_ID = os.getenv("TELEGRAM_API_ID")
    TG_API_HASH = os.getenv("TELEGRAM_API_HASH")

    # --- Telegram channels per universe ---
    TG_CRYPTO_CHANNELS = [
        'tree_of_alpha',
        'unusual_whales',
        'WatcherGuru',
        'Tier10k',
        'WalterBloomberg',
        'Cointelegraph',
        'CryptoTownEU',
    ]

    TG_STOCK_CHANNELS = [
        'smartlab_ru',
        'moex_official',
        'finam_ru',
        'rbc_invest',
    ]

    # --- Flags: enable / disable Telegram HTF per asset class ---
    # Читаются из ENV или .env через load_dotenv() в начале файла.
    USE_TG_CRYPTO = os.getenv("USE_TG_CRYPTO", "1") == "1"
    USE_TG_STOCKS = os.getenv("USE_TG_STOCKS", "1") == "1"

    @classmethod
    def equity_symbols(cls) -> list[str]:
        return cls.EQUITY_UNIVERSE

    @classmethod
    def get_leader_for_symbol(cls, symbol: str) -> str:
        """
        Выбирает лидера в зависимости от класса актива.
        """
        if symbol in cls.crypto_symbols():
            return cls.LEADER_SYMBOL_CRYPTO
        if symbol in cls.equity_symbols():
            return cls.LEADER_SYMBOL_EQUITY
        # Fallback: считаем всё остальное криптой/общим рынком
        return cls.LEADER_SYMBOL_CRYPTO

    @classmethod
    def get_strategy_params(cls):
        if os.path.exists(cls.STRATEGY_FILE):
            try:
                with open(cls.STRATEGY_FILE, "r") as f:
                    params = json.load(f)
                    # Fallback для новых параметров
                    if 'max_hold' not in params: params['max_hold'] = 48
                    return params
            except: pass
        return cls.DEFAULT_STRATEGY.copy()