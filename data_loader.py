# data_loader.py
import pandas as pd
import requests
import time
import os
import numpy as np
import redis
import json
from datetime import datetime, timedelta
from config import Config
from brokers import get_broker
from indicators import FeatureEngineer

class DataLoader:
    CACHE_DIR = "data_cache"
    NEWS_FILE = "data_cache/news_sentiment.csv"
    
    def __init__(self):
        self.redis_client = None
        if Config.USE_REDIS:
            try:
                self.redis_client = redis.Redis(host=Config.REDIS_HOST, port=Config.REDIS_PORT, db=0)
                self.redis_client.ping()
                print("🧠 [DATA] Redis подключен.")
            except:
                print("⚠️ [DATA] Redis недоступен. Работаем с файлами.")
    
    @staticmethod
    def _ensure_cache_dir():
        if not os.path.exists(DataLoader.CACHE_DIR):
            os.makedirs(DataLoader.CACHE_DIR)

    @staticmethod
    def get_funding_history(symbol, start_ts, end_ts):
        base_url = "https://fapi.binance.com/fapi/v1/fundingRate"
        all_funding = []
        current_start = start_ts
        
        print(f"   💸 [FUNDING] Загрузка ставки для {symbol}...")
        
        while current_start < end_ts:
            params = {
                'symbol': symbol,
                'startTime': current_start,
                'endTime': end_ts,
                'limit': 1000
            }
            try:
                r = requests.get(base_url, params=params, timeout=5)
                if r.status_code != 200:
                    break
                data = r.json()
                if not data: break
                
                all_funding.extend(data)
                current_start = data[-1]['fundingTime'] + 1
                time.sleep(0.05)
            except:
                break
                
        if not all_funding:
            return pd.DataFrame()
            
        df = pd.DataFrame(all_funding)
        df['fundingRate'] = df['fundingRate'].astype(float)
        df['datetime'] = pd.to_datetime(df['fundingTime'], unit='ms')
        df.set_index('datetime', inplace=True)
        return df[['fundingRate']]

    @staticmethod
    def get_binance_data(symbol, start_date, end_date, interval):
        endpoints = [
            ("Futures Global", "https://fapi.binance.com/fapi/v1/klines"),
            ("Spot Global", "https://api.binance.com/api/v3/klines")
        ]

        start_ts = int(start_date.timestamp() * 1000)
        end_ts = int(end_date.timestamp() * 1000)
        
        df_candles = pd.DataFrame()
        
        for region, base_url in endpoints:
            print(f"📥 [BINANCE {region}] Загрузка {symbol}...")
            
            all_candles = []
            current_start = start_ts
            failed = False
            
            while current_start < end_ts:
                params = {
                    'symbol': symbol,
                    'interval': interval,
                    'startTime': current_start,
                    'endTime': end_ts,
                    'limit': 1500
                }
                
                try:
                    response = requests.get(base_url, params=params, timeout=5)
                    if response.status_code != 200:
                        failed = True; break
                    
                    data = response.json()
                    if not data: break
                        
                    all_candles.extend(data)
                    current_start = data[-1][6] + 1
                    time.sleep(0.05)
                    
                except Exception as e:
                    print(f"⚠️ Ошибка {region}: {e}")
                    failed = True; break
            
            if not failed and all_candles:
                df = pd.DataFrame(all_candles, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume', 
                    'close_time', 'q_vol', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'
                ])
                
                cols = ['open', 'high', 'low', 'close', 'volume', 'taker_buy_base']
                df[cols] = df[cols].astype(float)
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('datetime', inplace=True)
                
                df_candles = df
                print(f"✅ Свечи загружены: {len(df)}")
                break

        if df_candles.empty:
            print("❌ Не удалось загрузить свечи.")
            return pd.DataFrame()

        taker_buy = df_candles['taker_buy_base']
        total_vol = df_candles['volume']
        taker_sell = total_vol - taker_buy
        df_candles['imbalance'] = (taker_buy - taker_sell) / total_vol.replace(0, 1)
        
        try:
            df_funding = DataLoader.get_funding_history(symbol, start_ts, end_ts)
            if not df_funding.empty:
                df_candles = df_candles.sort_index()
                df_funding = df_funding.sort_index()
                combined = pd.merge_asof(
                    df_candles, 
                    df_funding, 
                    left_index=True, 
                    right_index=True, 
                    direction='backward'
                )
                combined['fundingRate'] = combined['fundingRate'].fillna(method='bfill').fillna(0)
                df_candles['funding_rate'] = combined['fundingRate']
            else:
                df_candles['funding_rate'] = 0.0
        except Exception as e:
            print(f"⚠️ Ошибка мерджа фандинга: {e}")
            df_candles['funding_rate'] = 0.0

        return df_candles[['open', 'high', 'low', 'close', 'volume', 'taker_buy_base', 'funding_rate', 'imbalance']]
    
    @staticmethod
    def get_exchange_data(symbol, start_date, end_date, interval):
        """
        Унифицированная точка входа для загрузки свечей с биржи.

        Логика:
        - смотрим Config.ASSET_ROUTING → определяем брокера;
        - ЕСЛИ символ НЕ прописан явно → сразу используем Binance-фоллбэк;
        - если брокер = bitget/tinkoff, пробуем его,
            при ошибке / неподдерживаемом инструменте — fallback на Binance.
        """
        # Берём ЯВНЫЙ маршрут, без дефолта
        broker_name = Config.ASSET_ROUTING.get(symbol, Config.DEFAULT_BROKER)
        uname = str(broker_name).lower() if broker_name else None

        # --- BITGET (крипта, только явно маршрутизированные тикеры) ---
        if uname == "bitget":
            try:
                broker = get_broker("bitget")
                return broker.get_historical_klines(
                    symbol=symbol,
                    interval=interval,
                    start=start_date,
                    end=end_date,
                )
            except Exception as e:
                print(f"⚠️ [DATA] Bitget failed for {symbol}, fallback to Binance: {e}")
                return DataLoader.get_binance_data(symbol, start_date, end_date, interval)

        # --- TINKOFF (акции МОЕХ, тоже только явно маршрутизированные) ---
        if uname == "tinkoff":
            print(f"📥 [TINKOFF] Загрузка {symbol}...")
            try:    
                broker = get_broker("tinkoff")
                return broker.get_historical_klines(
                    symbol=symbol,
                    interval=interval,
                    start=start_date,
                    end=end_date,
                )
            except Exception as e:
                print(f"⚠️ [DATA] Tinkoff failed for {symbol}, fallback to Binance: {e}")
                return DataLoader.get_binance_data(symbol, start_date, end_date, interval)

        # --- ВСЁ ОСТАЛЬНОЕ → Binance как универсальный поставщик истории ---
        return DataLoader.get_binance_data(symbol, start_date, end_date, interval)

    def load_news_sentiment(self):
        if self.redis_client:
            try:
                cached = self.redis_client.lrange("news_sentiment", 0, -1)
                if cached:
                    data = [json.loads(x) for x in cached]
                    df = pd.DataFrame(data)
                    df['datetime'] = pd.to_datetime(df['datetime'])
                    df.set_index('datetime', inplace=True)
                    try:
                        # Ресемплим под конфиг (1h или 15m)
                        return df.resample(Config.TIMEFRAME_LTF)['sentiment'].mean().to_frame()
                    except: return None
            except: pass

        if not os.path.exists(DataLoader.NEWS_FILE): return None
        try:
            df_news = pd.read_csv(DataLoader.NEWS_FILE, index_col='datetime', parse_dates=True)
            df_news['sentiment_ema'] = df_news['sentiment'].ewm(span=12).mean()
            return df_news[['sentiment', 'sentiment_ema']]
        except: return None

    @staticmethod
    def _fetch_and_cache(symbol, start_date, end_date, interval):
        DataLoader._ensure_cache_dir()
        safe_symbol = symbol.replace("-", "").replace("/", "")
        
        # v7 - версия кэша для 1H/4H
        filename = f"{DataLoader.CACHE_DIR}/{safe_symbol}_{interval}_v7_WAR.csv"
        
        if os.path.exists(filename):
            try:
                df = pd.read_csv(filename, index_col='datetime', parse_dates=True)
                if len(df) > 100:
                    mask = (df.index >= start_date) & (df.index <= end_date)
                    if len(df.loc[mask]) > 0: return df.loc[mask]
            except: pass 
        
        # 👉 здесь вместо прямого бинанса теперь используем абстрактный метод
        df = DataLoader.get_exchange_data(safe_symbol, start_date, end_date, interval)
        if not df.empty:
            df.to_csv(filename)
        return df

    @staticmethod
    def merge_mtf(df_ltf, df_htf):
        """
        Прокидывает контекст старшего ТФ в младший.

        Теперь используем полноценный пайплайн из FeatureEngineer.add_htf_features:
        - считаем уровни/канал/squeeze на HTF;
        - мержим их в LTF с префиксом htf_;
        - при отсутствии HTF аккуратно подставляем нули.
        """
        from indicators import FeatureEngineer

        # Если HTF нет — просто создаём нужные колонки с нулями,
        # чтобы UNIVERSAL_FEATURE_COLS всегда находились в df.
        if df_htf is None or df_htf.empty:
            htf_cols = [
                "htf_volatility",
                "htf_sup_strength",
                "htf_res_strength",
                "htf_sup_dist_atr",
                "htf_res_dist_atr",
                "htf_channel_pos",
                "htf_squeeze_factor",
            ]
            for c in htf_cols:
                if c not in df_ltf.columns:
                    df_ltf[c] = 0.0
            return df_ltf

        df_ltf = df_ltf.sort_index()
        df_htf = df_htf.sort_index()

        try:
            # Здесь внутри:
            # 1) StructureFeatures.process_all на df_htf
            # 2) выбор ['volatility', 'sup_strength', ...]
            # 3) add_prefix('htf_') и merge_asof в LTF
            df_merged = FeatureEngineer.add_htf_features(df_ltf, df_htf)
            return df_merged
        except Exception as e:
            print(f"⚠️ [HTF] Ошибка при merge_mtf: {e}")
            # Аварийный фоллбэк — хотя бы нули, чтобы не падало обучение
            htf_cols = [
                "htf_volatility",
                "htf_sup_strength",
                "htf_res_strength",
                "htf_sup_dist_atr",
                "htf_res_dist_atr",
                "htf_channel_pos",
                "htf_squeeze_factor",
            ]
            for c in htf_cols:
                if c not in df_ltf.columns:
                    df_ltf[c] = 0.0
            return df_ltf

    @staticmethod
    def get_portfolio_data(
        assets,
        leader_symbol,          # str ИЛИ dict[str, str]
        start_date,
        end_date,
        interval_ltf,
        interval_htf,
    ):
        """
        Загружает портфель с фичами, новостями и колонкой leader_close.

        leader_symbol:
            - str  -> один лидер для всех (старое поведение)
            - dict -> {symbol: leader_symbol} для каждого инструмента отдельно
        """
        import pandas as pd

        dl = DataLoader()
        portfolio_data: dict[str, pd.DataFrame] = {}

        # --- 0. Подготовка мапы лидеров ---
        if isinstance(leader_symbol, dict):
            leader_map: dict[str, str] = leader_symbol
            print(f"👑 [DATA] Лидеры по классам: {leader_map}")
            unique_leaders = sorted(set(leader_map.values()))
            print(f"👑 [DATA] Лидеры рынков: {unique_leaders} "
                f"(tickers={len(leader_map)})")
        else:
            # Старый режим: один лидер на всех
            leader_map = {sym: leader_symbol for sym in assets}
            print(f"👑 [DATA] Лидер для всего портфеля: {leader_symbol}")

        # Кэшируем загруженные ряды лидеров, чтобы не грузить один и тот же тикер по 10 раз
        leader_cache: dict[str, pd.DataFrame] = {}

        # Новости общие для всех
        df_news = dl.load_news_sentiment()

        for symbol in assets:
            # 1. Младший ТФ
            df = dl._fetch_and_cache(symbol, start_date, end_date, interval_ltf)
            if df.empty:
                print(f"❌ Не удалось загрузить {symbol}. Пропуск.")
                continue

            # 2. Старший ТФ
            df_htf = dl._fetch_and_cache(symbol, start_date, end_date, interval_htf)

            # 3. MTF merge + HTF-фичи
            df = dl.merge_mtf(df, df_htf)

            # 4. Лидер для этого инструмента
            sym_leader = leader_map.get(symbol)
            if sym_leader:
                if sym_leader not in leader_cache:
                    print(f"👑 [DATA] Загрузка лидера {sym_leader} (для {symbol} и других).")
                    leader_cache[sym_leader] = dl._fetch_and_cache(
                        sym_leader, start_date, end_date, interval_ltf
                    )

                df_leader = leader_cache.get(sym_leader, None)
            else:
                df_leader = None

            if df_leader is not None and not df_leader.empty:
                leader_cls = df_leader[["close"]].rename(columns={"close": "leader_close"})
                df = df.join(leader_cls).ffill()

                # Если инструмент сам является своим лидером
                if symbol == sym_leader:
                    df["leader_close"] = df["close"]
                else:
                    df["leader_close"] = df["leader_close"].fillna(df["close"])
            else:
                # Фолбэк – просто дублируем close
                df["leader_close"] = df["close"]

            # 5. Новости
            if df_news is not None and not df_news.empty:
                df = df.join(df_news).fillna(0)
            else:
                df["sentiment"] = 0.0
                df["sentiment_ema"] = 0.0

            portfolio_data[symbol] = df

        if portfolio_data:
            lengths = {sym: len(df) for sym, df in portfolio_data.items()}
            min_len = min(lengths.values()) if lengths else 0
            print(
                f"✅ Данные готовы. Активов: {len(portfolio_data)}, "
                f"минимум баров на актив: {min_len}"
            )

        return portfolio_data
