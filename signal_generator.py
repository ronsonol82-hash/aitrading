# signal_generator.py
import pandas as pd
import numpy as np
import pickle
import os
import sys
import asyncio
from tqdm import tqdm
from datetime import datetime, timedelta
from config import Config, UniverseMode
from data_loader import DataLoader
from indicators import FeatureEngineer
from model_engine import MLEngine
from risk_utils import calc_position_size
import argparse
from joblib import Parallel, delayed
import multiprocessing

# --- HELPER FOR PARALLEL TRAINING ---
def train_wrapper(sym, model_obj, data_slice, features):
    # This runs in a separate process
    try:
        model_obj.train(data_slice, features)
        return sym, model_obj
    except Exception as e:
        print(f"❌ Error training {sym}: {e}")
        return sym, model_obj # Return old model if failed

class UniversalSignalFactory:
    """
    Фабрика Сигналов v3.0 (Universal Brain Edition).

    Обучает ЕДИНУЮ модель на общем паттерне:
      - используем всех учителей из портфеля,
      - применяем её ко всему списку Config.ASSETS.
    """
    OUTPUT_FILE = "data_cache/production_signals_v1.pkl"

    def __init__(
        self,
        regime_preset: str = "classic",
        cross_asset_wf: bool = False,
        train_window: int | None = None,
        trade_window: int | None = None,
        ):
        self.preset = regime_preset
        self.cross_asset_wf = cross_asset_wf
        self.train_window = train_window
        self.trade_window = trade_window

        self.data: dict[str, pd.DataFrame] = {}

        # Универсальный набор фич без абсолютных цен
        # (align с Config.UNIVERSAL_FEATURE_COLS)
        self.feature_cols = Config.UNIVERSAL_FEATURE_COLS

        # 🧠 Учителя завязываем на UNIVERSE_MODE + cross_asset_wf
        mode = Config.UNIVERSE_MODE

        if self.cross_asset_wf:
            # Кросс-активный WF внутри выбранной вселенной
            if mode == UniverseMode.CRYPTO:
                # только криптовалюты
                self.teachers = Config.crypto_symbols()
            elif mode == UniverseMode.STOCKS:
                # только акции/валюты биржи
                self.teachers = Config.equity_symbols()
            else:
                # BOTH: берём все классы
                self.teachers = Config.crypto_symbols() + Config.equity_symbols()
        else:
            # Без cross-asset: учителя = текущий торговый портфель
            # (то, что выбрано через UNIVERSE_MODE / GUI)
            self.teachers = ["MOEX"]

        # Если захочешь сузить обучение, можно вручную задать подмножество:
        # self.teachers = ["MOEX"] //// self.teachers = Config.ASSETS
            
    # -------- ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ --------
    def load_data(self) -> None:
        print("🏗 [UNIVERSAL FACTORY] Загрузка рыночных данных.")
        end = datetime.now()
        # Глубокая история для обучения (≈ 6–7 лет)
        start = end - timedelta(days=2500)

        mode = Config.UNIVERSE_MODE
        if mode == UniverseMode.CRYPTO:
            # Только крипта
            all_assets = Config.crypto_symbols()
        elif mode == UniverseMode.STOCKS:
            # Только биржа (MOEX / Тинькофф)
            all_assets = Config.equity_symbols()
        else:
            # BOTH — объединяем два списка
            all_assets = Config.crypto_symbols() + Config.equity_symbols()

        # На всякий случай уберём дубликаты
        all_assets = list(sorted(set(all_assets)))
        print(f"   📥 Запрос данных для ({mode.value}): {all_assets}")

        leader_map = {sym: Config.get_leader_for_symbol(sym) for sym in all_assets}

        self.data = DataLoader.get_portfolio_data(
            all_assets,
            leader_map,
            start,
            end,
            Config.TIMEFRAME_LTF,
            Config.TIMEFRAME_HTF,
        )

        if not self.data:
            print("❌ [UNIVERSAL] Ошибка: DataLoader вернул пустой словарь.")
            sys.exit(1)

        print("🛠 Расчёт фичей (price-agnostic структура + психология уровней)...")
        for sym, df in tqdm(self.data.items(), desc="Feature Engineering"):
            try:
                # 1) Фичи (уровни, канал, конfluence score и т.д.)
                df_feat = FeatureEngineer.add_features(df)

                # 2) Разметка triple-barrier (target)
                df_labeled = FeatureEngineer.label_data(
                    df_feat, Config.LOOK_AHEAD, Config.RR_RATIO
                )

                # Drop NaN из-за роллингов/ATR
                self.data[sym] = df_labeled.dropna()
            except Exception as e:
                print(f"⚠️ [UNIVERSAL] Ошибка обработки {sym}: {e}")

        print(f"✅ [UNIVERSAL] Данные готовы. Активов в памяти: {len(self.data)}")

    # -------- ОБУЧЕНИЕ ГЛОБАЛЬНОГО МОЗГА --------
    def run_universal_training(self) -> None:
        if not self.data:
            print("❌ [UNIVERSAL] Нет данных. Сначала запусти load_data().")
            return

        print("\n🎓 [UNIVERSAL] Запуск обучения Глобального Мозга...")
        print(f"   👨‍🏫 Учителя: {self.teachers}")
        if self.cross_asset_wf:
            print("   🌉 Режим: CROSS-ASSET WALK-FORWARD включен.")
            if self.train_window or self.trade_window:
                print(
                    f"   📐 Окна: train_window={self.train_window}, "
                    f"trade_window={self.trade_window}"
                )
        teacher_frames: list[pd.DataFrame] = []
        min_len = float("inf")

        # 1) Собираем учителей и находим минимальную длину истории
        for sym in self.teachers:
            if sym not in self.data:
                print(f"   ⚠️ Учитель {sym} не загрузился — пропускаем.")
                continue

            df = self.data[sym]
            if len(df) < 100:
                print(f"   ⚠️ {sym}: слишком мало баров ({len(df)}), пропуск.")
                continue

            teacher_frames.append(df)
            if len(df) < min_len:
                min_len = len(df)

            print(f"   ✔️ {sym}: {len(df)} баров после очистки.")

        if not teacher_frames:
            print("❌ [UNIVERSAL] Нет валидных учителей для обучения.")
            return

        lengths = [len(x) for x in teacher_frames]
        print(f"   📏 Общий размер учительских выборок: {lengths}")
        
        if self.cross_asset_wf:
            total_bars = min_len

            # Определяем длину train / OOS
            if self.train_window is not None and self.trade_window is not None:
                train_len = min(self.train_window, total_bars - 50)
                oos_len = min(self.trade_window, total_bars - train_len)
            elif self.train_window is not None:
                train_len = min(self.train_window, total_bars - 50)
                oos_len = total_bars - train_len
            elif self.trade_window is not None:
                oos_len = min(self.trade_window, max(50, total_bars // 3))
                train_len = total_bars - oos_len
            else:
                train_len = int(total_bars * 0.7)
                oos_len = total_bars - train_len

            if train_len <= 0:
                print("❌ [UNIVERSAL-CA] Некорректные окна train/test. "
                      "Уменьши trade_window или увеличь train_window.")
                return

            print(
                f"   🧪 [CA-WF] Используем первые {train_len} баров учителей для обучения "
                f"и откладываем ~{oos_len} баров как будущее (OOS)."
            )
            # ВАЖНО: берём ИМЕННО начало истории (прошлое), а не конец
            balanced_frames = [df.iloc[:train_len].copy() for df in teacher_frames]
        else:
            print(f"   🔎 Балансируем по последним {min_len} барам (equal history).")
            balanced_frames = [df.iloc[-min_len:].copy() for df in teacher_frames]

        df_train_full = (
            pd.concat(balanced_frames, axis=0)
            .sample(frac=1.0, random_state=42)
        )

        print(f"📊 [UNIVERSAL] Обучающая выборка: {len(df_train_full)} строк.")
        print(f"   🧬 Фичи ({len(self.feature_cols)}): {self.feature_cols}")

        # 3.0) Оставляем только реально существующие фичи
        cols = [c for c in self.feature_cols if c in df_train_full.columns]
        missing = [c for c in self.feature_cols if c not in df_train_full.columns]

        if missing:
            print(f"   ⚠️ [UNIVERSAL] Пропущены колонки (нет в df): {missing}")

        if len(cols) < 3:
            print("   ❌ [UNIVERSAL] Слишком мало фич после фильтрации, обучение отменено.")
            return

        self.feature_cols = cols
        print(f"   ✅ [UNIVERSAL] Будем использовать {len(self.feature_cols)} фич: {self.feature_cols}")

        # 3.1) Защита от случайного попадания цен в фичи
        forbidden = {"close", "open", "high", "low"}
        for col in self.feature_cols:
            if col in forbidden:
                print(
                    f"🚨 КРИТИЧЕСКАЯ ОШИБКА: В FEATURE_COLS есть ценовой столбец '{col}'.\n"
                    f"    Убери его из Config.FEATURE_COLS для универсальной модели."
                )
                return

        # 4) Обучение MLEngine на МЕГА-датасете
        mode = Config.UNIVERSE_MODE
        model_name = f"UNIVERSAL_BRAIN_{mode.value}"
        model_path = os.path.join(Config.MODEL_DIR, model_name)
        os.makedirs(model_path, exist_ok=True)

        engine = MLEngine(model_path, regime_preset=self.preset)
        engine.train(df_train_full, self.feature_cols)

        print(f"✅ [UNIVERSAL] Глобальная модель для {mode.value} обучена и сохранена в {model_name}.")

        # -------- ИНФЕРЕНС НА ВСЁМ ПОРТФЕЛЕ --------
        print("\n🔮 [UNIVERSAL] Экзамен: генерируем сигналы по всему портфелю...")

        production_data: dict[str, pd.DataFrame] = {}

        for sym, df in self.data.items():
            # Предсказание вероятностей и режима
            probs, regimes = engine.predict_batch(df, self.feature_cols)
            if probs is None:
                print(f"   ⚠️ {sym}: predict_batch вернул None, пропуск.")
                continue

            # Формируем результат в том же формате, что и walk-forward
            df_res = pd.DataFrame(index=df.index)
            df_res["p_long"] = probs[:, 1]
            df_res["p_short"] = probs[:, 2]
            df_res["regime"] = regimes

            # Копируем цены (нужны бэктестеру)
            for col in ["open", "high", "low", "close", "atr"]:
                if col in df.columns:
                    df_res[col] = df[col]

            production_data[sym] = df_res
            # Можно включить лог:
            # print(f"   ✅ {sym}: сигналы сгенерированы ({len(df_res)} строк).")

        # 5) Мерджим с существующими сигналами и сохраняем
        merged_data = self._merge_with_existing_signals(production_data)

        with open(self.OUTPUT_FILE, "wb") as f:
            pickle.dump(merged_data, f)

        mode = Config.UNIVERSE_MODE
        print(f"💾 [UNIVERSAL] Сигналы для {mode.value} сохранены/обновлены в {self.OUTPUT_FILE}")
        print("➡️  Дальше можно запускать debug_replayer.py / backtester.py")

    def _merge_with_existing_signals(self, new_signals: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        """
        Обновляет production_signals_v1.pkl только по текущему юниверсу.
        Остальные активы (другая биржа / другие режимы) оставляем как есть.
        """
        result: dict[str, pd.DataFrame] = {}

        # 1) Пытаемся прочитать старый файл сигналов
        try:
            if os.path.exists(self.OUTPUT_FILE):
                with open(self.OUTPUT_FILE, "rb") as f:
                    old = pickle.load(f)
                if isinstance(old, dict):
                    result.update(old)
        except Exception as e:
            print(f"⚠️ [UNIVERSAL] Не удалось прочитать старый файл сигналов: {e}")

        # 2) Обновляем / добавляем новые сигналы текущего юниверса
        for sym, df in new_signals.items():
            result[sym] = df

        return result

class SignalFactory:
    """
    Цех по производству вероятностей (v2.0 - Incremental Edition).
    Умный Walk-Forward: дописывает историю, а не пересчитывает мир с сотворения.
    """
    OUTPUT_FILE = "data_cache/production_signals_v1.pkl"

    def __init__(
        self,
        regime_preset: str = 'classic',
        force_reset: bool = False,
        train_window: int | None = None,
        trade_window: int | None = None,
    ):
        self.data = {}
        self.models = {}
        self.feature_cols = Config.FEATURE_COLS
        self.preset = regime_preset
        self.force_reset = force_reset
        # NEW: окна из GUI
        self.train_window = train_window
        self.trade_window = trade_window

    def load_data(self):
        mode = Config.UNIVERSE_MODE
        print(f"🏗 [FACTORY] Загрузка свежих рыночных данных для юниверса: {mode.value}.")
        end = datetime.now()
        # Грузим историю с запасом, чтобы индикаторы не поломались
        start = end - timedelta(days=2500) 
        
        leader_map = {sym: Config.get_leader_for_symbol(sym) for sym in Config.ASSETS}

        self.data = DataLoader.get_portfolio_data(
            Config.ASSETS,
            leader_map,
            start,
            end,
            Config.TIMEFRAME_LTF,
            Config.TIMEFRAME_HTF,
        )
        
        # Feature Engineering
        for sym, df in self.data.items():
            # Оптимизация вывода в консоль
            # print(f"   🛠 Features: {sym}") 
            df = FeatureEngineer.add_features(df)
            
            # Labeling создает 'target'. Нужно для обучения.
            # Мы используем Embargo в run_walk_forward, так что тут просто разметка.
            df = FeatureEngineer.label_data(df, Config.LOOK_AHEAD, Config.RR_RATIO)
            self.data[sym] = df
            
            # Инициализируем движки (пустые, они будут обучены внутри цикла)
            self.models[sym] = MLEngine(f"{Config.MODEL_DIR}/{sym}", regime_preset=self.preset)
        
        print(f"✅ Данные в памяти. Инструментов: {len(self.data)}")

    def _load_existing_signals(self, master_index):
        """
        Пытается загрузить старый кэш и синхронизировать его с новыми данными.
        Возвращает:
           - production_data (структуру, заполненную старыми данными)
           - start_idx (откуда начинать считать новые)
        """
        if self.force_reset or not os.path.exists(self.OUTPUT_FILE):
            return None, None

        try:
            with open(self.OUTPUT_FILE, "rb") as f:
                old_data = pickle.load(f)

            if not old_data:
                return None, None

            # Проверка консистентности: есть ли все активы?
            first_sym = list(self.data.keys())[0]
            if first_sym not in old_data:
                print("⚠️ [CACHE] Структура активов изменилась. Полный пересчет.")
                return None, None

            # Находим последнюю дату в кэше
            last_cache_date = old_data[first_sym].index[-1]
            
            # Проверяем, есть ли эта дата в новых загруженных данных
            if last_cache_date not in master_index:
                print(f"⚠️ [CACHE] Кэш ({last_cache_date}) не стыкуется с новыми данными. Полный пересчет.")
                return None, None

            # Ищем integer index этой даты в новом массиве
            # get_loc может вернуть slice или int, берем аккуратно
            idx_loc = master_index.get_loc(last_cache_date)
            if isinstance(idx_loc, slice):
                resume_idx = idx_loc.stop
            else:
                resume_idx = idx_loc + 1

            if resume_idx >= len(master_index):
                print("✅ [CACHE] Новых данных нет. Сигналы актуальны.")
                return old_data, len(master_index) # Stop immediately

            print(f"♻️ [CACHE] Найден кэш. Возобновляем расчет с {master_index[resume_idx]} (skip {resume_idx} bars).")
            
            # Подготовка структуры: берем старое, расширяем новым пустым местом
            production_data = {}
            for sym in self.data:
                # Создаем полный фрейм по размеру НОВЫХ данных
                full_index = self.data[sym].index
                df_res = pd.DataFrame(index=full_index)
                df_res['p_long'] = 0.0
                df_res['p_short'] = 0.0
                df_res['regime'] = 0
                
                # Копируем цены (они всегда свежие из load_data)
                src = self.data[sym]
                for col in ['open', 'high', 'low', 'close', 'atr']:
                    if col in src.columns:
                        df_res[col] = src[col]

                # Вставляем СТАРЫЕ предсказания в начало
                if sym in old_data:
                    old_df = old_data[sym]
                    # Используем update или прямое присваивание по индексу
                    # Важно: берем только intersection индексов на случай дыр
                    common_idx = old_df.index.intersection(full_index)
                    
                    df_res.loc[common_idx, 'p_long'] = old_df.loc[common_idx, 'p_long']
                    df_res.loc[common_idx, 'p_short'] = old_df.loc[common_idx, 'p_short']
                    df_res.loc[common_idx, 'regime'] = old_df.loc[common_idx, 'regime']
                
                production_data[sym] = df_res
            
            return production_data, resume_idx

        except Exception as e:
            print(f"❌ [CACHE ERROR] {e}. Начинаем с нуля.")
            return None, None

    def run_walk_forward(self):
        """
        Инкрементальная генерация.
        """
        if not self.data:
            print("❌ Нет данных. Сначала запустите load_data().")
            return

        master_sym = list(self.data.keys())[0]
        master_index = self.data[master_sym].index
        total_steps = len(master_index)
        
        # Настройки окна
        window_size = self.train_window or Config.WALK_FORWARD_WINDOW  # Окно обучения
        if self.trade_window is not None and self.trade_window > 0:
            step_size = int(self.trade_window)
            if step_size > window_size:
                step_size = window_size
        else:
            step_size = int(window_size * 0.25)  # дефолтное поведение
        
        step_size = max(1, step_size)
        
        # --- 1. ПОПЫТКА ЗАГРУЗКИ КЭША ---
        production_data, start_idx = self._load_existing_signals(master_index)
        
        if production_data is None:
            # Если кэша нет, инициализируем с нуля
            start_idx = window_size + Config.LOOK_AHEAD
            production_data = {}
            for sym in self.data:
                idx = self.data[sym].index
                df_res = pd.DataFrame(index=idx)
                df_res['p_long'] = 0.0; df_res['p_short'] = 0.0; df_res['regime'] = 0
                
                src = self.data[sym]
                for col in ['open', 'high', 'low', 'close', 'atr']:
                    if col in src.columns: df_res[col] = src[col]
                production_data[sym] = df_res

        # Если уже все посчитано
        if start_idx >= total_steps:
            return

        print(f"🚀 [FACTORY] Старт генерации: {total_steps - start_idx} новых баров...")
        
        current_idx = start_idx
        pbar = tqdm(total=total_steps - start_idx)
        
        # --- 2. ЦИКЛ (Только по новым данным) ---
        while current_idx < total_steps:
            # A. TRAIN PHASE
            train_start = max(0, current_idx - window_size - Config.LOOK_AHEAD)
            train_end = current_idx - Config.LOOK_AHEAD 
            
            if train_end > train_start + 500:
                # Parallel Training
                n_cores = max(1, multiprocessing.cpu_count() - 1)
                
                # We pass the EXTERNAL function train_wrapper here
                results = Parallel(n_jobs=n_cores, backend="loky")(
                    delayed(train_wrapper)(
                        sym, 
                        self.models[sym], 
                        self.data[sym].iloc[train_start:train_end], 
                        self.feature_cols
                    ) for sym in self.data
                )

                # Collect results back to main process
                for sym, trained_model in results:
                    self.models[sym] = trained_model
            
            # B. PREDICT PHASE
            test_end = min(total_steps, current_idx + step_size)
            
            for sym in self.data:
                df_full = self.data[sym]
                df_test_chunk = df_full.iloc[current_idx:test_end]
                
                if df_test_chunk.empty: continue
                
                # Векторизированный предикт
                probs, regimes = self.models[sym].predict_batch(df_test_chunk, self.feature_cols)
                
                if probs is not None:
                    # Записываем в production_data
                    # Важно: используем .iloc для надежности, так как индексы совпадают
                    # Но production_data[sym] это полный фрейм.
                    # Проще через .loc по индексу чанка
                    target_idx = df_test_chunk.index
                    
                    production_data[sym].loc[target_idx, 'p_long'] = probs[:, 1]
                    production_data[sym].loc[target_idx, 'p_short'] = probs[:, 2]
                    production_data[sym].loc[target_idx, 'regime'] = regimes

            processed_count = test_end - current_idx
            pbar.update(processed_count)
            current_idx = test_end
            
            # C. INTERMEDIATE SAVE (Каждые 5 шагов цикла или в конце)
            # Чтобы не потерять прогресс при краше
            # (Для простоты сохраняем в конце, но можно раскомментировать для параноидального режима)
            # if current_idx % (step_size * 5) == 0:
            #     self._save_to_disk(production_data)

        pbar.close()
        
        # --- 3. ФИНАЛЬНОЕ СОХРАНЕНИЕ ---
        self._save_to_disk(production_data)

    def _save_to_disk(self, production_data):
        # Обрезка "холодного старта" (где нули в начале самом-самом)
        # Если мы дописывали кэш, то начало уже нормальное.
        # Найдем первую дату, где p_long != 0
        
        final_output = {}
        # Эвристика: ищем первую запись с ненулевой вероятностью у лидера
        master_sym = list(production_data.keys())[0]
        df_master = production_data[master_sym]
        
        # Берем индекс, где впервые появились данные (или начало кэша, или начало расчета)
        # Но чтобы не усложнять, сохраняем всё, кроме откровенной пустоты в начале истории
        
        valid_idx = df_master[(df_master['p_long'] != 0) | (df_master['p_short'] != 0)].index
        if not valid_idx.empty:
            start_date = valid_idx[0]
        else:
            start_date = df_master.index[0]

        print(f"💾 Сохранение сигналов (c {start_date})...")
        
        for sym, df in production_data.items():
            final_output[sym] = df.loc[start_date:]
            
        with open(self.OUTPUT_FILE, "wb") as f:
            pickle.dump(final_output, f)
            
        print(f"✅ Успешно сохранено: {self.OUTPUT_FILE}")

    def _merge_with_existing_signals(self, new_signals: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        """
        Обновляет production_signals_v1.pkl только по текущему юниверсу.
        Остальные активы (другая биржа) оставляем как есть.
        """
        result: dict[str, pd.DataFrame] = {}

        # 1) Пытаемся прочитать старый файл
        try:
            if os.path.exists(self.OUTPUT_FILE):
                with open(self.OUTPUT_FILE, "rb") as f:
                    old = pickle.load(f)
                if isinstance(old, dict):
                    result.update(old)
        except Exception as e:
            print(f"⚠️ [UNIVERSAL] Не удалось прочитать старый файл сигналов: {e}")

        # 2) Обновляем / добавляем новые сигналы текущего юниверса
        for sym, df in new_signals.items():
            result[sym] = df

        return result

# =================== АСИНХРОННЫЙ ИНТЕРФЕЙС ДЛЯ НОВОГО КОДА ===================

async def async_main(args):
    """
    Асинхронная точка входа для новой системы.
    Поддерживает параметры из GUI / CLI.
    args — уже распарсенный argparse.Namespace
    """
    from config import Config
    from execution_router import ExecutionRouter

    print(f"🚀 [ASYNC] Запуск в асинхронном режиме")
    print(f"   📊 Брокер: {args.broker}")
    print(f"   🎮 Режим: {args.mode}")
    print(f"   ⚙️  Пресет: {args.preset}")
    print(f"   🧠 Cross-asset WF: {'Включен' if args.cross_asset_wf else 'Выключен'}")

    # Инициализируем роутер исполнения
    router = ExecutionRouter()
    await router.initialize(broker_name=args.broker)
    
    try:
        # Режим UNIVERSAL (Глобальный мозг)
        if args.mode == "universal":
            print("\n🧠 [ASYNC] Запуск UniversalSignalFactory...")
            
            u_factory = UniversalSignalFactory(
                regime_preset=args.preset,
                cross_asset_wf=args.cross_asset_wf,
                train_window=args.train_window,
                trade_window=args.trade_window,
            )
            
            # 1. Загрузка данных
            print("📥 Загрузка данных...")
            u_factory.load_data()
            
            # 2. Обучение (если не только инференс)
            if not args.inference_only:
                print("🎓 Обучение глобальной модели...")
                u_factory.run_universal_training()
            
            # 3. Инференс на портфеле (если не только обучение)
            if not args.universal_only:
                print("🔮 Генерация сигналов на портфеле...")

                try:
                    with open(u_factory.OUTPUT_FILE, "rb") as f:
                        signals = pickle.load(f)
                    
                    # Анализ сигналов и выбор портфеля
                    portfolio_decisions = analyze_portfolio_signals(
                        signals,
                        portfolio_size=args.portfolio_size,
                        risk_level=args.risk_level,
                    )

                    if portfolio_decisions:
                        print(f"\n📊 Топ-{len(portfolio_decisions)} сигналов выбраны для портфеля.")
                        await router.execute_portfolio_decisions(portfolio_decisions)
                        print(f"✅ Поручения отправлены: {len(portfolio_decisions)} позиций")
                    else:
                        print("ℹ️ Подходящих сигналов для портфеля не найдено.")
                
                except Exception as e:
                    print(f"⚠️ Ошибка при анализе сигналов: {e}")
            
            print("✅ [ASYNC] Универсальная модель: все операции завершены")
            
        else:
            # WALK-FORWARD режим
            print("\n🚶 [ASYNC] Запуск SignalFactory (walk-forward)...")
            
            factory = SignalFactory(
                regime_preset=args.preset,
                force_reset=args.reset,
                train_window=args.train_window,
                trade_window=args.trade_window,
            )
            
            print("📥 Загрузка данных...")
            factory.load_data()
            
            print("🔄 Запуск walk-forward...")
            factory.run_walk_forward()
            
            print("📈 Анализ свежих сигналов и подготовка ордеров...")
            try:
                with open(factory.OUTPUT_FILE, "rb") as f:
                    signals = pickle.load(f)
                
                latest_signals = get_latest_signals(signals)
                orders = prepare_orders_from_signals(
                    latest_signals,
                    risk_level=args.risk_level,
                )
                
                if orders:
                    await router.execute_batch_orders(orders)
                    print(f"✅ Ордера отправлены: {len(orders)}")
                else:
                    print("ℹ️ Нет ордеров, удовлетворяющих фильтрам по силе сигнала.")
            
            except Exception as e:
                print(f"⚠️ Ошибка при подготовке ордеров: {e}")
            
            print("✅ [ASYNC] Walk-forward завершен")
            
    except Exception as e:
        print(f"❌ [ASYNC] Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        await router.close()
        print("🔒 [ASYNC] Ресурсы освобождены")


# =================== КОНЕЦ АСИНХРОННОГО КОДА ===================

# ====== ХЕЛПЕРЫ ДЛЯ АНАЛИЗА СИГНАЛОВ И ПОДГОТОВКИ ОРДЕРОВ ======

def analyze_portfolio_signals(
    signals: dict,
    portfolio_size: int = 10,
    risk_level: float = 0.02,
):
    """
    Анализирует универсальные сигналы и выбирает лучшие позиции для портфеля.
    (полностью синхронная логика)
    """
    decisions = []
    
    for symbol, df in signals.items():
        if df.empty:
            continue
            
        last_row = df.iloc[-1]
        signal_strength = calculate_signal_strength(last_row)
        direction = "BUY" if last_row['p_long'] > last_row['p_short'] else "SELL"
        
        position_size = calculate_position_size(
            symbol,
            last_row.get('atr', 0.01),
            risk_level,
        )
        
        decisions.append({
            'symbol': symbol,
            'direction': direction,
            'strength': signal_strength,
            'size': position_size,
            'timestamp': df.index[-1],
            'p_long': last_row['p_long'],
            'p_short': last_row['p_short'],
            'regime': last_row['regime'],
        })
    
    decisions.sort(key=lambda x: x['strength'], reverse=True)
    return decisions[:portfolio_size]


def calculate_signal_strength(row):
    """Вычисляет силу сигнала на основе вероятностей и режима."""
    base_strength = abs(row['p_long'] - row['p_short'])
    
    regime_modifier = {
        0: 0.5,  # нейтральный
        1: 1.0,  # трендовый
        2: 0.8,  # волатильный
        3: 0.6,  # флэтовый
    }.get(row['regime'], 0.5)
    
    return base_strength * regime_modifier


def calculate_position_size(symbol, atr, risk_level):
    """
    СИНХРОННЫЙ расчёт размера позиции на основе ATR и уровня риска.
    Использует тот же движок, что и боевой риск-менеджмент (calc_position_size).
    """
    from data_loader import DataLoader  # локальный импорт, чтобы избежать циклов

    equity = getattr(Config, "DEPOSIT", 1000.0)
    sl_mult = Config.DEFAULT_STRATEGY.get("sl", 2.0)
    max_notional = getattr(Config, "MAX_POSITION_NOTIONAL", None)

    try:
        end = datetime.now()
        start = end - timedelta(days=1)
        data = DataLoader.get_symbol_data(symbol, start, end, "1h")

        if not data.empty:
            current_price = float(data["close"].iloc[-1])

            # Если ATR в сигналах не задан или некорректен — пересчитаем грубо.
            if atr is None or atr <= 0:
                atr_value = (
                    data["close"]
                    .diff()
                    .abs()
                    .rolling(14)
                    .mean()
                    .iloc[-1]
                )
            else:
                atr_value = float(atr)

            ps = calc_position_size(
                equity=equity,
                risk_per_trade=risk_level,
                atr=atr_value,
                sl_mult=sl_mult,
                price=current_price,
                max_notional=max_notional,
            )

            if ps.size > 0:
                return ps.size

    except Exception:
        # В случае любой ошибки просто возвращаем 0 — не лезем в сделку
        return 0.0

    return 0.0

def prepare_orders_from_signals(latest_signals: dict, risk_level: float):
    """Подготавливает ордера на основе последних сигналов (синхронно)."""
    orders = []
    
    for symbol, row in latest_signals.items():
        # фильтр по силе сигнала
        if row['p_long'] < 0.6 and row['p_short'] < 0.6:
            continue
            
        direction = "BUY" if row['p_long'] > row['p_short'] else "SELL"
        
        size = calculate_position_size(
            symbol,
            row.get('atr', 0.01),
            risk_level,
        )
        
        orders.append({
            'symbol': symbol,
            'side': direction,
            'quantity': size,
            'order_type': 'MARKET',
            'signal_strength': max(row['p_long'], row['p_short']),
        })
    
    return orders


def get_latest_signals(signals: dict):
    """Извлекает последние сигналы из словаря."""
    latest = {}
    for symbol, df in signals.items():
        if not df.empty:
            latest[symbol] = df.iloc[-1]
    return latest


if __name__ == "__main__":
    # ЕДИНЫЙ парсер аргументов для sync+async режимов
    parser = argparse.ArgumentParser(description="Signal Factory & ML Trainer")

    # Общие параметры
    parser.add_argument(
        "--preset",
        type=str,
        default="classic",
        choices=["classic", "grinder", "sniper", "loose"],
        help="Market regime preset",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Force delete cache and recalculate everything (walk mode only)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="walk",
        choices=["walk", "universal"],
        help="Signal generation mode: walk (per-symbol WF) or universal (global brain)",
    )
    parser.add_argument(
        "--train_window",
        type=int,
        default=None,
        help="Train window in candles (WF & Universal CA-WF)",
    )
    parser.add_argument(
        "--trade_window",
        type=int,
        default=None,
        help="Trade/OOS window in candles",
    )
    parser.add_argument(
        "--cross_asset_wf",
        action="store_true",
        help="Enable Cross-Asset Walk-Forward for UNIVERSAL mode",
    )

    # Флаг асинхронного режима
    parser.add_argument(
        "--async_mode",
        action="store_true",
        help="Run in async mode with GUI support",
    )

    # Дополнительные параметры только для async-ветки
    parser.add_argument(
        "--broker",
        type=str,
        default="bitget",
        help="Broker name (async mode only)",
    )
    parser.add_argument(
        "--portfolio_size",
        type=int,
        default=10,
        help="Number of assets in portfolio (async mode only)",
    )
    parser.add_argument(
        "--risk_level",
        type=float,
        default=0.02,
        help="Risk per trade (async mode only)",
    )
    parser.add_argument(
        "--universal_only",
        action="store_true",
        help="Run only universal training (async mode only)",
    )
    parser.add_argument(
        "--inference_only",
        action="store_true",
        help="Run only inference (async mode only)",
    )

    args = parser.parse_args()
    
    if args.async_mode:
        print("🚀 Запуск в асинхронном режиме...")
        asyncio.run(async_main(args))
    else:
        # Существующий синхронный код
        if args.mode == "walk":
            print(f"\n🏭 [FACTORY] Запуск WALK-FORWARD. Preset: {args.preset.upper()}")
            if args.reset:
                print("⚠️ FORCE RESET: кэш сигналов будет проигнорирован.")

            factory = SignalFactory(
                regime_preset=args.preset,
                force_reset=args.reset,
                train_window=args.train_window,
                trade_window=args.trade_window,
            )
            factory.load_data()
            factory.run_walk_forward()
            print("\n➡️ Следующий шаг: python optimizer.py --mode sniper")

        else:
            print(f"\n🧠 [UNIVERSAL] Запуск универсального мозга. Preset: {args.preset.upper()}")
            u_factory = UniversalSignalFactory(
                regime_preset=args.preset,
                cross_asset_wf=args.cross_asset_wf,
                train_window=args.train_window,
                trade_window=args.trade_window,
            )
            u_factory.load_data()
            u_factory.run_universal_training()