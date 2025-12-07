# signal_generator.py
import pandas as pd
import numpy as np
import pickle
import os
import sys
from tqdm import tqdm
from datetime import datetime, timedelta
from config import Config, UniverseMode
from data_loader import DataLoader
from indicators import FeatureEngineer
from model_engine import MLEngine
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Signal Factory & ML Trainer")
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
    # NEW: окна WF из GUI
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
    # NEW: флаг включения Cross-Asset WF
    parser.add_argument(
        "--cross_asset_wf",
        action="store_true",
        help="Enable Cross-Asset Walk-Forward for UNIVERSAL mode",
    )

    args = parser.parse_args()

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
        # Дальше можно сразу гонять backtester по production_signals_v1.pkl