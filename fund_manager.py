# fund_manager.py
import sys
import os
import json
import asyncio
import numpy as np
import pandas as pd
import pyqtgraph as pg
import threading
from datetime import datetime, timedelta

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
    QWidget, QLabel, QComboBox, QGroupBox, QTabWidget, QPushButton,
    QTextEdit, QSplitter, QDoubleSpinBox, QGridLayout, QFrame, QSlider,
    QTableWidget, QHeaderView, QTableWidgetItem, QRadioButton, QCheckBox,
    QTableView,                    # <-- добавили
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QObject, QTimer
from PyQt5.QtGui import (
    QPainter, QPicture, QColor, QFont,
    QStandardItemModel, QStandardItem   # <-- добавили
)
from async_strategy_runner import AsyncStrategyRunner

# --- ИМПОРТЫ ЛОГИКИ ПРОЕКТА ---
from config import Config, UniverseMode, get_assets_for_universe
from data_loader import DataLoader
from indicators import FeatureEngineer
from execution_router import ExecutionRouter

# ==========================================
# 🎨 GLOBAL STYLESHEET (PROFESSIONAL DARK)
# ==========================================
STYLESHEET = """
QMainWindow { background-color: #1e1e1e; color: #d4d4d4; }
QWidget { font-family: 'Segoe UI', sans-serif; font-size: 10pt; color: #d4d4d4; }
QTabWidget::pane { border: 1px solid #333333; background-color: #252526; }
QTabWidget::tab-bar { left: 5px; }
QTabBar::tab { background: #2d2d2d; color: #888888; padding: 8px 20px; margin-right: 2px; min-width: 100px; }
QTabBar::tab:selected { background: #3e3e3e; color: #ffffff; border-bottom: 2px solid #007acc; font-weight: bold; }
QGroupBox { border: 1px solid #3e3e3e; border-radius: 4px; margin-top: 20px; background-color: #252526; font-weight: bold; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; background-color: #252526; }
QPushButton { background-color: #3c3c3c; border: 1px solid #555555; color: #ffffff; padding: 6px 12px; border-radius: 3px; }
QPushButton:hover { background-color: #4a4a4a; border-color: #007acc; }
QPushButton#ActionBtn { background-color: #0e639c; border: 1px solid #1177bb; font-weight: bold; }
QPushButton#ActionBtn:hover { background-color: #1177bb; }
QTableWidget { gridline-color: #444; background-color: #1e1e1e; selection-background-color: #007acc; }
QHeaderView::section { background-color: #2d2d2d; padding: 4px; border: 1px solid #444; }
QTextEdit { background-color: #1e1e1e; color: #d4d4d4; border: 1px solid #333333; font-family: 'Consolas', monospace; }
"""

pg.setConfigOption('background', '#1e1e1e')
pg.setConfigOption('foreground', '#888888')
pg.setConfigOptions(antialias=True)


# ==========================================
# 1. CHART COMPONENTS
# ==========================================
class DateAxis(pg.AxisItem):
    def __init__(self, dates, orientation='bottom', **kwargs):
        super().__init__(orientation=orientation, **kwargs)
        self.dates = dates

    def tickStrings(self, values, scale, spacing):
        strings = []
        for v in values:
            idx = int(v)
            if 0 <= idx < len(self.dates):
                try:
                    strings.append(self.dates[idx].strftime('%d %b %H:%M'))
                except:
                    strings.append('')
            else:
                strings.append('')
        return strings

class CandlestickItem(pg.GraphicsObject):
    def __init__(self, data):
        pg.GraphicsObject.__init__(self)
        self.data = data
        self.generatePicture()

    def generatePicture(self):
        self.picture = QPicture()
        p = QPainter(self.picture)
        w = 0.4
        pen_up = pg.mkPen('#26a69a', width=1)
        brush_up = pg.mkBrush('#26a69a')
        pen_down = pg.mkPen('#ef5350', width=1)
        brush_down = pg.mkBrush('#ef5350')
        
        for (t, open_p, close_p, low_p, high_p) in self.data:
            if close_p >= open_p:
                p.setPen(pen_up); p.setBrush(brush_up)
            else:
                p.setPen(pen_down); p.setBrush(brush_down)
            p.drawLine(pg.QtCore.QPointF(t, low_p), pg.QtCore.QPointF(t, high_p))
            body_h = close_p - open_p
            if abs(body_h) < 1e-5: body_h = 0.0001 
            p.drawRect(pg.QtCore.QRectF(t - w, open_p, w * 2, body_h))
        p.end()

    def paint(self, p, *args):
        p.drawPicture(0, 0, self.picture)

    def boundingRect(self):
        return pg.QtCore.QRectF(self.picture.boundingRect())


# ==========================================
# 2. WORKERS (LOGIC)
# ==========================================
class Signaller(QObject):
    text_written = pyqtSignal(str)

class QtLogger(object):
    def __init__(self, signaller):
        self.signaller = signaller
        self.terminal = sys.stdout

    def write(self, message):
        self.terminal.write(message)
        self.signaller.text_written.emit(message)

    def flush(self):
        self.terminal.flush()

class UtilityWorker(QThread):
    finished = pyqtSignal(str)
    
    def __init__(self, script_name, args=[]):
        super().__init__()
        self.script_name = script_name
        self.args = args

    def run(self):
        print(f"\n[SYSTEM] Executing: {self.script_name} {' '.join(self.args)}.")
        try:
            import subprocess
            from config import Config, UniverseMode

            python_exe = sys.executable 
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"

            # 🔁 Прокидываем выбранный юниверс в дочерний процесс
            try:
                mode_obj = getattr(Config, "UNIVERSE_MODE", None)
                if isinstance(mode_obj, UniverseMode):
                    env["UNIVERSE_MODE"] = mode_obj.value
                    print(f"[SYSTEM] Passing UNIVERSE_MODE={mode_obj.value} to child process")
            except Exception:
                pass

            # 🔁 Прокидываем флаги использования лидеров
            env["USE_LEADER_CRYPTO"] = "1" if getattr(Config, "USE_LEADER_CRYPTO", True) else "0"
            env["USE_LEADER_STOCKS"] = "1" if getattr(Config, "USE_LEADER_STOCKS", True) else "0"

            cmd = [python_exe, "-u", self.script_name] + self.args
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8', 
                errors='replace',
                env=env
            )
            for line in process.stdout:
                print(line.strip())
            for line in process.stderr:
                print(f"STDERR: {line.strip()}")
            process.wait()
            self.finished.emit("Done")
        except Exception as e:
            print(f"[ERROR] Launch failed: {e}")
            self.finished.emit("Error")

class BacktestLoader(QThread):
    data_loaded = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, assets):
        super().__init__()
        self.assets = assets

    def run(self):
        end = datetime.now()
        start = end - timedelta(days=90)
        print("[DATA] Loading Portfolio Data.")
        try:
            # Строим ту же карту лидеров, что и в SignalFactory / Universal
            leader_map = {
                sym: Config.get_leader_for_symbol(sym)
                for sym in self.assets
            }
            portfolio = DataLoader.get_portfolio_data(
                self.assets,
                leader_map,
                start,
                end,
                "15m",
                "1h",
            )
            if not portfolio:
                self.error_occurred.emit("No data returned from DataLoader.")
                return
            
            import pickle
            signals = {}
            if os.path.exists("data_cache/production_signals_v1.pkl"):
                 try:
                     with open("data_cache/production_signals_v1.pkl", "rb") as f: 
                         signals = pickle.load(f)
                 except: pass
            
            processed_data = {}
            for sym, df in portfolio.items():
                df = FeatureEngineer.add_features(df)
                if sym in signals:
                    sig_df = signals[sym]
                    df = df.join(sig_df[['p_long', 'p_short', 'regime']], rsuffix='_sig')
                    if 'p_long_sig' in df.columns:
                        df['p_long'] = df['p_long_sig'].fillna(0)
                        df['p_short'] = df['p_short_sig'].fillna(0)
                        df['regime'] = df['regime_sig'].fillna(0).astype(int)
                else:
                    df['p_long'] = 0.0; df['p_short'] = 0.0; df['regime'] = 0
                processed_data[sym] = df
            
            self.data_loaded.emit(processed_data)
        except Exception as e:
             import traceback
             print(traceback.format_exc())
             self.error_occurred.emit(f"Loader Error: {str(e)}")


class WFOSettingsWidget(QGroupBox):
    def __init__(self, parent=None):
        super().__init__("🧠 Brain Surgery (WFO Settings)", parent)
        self.setStyleSheet(
            "QGroupBox { font-weight: bold; border: 1px solid #555; margin-top: 10px; } "
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }"
        )
        
        layout = QVBoxLayout()
        
        # --- SLIDER 1: TRAINING WINDOW (MEMORY) ---
        self.lbl_train = QLabel("📚 Memory (Train Window): 800 candles")
        self.slider_train = QSlider(Qt.Horizontal)
        self.slider_train.setRange(200, 5000)  # От 200 свечей до 5000
        self.slider_train.setValue(800)
        self.slider_train.setTickPosition(QSlider.TicksBelow)
        self.slider_train.setTickInterval(200)
        self.slider_train.valueChanged.connect(self.update_labels)
        
        # --- SLIDER 2: TESTING WINDOW (RE-TRAIN FREQUENCY) ---
        self.lbl_test = QLabel("⚔️ Courage (Trade Window): 200 candles")
        self.slider_test = QSlider(Qt.Horizontal)
        self.slider_test.setRange(50, 1000)  # От 50 свечей до 1000
        self.slider_test.setValue(200)
        self.slider_test.setTickPosition(QSlider.TicksBelow)
        self.slider_test.setTickInterval(50)
        self.slider_test.valueChanged.connect(self.update_labels)
        
        # --- INFO LABEL (DAYS / MONTHS) ---
        self.lbl_info = QLabel("")
        self.lbl_info.setStyleSheet("color: #888; font-style: italic; font-size: 9pt;")
        
        # --- ACTION BUTTON ---
        self.btn_retrain = QPushButton("🧬 RE-LOBOTOMIZE (Retrain Model)")
        self.btn_retrain.setStyleSheet(
            "background-color: #d32f2f; color: white; font-weight: bold; padding: 5px;"
        )
        self.btn_retrain.clicked.connect(self.on_retrain_click)
        
        layout.addWidget(self.lbl_train)
        layout.addWidget(self.slider_train)
        layout.addWidget(self.lbl_test)
        layout.addWidget(self.slider_test)
        layout.addWidget(self.lbl_info)
        layout.addWidget(self.btn_retrain)
        
        self.setLayout(layout)
        # Сразу приводим подписи в соответствие с текущими значениями
        self.update_labels()
        
    def update_labels(self):
        train_val = self.slider_train.value()
        test_val = self.slider_test.value()
        
        self.lbl_train.setText(f"📚 Memory (Train Window): {train_val} candles")
        self.lbl_test.setText(f"⚔️ Courage (Trade Window): {test_val} candles")
        
        # 4h таймфрейм
        hours_per_candle = 4
        
        train_days = (train_val * hours_per_candle) / 24
        test_days = (test_val * hours_per_candle) / 24
        
        train_months = train_days / 30.0
        test_months = test_days / 30.0
        
        self.lbl_info.setText(
            f"⏳ Horizon (4h): "
            f"Train ≈ {train_days:.1f} d (~{train_months:.1f} m) | "
            f"Trade ≈ {test_days:.1f} d (~{test_months:.1f} m)"
        )

    def get_values(self):
        return self.slider_train.value(), self.slider_test.value()

    def on_retrain_click(self):
        train, test = self.get_values()
        print(f"🔪 Starting Lobotomy... Train: {train}, Test: {test}")
        # Тут мы будем вызывать сигнал для запуска потока
        # self.parent().start_optimization(train, test)

# ==========================================
# 3. MAIN APPLICATION WINDOW
# ==========================================
class FundManagerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QUANTUM FUND MANAGER | PRO TERMINAL")
        self.resize(1600, 950)
        self.setStyleSheet(STYLESHEET)

        # Сначала создаём структуру UI (в т.ч. self.console)
        self.workers = {}

        # 🔌 ExecutionRouter: единая точка входа к Bitget/Tinkoff/Simulated
        self.execution_router = ExecutionRouter()
        self._router_initialized = False  # --- NEW: флаг инициализации брокеров

        # 🕒 Таймер для LIVE MONITOR + история equity за сессию
        self.live_timer = QTimer(self)
        self.live_timer.setInterval(5000)  # каждые 5 секунд
        self.live_timer.timeout.connect(self.refresh_live_monitor_snapshot)
        self.live_equity_history = []  # (t_index, equity)

        # Флаг активной торговой сессии
                # --- LIVE TRADING STATE ---
        self.trading_session_active = False
        self.live_trader = None              # AsyncStrategyRunner
        self.live_trader_task = None         # asyncio.Task (в отдельном event loop)

        # --- Async event loop в отдельном потоке ---
        self._async_loop = asyncio.new_event_loop()
        self._async_thread = threading.Thread(
            target=self._run_async_loop, daemon=True
        )
        self._async_thread.start()

        # Текущий выбранный юниверс (крипта/биржа/оба)
        self.current_universe_mode = Config.UNIVERSE_MODE

        # --- NEW: профиль оптимизатора (AUTO / CRYPTO / STOCKS / BOTH) ---
        env_profile = os.getenv("OPTIMIZER_PROFILE", "auto").lower()
        if env_profile not in ("crypto", "stocks", "both", "auto"):
            env_profile = "auto"
        self.optimizer_profile_mode = env_profile  # храним в окне

        # 1) Строим UI (создаётся self.console)
        self.setup_ui()

        # 2) Настраиваем перехват stdout/stderr в SYSTEM TERMINAL
        self.signaller = Signaller()
        self.signaller.text_written.connect(self.log_message)

        self.qt_logger = QtLogger(self.signaller)

        # Перенаправляем stdout и stderr в наш логгер
        sys.stdout = self.qt_logger
        sys.stderr = self.qt_logger

        # Синхронизируем режим исполнения при старте
        self.sync_execution_mode_from_config()

    # --- NEW: helper для вызова async-методов из GUI ---

    def _run_async_loop(self):
        """
        Фоновый поток: крутит asyncio-цикл для ExecutionRouter и AsyncStrategyRunner.
        """
        asyncio.set_event_loop(self._async_loop)
        self._async_loop.run_forever()

    def _await_async(self, coro):
        """
        Запускает async-код из Qt-GUI через существующий event loop
        в фоне. Блокирует текущий поток, пока корутина не выполнится.
        """
        if not hasattr(self, "_async_loop") or self._async_loop is None:
            # Фолбэк, если вдруг loop ещё не поднят
            return asyncio.run(coro)

        future = asyncio.run_coroutine_threadsafe(coro, self._async_loop)
        return future.result()
        
    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        

        # Внешний layout (одна строка с QSplitter)
        outer_layout = QVBoxLayout(central_widget)
        outer_layout.setContentsMargins(10, 10, 10, 10)
        outer_layout.setSpacing(10)

        # --- ГЛАВНЫЙ СПЛИТТЕР: СЛЕВА ТАБЫ, СПРАВА ТЕРМИНАЛ ---
        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.setHandleWidth(4)

        # ---- LEFT: ВКЛАДКИ (как и раньше) ----
        tabs_container = QWidget()
        tabs_layout = QVBoxLayout(tabs_container)
        tabs_layout.setContentsMargins(0, 0, 0, 0)
        tabs_layout.setSpacing(0)

        self.tabs = QTabWidget()
        self.tabs.addTab(self.create_control_tab(), "CONTROL CENTER")
        self.tabs.addTab(self.create_war_room_tab(), "WAR ROOM")
        self.tabs.addTab(self.create_live_monitor_tab(), "LIVE MONITOR")
        self.tabs.addTab(self.create_factory_info_tab(), "DATA FACTORY")

        tabs_layout.addWidget(self.tabs)

        # ---- RIGHT: SYSTEM TERMINAL ----
        log_group = QGroupBox("SYSTEM TERMINAL")
        # Больше НЕ фиксируем высоту – он на всю высоту сплиттера
        log_layout = QVBoxLayout(log_group)
        log_layout.setContentsMargins(5, 15, 5, 5)

        self.console = QTextEdit()
        self.console.setReadOnly(True)
        log_layout.addWidget(self.console)

        # Развешиваем по сплиттеру
        main_splitter.addWidget(tabs_container)
        main_splitter.addWidget(log_group)

        # Пропорции по умолчанию: левый широкий, правый уже, но можно тянуть
        main_splitter.setSizes([1200, 400])
        main_splitter.setStretchFactor(0, 3)
        main_splitter.setStretchFactor(1, 1)

        outer_layout.addWidget(main_splitter)

    def log_message(self, text):
        # Если по какой-то причине console ещё не создан – тихо выходим
        if not hasattr(self, "console") or self.console is None:
            return

        cursor = self.console.textCursor()
        cursor.movePosition(cursor.End)
        cursor.insertText(text)
        self.console.setTextCursor(cursor)
        self.console.ensureCursorVisible()
        self.sync_execution_mode_from_config()
    # ------------------------------------------
    # TAB 1: CONTROL CENTER (Full Pipeline)
    # ------------------------------------------
    def create_control_tab(self):
        tab = QWidget()
        layout = QHBoxLayout(tab)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)
        
        # --- LEFT: GENOME SETTINGS ---
        settings_group = QGroupBox("OPTIMIZER CONFIGURATION (GENOME)")
        settings_layout = QVBoxLayout(settings_group)
        grid = QGridLayout()
        grid.setVerticalSpacing(15)
        grid.setHorizontalSpacing(15)
        
        self.spin_sl_min = self._make_spin(0.5, 5.0, 1.5)
        self.spin_sl_max = self._make_spin(0.5, 5.0, 2.5)
        self.spin_tp_min = self._make_spin(1.0, 10.0, 3.0)
        self.spin_tp_max = self._make_spin(1.0, 15.0, 6.0)
        self.spin_pull_min = self._make_spin(0.0, 1.0, 0.0)
        self.spin_pull_max = self._make_spin(0.0, 1.0, 0.15)
        self.spin_conf_min = self._make_spin(0.1, 1.0, 0.65)
        self.spin_conf_max = self._make_spin(0.1, 1.0, 0.85)

        grid.addWidget(QLabel("Stop Loss (Min/Max):"), 0, 0)
        grid.addWidget(self.spin_sl_min, 0, 1); grid.addWidget(self.spin_sl_max, 0, 2)
        grid.addWidget(QLabel("Take Profit (Min/Max):"), 1, 0)
        grid.addWidget(self.spin_tp_min, 1, 1); grid.addWidget(self.spin_tp_max, 1, 2)
        grid.addWidget(QLabel("Pullback (Min/Max):"), 2, 0)
        grid.addWidget(self.spin_pull_min, 2, 1); grid.addWidget(self.spin_pull_max, 2, 2)
        grid.addWidget(QLabel("Confidence (Min/Max):"), 3, 0)
        grid.addWidget(self.spin_conf_min, 3, 1); grid.addWidget(self.spin_conf_max, 3, 2)
        
        settings_layout.addLayout(grid)
        
        # --- NEW: Strategy Profile indicator ---
        try:
            mode = getattr(self, "current_universe_mode", Config.UNIVERSE_MODE)
            profile_key = getattr(mode, "value", "both")
        except Exception:
            profile_key = "both"

        effective_profile = self._get_effective_optimizer_profile()
        self.lbl_optimizer_profile = QLabel(f"Optimizer profile: {effective_profile.upper()}")
        self.lbl_optimizer_profile.setStyleSheet("color: #aaaaaa; font-size: 11px;")
        settings_layout.addWidget(self.lbl_optimizer_profile)


        # --- NEW: COMBOBOX ДЛЯ ВЫБОРА ПРОФИЛЯ ---
        self.cbo_optimizer_profile = QComboBox()
        self.cbo_optimizer_profile.addItems(["AUTO", "CRYPTO", "STOCKS", "BOTH"])

        # Инициализация по текущему состоянию
        current_profile = self.optimizer_profile_mode.upper()
        if current_profile not in ("AUTO", "CRYPTO", "STOCKS", "BOTH"):
            current_profile = "AUTO"
        self.cbo_optimizer_profile.setCurrentText(current_profile)

        self.cbo_optimizer_profile.currentTextChanged.connect(
            self.on_optimizer_profile_changed
        )
        settings_layout.addWidget(self.cbo_optimizer_profile)

        # --- NEW: Telegram HTF feature toggles ---
        # Эти флаги прокидываются через ENV в оптимизатор / генератор / дебаг реплеер.
        self.chk_tg_crypto = QCheckBox("Telegram HTF → Crypto")
        self.chk_tg_stocks = QCheckBox("Telegram HTF → Stocks")

        # Инициализация по Config/ENV (по умолчанию включено, если переменных нет)
        use_tg_crypto = getattr(
            Config, "USE_TG_CRYPTO", os.getenv("USE_TG_CRYPTO", "1") == "1"
        )
        use_tg_stocks = getattr(
            Config, "USE_TG_STOCKS", os.getenv("USE_TG_STOCKS", "1") == "1"
        )
        self.chk_tg_crypto.setChecked(bool(use_tg_crypto))
        self.chk_tg_stocks.setChecked(bool(use_tg_stocks))

        settings_layout.addWidget(self.chk_tg_crypto)
        settings_layout.addWidget(self.chk_tg_stocks)

        # --- NEW: MARKET LEADERS (флаги + выбор тикера) ---
        leader_group = QGroupBox("MARKET LEADERS")
        leader_layout = QGridLayout(leader_group)

        # Флаги "использовать лидера"
        self.chk_leader_crypto = QCheckBox("Use leader for Crypto")
        self.chk_leader_stocks = QCheckBox("Use leader for Stocks")

        use_leader_crypto = getattr(
            Config, "USE_LEADER_CRYPTO", os.getenv("USE_LEADER_CRYPTO", "1") == "1"
        )
        use_leader_stocks = getattr(
            Config, "USE_LEADER_STOCKS", os.getenv("USE_LEADER_STOCKS", "1") == "1"
        )

        self.chk_leader_crypto.setChecked(bool(use_leader_crypto))
        self.chk_leader_stocks.setChecked(bool(use_leader_stocks))

        # Выбор лидера для крипты
        self.cbo_leader_crypto = QComboBox()
        self.cbo_leader_crypto.addItems(["BTCUSDT", "ETHUSDT", "NONE"])
        current_crypto_leader = getattr(Config, "LEADER_SYMBOL_CRYPTO", "BTCUSDT")
        if current_crypto_leader not in ("BTCUSDT", "ETHUSDT"):
            # если что-то экзотическое — по умолчанию BTC
            current_crypto_leader = "BTCUSDT"
        if not use_leader_crypto:
            current_crypto_leader = "NONE"
        self.cbo_leader_crypto.setCurrentText(current_crypto_leader)

        # Выбор лидера для акций
        self.cbo_leader_stocks = QComboBox()
        self.cbo_leader_stocks.addItems(["MOEX", "RTS", "SBER", "NONE"])
        current_stock_leader = getattr(Config, "LEADER_SYMBOL_EQUITY", "MOEX")
        if current_stock_leader not in ("MOEX", "RTS", "SBER"):
            current_stock_leader = "MOEX"
        if not use_leader_stocks:
            current_stock_leader = "NONE"
        self.cbo_leader_stocks.setCurrentText(current_stock_leader)

        # Разворачиваем в сетку
        leader_layout.addWidget(QLabel("Crypto leader:"), 0, 0)
        leader_layout.addWidget(self.chk_leader_crypto, 0, 1)
        leader_layout.addWidget(self.cbo_leader_crypto, 0, 2)

        leader_layout.addWidget(QLabel("Stocks leader:"), 1, 0)
        leader_layout.addWidget(self.chk_leader_stocks, 1, 1)
        leader_layout.addWidget(self.cbo_leader_stocks, 1, 2)

        settings_layout.addWidget(leader_group)

        # --- NEW: WFO SLIDERS (TRAIN / TRADE WINDOWS) ---
        self.wfo_widget = WFOSettingsWidget()
        settings_layout.addWidget(self.wfo_widget)
        
        settings_layout.addStretch()
        
        btn_save = QPushButton("SAVE CONFIGURATION")
        btn_save.setObjectName("ActionBtn")
        btn_save.clicked.connect(self.save_optimizer_settings)
        settings_layout.addWidget(btn_save)
        
        # --- RIGHT: EXECUTION PANEL (Split into Diag & Core) ---
        exec_widget = QWidget()
        exec_layout_main = QVBoxLayout(exec_widget)
        exec_layout_main.setContentsMargins(0,0,0,0)
        
        # 1. DIAGNOSTICS & ANALYTICS
        diag_group = QGroupBox("DIAGNOSTICS & ANALYTICS")
        diag_layout = QGridLayout(diag_group)
        
        # Создаем кнопки диагностики (Серый стиль)
        btn_gpu   = QPushButton("GPU Check");       btn_gpu.setObjectName("DiagBtn")
        btn_leak  = QPushButton("Leak Test");       btn_leak.setObjectName("DiagBtn")
        btn_noise = QPushButton("Noise Radar");     btn_noise.setObjectName("DiagBtn")
        btn_stats = QPushButton("Stat Analyzer");   btn_stats.setObjectName("DiagBtn")
        btn_bal   = QPushButton("Balance Check");   btn_bal.setObjectName("DiagBtn")
        btn_probs = QPushButton("Prob Audit");      btn_probs.setObjectName("DiagBtn")
        btn_core  = QPushButton("Core Debug");      btn_core.setObjectName("DiagBtn")
        btn_feat  = QPushButton("Feature Lab");     btn_feat.setObjectName("DiagBtn")
        btn_plot  = QPushButton("Plot");            btn_plot.setObjectName("DiagBtn")
        btn_validation = QPushButton("Valid Rep");  btn_validation.setObjectName("DiagBtn")
        btn_full_cycle = QPushButton("Full Cycle Test");    btn_full_cycle.setObjectName("DiagBtn")
        btn_async_bg   = QPushButton("Async Bitget");       btn_async_bg.setObjectName("DiagBtn")
        btn_no_look    = QPushButton("Core No-Lookahead"); btn_no_look.setObjectName("DiagBtn")

        # 🔹 Новые кнопки для Debug Replay
        btn_debug = QPushButton("Debug Replay (no plots)")
        btn_debug.setObjectName("DiagBtn")
        btn_debug_plots = QPushButton("Debug Replay + Charts")
        btn_debug_plots.setObjectName("DiagBtn")

        # 🔹 Новые кнопки: инструменты и тест коннекта
        btn_get_instruments = QPushButton("Get Instruments")
        btn_get_instruments.setObjectName("DiagBtn")
        btn_test_conn = QPushButton("Test Connections")
        btn_test_conn.setObjectName("DiagBtn")

        # Привязка к run_script
        btn_gpu.clicked.connect(   lambda: self.run_script("test_gpu.py",          []))
        btn_leak.clicked.connect(  lambda: self.run_script("leak_test.py",         []))
        btn_noise.clicked.connect( lambda: self.run_script("noise_radar.py",       []))
        btn_stats.clicked.connect( lambda: self.run_script("stat_analyzer.py",     []))
        btn_bal.clicked.connect(   lambda: self.run_script("check_balance.py",     []))
        btn_probs.clicked.connect( lambda: self.run_script("inspect_probs.py",     []))
        btn_core.clicked.connect(  lambda: self.run_script("debug_core.py",        []))
        btn_feat.clicked.connect(  lambda: self.run_script("feature_benchmark.py", []))
        btn_plot.clicked.connect(  lambda: self.run_script("plot_equity.py",       []))
        btn_validation.clicked.connect(
            lambda: self.run_script("validation_report.py", [])
        )

        # Новые привязки
        btn_get_instruments.clicked.connect(
            lambda: self.run_script("get_instruments.py", [])
        )
        btn_test_conn.clicked.connect(
            lambda: self.run_script("test_connections.py", [])
        )

        # 🔹 Запуск дебаг-реплеера:
        btn_debug.clicked.connect(
            lambda: self.run_script("debug_replayer.py", [])
        )
        btn_debug_plots.clicked.connect(
            lambda: self.run_script("debug_replayer.py", ["--plot"])
        )
        btn_full_cycle.clicked.connect(
            lambda: self.run_script("test_full_cycle.py", [])
        )
        btn_async_bg.clicked.connect(
            lambda: self.run_script("test_async_bitget.py", [])
        )
        btn_no_look.clicked.connect(
            lambda: self.run_script("test_core_no_lookahead.py", [])
        )
        # Расставляем сеткой 4x4
        diag_layout.addWidget(btn_gpu,   0, 0)
        diag_layout.addWidget(btn_leak,  0, 1)
        diag_layout.addWidget(btn_noise, 0, 2)
        diag_layout.addWidget(btn_stats, 0, 3)

        diag_layout.addWidget(btn_bal,   1, 0)
        diag_layout.addWidget(btn_probs, 1, 1)
        diag_layout.addWidget(btn_core,  1, 2)
        diag_layout.addWidget(btn_feat,  1, 3)

        # Третий ряд – Debug Replay
        diag_layout.addWidget(btn_debug,       2, 0)
        diag_layout.addWidget(btn_debug_plots, 2, 1)
        diag_layout.addWidget(btn_plot,        2, 2)
        diag_layout.addWidget(btn_validation,  2, 3)

        # Четвёртый ряд – инфраструктурные проверки
        diag_layout.addWidget(btn_get_instruments, 3, 0)
        diag_layout.addWidget(btn_test_conn,       3, 1)
        # 4-й ряд – high-level тесты
        diag_layout.addWidget(btn_full_cycle, 3, 2)
        diag_layout.addWidget(btn_async_bg,   3, 3)
        diag_layout.addWidget(btn_no_look,    4, 0)

        # 2. EXECUTION MODE & TRADING CONTROL
        mode_group = QGroupBox("EXECUTION MODE & TRADING CONTROL")
        mode_layout = QVBoxLayout(mode_group)

        # Радио-кнопки режима
        mode_row = QHBoxLayout()
        self.radio_mode_backtest = QRadioButton("BACKTEST")
        self.radio_mode_paper    = QRadioButton("PAPER / DEMO")
        self.radio_mode_live     = QRadioButton("LIVE / REAL")

        mode_row.addWidget(self.radio_mode_backtest)
        mode_row.addWidget(self.radio_mode_paper)
        mode_row.addWidget(self.radio_mode_live)
        mode_row.addStretch()
        mode_layout.addLayout(mode_row)

        # --- 2. Новый блок: выбор юниверса (крипта / биржа / совместно)
        universe_group = QGroupBox("Торгуемый рынок")
        universe_layout = QHBoxLayout()

        # Радиокнопки выбора рынка
        self.radio_universe_crypto = QRadioButton("Крипта")
        self.radio_universe_stocks = QRadioButton("Биржа (MOEX)")
        self.radio_universe_both   = QRadioButton("Крипта + Биржа")

        # Начальное состояние — из Config.UNIVERSE_MODE
        current_mode = Config.UNIVERSE_MODE
        if current_mode == UniverseMode.CRYPTO:
            self.radio_universe_crypto.setChecked(True)
        elif current_mode == UniverseMode.STOCKS:
            self.radio_universe_stocks.setChecked(True)
        else:
            self.radio_universe_both.setChecked(True)

        # Сигналы — все ведут в один handler
        self.radio_universe_crypto.toggled.connect(self.on_universe_mode_changed)
        self.radio_universe_stocks.toggled.connect(self.on_universe_mode_changed)
        self.radio_universe_both.toggled.connect(self.on_universe_mode_changed)

        universe_layout.addWidget(self.radio_universe_crypto)
        universe_layout.addWidget(self.radio_universe_stocks)
        universe_layout.addWidget(self.radio_universe_both)
        universe_layout.addStretch()
        universe_group.setLayout(universe_layout)

        # ВАЖНО: кладём группу внутрь mode_group, а не в общий layout
        mode_layout.addWidget(universe_group)

        # Текстовый статус
        self.lbl_mode_status = QLabel("Current mode: BACKTEST")
        self.lbl_mode_status.setStyleSheet("color: #aaaaaa;")
        mode_layout.addWidget(self.lbl_mode_status)

        # Кнопки управления сессией
        btn_row = QHBoxLayout()
        self.btn_start_trading = QPushButton("▶ START TRADING SESSION")
        self.btn_stop_trading  = QPushButton("⏹ STOP TRADING")

        btn_row.addWidget(self.btn_start_trading)
        btn_row.addWidget(self.btn_stop_trading)
        btn_row.addStretch()
        mode_layout.addLayout(btn_row)

        # Связка сигналов
        self.radio_mode_backtest.toggled.connect(self.on_execution_mode_changed)
        self.radio_mode_paper.toggled.connect(self.on_execution_mode_changed)
        self.radio_mode_live.toggled.connect(self.on_execution_mode_changed)
        self.btn_start_trading.clicked.connect(self.on_start_trading_clicked)
        self.btn_stop_trading.clicked.connect(self.on_stop_trading_clicked)

        # 3. CORE PRODUCTION PIPELINE
        prod_group = QGroupBox("CORE PRODUCTION PIPELINE")
        prod_layout = QVBoxLayout(prod_group)
        prod_layout.setSpacing(10)
        
        # Классический WALK-FORWARD генератор
        btn_gen = QPushButton("1. SIGNAL GENERATOR (WALK, Full Reset)")
        btn_gen.clicked.connect(self.run_walk_generator)
        
        # 1U. УНИВЕРСАЛЬНЫЙ МОЗГ
        btn_gen_universal = QPushButton("1U. UNIVERSAL BRAIN (Cross-Asset WF)")
        btn_gen_universal.setObjectName("ActionBtn")
        btn_gen_universal.setToolTip(
            "Обучить единый Universal Brain и сгенерировать сигналы по всему портфелю."
        )
        btn_gen_universal.clicked.connect(self.run_universal_generator)

        # --- НОВОЕ: отдельные кнопки для крипты и стоков ---
        btn_gen_universal_crypto = QPushButton("1U-C. Train Crypto Brain")
        btn_gen_universal_crypto.setToolTip(
            "Переключиться на крипто-юниверс и обучить Universal Brain только на крипте."
        )
        btn_gen_universal_crypto.clicked.connect(self.run_universal_crypto_brain)

        btn_gen_universal_stocks = QPushButton("1U-S. Train Stocks Brain")
        btn_gen_universal_stocks.setToolTip(
            "Переключиться на биржевой юниверс и обучить Universal Brain только на акциях/валютах."
        )
        btn_gen_universal_stocks.clicked.connect(self.run_universal_stocks_brain)
        
        # 2. Генетический оптимизатор
        btn_opt = QPushButton("2. GENETIC OPTIMIZER (Sniper Mode)")
        btn_opt.setObjectName("ActionBtn")  # Blue Highlight
        btn_opt.clicked.connect(self.run_optimizer_with_save)
        
        # 3. Дебаг реплеер (все сделки)
        btn_replay = QPushButton("3. DEBUG REPLAYER (Trace Report)")
        btn_replay.setObjectName("ActionBtn")
        btn_replay.setStyleSheet("border-color: #ffd700; color: #ffd700;")
        btn_replay.clicked.connect(lambda: self.run_script("debug_replayer.py", []))

        # --- NEW: дебаг только по крипте ---
        btn_replay_crypto = QPushButton("3C. DEBUG REPLAYER – CRYPTO ONLY")
        btn_replay_crypto.setObjectName("ActionBtn")
        btn_replay_crypto.setToolTip("Реплейер только по криптовым инструментам (asset_class=crypto).")
        btn_replay_crypto.clicked.connect(
            lambda: self.run_script("debug_replayer.py", ["--asset_class", "crypto"])
        )

        # --- NEW: дебаг только по стокам ---
        btn_replay_stocks = QPushButton("3S. DEBUG REPLAYER – STOCKS ONLY")
        btn_replay_stocks.setObjectName("ActionBtn")
        btn_replay_stocks.setToolTip("Реплейер только по биржевым инструментам (asset_class=stocks).")
        btn_replay_stocks.clicked.connect(
            lambda: self.run_script("debug_replayer.py", ["--asset_class", "stocks"])
        )
        
        # Добавляем в правильном порядке
        prod_layout.addWidget(btn_gen)
        prod_layout.addWidget(btn_gen_universal)
        prod_layout.addWidget(btn_gen_universal_crypto)
        prod_layout.addWidget(btn_gen_universal_stocks)
        prod_layout.addWidget(btn_opt)
        prod_layout.addWidget(btn_replay)
        prod_layout.addWidget(btn_replay_crypto)
        prod_layout.addWidget(btn_replay_stocks)
        
        exec_layout_main.addWidget(diag_group)
        exec_layout_main.addWidget(mode_group)
        exec_layout_main.addWidget(prod_group)
        exec_layout_main.addStretch()
        
        layout.addWidget(settings_group, 6)
        layout.addWidget(exec_widget, 4)
        
        
        self.load_optimizer_settings()
        return tab

    def _make_spin(self, min_v, max_v, def_v):
        s = QDoubleSpinBox()
        s.setRange(min_v, max_v)
        s.setValue(def_v)
        s.setSingleStep(0.05)
        s.setDecimals(2)
        return s
    
    def get_selected_assets(self):
        """
        Возвращает список тикеров в зависимости от текущего юниверса в GUI.
        Используется для WAR ROOM и внутренних загрузчиков.
        """
        mode = getattr(self, "current_universe_mode", Config.UNIVERSE_MODE)
        assets = get_assets_for_universe(mode)
        print(f"[GUI] get_selected_assets: mode={mode.value}, n={len(assets)}")
        return assets

    def refresh_asset_combo(self):
        """
        Перезаполняет селектор ASSET SELECTION в WAR ROOM под текущий юниверс.
        Если комбобокс ещё не создан — выходим тихо.
        """
        if not hasattr(self, "asset_combo"):
            return

        symbols = self.get_selected_assets()
        self.asset_combo.blockSignals(True)
        self.asset_combo.clear()
        self.asset_combo.addItems(symbols)
        self.asset_combo.blockSignals(False)

        # Обновляем график, если есть данные
        self.update_chart()    

    def on_universe_mode_changed(self, checked: bool = False):
        """
        Хэндлер радиокнопок выбора рынка:
        крипта / биржа (MOEX) / совместно.
        Сигнал toggled(bool) передаёт флаг, но нам он не нужен.
        """
        # Если UI ещё не до конца инициализирован – выходим тихо
        if not hasattr(self, "radio_universe_crypto"):
            return

        # Определяем выбранный режим по состоянию радиокнопок
        if self.radio_universe_crypto.isChecked():
            mode = UniverseMode.CRYPTO
        elif self.radio_universe_stocks.isChecked():
            mode = UniverseMode.STOCKS
        elif self.radio_universe_both.isChecked():
            mode = UniverseMode.BOTH
        else:
            mode = UniverseMode.BOTH  # safety fallback

        # Обновляем режим в конфиге и локально
        Config.UNIVERSE_MODE = mode
        self.current_universe_mode = mode

        # Дублируем в ENV, чтобы дочерние процессы видели тот же юниверс
        os.environ["UNIVERSE_MODE"] = mode.value

        print(
            f"[GUI] Universe mode set to: {mode.value} "
            f"({'крипта' if mode == UniverseMode.CRYPTO else 'биржа' if mode == UniverseMode.STOCKS else 'совместно'})"
        )

        # Перезаполняем список инструментов в WAR ROOM
        self.refresh_asset_combo()

        # Подгружаем оптимизатор-профиль для выбранного юниверса
        if hasattr(self, "spin_sl_min"):
            self.load_optimizer_settings()

    def _get_effective_optimizer_profile(self) -> str:
        """
        Возвращает реально используемый профиль:
        - если optimizer_profile_mode != auto → берём его прямо;
        - если auto → берём из current_universe_mode / Config.UNIVERSE_MODE.
        """
        if self.optimizer_profile_mode in ("crypto", "stocks", "both"):
            return self.optimizer_profile_mode

        # auto → профиль от текущего юниверса
        try:
            mode_obj = getattr(self, "current_universe_mode", Config.UNIVERSE_MODE)
            return getattr(mode_obj, "value", "both")
        except Exception:
            return "both"

    def on_optimizer_profile_changed(self, text: str):
        """
        Вызывается при смене профиля в комбобоксе.
        Обновляем env и подпись.
        """
        mode = text.lower()
        if mode not in ("auto", "crypto", "stocks", "both"):
            mode = "auto"

        self.optimizer_profile_mode = mode
        os.environ["OPTIMIZER_PROFILE"] = mode  # увидят optimizer.py / signal_generator.py

        effective_profile = self._get_effective_optimizer_profile()
        self.lbl_optimizer_profile.setText(f"Optimizer profile: {effective_profile.upper()}")

    # ------------------------------------------
    # TAB 2: WAR ROOM
    # ------------------------------------------
    def create_war_room_tab(self):
        tab = QWidget()
        layout = QHBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)
        splitter = QSplitter(Qt.Horizontal)
        
        # Sidebar
        sidebar = QFrame()
        sidebar.setStyleSheet("background-color: #252526;")
        side_layout = QVBoxLayout(sidebar)
        side_layout.setContentsMargins(10, 20, 10, 10)
                
        side_layout.addWidget(QLabel("<b>ASSET SELECTION</b>"))
        self.asset_combo = QComboBox()
        self.asset_combo.addItems(self.get_selected_assets())
        self.asset_combo.currentIndexChanged.connect(self.update_chart)
        side_layout.addWidget(self.asset_combo)
        
        btn_load = QPushButton("LOAD MARKET DATA")
        btn_load.setObjectName("ActionBtn")
        btn_load.clicked.connect(self.load_backtest_data)
        side_layout.addWidget(btn_load)
        
        self.lbl_status = QLabel("Status: Idle")
        self.lbl_status.setStyleSheet("color: #888; font-style: italic;")
        side_layout.addWidget(self.lbl_status)
        side_layout.addStretch()
        
        # Chart
        chart_area = QWidget()
        chart_layout = QVBoxLayout(chart_area)
        chart_layout.setContentsMargins(0, 0, 0, 0)
        chart_layout.setSpacing(0)
        
        self.date_axis = DateAxis(dates=[], orientation='bottom')
        self.plot_widget = pg.PlotWidget(axisItems={'bottom': self.date_axis})
        self.plot_widget.showGrid(x=True, y=True, alpha=0.2)
        self.plot_widget.getAxis('left').setWidth(50)

        self.prob_axis = DateAxis(dates=[], orientation='bottom')
        self.prob_plot = pg.PlotWidget(axisItems={'bottom': self.prob_axis})
        self.prob_plot.setMaximumHeight(200)
        self.prob_plot.setXLink(self.plot_widget)
        self.prob_plot.showGrid(x=True, y=True, alpha=0.2)
        self.prob_plot.getAxis('left').setWidth(50)

        # NEW: INDICATOR PANEL (ATR)
        self.ind_axis = DateAxis(dates=[], orientation='bottom')
        self.ind_plot = pg.PlotWidget(axisItems={'bottom': self.ind_axis})
        self.ind_plot.setMaximumHeight(150)
        self.ind_plot.setXLink(self.plot_widget)
        self.ind_plot.showGrid(x=True, y=True, alpha=0.2)
        self.ind_plot.getAxis('left').setWidth(50)

        chart_layout.addWidget(self.plot_widget, stretch=3)
        chart_layout.addWidget(self.prob_plot, stretch=1)
        chart_layout.addWidget(self.ind_plot, stretch=1)
        
        splitter.addWidget(sidebar)
        splitter.addWidget(chart_area)
        splitter.setSizes([250, 1350])
        splitter.setHandleWidth(1)
        
        layout.addWidget(splitter)
        return tab

    # ------------------------------------------
    # TAB 3: DATA FACTORY
    # ------------------------------------------
    def create_factory_info_tab(self):
        """
        DATA FACTORY:
        - Сводка по signals (production_signals_v1.pkl)
        - Информация о пайплайне (без запуска)
        - Snapshot валидации (validation_report.json)
        """
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # 1) DATA OVERVIEW
        overview_group = QGroupBox("DATA OVERVIEW")
        ov_layout = QVBoxLayout(overview_group)

        self.lbl_data_overview = QLabel("No signals snapshot yet.")
        self.lbl_data_overview.setWordWrap(True)
        ov_layout.addWidget(self.lbl_data_overview)

        self.tbl_assets_overview = QTableWidget()
        self.tbl_assets_overview.setColumnCount(6)
        self.tbl_assets_overview.setHorizontalHeaderLabels([
            "Symbol", "From", "To", "Bars", "Has Signals", "Has Regimes"
        ])
        self.tbl_assets_overview.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl_assets_overview.setSelectionBehavior(self.tbl_assets_overview.SelectRows)
        self.tbl_assets_overview.setEditTriggers(self.tbl_assets_overview.NoEditTriggers)
        ov_layout.addWidget(self.tbl_assets_overview)

        btn_refresh_data = QPushButton("REFRESH DATA SNAPSHOT")
        btn_refresh_data.setObjectName("ActionBtn")
        btn_refresh_data.clicked.connect(self.refresh_data_factory_snapshot)
        ov_layout.addWidget(btn_refresh_data, alignment=Qt.AlignLeft)

        layout.addWidget(overview_group)

        # 2) PIPELINE STATUS (информация, без кнопок)
        pipeline_group = QGroupBox("PIPELINE STATUS")
        pipe_layout = QVBoxLayout(pipeline_group)

        lbl_pipeline_info = QLabel(
        "Генерация, оптимизация и валидация сигналов на вкладке CONTROL CENTER (раздел CORE PRODUCTION PIPELINE).\n\n"
        "Вкладка DATA FACTORY — это аналитический дашборд (read-only):\n"
        "• Охват сигналов по активам (Data Coverage)\n"
        "• Наличие рыночных режимов\n"
        "• Снэпшот отчетов валидации."
        )
        lbl_pipeline_info.setWordWrap(True)
        pipe_layout.addWidget(lbl_pipeline_info)

        layout.addWidget(pipeline_group)

        # 3) VALIDATION SNAPSHOT
        validation_group = QGroupBox("VALIDATION SNAPSHOT")
        val_layout = QVBoxLayout(validation_group)

        self.lbl_validation_overview = QLabel("No validation report yet.")
        self.lbl_validation_overview.setWordWrap(True)
        val_layout.addWidget(self.lbl_validation_overview)

        self.txt_validation_detail = QTextEdit()
        self.txt_validation_detail.setReadOnly(True)
        val_layout.addWidget(self.txt_validation_detail)

        btn_refresh_val = QPushButton("REFRESH VALIDATION SNAPSHOT")
        btn_refresh_val.setObjectName("DiagBtn")
        btn_refresh_val.clicked.connect(self.refresh_validation_snapshot)
        val_layout.addWidget(btn_refresh_val, alignment=Qt.AlignLeft)

        layout.addWidget(validation_group)

        layout.addStretch()

        # Первичная загрузка снимков
        self.refresh_data_factory_snapshot()
        self.refresh_validation_snapshot()

        return tab

    # ==========================================
    # DATA FACTORY HELPERS
    # ==========================================
    def refresh_data_factory_snapshot(self):
        """
        Читает production_signals_v1.pkl и заполняет таблицу по активам.
        """
        import pickle

        try:
            base_dir = Config.BASE_DIR
            signals_path = os.path.join(base_dir, "data_cache", "production_signals_v1.pkl")

            if not os.path.exists(signals_path):
                self.lbl_data_overview.setText(
                    "Signals file not found: data_cache/production_signals_v1.pkl\n"
                    "Run signal_generator.py to build universal signals."
                )
                self.tbl_assets_overview.setRowCount(0)
                return

            with open(signals_path, "rb") as f:
                signals = pickle.load(f)

            if not isinstance(signals, dict) or not signals:
                self.lbl_data_overview.setText("Signals file loaded, but dictionary is empty.")
                self.tbl_assets_overview.setRowCount(0)
                return

            symbols = sorted(signals.keys())
            self.tbl_assets_overview.setRowCount(len(symbols))

            global_min = None
            global_max = None
            total_bars = 0

            for row, sym in enumerate(symbols):
                df = signals[sym]
                if df is None or df.empty:
                    from_str = "-"
                    to_str = "-"
                    bars = 0
                else:
                    idx = df.index
                    from_dt = idx[0]
                    to_dt = idx[-1]
                    from_str = str(from_dt)
                    to_str = str(to_dt)
                    bars = len(df)

                    total_bars += bars
                    if global_min is None or from_dt < global_min:
                        global_min = from_dt
                    if global_max is None or to_dt > global_max:
                        global_max = to_dt

                has_signals = "p_long" in df.columns if df is not None and not df.empty else False
                has_regime = "regime" in df.columns if df is not None and not df.empty else False

                self.tbl_assets_overview.setItem(row, 0, QTableWidgetItem(sym))
                self.tbl_assets_overview.setItem(row, 1, QTableWidgetItem(from_str))
                self.tbl_assets_overview.setItem(row, 2, QTableWidgetItem(to_str))
                self.tbl_assets_overview.setItem(row, 3, QTableWidgetItem(str(bars)))
                self.tbl_assets_overview.setItem(row, 4, QTableWidgetItem("YES" if has_signals else "NO"))
                self.tbl_assets_overview.setItem(row, 5, QTableWidgetItem("YES" if has_regime else "NO"))

            if global_min is not None and global_max is not None:
                self.lbl_data_overview.setText(
                    f"Signals loaded for {len(symbols)} assets | "
                    f"{global_min.date()} → {global_max.date()} | "
                    f"Total bars: ~{total_bars}"
                )
            else:
                self.lbl_data_overview.setText(
                    f"Signals dictionary has {len(symbols)} keys, but all DataFrames are empty."
                )

        except Exception as e:
            self.lbl_data_overview.setText(f"Error while reading signals: {e}")
            self.tbl_assets_overview.setRowCount(0)

    def refresh_validation_snapshot(self):
        """
        Читает validation_report.json и рисует текстовый отчёт.
        """
        try:
            base_dir = Config.BASE_DIR
            report_path = os.path.join(base_dir, "validation_report.json")

            if not os.path.exists(report_path):
                self.lbl_validation_overview.setText(
                    "validation_report.json not found. "
                    "Run validation_report.py from PIPELINE section."
                )
                self.txt_validation_detail.clear()
                return

            with open(report_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not data:
                self.lbl_validation_overview.setText("validation_report.json is empty.")
                self.txt_validation_detail.clear()
                return

            # Берём FULL_HISTORY как основную сводку, если есть
            full = data.get("FULL_HISTORY", None)
            if full:
                self.lbl_validation_overview.setText(
                    f"FULL_HISTORY → Return: {full.get('total_return_pct', 0):.2f}% | "
                    f"MaxDD: {full.get('max_drawdown_pct', 0):.2f}% | "
                    f"PF: {full.get('profit_factor', 0):.2f} | "
                    f"Trades: {full.get('total_trades', 0)}"
                )
            else:
                # Берём первый срез
                first_key = next(iter(data.keys()))
                s = data[first_key]
                self.lbl_validation_overview.setText(
                    f"{first_key} → Return: {s.get('total_return_pct', 0):.2f}% | "
                    f"MaxDD: {s.get('max_drawdown_pct', 0):.2f}% | "
                    f"PF: {s.get('profit_factor', 0):.2f} | "
                    f"Trades: {s.get('total_trades', 0)}"
                )

            # Текстовая табличка по всем срезам
            lines = []
            for key, s in data.items():
                line = (
                    f"{key:12} | "
                    f"Ret {s.get('total_return_pct', 0):7.1f}% | "
                    f"MaxDD {s.get('max_drawdown_pct', 0):7.1f}% | "
                    f"PF {s.get('profit_factor', 0):5.2f} | "
                    f"Trades {s.get('total_trades', 0):5d}"
                )
                lines.append(line)

            self.txt_validation_detail.setPlainText("\n".join(lines))

        except Exception as e:
            self.lbl_validation_overview.setText(f"Error while reading validation report: {e}")
            self.txt_validation_detail.clear()

    # ------------------------------------------
    # TAB 4: LIVE MONITOR (каркас)
    # ------------------------------------------
    def create_live_monitor_tab(self):
        """
        LIVE MONITOR v1:
        - Каркас для real-time мониторинга счёта и ордеров.
        - Пока работает как placeholder (DISCONNECTED).
        """        
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # --- Верхняя строка статуса ---
        self.lbl_live_status = QLabel("LIVE MONITOR v1 — DISCONNECTED")
        self.lbl_live_status.setObjectName("LiveStatusLabel")
        layout.addWidget(self.lbl_live_status)

        # 1) ACCOUNT OVERVIEW - новая версия с GridLayout
        account_group = QGroupBox("ACCOUNT OVERVIEW (LIVE)")
        info_grid = QGridLayout(account_group)
        
        # Первая строка
        self.lbl_live_equity = QLabel("Total Equity: —")
        self.lbl_live_pnl = QLabel("Total PnL: —")
        info_grid.addWidget(self.lbl_live_equity, 0, 0)
        info_grid.addWidget(self.lbl_live_pnl, 0, 1)
        
        # Вторая строка
        self.lbl_live_bitget = QLabel("Bitget: —")
        self.lbl_live_tinkoff = QLabel("Tinkoff: —")
        info_grid.addWidget(self.lbl_live_bitget, 1, 0)
        info_grid.addWidget(self.lbl_live_tinkoff, 1, 1)

        # Кнопка ручного обновления (оригинальная)
        btn_refresh_live = QPushButton("REFRESH NOW")
        btn_refresh_live.setObjectName("DiagBtn")
        btn_refresh_live.clicked.connect(self.refresh_live_monitor_snapshot)
        info_grid.addWidget(btn_refresh_live, 0, 2, 2, 1)  # Занимает 2 строки

        layout.addWidget(account_group)

        # 1c) RISK CONTROL
        risk_group = QGroupBox("RISK CONTROL (LIVE/PAPER)")
        risk_grid = QGridLayout(risk_group)

        self.chk_allow_live = QCheckBox("ALLOW_LIVE (arm real trading)")
        self.spin_risk = QDoubleSpinBox(); self.spin_risk.setRange(0.0001, 0.10); self.spin_risk.setSingleStep(0.001)
        self.spin_max_pos = QDoubleSpinBox(); self.spin_max_pos.setRange(1, 200); self.spin_max_pos.setDecimals(0)
        self.spin_max_notional = QDoubleSpinBox(); self.spin_max_notional.setRange(0, 1e9); self.spin_max_notional.setDecimals(2)
        self.spin_max_dd = QDoubleSpinBox(); self.spin_max_dd.setRange(0.0, 0.99); self.spin_max_dd.setSingleStep(0.005)

        btn_apply_risk = QPushButton("APPLY & SAVE")
        btn_apply_risk.setObjectName("ActionBtn")

        risk_grid.addWidget(self.chk_allow_live, 0, 0, 1, 2)

        risk_grid.addWidget(QLabel("RISK_PER_TRADE"), 1, 0)
        risk_grid.addWidget(self.spin_risk,           1, 1)

        risk_grid.addWidget(QLabel("MAX_OPEN_POSITIONS"), 2, 0)
        risk_grid.addWidget(self.spin_max_pos,            2, 1)

        risk_grid.addWidget(QLabel("MAX_POSITION_NOTIONAL"), 3, 0)
        risk_grid.addWidget(self.spin_max_notional,           3, 1)

        risk_grid.addWidget(QLabel("MAX_DAILY_DRAWDOWN"), 4, 0)
        risk_grid.addWidget(self.spin_max_dd,            4, 1)

        risk_grid.addWidget(btn_apply_risk, 5, 0, 1, 2)

        layout.addWidget(risk_group)

        def _load_risk_ui_from_config():
            self.chk_allow_live.setChecked(bool(getattr(Config, "ALLOW_LIVE", False)))
            self.spin_risk.setValue(float(getattr(Config, "RISK_PER_TRADE", 0.02)))
            self.spin_max_pos.setValue(float(getattr(Config, "MAX_OPEN_POSITIONS", 5)))
            self.spin_max_notional.setValue(float(getattr(Config, "MAX_POSITION_NOTIONAL", 0.0)))
            self.spin_max_dd.setValue(float(getattr(Config, "MAX_DAILY_DRAWDOWN", 0.05)))

        def _apply_risk_ui_to_config():
            from config import ExecutionMode

            Config.set_runtime("ALLOW_LIVE", bool(self.chk_allow_live.isChecked()))
            Config.set_runtime("RISK_PER_TRADE", float(self.spin_risk.value()))
            Config.set_runtime("MAX_OPEN_POSITIONS", int(self.spin_max_pos.value()))
            Config.set_runtime("MAX_POSITION_NOTIONAL", float(self.spin_max_notional.value()))
            Config.set_runtime("MAX_DAILY_DRAWDOWN", float(self.spin_max_dd.value()))

            # если режим LIVE выбран, а ALLOW_LIVE снят — guard откатит
            # обновим UI режима, чтобы пользователь сразу увидел правду
            self.sync_execution_mode_from_config()

            if hasattr(self, "live_log"):
                self.live_log.append("[RISK] Settings applied & saved to runtime_settings.json")

        btn_apply_risk.clicked.connect(_apply_risk_ui_to_config)
        _load_risk_ui_from_config()

        # 1b) EQUITY CURVE (SESSION) - оригинальный график
        equity_group = QGroupBox("EQUITY CURVE (SESSION)")
        eq_layout = QVBoxLayout(equity_group)

        self.live_equity_plot = pg.PlotWidget()
        self.live_equity_plot.showGrid(x=True, y=True, alpha=0.2)
        self.live_equity_plot.getAxis("left").setWidth(60)
        eq_layout.addWidget(self.live_equity_plot)

        layout.addWidget(equity_group, stretch=1)

        # 2) POSITIONS & ORDERS - с новой таблицей позиций
        tables_container = QWidget()
        tables_layout = QHBoxLayout(tables_container)
        tables_layout.setContentsMargins(0, 0, 0, 0)
        tables_layout.setSpacing(10)

        # Новая таблица позиций с QTableView
        positions_group = QGroupBox("POSITIONS (LIVE)")
        pos_layout = QVBoxLayout(positions_group)
        
        self.tbl_live_positions = QTableView()
        self.model_live_positions = QStandardItemModel()
        self.model_live_positions.setHorizontalHeaderLabels([
            "Broker", "Symbol", "Side", "Qty", "Avg Price",
            "Last Price", "PnL", "PnL %"
        ])
        self.tbl_live_positions.setModel(self.model_live_positions)
        pos_layout.addWidget(self.tbl_live_positions)
        
        tables_layout.addWidget(positions_group)

        # Оригинальная таблица ордеров (оставляем как было)
        orders_group = QGroupBox("ORDERS (LIVE)")
        ord_layout = QVBoxLayout(orders_group)
        
        self.tbl_orders = QTableWidget()
        self.tbl_orders.setColumnCount(7)
        self.tbl_orders.setHorizontalHeaderLabels([
            "Symbol", "Type", "Side", "Price", "Qty", "Status", "Age"
        ])
        self.tbl_orders.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl_orders.setSelectionBehavior(self.tbl_orders.SelectRows)
        self.tbl_orders.setEditTriggers(self.tbl_orders.NoEditTriggers)
        ord_layout.addWidget(self.tbl_orders)
        
        tables_layout.addWidget(orders_group)

        layout.addWidget(tables_container, stretch=2)

        # 3) LIVE EVENT LOG - оригинальный лог
        live_log_group = QGroupBox("LIVE EVENTS")
        log_layout = QVBoxLayout(live_log_group)

        self.live_log = QTextEdit()
        self.live_log.setReadOnly(True)
        log_layout.addWidget(self.live_log)

        layout.addWidget(live_log_group, stretch=1)

        # Первичная инициализация
        self.refresh_live_monitor_snapshot()
        self.live_log.append(
            "Live monitor is not connected to an exchange yet.\n"
            "Wire your ExchangeClient implementation into refresh_live_monitor_snapshot()."
        )

        return tab

    # ==========================================
    # LOGIC
    # ==========================================
    def save_optimizer_settings(self):
        # 1) Собираем настройки текущего профиля
        profile = {
            "sl_min": self.spin_sl_min.value(), "sl_max": self.spin_sl_max.value(),
            "tp_min": self.spin_tp_min.value(), "tp_max": self.spin_tp_max.value(),
            "pullback_min": self.spin_pull_min.value(), "pullback_max": self.spin_pull_max.value(),
            "conf_min": self.spin_conf_min.value(), "conf_max": self.spin_conf_max.value(),
            "trail_act_min": 1.2, "trail_act_max": 2.5,
            "max_hold_min": 24, "max_hold_max": 72,
        }

        # Окна WALK-FORWARD
        if hasattr(self, "wfo_widget"):
            train_bars, test_bars = self.wfo_widget.get_values()
            profile["train_window"] = int(train_bars)
            profile["test_window"] = int(test_bars)

        # 2) Определяем ключ профиля по текущему юниверсу
        try:
            mode = getattr(self, "current_universe_mode", Config.UNIVERSE_MODE)
            profile_key = getattr(mode, "value", "both")
        except Exception:
            profile_key = "both"

        # 3) Читаем существующий файл (если есть)
        data = {}
        if os.path.exists("optimizer_settings.json"):
            try:
                with open("optimizer_settings.json", "r") as f:
                    data = json.load(f)
                    if not isinstance(data, dict):
                        data = {}
            except Exception:
                data = {}

        # 4) Обратная совместимость: старый формат (плоский словарь)
        if "sl_min" in data or "tp_min" in data:
            # Заворачиваем старый формат в профиль 'both'
            data = {"both": data}

        # 5) Обновляем профиль текущего юниверса
        if profile_key not in data or not isinstance(data[profile_key], dict):
            data[profile_key] = {}
        data[profile_key].update(profile)

        # 6) Сохраняем override профиля оптимизатора (чтобы не потерялся между сессиями)
        data["optimizer_profile_override"] = self.optimizer_profile_mode

        try:
            with open("optimizer_settings.json", "w") as f:
                json.dump(data, f, indent=4)
            print(f"✅ Configuration saved for profile '{profile_key}' in optimizer_settings.json")
        except Exception as e:
            print(f"❌ Save Error: {e}")

        # 7) Обновляем ENV для Telegram HTF.
        if hasattr(self, "chk_tg_crypto"):
            os.environ["USE_TG_CRYPTO"] = "1" if self.chk_tg_crypto.isChecked() else "0"
        if hasattr(self, "chk_tg_stocks"):
            os.environ["USE_TG_STOCKS"] = "1" if self.chk_tg_stocks.isChecked() else "0"

        # 8) Обновляем ENV + Config для лидеров рынка.
        if hasattr(self, "chk_leader_crypto") and hasattr(self, "cbo_leader_crypto"):
            use_leader_crypto = self.chk_leader_crypto.isChecked()
            use_leader_stocks = self.chk_leader_stocks.isChecked()

            sym_leader_crypto = self.cbo_leader_crypto.currentText().strip()
            sym_leader_stocks = self.cbo_leader_stocks.currentText().strip()

            # NONE → фактически выключаем лидера
            if sym_leader_crypto.upper() == "NONE":
                use_leader_crypto = False
            if sym_leader_stocks.upper() == "NONE":
                use_leader_stocks = False

            # Обновляем Config (чтобы текущий процесс видел новые значения)
            Config.USE_LEADER_CRYPTO = use_leader_crypto
            Config.USE_LEADER_STOCKS = use_leader_stocks
            os.environ["USE_LEADER_CRYPTO"] = "1" if use_leader_crypto else "0"
            os.environ["USE_LEADER_STOCKS"] = "1" if use_leader_stocks else "0"

            if use_leader_crypto and sym_leader_crypto.upper() != "NONE":
                Config.LEADER_SYMBOL_CRYPTO = sym_leader_crypto
                os.environ["LEADER_SYMBOL_CRYPTO"] = sym_leader_crypto
            if use_leader_stocks and sym_leader_stocks.upper() != "NONE":
                Config.LEADER_SYMBOL_EQUITY = sym_leader_stocks
                os.environ["LEADER_SYMBOL_EQUITY"] = sym_leader_stocks

            # (опционально) сохраняем в профиль для будущего использования
            data[profile_key]["use_leader_crypto"] = use_leader_crypto
            data[profile_key]["use_leader_stocks"] = use_leader_stocks
            data[profile_key]["leader_symbol_crypto"] = getattr(Config, "LEADER_SYMBOL_CRYPTO", "BTCUSDT")
            data[profile_key]["leader_symbol_stocks"] = getattr(Config, "LEADER_SYMBOL_EQUITY", "MOEX")
            

    def load_optimizer_settings(self):
        if not os.path.exists("optimizer_settings.json"):
            # Если файла нет — хотя бы подпись профиля обновим по текущему effective-профилю
            try:
                if hasattr(self, "lbl_optimizer_profile"):
                    effective = self._get_effective_optimizer_profile()
                    self.lbl_optimizer_profile.setText(f"Optimizer profile: {effective.upper()}")
            except Exception:
                pass
            return

        try:
            with open("optimizer_settings.json", "r") as f:
                data = json.load(f)
        except Exception:
            return

        # --- NEW: восстановление override-профиля оптимизатора ---
        if isinstance(data, dict):
            override = data.get("optimizer_profile_override")
            if override in ("crypto", "stocks", "both", "auto"):
                # сохраняем режим в окне
                self.optimizer_profile_mode = override
                # и сразу прокидываем в ENV, чтобы optimizer.py его увидел
                os.environ["OPTIMIZER_PROFILE"] = override
                # синхронизируем комбобокс, если он уже создан
                if hasattr(self, "cbo_optimizer_profile"):
                    self.cbo_optimizer_profile.setCurrentText(override.upper())

        # Определяем текущий юниверс (для выбора профиля настроек из файла)
        try:
            mode = getattr(self, "current_universe_mode", Config.UNIVERSE_MODE)
            profile_key = getattr(mode, "value", "both")
        except Exception:
            profile_key = "both"

        # Старый формат (плоский словарь)
        if isinstance(data, dict) and ("sl_min" in data or "tp_min" in data):
            s = data
        else:
            # Новый формат: словарь профилей
            if not isinstance(data, dict):
                return
            s = data.get(profile_key) or data.get("both") or {}

        # Обновляем спины
        self.spin_sl_min.setValue(s.get("sl_min", 1.5))
        self.spin_sl_max.setValue(s.get("sl_max", 2.5))
        self.spin_tp_min.setValue(s.get("tp_min", 3.0))
        self.spin_tp_max.setValue(s.get("tp_max", 6.0))
        self.spin_pull_min.setValue(s.get("pullback_min", 0.0))
        self.spin_pull_max.setValue(s.get("pullback_max", 0.15))
        self.spin_conf_min.setValue(s.get("conf_min", 0.65))
        self.spin_conf_max.setValue(s.get("conf_max", 0.85))

        # Восстанавливаем слайдеры WFO
        if hasattr(self, "wfo_widget"):
            default_train, default_test = self.wfo_widget.get_values()
            self.wfo_widget.slider_train.setValue(int(s.get("train_window", default_train)))
            self.wfo_widget.slider_test.setValue(int(s.get("test_window", default_test)))
            self.wfo_widget.update_labels()

        # Обновляем подпись активного профиля (с учётом режима AUTO)
        if hasattr(self, "lbl_optimizer_profile"):
            effective = self._get_effective_optimizer_profile()
            self.lbl_optimizer_profile.setText(f"Optimizer profile: {effective.upper()}")

    def run_optimizer_with_save(self):
        self.save_optimizer_settings()
        self.run_script("optimizer.py", ["--mode", "sniper"])

    def refresh_live_monitor_snapshot(self):
        """
        Обновляет LIVE MONITOR, используя ExecutionRouter:
        - агрегированный equity/balance и PnL по брокерам;
        - список открытых позиций;
        - обновляет equity-кривую за сессию.

        Все async-вызовы идут через self._async_loop, чтобы не блокировать GUI.
        """
        import asyncio
        from config import ExecutionMode

        # 0) Если роутера нет – показываем DISCONNECTED и чистим UI
        router = getattr(self, "execution_router", None)
        if router is None:
            if hasattr(self, "lbl_live_status"):
                self.lbl_live_status.setText("LIVE MONITOR v1 — DISCONNECTED")
            if hasattr(self, "lbl_live_equity"):
                self.lbl_live_equity.setText("Total Equity: —")
            if hasattr(self, "lbl_live_pnl"):
                self.lbl_live_pnl.setText("Total PnL: —")
            if hasattr(self, "lbl_live_bitget"):
                self.lbl_live_bitget.setText("Bitget: —")
            if hasattr(self, "lbl_live_tinkoff"):
                self.lbl_live_tinkoff.setText("Tinkoff: —")
            if hasattr(self, "model_live_positions"):
                self.model_live_positions.removeRows(
                    0, self.model_live_positions.rowCount()
                )
            if hasattr(self, "tbl_orders"):
                self.tbl_orders.setRowCount(0)
            if hasattr(self, "live_equity_plot"):
                self.live_equity_plot.clear()
            return

        # 1) В режиме BACKTEST монитор не трогаем
        mode_obj = getattr(Config, "EXECUTION_MODE", ExecutionMode.BACKTEST)
        if isinstance(mode_obj, ExecutionMode):
            mode = mode_obj
        else:
            try:
                mode = ExecutionMode(mode_obj)
            except Exception:
                mode = ExecutionMode.BACKTEST

        if mode == ExecutionMode.BACKTEST:
            return

        # 2) Проверяем event loop
        loop = getattr(self, "_async_loop", None)
        if loop is None:
            return

        # 3) Однократная инициализация брокеров
        if not getattr(self, "_router_initialized", False):
            try:
                fut_init = asyncio.run_coroutine_threadsafe(
                    router.initialize(),
                    loop,
                )
                fut_init.result(timeout=10.0)
                self._router_initialized = True
            except Exception as e:
                if hasattr(self, "lbl_live_status"):
                    self.lbl_live_status.setText("LIVE MONITOR v1 — ERROR (init)")
                if hasattr(self, "live_log"):
                    self.live_log.append(f"[ERROR] Router init failed: {e}")
                return

        # 4) Тянем агрегированное состояние счёта
        try:
            fut_state = asyncio.run_coroutine_threadsafe(
                router.get_global_account_state(),
                loop,
            )
            state = fut_state.result(timeout=5.0)
        except Exception as e:
            if hasattr(self, "lbl_live_status"):
                self.lbl_live_status.setText("LIVE MONITOR v1 — ERROR (state)")
            if hasattr(self, "lbl_live_equity"):
                self.lbl_live_equity.setText("Total Equity: ERR")
            if hasattr(self, "lbl_live_pnl"):
                self.lbl_live_pnl.setText("Total PnL: ERR")
            if hasattr(self, "lbl_live_bitget"):
                self.lbl_live_bitget.setText("Bitget: ERR")
            if hasattr(self, "lbl_live_tinkoff"):
                self.lbl_live_tinkoff.setText("Tinkoff: ERR")
            if hasattr(self, "model_live_positions"):
                self.model_live_positions.removeRows(
                    0, self.model_live_positions.rowCount()
                )
            if hasattr(self, "tbl_orders"):
                self.tbl_orders.setRowCount(0)
            if hasattr(self, "live_log"):
                self.live_log.append(f"[ERROR] Failed to refresh account snapshot: {e}")
            return

        # ---- 4a. Агрегаты по счёту ----
        total_equity = float(getattr(state, "equity", 0.0) or 0.0)

        total_upnl = 0.0
        bitget_eq = bitget_upnl = 0.0
        tink_eq = tink_upnl = 0.0

        details = getattr(state, "details", {}) or {}

        for name, st in details.items():
            eq = float(getattr(st, "equity", 0.0) or 0.0)
            upnl = float(getattr(st, "unrealized_pnl", 0.0) or 0.0)
            total_upnl += upnl

            lname = str(name).lower()
            if lname.startswith("bitget"):
                bitget_eq += eq
                bitget_upnl += upnl
            if lname.startswith("tinkoff"):
                tink_eq += eq
                tink_upnl += upnl

        # ---- 4b. Обновляем верхние лейблы ----
        if hasattr(self, "lbl_live_status"):
            self.lbl_live_status.setText("LIVE MONITOR v1 — CONNECTED")

        if hasattr(self, "lbl_live_equity"):
            self.lbl_live_equity.setText(f"Total Equity: {total_equity:,.2f}")

        if hasattr(self, "lbl_live_pnl"):
            self.lbl_live_pnl.setText(f"Total PnL: {total_upnl:,.2f}")

        if hasattr(self, "lbl_live_bitget"):
            if bitget_eq > 0.0 or bitget_upnl != 0.0:
                self.lbl_live_bitget.setText(
                    f"Bitget: eq={bitget_eq:,.2f}, uPnL={bitget_upnl:,.2f}"
                )
            else:
                self.lbl_live_bitget.setText("Bitget: —")

        if hasattr(self, "lbl_live_tinkoff"):
            if tink_eq > 0.0 or tink_upnl != 0.0:
                self.lbl_live_tinkoff.setText(
                    f"Tinkoff: eq={tink_eq:,.2f}, uPnL={tink_upnl:,.2f}"
                )
            else:
                self.lbl_live_tinkoff.setText("Tinkoff: —")

        # 5) Equity-кривая за сессию
        try:
            if not hasattr(self, "live_equity_history"):
                self.live_equity_history = []
            t_idx = len(self.live_equity_history)
            self.live_equity_history.append((t_idx, total_equity))

            if hasattr(self, "live_equity_plot") and len(self.live_equity_history) >= 2:
                xs = np.array([t for (t, _) in self.live_equity_history], dtype=float)
                ys = np.array([v for (_, v) in self.live_equity_history], dtype=float)
                self.live_equity_plot.clear()
                self.live_equity_plot.plot(xs, ys, pen=pg.mkPen('#26a69a', width=2))
        except Exception as e:
            print(f"[LIVE] Failed to update equity curve: {e}")

        # 6) Позиции → POSITIONS (LIVE)
        try:
            fut_pos = asyncio.run_coroutine_threadsafe(
                router.list_all_positions(),
                loop,
            )
            positions = fut_pos.result(timeout=5.0)
        except Exception as e:
            if hasattr(self, "model_live_positions"):
                self.model_live_positions.removeRows(
                    0, self.model_live_positions.rowCount()
                )
            if hasattr(self, "tbl_orders"):
                self.tbl_orders.setRowCount(0)
            if hasattr(self, "live_log"):
                self.live_log.append(f"[ERROR] Failed to fetch positions: {e}")
            return

        if hasattr(self, "model_live_positions"):
            # очистить таблицу
            self.model_live_positions.removeRows(
                0, self.model_live_positions.rowCount()
            )

            from PyQt5.QtGui import QColor, QStandardItem

            for p in positions or []:
                symbol = getattr(p, "symbol", "")
                broker_name = getattr(p, "broker", "")
                qty = float(getattr(p, "quantity", 0.0) or 0.0)
                avg_price = float(getattr(p, "avg_price", 0.0) or 0.0)
                last_price = float(getattr(p, "last_price", avg_price) or 0.0)
                upnl = float(getattr(p, "unrealized_pnl", 0.0) or 0.0)

                side = "LONG" if qty > 0 else "SHORT"
                size_abs = abs(qty)

                if avg_price > 0 and size_abs > 0:
                    try:
                        upnl_pct = (upnl / (avg_price * size_abs)) * 100.0
                    except ZeroDivisionError:
                        upnl_pct = 0.0
                else:
                    upnl_pct = 0.0

                color = None
                if upnl > 0:
                    color = QColor("#26a69a")
                elif upnl < 0:
                    color = QColor("#ef5350")

                row_values = [
                    str(broker_name),
                    str(symbol),
                    side,
                    f"{size_abs:.4f}",
                    f"{avg_price:,.4f}",
                    f"{last_price:,.4f}",
                    f"{upnl:,.2f}",
                    f"{upnl_pct:,.2f}%",
                ]

                items = [QStandardItem(text) for text in row_values]
                if color is not None and len(items) >= 8:
                    # Подсветим PnL и PnL %
                    items[6].setForeground(color)
                    items[7].setForeground(color)

                self.model_live_positions.appendRow(items)

        # 7) Таблица ордеров — пока заглушка
        if hasattr(self, "tbl_orders"):
            self.tbl_orders.setRowCount(0)

        # 8) Краткий лог-снимок
        if hasattr(self, "live_log"):
            self.live_log.append(
                f"[SNAPSHOT] Equity: {total_equity:,.2f} | uPnL: {total_upnl:,.2f} | "
                f"Positions: {len(positions) if positions is not None else 0}"
            )

    def get_selected_assets(self) -> list[str]:
        """
        Единая точка выбора списка тикеров
        для тестов (backtest/debug_replayer) и реальной торговли.

        Опирается на текущий режим юниверса:
        - crypto      -> только крипта
        - stocks      -> только МОЕХ / Тинькофф
        - both        -> совместный портфель
        """
        # Берём локальное поле, если есть, иначе — из Config
        mode = getattr(self, "current_universe_mode", Config.UNIVERSE_MODE)

        # Используем вспомогательную функцию из config.py
        assets = get_assets_for_universe(mode)

        print(f"[GUI] get_selected_assets: mode={mode.value}, n={len(assets)}")
        return assets

    def refresh_asset_combo(self):
        """
        Перезаполняет комбобокс в WAR ROOM под текущий юниверс.
        Если asset_combo ещё не создан (например, вкладка не открыта) — тихо выходим.
        """
        if not hasattr(self, "asset_combo"):
            return

        symbols = self.get_selected_assets()

        self.asset_combo.blockSignals(True)
        self.asset_combo.clear()
        self.asset_combo.addItems(symbols)
        self.asset_combo.blockSignals(False)

        # Обновляем график, если данные уже загружены
        self.update_chart()

    def sync_execution_mode_from_config(self):
        """
        Читает Config.EXECUTION_MODE и синхронизирует:
          - radio-кнопки
          - label
          - live_timer (автообновление мониторинга)
        """
        from config import Config, ExecutionMode

        mode_obj = getattr(Config, "EXECUTION_MODE", ExecutionMode.BACKTEST)
        mode = mode_obj.value if isinstance(mode_obj, ExecutionMode) else str(mode_obj).lower()
        mode = mode.lower()

        if hasattr(self, "radio_mode_backtest"):
            if mode == "paper":
                self.radio_mode_paper.setChecked(True)
            elif mode == "live":
                self.radio_mode_live.setChecked(True)
            else:
                self.radio_mode_backtest.setChecked(True)

        if hasattr(self, "lbl_mode_status"):
            self.lbl_mode_status.setText(f"Current mode: {mode.upper()}")

        # Авто-обновление Live Monitor только в paper/live
        if mode in ("paper", "live"):
            if not self.live_timer.isActive():
                self.live_timer.start()
        else:
            if self.live_timer.isActive():
                self.live_timer.stop()

    def on_execution_mode_changed(self):
        """
        Хэндлер radio-кнопок: обновляет Config.EXECUTION_MODE, ENV и таймер.
        """
        from config import Config, ExecutionMode

        if not (hasattr(self, "radio_mode_backtest") and hasattr(self, "radio_mode_paper")):
            return  # UI ещё не готов

        if self.radio_mode_backtest.isChecked():
            mode = "backtest"
        elif self.radio_mode_paper.isChecked():
            mode = "paper"
        elif self.radio_mode_live.isChecked():
            mode = "live"
        else:
            mode = "backtest"

        try:
            enum_val = ExecutionMode(mode)
        except Exception:
            enum_val = ExecutionMode.BACKTEST
            mode = "backtest"

        Config.EXECUTION_MODE = enum_val
        os.environ["EXECUTION_MODE"] = enum_val.value

        if hasattr(self, "lbl_mode_status"):
            self.lbl_mode_status.setText(f"Current mode: {enum_val.value.upper()}")

        # Запускаем / останавливаем автообновление LIVE MONITOR
        if mode in ("paper", "live"):
            if not self.live_timer.isActive():
                self.live_timer.start()
        else:
            if self.live_timer.isActive():
                self.live_timer.stop()

        print(f"[MODE] Execution mode set to {enum_val.value}")
        if hasattr(self, "live_log"):
            self.live_log.append(f"[MODE] Execution mode switched to {enum_val.value}")

    def on_start_trading_clicked(self):
        """
        Запускает live-цикл AsyncStrategyRunner в отдельном asyncio-loop'е.
        Разрешено только в режимах PAPER / LIVE.
        """
        from config import ExecutionMode

        mode_obj = getattr(Config, "EXECUTION_MODE", ExecutionMode.BACKTEST)
        mode = mode_obj.value if isinstance(mode_obj, ExecutionMode) else str(mode_obj).lower()
        mode = mode.lower()

        if mode == "backtest":
            msg = "[TRADING] EXECUTION_MODE=backtest — live-сессию не запускаем. Переключись на PAPER или LIVE."
            print(msg)
            if hasattr(self, "live_log"):
                self.live_log.append(msg)
            return

        if self.trading_session_active:
            print("[TRADING] Session already active.")
            return

        assets = self.get_selected_assets()
        if not assets:
            print("[TRADING] No assets selected — abort.")
            if hasattr(self, "live_log"):
                self.live_log.append("[TRADING] No assets selected.")
            return

        print(f"[TRADING] Starting session with universe={Config.UNIVERSE_MODE.value}, assets={assets}")

        # Создаем раннер, если его ещё нет
        if self.live_trader is None:
            self.live_trader = AsyncStrategyRunner()

        # Обновляем фильтр по активам
        self.live_trader.set_assets(assets)

        if not hasattr(self, "_async_loop") or self._async_loop is None:
            print("[TRADING] Async loop is not initialized.")
            return

        import asyncio

        async def _runner_main():
            try:
                await self.live_trader.initialize()
                await self.live_trader.run_forever()
            except asyncio.CancelledError:
                print("[TRADING] Live trader cancelled.")
            except Exception as e:
                print(f"[TRADING] Live trader error: {type(e).__name__}: {e}")

        # Запускаем корутину в фоне
        self.live_trader_task = asyncio.run_coroutine_threadsafe(
            _runner_main(),
            self._async_loop,
        )

        self.trading_session_active = True

        if hasattr(self, "live_log"):
            self.live_log.append("[TRADING] Session started (AsyncStrategyRunner running).")
        print("[TRADING] Session started.")

    def on_stop_trading_clicked(self):
        """
        Остановка торговой сессии:
        - просим AsyncStrategyRunner остановиться;
        - отменяем задачу;
        - оставляем ExecutionRouter живым (монитор можно крутить дальше).
        """
        if not self.trading_session_active:
            print("[TRADING] No active session.")
            if hasattr(self, "live_log"):
                self.live_log.append("[TRADING] No active session.")
            return

        print("[TRADING] Stopping session...")

        if self.live_trader is not None:
            try:
                self.live_trader.request_stop()
            except AttributeError:
                pass

        if self.live_trader_task is not None:
            self.live_trader_task.cancel()
            self.live_trader_task = None

        self.trading_session_active = False

        if hasattr(self, "live_log"):
            self.live_log.append("[TRADING] Session stopped.")
        print("[TRADING] Session stopped.")

    def _build_wfo_cli(self):
        """
        Формирует CLI-аргументы вида:
            --train_window N --trade_window M
        исходя из значений слайдеров.
        """
        if hasattr(self, "wfo_widget"):
            train_bars, test_bars = self.wfo_widget.get_values()
            return [
                "--train_window", str(int(train_bars)),
                "--trade_window", str(int(test_bars)),
            ]
        return []

    def run_walk_generator(self):
        args = ["--mode", "walk", "--reset"]
        args += self._build_wfo_cli()
        self.run_script("signal_generator.py", args)

    def run_universal_generator(self):
        """
        Запуск Universal Brain из GUI.

        BACKTEST  -> старый sync-режим (только генерация сигналов).
        PAPER/LIVE -> async-режим с ExecutionRouter и брокером.
        """
        from config import ExecutionMode  # локальный импорт, чтобы не тянуть наверх

        # Базовые аргументы, как раньше
        args = ["--mode", "universal", "--preset", "grinder", "--cross_asset_wf"]
        args += self._build_wfo_cli()

        # Определяем текущий режим исполнения
        mode_obj = getattr(Config, "EXECUTION_MODE", ExecutionMode.BACKTEST)

        # По умолчанию работаем в синхронном backtest-режиме
        use_async = isinstance(mode_obj, ExecutionMode) and mode_obj in (
            ExecutionMode.PAPER,
            ExecutionMode.LIVE,
        )

        if use_async:
            # Берём брокера из конфига (если нет — падаем на bitget)
            broker_name = getattr(Config, "DEFAULT_BROKER", "bitget")

            args += [
                "--async_mode",
                "--broker", broker_name,
                # Параметры ниже можно не передавать, т.к. в скрипте есть дефолты.
                # "--portfolio_size", "10",
                # "--risk_level", "0.02",
            ]

            print(
                f"[GUI] Universal generator: ASYNC mode via broker={broker_name} "
                f"(execution_mode={mode_obj.value})"
            )
        else:
            print("[GUI] Universal generator: SYNC BACKTEST mode")

        self.run_script("signal_generator.py", args)

    # --- НОВОЕ: кнопки Train Crypto / Train Stocks Brain ---

    def run_universal_crypto_brain(self):
        """
        Быстрый хелпер:
        - переключает юниверс на CRYPTO через радиокнопку,
        - запускает универсальный мозг.
        """
        if hasattr(self, "radio_universe_crypto"):
            self.radio_universe_crypto.setChecked(True)
        self.run_universal_generator()

    def run_universal_stocks_brain(self):
        """
        Быстрый хелпер:
        - переключает юниверс на STOCKS через радиокнопку,
        - запускает универсальный мозг.
        """
        if hasattr(self, "radio_universe_stocks"):
            self.radio_universe_stocks.setChecked(True)
        self.run_universal_generator()

    def run_script(self, script_name, args):
        self.console.clear()
        print(f"--- STARTING {script_name} ---")
        self.workers['util'] = UtilityWorker(script_name, args)
        self.workers['util'].finished.connect(lambda: print(f"--- FINISHED {script_name} ---"))
        self.workers['util'].start()

    def load_backtest_data(self):
        self.lbl_status.setText("Status: Loading Data...")

        assets = self.get_selected_assets()

        self.workers['loader'] = BacktestLoader(assets, Config.LEADER_SYMBOL)
        self.workers['loader'].data_loaded.connect(self.on_data_ready)
        self.workers['loader'].error_occurred.connect(
            lambda e: self.lbl_status.setText(f"Error: {e}")
        )
        self.workers['loader'].start()

    def on_data_ready(self, data):
        self.data_store = data
        self.lbl_status.setText(f"Status: Data Ready ({len(data)} assets)")
        self.update_chart()

    def update_chart(self):
        if not hasattr(self, 'data_store'):
            return
        sym = self.asset_combo.currentText()
        if sym not in self.data_store:
            return
        df = self.data_store[sym]

        self.plot_widget.clear()
        self.prob_plot.clear()
        if hasattr(self, "ind_plot"):
            self.ind_plot.clear()

        if df.empty:
            return

        dates = df.index.tolist()
        self.date_axis.dates = dates
        self.prob_axis.dates = dates
        if hasattr(self, "ind_axis"):
            self.ind_axis.dates = dates

        # --- Свечи ---
        candles = []
        for i in range(len(df)):
            candles.append((
                float(i),
                float(df['open'].iloc[i]),
                float(df['close'].iloc[i]),
                float(df['low'].iloc[i]),
                float(df['high'].iloc[i]),
            ))
        candlestick_item = CandlestickItem(candles)
        self.plot_widget.addItem(candlestick_item)

        # --- Вероятности ---
        if 'p_long' in df.columns and 'p_short' in df.columns:
            x = np.arange(len(df))
            self.prob_plot.plot(x, df['p_long'].values, pen=pg.mkPen('#26a69a', width=1), name="Long Prob")
            self.prob_plot.plot(x, df['p_short'].values, pen=pg.mkPen('#ef5350', width=1), name="Short Prob")
            self.prob_plot.addItem(pg.InfiniteLine(
                pos=0.75, angle=0, pen=pg.mkPen('#666666', width=1, style=Qt.DashLine)
            ))

        # --- NEW: ATR(14) на отдельной панели ---
        try:
            high = df['high'].astype(float)
            low = df['low'].astype(float)
            close = df['close'].astype(float)

            tr1 = (high - low).abs()
            tr2 = (high - close.shift(1)).abs()
            tr3 = (low - close.shift(1)).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

            atr = tr.rolling(14).mean()

            if hasattr(self, "ind_plot") and atr.notna().any():
                x = np.arange(len(df))
                self.ind_plot.plot(x, atr.values, pen=pg.mkPen('#ffaa00', width=1), name="ATR(14)")
        except Exception as e:
            print(f"[WAR ROOM] ATR calc failed for {sym}: {e}")

if __name__ == "__main__":
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    window = FundManagerWindow()
    window.show()
    sys.exit(app.exec_())