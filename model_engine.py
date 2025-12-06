# model_engine.py
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
import os
import warnings
from scipy.stats import mode
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator, ClassifierMixin
from config import Config

warnings.filterwarnings("ignore")

print("✅ MODEL ENGINE v2.4 LOADED (CPU TURBO: Hist + MultiCore)")

class ConstantModel(BaseEstimator, ClassifierMixin):
    def __init__(self, constant_class_id):
        self.constant_class_id = int(constant_class_id)
        self.classes_ = np.array([self.constant_class_id])

    def predict_proba(self, X):
        n_samples = X.shape[0]
        return np.ones((n_samples, 1), dtype=np.float64)

    def fit(self, X, y, sample_weight=None):
        return self

class QuantileRegimeSelector:
    """
    Advanced Regime Selector v3.0
    """
    def __init__(self, preset_name='auto', smoothing_window=3):
        self.smoothing_window = smoothing_window
        self.preset_name = preset_name
        
        self.presets = {
            'classic': {'atr_q': [0.40, 0.90], 'adx_q': 0.60},
            'sniper':  {'atr_q': [0.30, 0.85], 'adx_q': 0.70},
            'grinder': {'atr_q': [0.50, 0.95], 'adx_q': 0.50},
            'volatile':{'atr_q': [0.60, 0.98], 'adx_q': 0.55}
        }
        self.atr_thresholds = [0.0, 0.0]
        self.adx_threshold = 0.0
        self.active_params = {}

    def _auto_select_preset(self, df):
        if 'atr_rel' not in df.columns: return self.presets['classic']
        med_vol = df['atr_rel'].median()
        max_vol = df['atr_rel'].quantile(0.95)
        if med_vol > 0.02: return self.presets['volatile']
        elif max_vol < 0.01: return self.presets['sniper']
        else: return self.presets['classic']

    def fit(self, df: pd.DataFrame):
        if self.preset_name == 'auto':
            self.active_params = self._auto_select_preset(df)
        else:
            self.active_params = self.presets.get(self.preset_name, self.presets['classic'])
            
        if 'atr_rel' not in df.columns:
            df['atr_rel'] = df['atr'] / df['close']
        
        metric_vol = df['atr_rel'].replace([np.inf, -np.inf], np.nan).dropna()
        metric_trend = df['adx'] if 'adx' in df.columns else pd.Series(0, index=df.index)
        metric_trend = metric_trend.replace([np.inf, -np.inf], np.nan).dropna()
        
        if metric_vol.empty: return

        self.atr_thresholds[0] = metric_vol.quantile(self.active_params['atr_q'][0])
        self.atr_thresholds[1] = metric_vol.quantile(self.active_params['atr_q'][1])
        self.adx_threshold = metric_trend.quantile(self.active_params['adx_q'])

    def predict(self, df: pd.DataFrame, smooth=True) -> np.ndarray:
        if df.empty: return np.array([])
        vol = df['atr_rel'] if 'atr_rel' in df.columns else (df['atr'] / df['close'])
        trend = df['adx'] if 'adx' in df.columns else pd.Series(0, index=df.index)
        vol = vol.fillna(0).values
        trend = trend.fillna(0).values
        
        raw_regimes = np.zeros(len(df), dtype=int)
        
        is_panic = vol > self.atr_thresholds[1]
        raw_regimes[is_panic] = 2
        
        is_mid_vol = (vol > self.atr_thresholds[0]) & (vol <= self.atr_thresholds[1])
        is_strong_trend = trend > self.adx_threshold
        is_trend = is_mid_vol & is_strong_trend
        raw_regimes[is_trend] = 1
        
        if not smooth: return raw_regimes

        final_regimes = pd.Series(raw_regimes).rolling(window=self.smoothing_window, center=False).apply(
            lambda x: mode(x, keepdims=True)[0][0], raw=True
        ).fillna(0).astype(int).values
        return final_regimes

class MLEngine:
    def __init__(self, model_dir: str | None = None, regime_preset: str = 'auto'):
        self.model_dir = model_dir
        self.models = {} 
        self.scalers = {} 
        self.encoders = {}
        self.regime_selector = QuantileRegimeSelector(preset_name=regime_preset, smoothing_window=5)
        
        if self.model_dir:
            os.makedirs(self.model_dir, exist_ok=True)
            self._load_state()

    def _get_path(self, name):
        return os.path.join(self.model_dir, f"{name}.pkl")

    def save_state(self):
        if not self.model_dir: return
        joblib.dump(self.regime_selector, self._get_path("selector"))
        for r, m in self.models.items():
            joblib.dump(m, self._get_path(f"model_{r}"))
            joblib.dump(self.scalers[r], self._get_path(f"scaler_{r}"))
            if r in self.encoders:
                joblib.dump(self.encoders[r], self._get_path(f"encoder_{r}"))

    def _load_state(self):
        if not self.model_dir: return
        try:
            sel_path = self._get_path("selector")
            if os.path.exists(sel_path):
                self.regime_selector = joblib.load(sel_path)
            
            import glob
            model_files = glob.glob(os.path.join(self.model_dir, "model_*.pkl"))
            for f in model_files:
                base = os.path.basename(f)
                try:
                    r_id = int(base.split('_')[1].split('.')[0])
                    self.models[r_id] = joblib.load(f)
                    s_path = self._get_path(f"scaler_{r_id}")
                    if os.path.exists(s_path):
                        self.scalers[r_id] = joblib.load(s_path)
                    e_path = self._get_path(f"encoder_{r_id}")
                    if os.path.exists(e_path):
                        self.encoders[r_id] = joblib.load(e_path)
                except: continue
        except Exception: pass

    def train(self, df: pd.DataFrame, feature_cols: list[str], target_col: str = 'target'):
        self.regime_selector.fit(df)
        df['regime'] = self.regime_selector.predict(df, smooth=True)
        
        split_idx = int(len(df) * 0.85)
        df_train = df.iloc[:split_idx]
        df_calib = df.iloc[split_idx:]
        
        unique_regimes = sorted(df['regime'].unique())
        
        for r in unique_regimes:
            mask_train = df_train['regime'] == r
            if mask_train.sum() < 30: continue 
            
            X_train = df_train.loc[mask_train, feature_cols].values
            y_train = df_train.loc[mask_train, target_col].values
            
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            self.scalers[r] = scaler
            
            unique_y = np.unique(y_train)
            
            if len(unique_y) < 2:
                self.models[r] = ConstantModel(unique_y[0])
                continue

            le = LabelEncoder()
            y_train_encoded = le.fit_transform(y_train)
            self.encoders[r] = le
            
            # --- CPU TURBO CONFIGURATION ---
            clf = xgb.XGBClassifier(
                n_estimators=Config.MODEL_N_ESTIMATORS,
                max_depth=Config.MODEL_MAX_DEPTH, 
                learning_rate=Config.MODEL_LEARNING_RATE,
                objective='multi:softprob',
                num_class=len(unique_y),
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-4,             # ИСПОЛЬЗУЕМ 16+ ядер
                reg_lambda=1.5,
                tree_method='hist',    # ОЧЕНЬ БЫСТРЫЙ МЕТОД ДЛЯ CPU
                device='cpu'           # ЯВНОЕ УКАЗАНИЕ CPU
            )
            
            try:
                clf.fit(X_train_scaled, y_train_encoded)
                model_to_save = clf
                
                mask_calib = df_calib['regime'] == r
                if mask_calib.sum() > 20:
                    X_calib = df_calib.loc[mask_calib, feature_cols].values
                    y_calib = df_calib.loc[mask_calib, target_col].values
                    
                    valid_mask = np.isin(y_calib, le.classes_)
                    if valid_mask.sum() > 10:
                        X_calib = X_calib[valid_mask]
                        y_calib = y_calib[valid_mask]
                        
                        X_calib_scaled = scaler.transform(X_calib)
                        y_calib_encoded = le.transform(y_calib)
                        
                        calib_clf = CalibratedClassifierCV(clf, method='isotonic', cv='prefit')
                        calib_clf.fit(X_calib_scaled, y_calib_encoded)
                        model_to_save = calib_clf
                
                self.models[r] = model_to_save
                
            except Exception as e:
                print(f"⚠️ Ошибка обучения режима {r}: {e}")
                mode_y = mode(y_train, keepdims=True)[0][0]
                self.models[r] = ConstantModel(mode_y)

        self.save_state()

    def predict_batch(self, df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
        if df.empty: return None, None
        
        regimes = self.regime_selector.predict(df, smooth=True)
        final_probs = np.zeros((len(df), 3)) 
        unique_regimes = np.unique(regimes)
        
        for r in unique_regimes:
            if r not in self.models: continue
            
            mask = (regimes == r)
            X_chunk = df.loc[mask, feature_cols].values
            
            scaler = self.scalers.get(r)
            if not scaler: continue
            X_scaled = scaler.transform(X_chunk)
            
            model = self.models[r]
            le = self.encoders.get(r)
            
            try:
                probs = model.predict_proba(X_scaled)
                
                if isinstance(model, ConstantModel):
                    class_id = int(model.classes_[0])
                    if class_id < 3: final_probs[mask, class_id] = 1.0
                
                elif le is not None:
                    for i, original_class in enumerate(le.classes_):
                        if original_class < 3:
                            if i < probs.shape[1]:
                                final_probs[mask, int(original_class)] = probs[:, i]
                
                else:
                    if probs.shape[1] == 3:
                        final_probs[mask] = probs
                    
            except Exception as e:
                pass
                
        return final_probs, regimes

    def predict_proba(self, df: pd.DataFrame, feature_cols: list[str], timestamp) -> np.ndarray | None:
        # 1. Базовые проверки
        if not self.models: return None
        if timestamp not in df.index: return None
        
        # 2. Получение данных и определение режима
        try:
            row = df.loc[[timestamp]]
            regime = self.regime_selector.predict(row)[0]
        except Exception:
            return None
        
        # 3. Загрузка артефактов модели
        model = self.models.get(regime)
        scaler = self.scalers.get(regime)
        le = self.encoders.get(regime) 
        
        if model is None or scaler is None: return None
            
        # 4. Подготовка фичей
        try:
            X = row[feature_cols].values
            X_scaled = scaler.transform(X)
        except Exception:
            return None
        
        # 5. Предикт и маппинг вероятностей
        try:
            # --- CRITICAL FIX: Блокировка ConstantModel ---
            # Если модель "константная" (видела только 1 класс), она возвращает вероятность 1.0.
            # Это ловушка. Мы запрещаем торговать в таком случае.
            if isinstance(model, ConstantModel):
                return np.zeros(3, dtype=np.float64)

            # Получаем "сырые" вероятности от модели
            probs = model.predict_proba(X_scaled)[0]
            
            # Создаем стандартизированный массив [Neutral, Long, Short]
            full_probs = np.zeros(3, dtype=np.float64)
            
            # Маппинг через LabelEncoder (перевод локальных классов модели в глобальные 0, 1, 2)
            if le is not None:
                for i, class_label in enumerate(le.classes_):
                    class_int = int(class_label)
                    # Защита от выхода за границы массива (если вдруг классы кривые)
                    if class_int < 3 and i < len(probs): 
                        full_probs[class_int] = probs[i]
            else:
                # Если энкодера нет (редкий случай), пытаемся вставить как есть
                if len(probs) == 3:
                    full_probs = probs
            
            return full_probs

        except Exception as e:
            # Можно раскомментировать для отладки:
            # print(f"⚠️ Prediction Error at {timestamp}: {e}")
            return None