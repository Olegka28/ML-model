#!/usr/bin/env python3
"""
🤖 ModelManager - управление моделями

Обучение, сохранение, загрузка, валидация и версификация моделей (XGBoost, LightGBM, CatBoost).
"""

import os
import sys
import json
import pickle
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import mlflow
import mlflow.xgboost
import mlflow.lightgbm
import mlflow.catboost

from ..utils.config import Config
from ..utils.logger import Logger
from ..utils.validators import ModelValidator, ValidationError
from .model_versioning import ModelVersioning
from .baseline import BaselineModels

class ModelManager:
    """
    Менеджер моделей ML системы
    - Обучение (XGBoost, LightGBM, CatBoost)
    - Сохранение/загрузка моделей и метаданных
    - Версификация и сравнение моделей
    - Валидация моделей и метаданных
    - Интеграция с MLflow
    """
    def __init__(self, config: Config):
        self.config = config
        self.models_root = Path(config.models_root)
        self.models_root.mkdir(parents=True, exist_ok=True)
        self.logger = Logger('ModelManager', level=config.log_level)
        self.versioning = ModelVersioning(config.models_root)
        mlflow.set_tracking_uri(config.mlflow_tracking_uri)
        mlflow.set_experiment(config.mlflow_experiment_name)
        self._model_cache = {}

    def validate_input_data(self, X: np.ndarray, y: np.ndarray, task: str = 'regression') -> bool:
        """
        Валидация входных данных
        
        Args:
            X: np.ndarray - признаки
            y: np.ndarray - таргет
            task: тип задачи
            
        Returns:
            True если данные валидны
        """
        try:
            # Проверка размерностей
            if len(X) != len(y):
                raise ValueError(f"Размерности X ({len(X)}) и y ({len(y)}) не совпадают")
            
            if len(X) == 0:
                raise ValueError("Данные пустые")
            
            # Проверка на NaN и inf
            if np.any(np.isnan(X)) or np.any(np.isinf(X)):
                raise ValueError("Обнаружены NaN или inf значения в признаках")
            
            if np.any(np.isnan(y)) or np.any(np.isinf(y)):
                raise ValueError("Обнаружены NaN или inf значения в таргете")
            
            # Проверка типов данных
            if not isinstance(X, np.ndarray):
                raise ValueError("X должен быть numpy.ndarray")
            
            if not isinstance(y, np.ndarray):
                raise ValueError("y должен быть numpy.ndarray")
            
            # Проверка для классификации
            if task == 'classification':
                unique_classes = np.unique(y)
                if len(unique_classes) < 2:
                    raise ValueError("Для классификации нужно минимум 2 класса")
                
                # Проверяем, что классы целые числа
                if not np.issubdtype(y.dtype, np.integer):
                    raise ValueError("Для классификации таргет должен быть целыми числами")
            
            self.logger.info(f"✅ Валидация данных прошла успешно: X={X.shape}, y={y.shape}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка валидации данных: {e}")
            raise
    
    def _create_time_series_split(self, X: np.ndarray, y: np.ndarray, 
                                 cv_type: str = 'walk_forward', 
                                 n_splits: int = 5,
                                 test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Создает правильные сплиты для временных рядов
        
        Args:
            X: признаки
            y: таргет
            cv_type: тип кросс-валидации ('walk_forward', 'time_series_split', 'expanding_window')
            n_splits: количество сплитов
            test_size: размер тестового набора
            
        Returns:
            X_train, X_val, y_train, y_val
        """
        from sklearn.model_selection import TimeSeriesSplit
        
        if cv_type == 'walk_forward':
            # Walk-Forward CV - расширяющееся окно
            return self._walk_forward_split(X, y, n_splits, test_size)
            
        elif cv_type == 'time_series_split':
            # TimeSeriesSplit - фиксированное окно
            return self._time_series_split(X, y, n_splits)
            
        elif cv_type == 'expanding_window':
            # Расширяющееся окно с фиксированным тестовым набором
            return self._expanding_window_split(X, y, test_size)
            
        else:
            raise ValueError(f"Неподдерживаемый тип CV: {cv_type}")
    
    def _walk_forward_split(self, X: np.ndarray, y: np.ndarray, 
                           n_splits: int, test_size: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Walk-Forward CV - расширяющееся окно обучения
        
        Это наиболее подходящий метод для криптовалют, так как:
        1. Использует расширяющееся окно обучения
        2. Тестирует на будущих данных
        3. Имитирует реальные торговые условия
        """
        total_size = len(X)
        test_size_samples = int(total_size * test_size)
        
        # Используем последние данные для валидации
        split_point = total_size - test_size_samples
        
        # Обучаем на всех данных до split_point
        X_train = X[:split_point]
        y_train = y[:split_point]
        
        # Тестируем на оставшихся данных
        X_val = X[split_point:]
        y_val = y[split_point:]
        
        self.logger.info(f"🔄 Walk-Forward CV: обучаем на {len(X_train)} баров, тестируем на {len(X_val)} баров")
        self.logger.info(f"📈 Обучающий период: 0 → {split_point}, Тестовый период: {split_point} → {total_size}")
        
        return X_train, X_val, y_train, y_val
    
    def _time_series_split(self, X: np.ndarray, y: np.ndarray, 
                          n_splits: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        TimeSeriesSplit - фиксированное окно
        
        Использует sklearn TimeSeriesSplit для создания нескольких сплитов
        """
        from sklearn.model_selection import TimeSeriesSplit
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        # Берем последний сплит для обучения
        splits = list(tscv.split(X))
        train_idx, val_idx = splits[-1]
        
        X_train = X[train_idx]
        y_train = y[train_idx]
        X_val = X[val_idx]
        y_val = y[val_idx]
        
        self.logger.info(f"🔄 TimeSeriesSplit CV: обучаем на {len(X_train)} баров, тестируем на {len(X_val)} баров")
        self.logger.info(f"📈 Используем последний сплит из {n_splits} сплитов")
        
        return X_train, X_val, y_train, y_val
    
    def _expanding_window_split(self, X: np.ndarray, y: np.ndarray, 
                               test_size: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Расширяющееся окно с фиксированным тестовым набором
        
        Похоже на Walk-Forward, но с более гибким подходом
        """
        total_size = len(X)
        test_size_samples = int(total_size * test_size)
        
        # Минимальный размер обучающего набора
        min_train_size = int(total_size * 0.3)  # Минимум 30% данных для обучения
        
        # Используем расширяющееся окно
        split_point = max(min_train_size, total_size - test_size_samples)
        
        X_train = X[:split_point]
        y_train = y[:split_point]
        X_val = X[split_point:]
        y_val = y[split_point:]
        
        self.logger.info(f"🔄 Expanding Window CV: обучаем на {len(X_train)} баров, тестируем на {len(X_val)} баров")
        self.logger.info(f"📈 Обучающий период: 0 → {split_point}, Тестовый период: {split_point} → {total_size}")
        
        return X_train, X_val, y_train, y_val
    
    def cross_validate_time_series(self, X: np.ndarray, y: np.ndarray, 
                                 model_config: Optional[dict] = None,
                                 cv_type: str = 'walk_forward',
                                 n_splits: int = 5) -> Dict[str, Any]:
        """
        Полная кросс-валидация для временных рядов
        
        Args:
            X: признаки
            y: таргет
            model_config: конфигурация модели
            cv_type: тип CV
            n_splits: количество сплитов
            
        Returns:
            Результаты кросс-валидации
        """
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        if model_config is None:
            model_config = {
                'model_type': self.config.model.model_type,
                'target_type': self.config.model.target_type,
                'horizon': self.config.model.horizon
            }
        
        cv_scores = {
            'rmse': [],
            'mae': [],
            'r2': [],
            'splits': []
        }
        
        if cv_type == 'walk_forward':
            # Walk-Forward CV
            total_size = len(X)
            test_size = int(total_size * 0.2)  # 20% для тестирования
            
            for i in range(n_splits):
                # Вычисляем размер обучающего набора
                train_size = int(total_size * (0.3 + 0.1 * i))  # От 30% до 70%
                train_size = min(train_size, total_size - test_size)
                
                # Создаем сплит
                X_train = X[:train_size]
                y_train = y[:train_size]
                X_val = X[train_size:train_size + test_size]
                y_val = y[train_size:train_size + test_size]
                
                # Обучаем модель
                try:
                    model, _ = self.train_model(X_train, y_train, model_config, 'regression')
                    y_pred = self.predict(model, X_val)
                    
                    # Вычисляем метрики
                    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                    mae = mean_absolute_error(y_val, y_pred)
                    r2 = r2_score(y_val, y_pred)
                    
                    cv_scores['rmse'].append(rmse)
                    cv_scores['mae'].append(mae)
                    cv_scores['r2'].append(r2)
                    cv_scores['splits'].append({
                        'train_size': len(X_train),
                        'val_size': len(X_val),
                        'train_period': f"0 → {train_size}",
                        'val_period': f"{train_size} → {train_size + test_size}"
                    })
                    
                    self.logger.info(f"🔄 Сплит {i+1}/{n_splits}: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ Ошибка в сплите {i+1}: {e}")
                    continue
                    
        elif cv_type == 'time_series_split':
            # TimeSeriesSplit
            tscv = TimeSeriesSplit(n_splits=n_splits)
            
            for i, (train_idx, val_idx) in enumerate(tscv.split(X)):
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_val = X[val_idx]
                y_val = y[val_idx]
                
                try:
                    model, _ = self.train_model(X_train, y_train, model_config, 'regression')
                    y_pred = self.predict(model, X_val)
                    
                    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                    mae = mean_absolute_error(y_val, y_pred)
                    r2 = r2_score(y_val, y_pred)
                    
                    cv_scores['rmse'].append(rmse)
                    cv_scores['mae'].append(mae)
                    cv_scores['r2'].append(r2)
                    cv_scores['splits'].append({
                        'train_size': len(X_train),
                        'val_size': len(X_val),
                        'train_period': f"{train_idx[0]} → {train_idx[-1]}",
                        'val_period': f"{val_idx[0]} → {val_idx[-1]}"
                    })
                    
                    self.logger.info(f"🔄 Сплит {i+1}/{n_splits}: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ Ошибка в сплите {i+1}: {e}")
                    continue
        
        # Вычисляем итоговые метрики
        if cv_scores['rmse']:
            cv_results = {
                'mean_rmse': np.mean(cv_scores['rmse']),
                'std_rmse': np.std(cv_scores['rmse']),
                'mean_mae': np.mean(cv_scores['mae']),
                'std_mae': np.std(cv_scores['mae']),
                'mean_r2': np.mean(cv_scores['r2']),
                'std_r2': np.std(cv_scores['r2']),
                'n_splits': len(cv_scores['rmse']),
                'splits': cv_scores['splits']
            }
            
            self.logger.info(f"📊 CV результаты ({cv_type}):")
            self.logger.info(f"   RMSE: {cv_results['mean_rmse']:.4f} ± {cv_results['std_rmse']:.4f}")
            self.logger.info(f"   MAE: {cv_results['mean_mae']:.4f} ± {cv_results['std_mae']:.4f}")
            self.logger.info(f"   R²: {cv_results['mean_r2']:.4f} ± {cv_results['std_r2']:.4f}")
            
            return cv_results
        else:
            self.logger.error("❌ Не удалось выполнить кросс-валидацию")
            return {}
    
    def train_model(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None, model_config: Optional[dict] = None, task: str = 'regression') -> Tuple[Any, Dict]:
        """
        Обучение модели (XGBoost/LightGBM/CatBoost)
        Args:
            X: np.ndarray - признаки
            y: np.ndarray - таргет
            feature_names: List[str] - названия признаков (опционально)
            model_config: dict - параметры модели
            task: 'regression' или 'classification'
        Returns:
            model, metadata
        """
        import xgboost as xgb
        import lightgbm as lgb
        from catboost import CatBoostRegressor, CatBoostClassifier
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, f1_score
        import optuna

        # Валидация входных данных
        self.validate_input_data(X, y, task)

        if model_config is None:
            model_config = {
                'model_type': self.config.model.model_type,
                'target_type': self.config.model.target_type,
                'horizon': self.config.model.horizon,
                'n_trials': self.config.model.n_trials,
                'early_stopping_rounds': self.config.model.early_stopping_rounds,
                'random_state': self.config.model.random_state
            }
        model_type = model_config.get('model_type', 'xgboost')
        n_trials = model_config.get('n_trials', 50)
        early_stopping = model_config.get('early_stopping_rounds', 20)
        random_state = model_config.get('random_state', 42)
        # Используем правильную валидацию для временных рядов
        if self.config.model.use_time_series_cv:
            # TimeSeriesSplit или Walk-Forward CV
            X_train, X_val, y_train, y_val = self._create_time_series_split(
                X, y, 
                self.config.model.cv_type,
                self.config.model.cv_n_splits,
                self.config.model.cv_test_size
            )
        else:
            # Обычный train/test split
            train_size = int(len(X) * self.config.model.train_test_split)
            X_train, X_val = X[:train_size], X[train_size:]
            y_train, y_val = y[:train_size], y[train_size:]
        
        self.logger.info(f"📊 Train: {X_train.shape}, Val: {X_val.shape}")
        self.logger.info(f"📅 Временной диапазон: {len(X_train)} → {len(X_val)} баров")

        def xgb_objective(trial):
            params = {
                'objective': 'reg:squarederror' if task == 'regression' else 'binary:logistic',
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'random_state': random_state
            }
            model = xgb.XGBRegressor(**params) if task == 'regression' else xgb.XGBClassifier(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=early_stopping, verbose=False)
            y_pred = model.predict(X_val)
            if task == 'regression':
                return mean_squared_error(y_val, y_pred, squared=False)
            else:
                return 1 - accuracy_score(y_val, (y_pred > 0.5).astype(int))

        def lgb_objective(trial):
            params = {
                'objective': 'regression' if task == 'regression' else 'binary',
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'random_state': random_state
            }
            model = lgb.LGBMRegressor(**params) if task == 'regression' else lgb.LGBMClassifier(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(early_stopping)], verbose=False)
            y_pred = model.predict(X_val)
            if task == 'regression':
                return mean_squared_error(y_val, y_pred, squared=False)
            else:
                return 1 - accuracy_score(y_val, (y_pred > 0.5).astype(int))

        def cat_objective(trial):
            params = {
                'iterations': trial.suggest_int('iterations', 100, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                'depth': trial.suggest_int('depth', 3, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'random_seed': random_state,
                'loss_function': 'RMSE' if task == 'regression' else 'Logloss',
                'verbose': False
            }
            model = CatBoostRegressor(**params) if task == 'regression' else CatBoostClassifier(**params)
            model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=early_stopping, verbose=False)
            y_pred = model.predict(X_val)
            if task == 'regression':
                return mean_squared_error(y_val, y_pred, squared=False)
            else:
                return 1 - accuracy_score(y_val, (y_pred > 0.5).astype(int))

        # Optuna hyperparameter tuning
        if model_type == 'xgboost':
            study = optuna.create_study(direction='minimize')
            study.optimize(xgb_objective, n_trials=n_trials)
            best_params = study.best_params
            best_params['objective'] = 'reg:squarederror' if task == 'regression' else 'binary:logistic'
            best_params['random_state'] = random_state
            model = xgb.XGBRegressor(**best_params) if task == 'regression' else xgb.XGBClassifier(**best_params)
        elif model_type == 'lightgbm':
            study = optuna.create_study(direction='minimize')
            study.optimize(lgb_objective, n_trials=n_trials)
            best_params = study.best_params
            best_params['objective'] = 'regression' if task == 'regression' else 'binary'
            best_params['random_state'] = random_state
            model = lgb.LGBMRegressor(**best_params) if task == 'regression' else lgb.LGBMClassifier(**best_params)
        elif model_type == 'catboost':
            study = optuna.create_study(direction='minimize')
            study.optimize(cat_objective, n_trials=n_trials)
            best_params = study.best_params
            best_params['loss_function'] = 'RMSE' if task == 'regression' else 'Logloss'
            best_params['random_seed'] = random_state
            model = CatBoostRegressor(**best_params) if task == 'regression' else CatBoostClassifier(**best_params)
        else:
            raise ValueError(f"Неизвестный тип модели: {model_type}")

        # Финальное обучение
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)] if model_type != 'catboost' else (X_val, y_val), verbose=False)
        y_pred = model.predict(X_val)
        if task == 'regression':
            metrics = {
                'rmse': float(mean_squared_error(y_val, y_pred, squared=False)),
                'mae': float(mean_absolute_error(y_val, y_pred)),
                'r2': float(r2_score(y_val, y_pred)),
            }
        else:
            metrics = {
                'accuracy': float(accuracy_score(y_val, (y_pred > 0.5).astype(int))),
                'f1': float(f1_score(y_val, (y_pred > 0.5).astype(int))),
            }
        # Baseline
        baseline_models = BaselineModels()
        if task == 'regression':
            baseline_report = baseline_models.create_baseline_report(X_train, y_train, X_val, y_val, 'regression')
            baseline = {
                baseline_report['best_baseline']['name']: baseline_report['best_baseline']['metrics']
            }
        else:
            baseline_report = baseline_models.create_baseline_report(X_train, y_train, X_val, y_val, 'classification')
            baseline = {
                baseline_report['best_baseline']['name']: baseline_report['best_baseline']['metrics']
            }
        # MLflow logging
        with mlflow.start_run(run_name=f"{model_type}_{task}"):
            mlflow.log_params(best_params)
            for k, v in metrics.items():
                mlflow.log_metric(k, v)
            mlflow.log_param('task', task)
            mlflow.log_param('train_size', len(y_train))
            mlflow.log_param('val_size', len(y_val))
            mlflow.log_param('baseline', str(baseline))
            if model_type == 'xgboost':
                mlflow.xgboost.log_model(model, "model")
            elif model_type == 'lightgbm':
                mlflow.lightgbm.log_model(model, "model")
            elif model_type == 'catboost':
                mlflow.catboost.log_model(model, "model")
        # Метаданные
        metadata = {
            'model_type': model_type,
            'task': task,
            'symbol': 'SOL_USDT',  # Будет переопределено при сохранении
            'target_type': model_config.get('target_type', self.config.model.target_type),
            'horizon': model_config.get('horizon', self.config.model.horizon),
            'features': feature_names if feature_names is not None else list(range(X.shape[1])),
            'features_count': X.shape[1],
            'best_params': best_params,
            'metrics': metrics,
            'baseline': baseline,
            'train_size': len(y_train),
            'val_size': len(y_val),
            'train_date': datetime.now().isoformat(),
        }
        self.logger.info(f"✅ Модель обучена. Метрики: {metrics}")
        return model, metadata

    def save_model(self, model: Any, metadata: Dict[str, Any], symbol: str, task: str = 'regression') -> str:
        """
        Сохранение модели и метаданных
        """
        # Обновляем метаданные с информацией о символе
        metadata['symbol'] = symbol
        metadata['task'] = task
        
        model_dir = self.models_root / symbol / task
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / 'model.pkl'
        meta_path = model_dir / 'meta.json'
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        # Версификация
        if self.config.model.version_model:
            # Извлекаем список признаков из метаданных
            features = metadata.get('selected_features', [])
            if not features and 'features' in metadata:
                features = metadata['features']
            self.versioning.save_model_with_versioning(symbol, model, features, metadata, task)
        self.logger.info(f"💾 Модель сохранена: {model_path}")
        return str(model_path)
    
    def export_model(self, symbol: str, task: str = 'regression', 
                    export_format: str = 'pickle', export_path: Optional[str] = None) -> str:
        """
        Экспорт модели в разных форматах
        
        Args:
            symbol: символ модели
            task: тип задачи
            export_format: формат экспорта ('pickle', 'onnx', 'json')
            export_path: путь для экспорта (опционально)
            
        Returns:
            Путь к экспортированному файлу
        """
        try:
            # Загружаем модель
            model, metadata = self.load_model(symbol, task, use_cache=False)
            
            if export_path is None:
                export_path = f"exported_models/{symbol}_{task}_{export_format}"
            
            os.makedirs(os.path.dirname(export_path), exist_ok=True)
            
            if export_format == 'pickle':
                # Экспорт в pickle
                with open(f"{export_path}.pkl", 'wb') as f:
                    pickle.dump(model, f)
                
                # Экспорт метаданных
                with open(f"{export_path}_metadata.json", 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                self.logger.info(f"✅ Модель экспортирована в pickle: {export_path}.pkl")
                return f"{export_path}.pkl"
                
            elif export_format == 'json':
                # Экспорт только метаданных в JSON
                with open(f"{export_path}.json", 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                self.logger.info(f"✅ Метаданные экспортированы в JSON: {export_path}.json")
                return f"{export_path}.json"
                
            elif export_format == 'onnx':
                # Экспорт в ONNX (если поддерживается)
                try:
                    import onnx
                    from skl2onnx import convert_sklearn
                    from skl2onnx.common.data_types import FloatTensorType
                    
                    # Определяем тип входных данных
                    input_type = FloatTensorType([None, metadata.get('features_count', 5)])
                    
                    # Конвертируем модель
                    onx = convert_sklearn(model, initial_types=[('input', input_type)])
                    
                    # Сохраняем ONNX модель
                    with open(f"{export_path}.onnx", "wb") as f:
                        f.write(onx.SerializeToString())
                    
                    self.logger.info(f"✅ Модель экспортирована в ONNX: {export_path}.onnx")
                    return f"{export_path}.onnx"
                    
                except ImportError:
                    self.logger.error("❌ ONNX экспорт недоступен. Установите: pip install onnx skl2onnx")
                    raise ValueError("ONNX экспорт недоступен")
                    
            else:
                raise ValueError(f"Неподдерживаемый формат экспорта: {export_format}")
                
        except Exception as e:
            self.logger.error(f"❌ Ошибка экспорта модели: {e}")
            raise

    def load_model(self, symbol: str, task: str = 'regression', use_cache: bool = True) -> Tuple[Any, Dict[str, Any]]:
        """
        Загрузка модели и метаданных
        
        Args:
            symbol: символ модели
            task: тип задачи
            use_cache: использовать кэш
            
        Returns:
            (model, metadata)
        """
        # Проверяем кэш
        if use_cache:
            cached_result = self.get_cached_model(symbol, task)
            if cached_result:
                return cached_result
        
        # Загружаем с диска
        model_dir = self.models_root / symbol / task
        model_path = model_dir / 'model.pkl'
        meta_path = model_dir / 'meta.json'
        
        if not model_path.exists() or not meta_path.exists():
            raise FileNotFoundError(f"Модель или метаданные не найдены для {symbol} ({task})")
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
        
        ModelValidator.validate_metadata(metadata)
        
        # Сохраняем в кэш
        if use_cache:
            self.cache_model(symbol, task, model, metadata)
        
        self.logger.info(f"✅ Модель загружена: {model_path}")
        return model, metadata

    def predict(self, model: Any, X: np.ndarray) -> np.ndarray:
        """
        Получение предсказаний
        """
        return model.predict(X)

    def get_model_info(self, symbol: str, task: str = 'regression') -> Optional[Dict[str, Any]]:
        try:
            _, metadata = self.load_model(symbol, task)
            return metadata
        except Exception as e:
            self.logger.error(f"Ошибка получения информации о модели: {e}")
            return None

    def compare_models(self, symbol: str, task: str = 'regression') -> Dict[str, Any]:
        try:
            return self.versioning.get_model_statistics(symbol, task)
        except Exception as e:
            self.logger.error(f"Ошибка сравнения моделей: {e}")
            return {}

    def get_model_history(self, symbol: str, task: str = 'regression') -> List[Dict[str, Any]]:
        try:
            return self.versioning.get_model_history(symbol, task)
        except Exception as e:
            self.logger.error(f"Ошибка истории моделей: {e}")
            return []

    def restore_model_version(self, symbol: str, version: str, task: str = 'regression') -> bool:
        try:
            return self.versioning.restore_model_version(symbol, version, task)
        except Exception as e:
            self.logger.error(f"Ошибка восстановления версии: {e}")
            return False

    def get_cached_model(self, symbol: str, task: str = 'regression') -> Optional[Tuple[Any, Dict[str, Any]]]:
        """
        Получить модель из кэша
        
        Args:
            symbol: символ модели
            task: тип задачи
            
        Returns:
            (model, metadata) или None если нет в кэше
        """
        cache_key = f"{symbol}_{task}"
        if cache_key in self._model_cache:
            self.logger.info(f"📦 Модель {symbol} загружена из кэша")
            return self._model_cache[cache_key]
        return None
    
    def cache_model(self, symbol: str, task: str, model: Any, metadata: Dict[str, Any]):
        """
        Сохранить модель в кэш
        
        Args:
            symbol: символ модели
            task: тип задачи
            model: модель
            metadata: метаданные
        """
        cache_key = f"{symbol}_{task}"
        self._model_cache[cache_key] = (model, metadata)
        self.logger.info(f"💾 Модель {symbol} сохранена в кэш")
    
    def clear_cache(self):
        """Очистить кэш моделей"""
        cache_size = len(self._model_cache)
        self._model_cache.clear()
        self.logger.info(f"🗑️ Кэш моделей очищен ({cache_size} моделей)")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Получить информацию о кэше"""
        return {
            'cache_size': len(self._model_cache),
            'cached_models': list(self._model_cache.keys()),
            'memory_usage': sum(sys.getsizeof(model) + sys.getsizeof(metadata) 
                              for model, metadata in self._model_cache.values())
        }

    def compare_with_baseline(self, symbol: str, task: str = 'regression') -> Dict[str, Any]:
        """
        Сравнить модель с базовыми моделями
        
        Args:
            symbol: символ модели
            task: тип задачи
            
        Returns:
            Результаты сравнения
        """
        try:
            # Загружаем модель
            model, metadata = self.load_model(symbol, task)
            
            # Создаем базовые модели
            baseline_models = BaselineModels()
            
            # Получаем данные для сравнения (используем последние данные)
            # В реальном приложении здесь нужно загрузить тестовые данные
            X_test = np.random.random((100, metadata.get('features_count', 5)))
            y_test = np.random.random(100) if task == 'regression' else np.random.randint(0, 2, 100)
            
            # Получаем предсказания модели
            y_pred = self.predict(model, X_test)
            
            # Создаем отчет по базовым моделям
            baseline_report = baseline_models.create_baseline_report(
                X_test, y_test, X_test, y_test, task
            )
            
            # Сравниваем с лучшей базовой моделью
            best_baseline_metrics = baseline_report['best_baseline']['metrics']
            model_metrics = metadata.get('metrics', {})
            
            comparison = baseline_models.compare_with_baseline(
                model_metrics, best_baseline_metrics, task
            )
            
            return {
                'model_metrics': model_metrics,
                'baseline_metrics': best_baseline_metrics,
                'baseline_name': baseline_report['best_baseline']['name'],
                'comparison': comparison,
                'all_baselines': baseline_report['all_baselines']
            }
            
        except Exception as e:
            self.logger.error(f"Ошибка сравнения с базовой моделью: {e}")
            return {}
    
    def get_model_performance_summary(self, symbol: str, task: str = 'regression') -> Dict[str, Any]:
        """
        Получить полную сводку по производительности модели
        
        Args:
            symbol: символ модели
            task: тип задачи
            
        Returns:
            Сводка производительности
        """
        try:
            # Основная информация о модели
            model_info = self.get_model_info(symbol, task)
            if not model_info:
                return {'error': 'Модель не найдена'}
            
            # История версий
            history = self.get_model_history(symbol, task)
            
            # Статистика версификации
            versioning_stats = self.compare_models(symbol, task)
            
            # Сравнение с базовыми моделями
            baseline_comparison = self.compare_with_baseline(symbol, task)
            
            # Информация о кэше
            cache_info = self.get_cache_info()
            
            return {
                'model_info': model_info,
                'version_history': history,
                'versioning_stats': versioning_stats,
                'baseline_comparison': baseline_comparison,
                'cache_info': cache_info,
                'summary': {
                    'is_cached': symbol in [key.split('_')[0] for key in cache_info['cached_models']],
                    'total_versions': len(history),
                    'current_score': model_info.get('model_score', 0),
                    'is_better_than_baseline': baseline_comparison.get('comparison', {}).get('is_better_than_baseline', False)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Ошибка получения сводки производительности: {e}")
            return {'error': str(e)}
    
    def get_info(self) -> Dict[str, Any]:
        return {
            'models_root': str(self.models_root),
            'versioning_enabled': self.config.model.version_model,
            'save_models': self.config.model.save_model,
            'model_cache_size': len(self._model_cache),
            'mlflow_tracking_uri': self.config.mlflow_tracking_uri,
            'mlflow_experiment': self.config.mlflow_experiment_name,
            'cache_info': self.get_cache_info()
        } 