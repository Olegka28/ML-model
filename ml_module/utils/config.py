#!/usr/bin/env python3
"""
⚙️ Система конфигурации для ML Trading System

Централизованное управление настройками системы.
"""

import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class DataConfig:
    """Конфигурация для работы с данными"""
    data_root: str = 'data'
    cache_dir: str = 'cache'
    timeframes: List[str] = field(default_factory=lambda: ['15m', '1h', '4h', '1d'])
    years_back: int = 4
    force_download: bool = False
    validate_data: bool = True
    cache_data: bool = True

@dataclass
class FeatureConfig:
    """Конфигурация для генерации признаков"""
    include_technical_indicators: bool = True
    include_multi_timeframe: bool = True
    include_lag_features: bool = True
    include_rolling_features: bool = True
    lag_windows: List[int] = field(default_factory=lambda: list(range(1, 11)))
    roll_windows: List[int] = field(default_factory=lambda: [5, 10, 20, 50])
    cache_features: bool = True
    validate_features: bool = True
    handle_inf_nan: bool = True

@dataclass
class ModelConfig:
    """Конфигурация для обучения моделей"""
    model_type: str = 'xgboost'  # xgboost, lightgbm, catboost
    target_type: str = 'crypto_clipped'  # crypto_clipped, volume_weighted, vol_regime, etc.
    horizon: int = 10
    train_test_split: float = 0.8
    validation_split: float = 0.2
    n_trials: int = 50
    early_stopping_rounds: int = 20
    random_state: int = 42
    save_model: bool = True
    version_model: bool = True
    
    # Настройки временных рядов
    use_time_series_cv: bool = True  # Использовать временные ряды CV
    cv_type: str = 'walk_forward'  # walk_forward, time_series_split, expanding_window
    cv_n_splits: int = 5  # Количество сплитов для CV
    cv_test_size: float = 0.2  # Размер тестового набора для CV

@dataclass
class Config:
    """Основная конфигурация системы"""
    # Пути
    models_root: str = 'models'
    data_root: str = 'data'
    cache_root: str = 'cache'
    logs_root: str = 'logs'
    
    # Настройки логирования
    log_level: str = 'INFO'
    log_format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Настройки MLflow
    mlflow_tracking_uri: str = 'file:./mlruns'
    mlflow_experiment_name: str = 'ml_trading'
    
    # Настройки кэширования
    enable_cache: bool = True
    cache_ttl: int = 3600  # секунды
    
    # Настройки параллелизма
    n_jobs: int = -1  # -1 для всех ядер
    
    # Подконфигурации
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    
    def __post_init__(self):
        """Создаем необходимые директории"""
        for path in [self.models_root, self.data_root, self.cache_root, self.logs_root]:
            Path(path).mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Создать конфигурацию из словаря"""
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразовать конфигурацию в словарь"""
        return {
            'models_root': self.models_root,
            'data_root': self.data_root,
            'cache_root': self.cache_root,
            'logs_root': self.logs_root,
            'log_level': self.log_level,
            'log_format': self.log_format,
            'mlflow_tracking_uri': self.mlflow_tracking_uri,
            'mlflow_experiment_name': self.mlflow_experiment_name,
            'enable_cache': self.enable_cache,
            'cache_ttl': self.cache_ttl,
            'n_jobs': self.n_jobs,
            'data': {
                'data_root': self.data.data_root,
                'cache_dir': self.data.cache_dir,
                'timeframes': self.data.timeframes,
                'years_back': self.data.years_back,
                'force_download': self.data.force_download,
                'validate_data': self.data.validate_data,
                'cache_data': self.data.cache_data
            },
            'features': {
                'include_technical_indicators': self.features.include_technical_indicators,
                'include_multi_timeframe': self.features.include_multi_timeframe,
                'include_lag_features': self.features.include_lag_features,
                'include_rolling_features': self.features.include_rolling_features,
                'lag_windows': self.features.lag_windows,
                'roll_windows': self.features.roll_windows,
                'cache_features': self.features.cache_features,
                'validate_features': self.features.validate_features,
                'handle_inf_nan': self.features.handle_inf_nan
            },
            'model': {
                'model_type': self.model.model_type,
                'target_type': self.model.target_type,
                'horizon': self.model.horizon,
                'train_test_split': self.model.train_test_split,
                'validation_split': self.model.validation_split,
                'n_trials': self.model.n_trials,
                'early_stopping_rounds': self.model.early_stopping_rounds,
                'random_state': self.model.random_state,
                'save_model': self.model.save_model,
                'version_model': self.model.version_model,
                'use_time_series_cv': self.model.use_time_series_cv,
                'cv_type': self.model.cv_type,
                'cv_n_splits': self.model.cv_n_splits,
                'cv_test_size': self.model.cv_test_size
            }
        }
    
    def save(self, path: str):
        """Сохранить конфигурацию в файл"""
        import json
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'Config':
        """Загрузить конфигурацию из файла"""
        import json
        with open(path, 'r') as f:
            config_dict = json.load(f)
        
        # Восстанавливаем подконфигурации
        if 'data' in config_dict:
            config_dict['data'] = DataConfig(**config_dict['data'])
        if 'features' in config_dict:
            config_dict['features'] = FeatureConfig(**config_dict['features'])
        if 'model' in config_dict:
            config_dict['model'] = ModelConfig(**config_dict['model'])
        
        return cls(**config_dict)

# Предустановленные конфигурации
DEFAULT_CONFIG = Config()

INTRADAY_CONFIG = Config(
    model=ModelConfig(
        target_type='clipped',
        horizon=5,
        n_trials=30
    ),
    features=FeatureConfig(
        lag_windows=list(range(1, 6)),
        roll_windows=[5, 10, 20]
    )
)

SWING_CONFIG = Config(
    model=ModelConfig(
        target_type='clipped',
        horizon=20,
        n_trials=100
    ),
    features=FeatureConfig(
        lag_windows=list(range(1, 21)),
        roll_windows=[5, 10, 20, 50, 100]
    )
) 