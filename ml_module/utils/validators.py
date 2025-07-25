#!/usr/bin/env python3
"""
✅ Система валидации для ML Trading System

Валидация данных, признаков и моделей на всех этапах.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

class ValidationError(Exception):
    """Исключение для ошибок валидации"""
    pass

class DataValidator:
    """Валидатор данных"""
    
    @staticmethod
    def validate_ohlcv_data(df: pd.DataFrame) -> bool:
        """
        Валидация OHLCV данных
        
        Args:
            df: DataFrame с данными
            
        Returns:
            True если данные валидны
            
        Raises:
            ValidationError: если данные невалидны
        """
        # Проверяем наличие необходимых колонок
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValidationError(f"Отсутствуют необходимые колонки: {missing_columns}")
        
        # Проверяем типы данных
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise ValidationError(f"Колонка {col} должна быть числовой")
        
        # Проверяем логику OHLC
        invalid_ohlc = (
            (df['high'] < df['low']) |
            (df['open'] > df['high']) |
            (df['close'] > df['high']) |
            (df['open'] < df['low']) |
            (df['close'] < df['low'])
        )
        
        if invalid_ohlc.any():
            invalid_count = invalid_ohlc.sum()
            raise ValidationError(f"Найдено {invalid_count} строк с нелогичными OHLC данными")
        
        # Проверяем на отрицательные цены
        price_columns = ['open', 'high', 'low', 'close']
        negative_prices = (df[price_columns] <= 0).any(axis=1)
        
        if negative_prices.any():
            negative_count = negative_prices.sum()
            raise ValidationError(f"Найдено {negative_count} строк с отрицательными или нулевыми ценами")
        
        # Проверяем на отрицательный объем
        negative_volume = df['volume'] < 0
        if negative_volume.any():
            negative_count = negative_volume.sum()
            raise ValidationError(f"Найдено {negative_count} строк с отрицательным объемом")
        
        # Проверяем индекс времени
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValidationError("Индекс должен быть DatetimeIndex")
        
        # Проверяем на дубликаты в индексе
        if df.index.duplicated().any():
            duplicate_count = df.index.duplicated().sum()
            raise ValidationError(f"Найдено {duplicate_count} дублирующихся временных меток")
        
        return True
    
    @staticmethod
    def validate_data_completeness(df: pd.DataFrame, min_rows: int = 100) -> bool:
        """
        Проверка полноты данных
        
        Args:
            df: DataFrame с данными
            min_rows: Минимальное количество строк
            
        Returns:
            True если данные достаточно полные
        """
        if len(df) < min_rows:
            raise ValidationError(f"Недостаточно данных: {len(df)} строк (минимум {min_rows})")
        
        # Проверяем пропуски
        missing_data = df.isnull().sum()
        total_missing = missing_data.sum()
        
        if total_missing > 0:
            missing_percent = (total_missing / (len(df) * len(df.columns))) * 100
            if missing_percent > 5:  # Больше 5% пропусков
                raise ValidationError(f"Слишком много пропущенных данных: {missing_percent:.2f}%")
        
        return True
    
    @staticmethod
    def validate_data_freshness(df: pd.DataFrame, max_days_old: int = 7) -> bool:
        """
        Проверка свежести данных
        
        Args:
            df: DataFrame с данными
            max_days_old: Максимальный возраст данных в днях
            
        Returns:
            True если данные достаточно свежие
        """
        if df.empty:
            raise ValidationError("DataFrame пустой")
        
        latest_timestamp = df.index.max()
        current_time = pd.Timestamp.now()
        days_old = (current_time - latest_timestamp).days
        
        if days_old > max_days_old:
            raise ValidationError(f"Данные устарели: {days_old} дней (максимум {max_days_old})")
        
        return True

class FeatureValidator:
    """Валидатор признаков"""
    
    @staticmethod
    def validate_features(df: pd.DataFrame, expected_features: List[str]) -> bool:
        """
        Валидация наличия ожидаемых признаков
        
        Args:
            df: DataFrame с признаками
            expected_features: Список ожидаемых признаков
            
        Returns:
            True если все признаки присутствуют
        """
        missing_features = [f for f in expected_features if f not in df.columns]
        
        if missing_features:
            raise ValidationError(f"Отсутствуют признаки: {missing_features}")
        
        return True
    
    @staticmethod
    def validate_feature_quality(df: pd.DataFrame) -> bool:
        """
        Валидация качества признаков
        
        Args:
            df: DataFrame с признаками
            
        Returns:
            True если признаки качественные
        """
        # Проверяем на inf значения
        inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
        if inf_count > 0:
            raise ValidationError(f"Найдено {inf_count} inf значений в признаках")
        
        # Проверяем на NaN значения
        nan_count = df.isnull().sum().sum()
        if nan_count > 0:
            nan_percent = (nan_count / (len(df) * len(df.columns))) * 100
            if nan_percent > 10:  # Больше 10% NaN
                raise ValidationError(f"Слишком много NaN значений: {nan_percent:.2f}%")
        
        # Проверяем на константные признаки (разрешаем до 20% константных признаков)
        constant_features = []
        for col in df.columns:
            if df[col].nunique() <= 1:
                constant_features.append(col)
        
        constant_percent = (len(constant_features) / len(df.columns)) * 100
        if constant_percent > 20:  # Больше 20% константных признаков
            raise ValidationError(f"Слишком много константных признаков: {constant_percent:.1f}% ({constant_features})")
        elif constant_features:
            # Логируем предупреждение, но не прерываем выполнение
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"⚠️ Найдены константные признаки ({constant_percent:.1f}%): {constant_features}")
        
        return True
    
    @staticmethod
    def validate_feature_types(df: pd.DataFrame) -> bool:
        """
        Валидация типов признаков
        
        Args:
            df: DataFrame с признаками
            
        Returns:
            True если типы признаков корректны
        """
        # Проверяем, что все признаки числовые
        non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns.tolist()
        
        if non_numeric_columns:
            raise ValidationError(f"Некорректные типы признаков: {non_numeric_columns}")
        
        return True

class ModelValidator:
    """Валидатор моделей"""
    
    @staticmethod
    def validate_model_file(model_path: str) -> bool:
        """
        Валидация файла модели
        
        Args:
            model_path: Путь к файлу модели
            
        Returns:
            True если файл модели валиден
        """
        if not Path(model_path).exists():
            raise ValidationError(f"Файл модели не найден: {model_path}")
        
        # Проверяем размер файла
        file_size = Path(model_path).stat().st_size
        if file_size < 1000:  # Меньше 1KB
            raise ValidationError(f"Файл модели слишком маленький: {file_size} байт")
        
        return True
    
    @staticmethod
    def validate_metadata(metadata: Dict[str, Any]) -> bool:
        """
        Валидация метаданных модели
        
        Args:
            metadata: Словарь с метаданными
            
        Returns:
            True если метаданные валидны
        """
        required_fields = ['symbol', 'target_type', 'horizon', 'features', 'train_date']
        missing_fields = [field for field in required_fields if field not in metadata]
        
        if missing_fields:
            raise ValidationError(f"Отсутствуют обязательные поля в метаданных: {missing_fields}")
        
        # Проверяем типы полей
        if not isinstance(metadata['symbol'], str):
            raise ValidationError("Поле 'symbol' должно быть строкой")
        
        if not isinstance(metadata['features'], list):
            raise ValidationError("Поле 'features' должно быть списком")
        
        if not isinstance(metadata['horizon'], int):
            raise ValidationError("Поле 'horizon' должно быть целым числом")
        
        return True
    
    @staticmethod
    def validate_prediction_input(X: np.ndarray, expected_features: List[str]) -> bool:
        """
        Валидация входных данных для предсказания
        
        Args:
            X: Массив признаков
            expected_features: Список ожидаемых признаков
            
        Returns:
            True если входные данные валидны
        """
        if X is None:
            raise ValidationError("Входные данные не могут быть None")
        
        if not isinstance(X, np.ndarray):
            raise ValidationError("Входные данные должны быть numpy массивом")
        
        if X.ndim != 2:
            raise ValidationError("Входные данные должны быть 2D массивом")
        
        if X.shape[1] != len(expected_features):
            raise ValidationError(f"Неверное количество признаков: {X.shape[1]} (ожидается {len(expected_features)})")
        
        # Проверяем на NaN и inf
        if np.isnan(X).any():
            raise ValidationError("Входные данные содержат NaN значения")
        
        if np.isinf(X).any():
            raise ValidationError("Входные данные содержат inf значения")
        
        return True

class ConfigValidator:
    """Валидатор конфигурации"""
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> bool:
        """
        Валидация конфигурации
        
        Args:
            config: Словарь с конфигурацией
            
        Returns:
            True если конфигурация валидна
        """
        # Проверяем обязательные поля
        required_fields = ['models_root', 'data_root', 'model']
        missing_fields = [field for field in required_fields if field not in config]
        
        if missing_fields:
            raise ValidationError(f"Отсутствуют обязательные поля конфигурации: {missing_fields}")
        
        # Проверяем модель
        model_config = config.get('model', {})
        if not isinstance(model_config, dict):
            raise ValidationError("Поле 'model' должно быть словарем")
        
        # Проверяем параметры модели
        if 'horizon' in model_config and model_config['horizon'] <= 0:
            raise ValidationError("Горизонт должен быть положительным числом")
        
        if 'n_trials' in model_config and model_config['n_trials'] <= 0:
            raise ValidationError("Количество trials должно быть положительным числом")
        
        return True

# Функции-помощники для быстрой валидации
def validate_data_pipeline(df: pd.DataFrame, expected_features: List[str]) -> Tuple[bool, List[str]]:
    """
    Полная валидация пайплаина данных
    
    Returns:
        (is_valid, error_messages)
    """
    errors = []
    
    try:
        DataValidator.validate_ohlcv_data(df)
    except ValidationError as e:
        errors.append(f"Ошибка OHLCV данных: {e}")
    
    try:
        DataValidator.validate_data_completeness(df)
    except ValidationError as e:
        errors.append(f"Ошибка полноты данных: {e}")
    
    try:
        FeatureValidator.validate_features(df, expected_features)
    except ValidationError as e:
        errors.append(f"Ошибка признаков: {e}")
    
    try:
        FeatureValidator.validate_feature_quality(df)
    except ValidationError as e:
        errors.append(f"Ошибка качества признаков: {e}")
    
    return len(errors) == 0, errors 