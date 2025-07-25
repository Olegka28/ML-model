#!/usr/bin/env python3
"""
🏗️ Базовый класс ML системы

Центральный компонент, координирующий работу всех подсистем.
"""

import time
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import pandas as pd
import numpy as np

from ..utils.config import Config
from ..utils.logger import Logger
from ..utils.validators import ValidationError, DataValidator, FeatureValidator
from ..data_collector import DataManager
from ..features import FeatureManager
from .model_manager import ModelManager

class BaseSystem:
    """
    Базовый класс ML системы
    
    Координирует работу:
    - DataManager: управление данными
    - FeatureManager: управление признаками  
    - ModelManager: управление моделями
    """
    
    def __init__(self, config: Config):
        """
        Инициализация системы
        
        Args:
            config: Конфигурация системы
        """
        self.config = config
        
        # Инициализируем подсистемы
        self.data_manager = DataManager(config)
        self.feature_manager = FeatureManager(config)
        self.model_manager = ModelManager(config)
        
        # Инициализируем логгер
        log_file = Path(config.logs_root) / f"ml_system_{time.strftime('%Y%m%d')}.log"
        self.logger = Logger(
            name='BaseSystem',
            level=config.log_level,
            log_file=str(log_file),
            log_format=config.log_format
        )
        
        self.logger.info("🏗️ Базовая система инициализирована")
    
    def normalize_symbol(self, symbol: str) -> str:
        """
        Нормализация символа для работы с данными
        
        Args:
            symbol: Символ в любом формате (SOLUSDT, SOL_USDT, SOL/USDT)
            
        Returns:
            Нормализованный символ для API (например, SOL/USDT)
        """
        # Валидация входных данных
        if not symbol or not isinstance(symbol, str):
            raise ValueError("Symbol must be a non-empty string")
        
        symbol = symbol.strip().upper()
        if not symbol:
            raise ValueError("Symbol cannot be empty after trimming")
        
        # Если уже содержит /, возвращаем как есть
        if '/' in symbol:
            return symbol
        
        # Если содержит _, заменяем на /
        if '_' in symbol:
            return symbol.replace('_', '/')
        
        # Иначе добавляем / перед USDT
        if symbol.endswith('USDT'):
            return symbol.replace('USDT', '/USDT')
        
        # Если не USDT пара, возвращаем как есть
        return symbol
    
    def symbol_to_filename(self, symbol: str) -> str:
        """
        Преобразование символа в имя файла
        
        Args:
            symbol: Символ в любом формате
            
        Returns:
            Имя файла (например, SOL_USDT)
        """
        # Сначала нормализуем
        normalized = self.normalize_symbol(symbol)
        # Затем заменяем / на _ для имени файла
        return normalized.replace('/', '_')
    
    def load_and_validate_data(self, symbol: str, timeframes: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Загрузка и валидация данных
        
        Args:
            symbol: Символ монеты
            timeframes: Список таймфреймов
            
        Returns:
            Словарь с данными по таймфреймам
        """
        # Валидация входных данных
        if not symbol or not isinstance(symbol, str):
            raise ValueError("Symbol must be a non-empty string")
        
        if not timeframes or not isinstance(timeframes, list):
            raise ValueError("Timeframes must be a non-empty list")
        
        self.logger.info(f"📊 Загрузка данных для {symbol}")
        
        try:
            # Загружаем данные
            data = self.data_manager.load_data(symbol, timeframes)
            
            # Валидируем каждую таблицу
            for timeframe, df in data.items():
                self.logger.info(f"✅ Валидация данных {timeframe}")
                
                # Валидация OHLCV
                DataValidator.validate_ohlcv_data(df)
                
                # Валидация полноты
                DataValidator.validate_data_completeness(df)
                
                # Валидация свежести (если включена)
                if self.config.data.validate_data:
                    DataValidator.validate_data_freshness(df)
            
            self.logger.info(f"✅ Данные загружены и валидированы для {symbol}")
            return data
            
        except ValidationError as e:
            self.logger.error(f"❌ Ошибка валидации данных для {symbol}: {e}")
            raise
        except Exception as e:
            self.logger.error(f"❌ Ошибка загрузки данных для {symbol}: {e}")
            raise
    
    def generate_and_validate_features(self, data: Dict[str, pd.DataFrame], 
                                     feature_config: Optional[Dict] = None) -> pd.DataFrame:
        """
        Генерация и валидация признаков
        
        Args:
            data: Словарь с данными по таймфреймам
            feature_config: Конфигурация признаков (опционально)
            
        Returns:
            DataFrame с признаками
        """
        # Валидация входных данных
        if not data or not isinstance(data, dict):
            raise ValueError("Data must be a non-empty dictionary")
        
        self.logger.info("🔬 Генерация признаков")
        
        try:
            # Генерируем признаки
            features = self.feature_manager.generate_features(data, feature_config)
            
            # Валидируем признаки только если включено в конфигурации
            if self.config.features.validate_features:
                self.logger.info("✅ Валидация признаков")
                FeatureValidator.validate_feature_quality(features)
                FeatureValidator.validate_feature_types(features)
            
            self.logger.info(f"✅ Признаки сгенерированы: {features.shape}")
            return features
            
        except ValidationError as e:
            self.logger.error(f"❌ Ошибка валидации признаков: {e}")
            raise
        except Exception as e:
            self.logger.error(f"❌ Ошибка генерации признаков: {e}")
            raise
    
    def generate_and_select_features(self, data: Dict[str, pd.DataFrame], 
                                   target: pd.Series,
                                   feature_config: Optional[Dict] = None) -> pd.DataFrame:
        """
        Генерация и фильтрация признаков
        
        Args:
            data: Словарь с данными по таймфреймам
            target: Series с таргетом
            feature_config: Конфигурация признаков (опционально)
            
        Returns:
            DataFrame с отфильтрованными признаками
        """
        # Генерируем все признаки
        features = self.generate_and_validate_features(data, feature_config)
        
        # Выравниваем размеры features и target
        common_index = features.index.intersection(target.index)
        features = features.loc[common_index]
        target = target.loc[common_index]
        
        # Фильтрация признаков (если включена в конфигурации)
        if self.config.features.use_feature_selection:
            self.logger.info("🔍 Фильтрация признаков...")
            selected_features = self.feature_manager.select_features(
                features=features,
                target=target,
                method=self.config.features.selection_method,
                threshold=self.config.features.selection_threshold,
                remove_correlated=self.config.features.remove_correlated_features,
                correlation_threshold=self.config.features.correlation_threshold,
                n_features=self.config.features.n_features
            )
            
            if selected_features:
                # Возвращаем только отфильтрованные признаки
                features_filtered = features[selected_features]
                self.logger.info(f"✅ Отобрано {len(selected_features)} признаков из {len(features.columns)}")
                return features_filtered
            else:
                self.logger.warning("⚠️ Не удалось отобрать признаки, используем все")
        
        return features
    
    def create_target(self, df: pd.DataFrame, target_type: str, horizon: int) -> pd.Series:
        """
        Создание целевой переменной
        
        Args:
            df: DataFrame с данными
            target_type: Тип таргета
            horizon: Горизонт предсказания
            
        Returns:
            Series с таргетом
        """
        # Валидация входных данных
        if df is None or df.empty:
            raise ValueError("DataFrame cannot be None or empty")
        
        if not target_type or not isinstance(target_type, str):
            raise ValueError("Target type must be a non-empty string")
        
        if not isinstance(horizon, int) or horizon <= 0:
            raise ValueError("Horizon must be a positive integer")
        
        self.logger.info(f"🎯 Создание таргета: {target_type}, горизонт: {horizon}")
        
        try:
            from ..features import TargetCreator
            target_creator = TargetCreator()
            
            # Используем новый TargetCreator для всех типов таргетов
            target = target_creator.create_target(df, target_type, horizon)
            
            # Удаляем NaN значения
            target = target.dropna()
            
            # Проверяем, что таргет не пустой
            if len(target) == 0:
                raise ValueError(f"Target is empty after processing for {target_type} with horizon {horizon}")
            
            self.logger.info(f"✅ Таргет создан: {len(target)} значений")
            return target
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка создания таргета {target_type}: {e}")
            raise
    
    def prepare_training_data(self, features: pd.DataFrame, target: pd.Series) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Подготовка данных для обучения
        
        Args:
            features: DataFrame с признаками
            target: Series с таргетом
            
        Returns:
            (X, y, feature_names)
        """
        # Валидация входных данных
        if features is None or features.empty:
            raise ValueError("Features DataFrame cannot be None or empty")
        
        if target is None or target.empty:
            raise ValueError("Target Series cannot be None or empty")
        
        self.logger.info("🔧 Подготовка данных для обучения")
        
        try:
            # Проверяем совпадение индексов
            if not features.index.equals(target.index):
                self.logger.warning("⚠️ Индексы features и target не совпадают, выравниваем...")
                # Выравниваем индексы
                common_index = features.index.intersection(target.index)
                features = features.loc[common_index]
                target = target.loc[common_index]
            
            # Объединяем признаки и таргет
            df = pd.concat([features, target], axis=1)
            df.columns = list(features.columns) + ['target']
            
            # Удаляем строки с NaN
            df = df.dropna()
            
            # Проверяем, что данные не пустые
            if len(df) == 0:
                raise ValueError("No valid data after removing NaN values")
            
            # Разделяем на X и y
            X = df.drop('target', axis=1).values
            y = df['target'].values
            feature_names = df.drop('target', axis=1).columns.tolist()
            
            self.logger.info(f"✅ Данные подготовлены: X={X.shape}, y={y.shape}")
            return X, y, feature_names
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка подготовки данных: {e}")
            raise
    
    def train_model(self, X: np.ndarray, y: np.ndarray, 
                   feature_names: Optional[List[str]] = None,
                   model_config: Optional[Dict] = None, task: str = 'regression') -> Tuple[Any, Dict[str, Any]]:
        """
        Обучение модели
        
        Args:
            X: Признаки
            y: Таргет
            feature_names: Названия признаков (опционально)
            model_config: Конфигурация модели (опционально)
            task: Тип задачи ('regression' или 'classification')
            
        Returns:
            (model, metadata)
        """
        # Валидация входных данных
        if X is None or len(X) == 0:
            raise ValueError("Features array cannot be None or empty")
        
        if y is None or len(y) == 0:
            raise ValueError("Target array cannot be None or empty")
        
        if task not in ['regression', 'classification']:
            raise ValueError("Task must be 'regression' or 'classification'")
        
        self.logger.info(f"🤖 Обучение модели для задачи: {task}")
        
        try:
            # Обучаем модель
            model, metadata = self.model_manager.train_model(X, y, feature_names, model_config, task)
            
            self.logger.info("✅ Модель обучена")
            return model, metadata
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка обучения модели для {task}: {e}")
            raise
    
    def save_model(self, model: Any, metadata: Dict[str, Any], symbol: str, task: str = 'regression') -> str:
        """
        Сохранение модели
        
        Args:
            model: Обученная модель
            metadata: Метаданные модели
            symbol: Символ монеты
            task: Тип задачи ('regression' или 'classification')
            
        Returns:
            Путь к сохраненной модели
        """
        # Валидация входных данных
        if model is None:
            raise ValueError("Model cannot be None")
        
        if not metadata or not isinstance(metadata, dict):
            raise ValueError("Metadata must be a non-empty dictionary")
        
        if not symbol or not isinstance(symbol, str):
            raise ValueError("Symbol must be a non-empty string")
        
        if task not in ['regression', 'classification']:
            raise ValueError("Task must be 'regression' or 'classification'")
        
        self.logger.info(f"💾 Сохранение модели для {symbol} ({task})")
        
        try:
            model_path = self.model_manager.save_model(model, metadata, symbol, task)
            self.logger.info(f"✅ Модель сохранена: {model_path}")
            return model_path
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка сохранения модели для {symbol} ({task}): {e}")
            raise
    
    def load_model(self, symbol: str, task: str = 'regression') -> Tuple[Any, Dict[str, Any]]:
        """
        Загрузка модели
        
        Args:
            symbol: Символ монеты
            task: Тип задачи ('regression' или 'classification')
            
        Returns:
            (model, metadata)
        """
        # Валидация входных данных
        if not symbol or not isinstance(symbol, str):
            raise ValueError("Symbol must be a non-empty string")
        
        if task not in ['regression', 'classification']:
            raise ValueError("Task must be 'regression' or 'classification'")
        
        self.logger.info(f"📥 Загрузка модели для {symbol} ({task})")
        
        try:
            model, metadata = self.model_manager.load_model(symbol, task)
            self.logger.info("✅ Модель загружена")
            return model, metadata
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка загрузки модели для {symbol} ({task}): {e}")
            raise
    
    def predict(self, symbol: str, features: pd.DataFrame, task: str = 'regression') -> Tuple[float, float]:
        """
        Получение предсказания
        
        Args:
            symbol: Символ монеты
            features: DataFrame с признаками
            task: Тип задачи ('regression' или 'classification')
            
        Returns:
            (prediction, confidence)
        """
        # Валидация входных данных
        if not symbol or not isinstance(symbol, str):
            raise ValueError("Symbol must be a non-empty string")
        
        if features is None or features.empty:
            raise ValueError("Features DataFrame cannot be None or empty")
        
        if task not in ['regression', 'classification']:
            raise ValueError("Task must be 'regression' or 'classification'")
        
        self.logger.info(f"🔮 Получение предсказания для {symbol} ({task})")
        
        try:
            # Загружаем модель
            model, metadata = self.load_model(symbol, task)
            
            # Подготавливаем признаки
            expected_features = metadata.get('features', [])
            if not expected_features:
                raise ValueError("No features found in model metadata")
            
            # Оптимизированная проверка наличия признаков
            missing_features = set(expected_features) - set(features.columns)
            if missing_features:
                self.logger.warning(f"⚠️ Отсутствуют признаки: {missing_features}")
                return None, 0.0
            
            # Делаем предсказание
            X = features[expected_features].iloc[[-1]].values
            
            # Валидируем входные данные ПЕРЕД предсказанием
            from ..utils.validators import ModelValidator
            ModelValidator.validate_prediction_input(X, expected_features)
            
            # Делаем предсказание
            prediction = model.predict(X)[0]
            
            # Рассчитываем уверенность
            confidence = self._calculate_confidence(prediction, metadata)
            
            self.logger.info(f"✅ Предсказание: {prediction:.6f}, уверенность: {confidence:.1f}%")
            return prediction, confidence
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка получения предсказания для {symbol} ({task}): {e}")
            raise
    
    def _calculate_confidence(self, prediction: float, metadata: Dict[str, Any]) -> float:
        """
        Расчет уверенности в предсказании
        
        Args:
            prediction: Предсказание
            metadata: Метаданные модели
            
        Returns:
            Уверенность в процентах
        """
        # Простая эвристика: чем дальше от нуля, тем выше уверенность
        abs_pred = abs(prediction)
        
        # Используем стандартное отклонение таргета если доступно
        target_std = metadata.get('target_std', 0.01)
        
        # Защита от деления на ноль
        if target_std <= 0:
            target_std = 0.01
        
        z_score = abs_pred / target_std
        
        # Рассчитываем уверенность на основе z-score
        confidence = min(z_score * 20, 95.0)  # Максимум 95%
        
        return confidence
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Получение информации о системе
        
        Returns:
            Словарь с информацией о системе
        """
        return {
            'config': self.config.to_dict(),
            'data_manager': self.data_manager.get_info(),
            'feature_manager': self.feature_manager.get_info(),
            'model_manager': self.model_manager.get_info()
        } 