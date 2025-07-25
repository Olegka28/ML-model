#!/usr/bin/env python3
"""
🎯 ClassificationSystem - специализированная система для классификации
"""

from ..core.base_system import BaseSystem
from ..utils.config import Config
from typing import Optional, Dict, Any, List
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE

class ClassificationSystem(BaseSystem):
    """
    Система для классификационного обучения и предсказаний
    """
    def __init__(self, config: Config):
        super().__init__(config)
        self.logger.info("🎯 ClassificationSystem инициализирована")

    def create_classification_target(self, df: pd.DataFrame, percent: float = 0.025, horizon: int = 20) -> pd.Series:
        """
        Создание бинарного таргета: 1 если рост >= percent за horizon, иначе 0
        
        Args:
            df: DataFrame с данными
            percent: Процент роста для классификации (по умолчанию: 0.025 = 2.5%)
            horizon: Горизонт предсказания
            
        Returns:
            Series с бинарным таргетом
        """
        # Валидация входных данных
        if df is None or df.empty:
            raise ValueError("DataFrame cannot be None or empty")
        
        if 'close' not in df.columns:
            raise ValueError("DataFrame must contain 'close' column")
        
        if not isinstance(percent, (int, float)) or percent <= 0:
            raise ValueError("Percent must be a positive number")
        
        if not isinstance(horizon, int) or horizon <= 0:
            raise ValueError("Horizon must be a positive integer")
        
        self.logger.info(f"🎯 Создание классификационного таргета: {percent*100}% рост за {horizon} баров")
        
        try:
            # Создаем бинарный таргет
            future_price = df['close'].shift(-horizon)
            target = ((future_price - df['close']) / df['close'] >= percent).astype(int)
            
            # Удаляем NaN значения
            target = target.dropna()
            
            # Проверяем, что таргет не пустой
            if len(target) == 0:
                raise ValueError(f"Target is empty after processing for {percent*100}% growth with horizon {horizon}")
            
            # Проверяем баланс классов
            class_counts = target.value_counts()
            self.logger.info(f"📊 Баланс классов: {dict(class_counts)}")
            
            self.logger.info(f"✅ Классификационный таргет создан: {len(target)} значений")
            return target
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка создания классификационного таргета: {e}")
            raise

    def run_experiment(self, symbol: str, percent: float = 0.025, horizon: int = 20) -> Dict[str, Any]:
        """
        Запуск полного эксперимента по классификации
        
        Args:
            symbol: Символ монеты
            percent: Процент роста для классификации (по умолчанию: 0.025 = 2.5%)
            horizon: Горизонт предсказания
            
        Returns:
            Метаданные эксперимента
        """
        # Валидация входных данных
        if not symbol or not isinstance(symbol, str):
            raise ValueError("Symbol must be a non-empty string")
        
        if not isinstance(percent, (int, float)) or percent <= 0:
            raise ValueError("Percent must be a positive number")
        
        if not isinstance(horizon, int) or horizon <= 0:
            raise ValueError("Horizon must be a positive integer")
        
        self.logger.info(f"🚀 Запуск эксперимента классификации для {symbol}")
        self.logger.info(f"🎯 Процент роста: {percent*100}%, горизонт: {horizon}")
        
        try:
            # 1. Загрузка данных
            self.logger.info("📊 Шаг 1: Загрузка данных...")
            data = self.load_and_validate_data(symbol, self.config.data.timeframes)
            
            # 2. Генерация признаков
            self.logger.info("🔬 Шаг 2: Генерация признаков...")
            features = self.generate_and_validate_features(data)
            
            # 3. Создание таргета
            self.logger.info("🎯 Шаг 3: Создание классификационного таргета...")
            target = self.create_classification_target(features, percent, horizon)
            
            # 4. Подготовка данных
            self.logger.info("🔧 Шаг 4: Подготовка данных...")
            X, y, feature_names = self.prepare_training_data(features, target)
            
            # 5. Балансировка классов (SMOTE)
            self.logger.info("⚖️ Шаг 5: Балансировка классов (SMOTE)...")
            sm = SMOTE(random_state=42)
            X_bal, y_bal = sm.fit_resample(X, y)
            
            # Логируем информацию о балансировке
            original_counts = np.bincount(y)
            balanced_counts = np.bincount(y_bal)
            self.logger.info(f"📊 До балансировки: {dict(zip(range(len(original_counts)), original_counts))}")
            self.logger.info(f"📊 После балансировки: {dict(zip(range(len(balanced_counts)), balanced_counts))}")
            
            # 6. Обучение модели (ИСПРАВЛЕНО: добавлен task='classification')
            self.logger.info("🤖 Шаг 6: Обучение модели классификации...")
            model, metadata = self.train_model(X_bal, y_bal, task='classification')
            
            # 7. Сохранение модели (ИСПРАВЛЕНО: добавлен task='classification')
            self.logger.info("💾 Шаг 7: Сохранение модели...")
            self.save_model(model, metadata, symbol, task='classification')
            
            # 8. Возврат метрик
            self.logger.info("✅ Эксперимент классификации завершен успешно")
            return metadata
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка в эксперименте классификации для {symbol}: {e}")
            raise

    def predict_latest(self, symbol: str, latest_data: pd.DataFrame, 
                      timeframe: str = '15m') -> Optional[Dict[str, Any]]:
        """
        Получить предсказание для последних данных
        
        Args:
            symbol: Символ монеты
            latest_data: DataFrame с последними данными
            timeframe: Таймфрейм данных (по умолчанию: '15m')
            
        Returns:
            Словарь с предсказанием и уверенностью или None при ошибке
        """
        # Валидация входных данных
        if not symbol or not isinstance(symbol, str):
            self.logger.error("Symbol must be a non-empty string")
            return None
        
        if latest_data is None or latest_data.empty:
            self.logger.error("Latest data cannot be None or empty")
            return None
        
        if not timeframe or not isinstance(timeframe, str):
            self.logger.error("Timeframe must be a non-empty string")
            return None
        
        self.logger.info(f"🔮 Получение предсказания классификации для {symbol} ({timeframe})")
        
        try:
            # Генерируем признаки для последних данных
            self.logger.info("🔬 Генерация признаков для предсказания...")
            features = self.feature_manager.generate_features({timeframe: latest_data})
            
            # Получаем предсказание (ИСПРАВЛЕНО: добавлен task='classification')
            prediction, confidence = self.predict(symbol, features, task='classification')
            
            if prediction is None:
                self.logger.warning(f"⚠️ Не удалось получить предсказание для {symbol}")
                return None
            
            # Интерпретируем предсказание
            prediction_class = int(prediction)
            prediction_label = "РОСТ" if prediction_class == 1 else "ПАДЕНИЕ"
            
            result = {
                'prediction': prediction_class,
                'prediction_label': prediction_label,
                'confidence': confidence,
                'symbol': symbol,
                'timeframe': timeframe,
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
            self.logger.info(f"✅ Предсказание классификации: {prediction_label} (класс {prediction_class}), уверенность: {confidence:.1f}%")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка предсказания классификации для {symbol}: {e}")
            return None
    
    def get_model_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Получить информацию о модели классификации
        
        Args:
            symbol: Символ монеты
            
        Returns:
            Информация о модели или None при ошибке
        """
        # Валидация входных данных
        if not symbol or not isinstance(symbol, str):
            self.logger.error("Symbol must be a non-empty string")
            return None
        
        try:
            # Загружаем модель для получения метаданных (ИСПРАВЛЕНО: добавлен task='classification')
            model, metadata = self.load_model(symbol, task='classification')
            
            # Формируем информацию о модели
            model_info = {
                'symbol': symbol,
                'task': 'classification',
                'model_type': metadata.get('model_type', 'unknown'),
                'features_count': metadata.get('features_count', 0),
                'training_date': metadata.get('saved_at', 'unknown'),
                'metrics': {
                    'accuracy': metadata.get('accuracy', 0),
                    'f1_score': metadata.get('f1_score', 0),
                    'precision': metadata.get('precision', 0),
                    'recall': metadata.get('recall', 0)
                },
                'target_percent': metadata.get('target_percent', 0),
                'horizon': metadata.get('horizon', 0),
                'model_score': metadata.get('model_score', 0)
            }
            
            self.logger.info(f"✅ Информация о модели классификации {symbol} получена")
            return model_info
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка получения информации о модели классификации {symbol}: {e}")
            return None
    
    def compare_models(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Сравнить версии модели классификации
        
        Args:
            symbol: Символ монеты
            
        Returns:
            Результаты сравнения или None при ошибке
        """
        # Валидация входных данных
        if not symbol or not isinstance(symbol, str):
            self.logger.error("Symbol must be a non-empty string")
            return None
        
        try:
            # Используем ModelManager для сравнения (ИСПРАВЛЕНО: добавлен task='classification')
            comparison = self.model_manager.compare_models(symbol, task='classification')
            
            self.logger.info(f"✅ Сравнение моделей классификации {symbol} выполнено")
            return comparison
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка сравнения моделей классификации {symbol}: {e}")
            return None
    
    def get_model_history(self, symbol: str) -> Optional[List[Dict[str, Any]]]:
        """
        Получить историю версий модели классификации
        
        Args:
            symbol: Символ монеты
            
        Returns:
            История версий или None при ошибке
        """
        # Валидация входных данных
        if not symbol or not isinstance(symbol, str):
            self.logger.error("Symbol must be a non-empty string")
            return None
        
        try:
            # Используем ModelManager для получения истории (ИСПРАВЛЕНО: добавлен task='classification')
            history = self.model_manager.get_model_history(symbol, task='classification')
            
            self.logger.info(f"✅ История моделей классификации {symbol} получена")
            return history
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка получения истории моделей классификации {symbol}: {e}")
            return None
    
    def get_class_distribution(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Получить распределение классов для модели
        
        Args:
            symbol: Символ монеты
            
        Returns:
            Распределение классов или None при ошибке
        """
        # Валидация входных данных
        if not symbol or not isinstance(symbol, str):
            self.logger.error("Symbol must be a non-empty string")
            return None
        
        try:
            # Загружаем модель для получения метаданных
            model, metadata = self.load_model(symbol, task='classification')
            
            # Получаем информацию о распределении классов
            class_distribution = {
                'symbol': symbol,
                'task': 'classification',
                'target_percent': metadata.get('target_percent', 0),
                'horizon': metadata.get('horizon', 0),
                'original_class_counts': metadata.get('original_class_counts', {}),
                'balanced_class_counts': metadata.get('balanced_class_counts', {}),
                'class_balance_ratio': metadata.get('class_balance_ratio', 0)
            }
            
            self.logger.info(f"✅ Распределение классов для {symbol} получено")
            return class_distribution
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка получения распределения классов для {symbol}: {e}")
            return None 