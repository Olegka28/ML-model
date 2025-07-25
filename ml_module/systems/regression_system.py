#!/usr/bin/env python3
"""
📈 RegressionSystem - специализированная система для регрессии
"""

from ..core.base_system import BaseSystem
from ..utils.config import Config
from typing import Optional, Dict, Any, List
import numpy as np
import pandas as pd

class RegressionSystem(BaseSystem):
    """
    Система для регрессионного обучения и предсказаний
    """
    def __init__(self, config: Config):
        super().__init__(config)
        self.logger.info("📈 RegressionSystem инициализирована")

    def run_experiment(self, symbol: str, target_type: str = 'crypto_clipped', horizon: int = 10) -> Dict[str, Any]:
        """
        Запуск полного эксперимента по регрессии
        
        Args:
            symbol: Символ монеты
            target_type: Тип таргета (по умолчанию: crypto_clipped)
            horizon: Горизонт предсказания
            
        Returns:
            Метаданные эксперимента
        """
        # Валидация входных данных
        if not symbol or not isinstance(symbol, str):
            raise ValueError("Symbol must be a non-empty string")
        
        if not target_type or not isinstance(target_type, str):
            raise ValueError("Target type must be a non-empty string")
        
        if not isinstance(horizon, int) or horizon <= 0:
            raise ValueError("Horizon must be a positive integer")
        
        self.logger.info(f"🚀 Запуск эксперимента регрессии для {symbol}")
        self.logger.info(f"🎯 Таргет: {target_type}, горизонт: {horizon}")
        
        try:
            # 1. Загрузка данных
            self.logger.info("📊 Шаг 1: Загрузка данных...")
            data = self.load_and_validate_data(symbol, self.config.data.timeframes)
            
            # 2. Генерация признаков
            self.logger.info("🔬 Шаг 2: Генерация признаков...")
            features = self.generate_and_validate_features(data)
            
            # 3. Создание таргета
            self.logger.info("🎯 Шаг 3: Создание таргета...")
            target = self.create_target(features, target_type, horizon)
            
            # 4. Подготовка данных
            self.logger.info("🔧 Шаг 4: Подготовка данных...")
            X, y, feature_names = self.prepare_training_data(features, target)
            
            # 5. Обучение модели (ИСПРАВЛЕНО: добавлен task='regression')
            self.logger.info("🤖 Шаг 5: Обучение модели...")
            model, metadata = self.train_model(X, y, task='regression')
            
            # 6. Сохранение модели (ИСПРАВЛЕНО: добавлен task='regression')
            self.logger.info("💾 Шаг 6: Сохранение модели...")
            self.save_model(model, metadata, symbol, task='regression')
            
            # 7. Возврат метрик
            self.logger.info("✅ Эксперимент завершен успешно")
            return metadata
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка в эксперименте для {symbol}: {e}")
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
        
        self.logger.info(f"🔮 Получение предсказания для {symbol} ({timeframe})")
        
        try:
            # Генерируем признаки для последних данных
            self.logger.info("🔬 Генерация признаков для предсказания...")
            features = self.feature_manager.generate_features({timeframe: latest_data})
            
            # Получаем предсказание (ИСПРАВЛЕНО: добавлен task='regression')
            prediction, confidence = self.predict(symbol, features, task='regression')
            
            if prediction is None:
                self.logger.warning(f"⚠️ Не удалось получить предсказание для {symbol}")
                return None
            
            result = {
                'prediction': prediction,
                'confidence': confidence,
                'symbol': symbol,
                'timeframe': timeframe,
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
            self.logger.info(f"✅ Предсказание получено: {prediction:.6f}, уверенность: {confidence:.1f}%")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка предсказания для {symbol}: {e}")
            return None
    
    def get_model_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Получить информацию о модели
        
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
            # Загружаем модель для получения метаданных (ИСПРАВЛЕНО: добавлен task='regression')
            model, metadata = self.load_model(symbol, task='regression')
            
            # Формируем информацию о модели
            model_info = {
                'symbol': symbol,
                'task': 'regression',
                'model_type': metadata.get('model_type', 'unknown'),
                'features_count': metadata.get('features_count', 0),
                'training_date': metadata.get('saved_at', 'unknown'),
                'metrics': {
                    'rmse': metadata.get('rmse', 0),
                    'mae': metadata.get('mae', 0),
                    'r2': metadata.get('r2', 0)
                },
                'target_type': metadata.get('target_type', 'unknown'),
                'horizon': metadata.get('horizon', 0),
                'model_score': metadata.get('model_score', 0)
            }
            
            self.logger.info(f"✅ Информация о модели {symbol} получена")
            return model_info
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка получения информации о модели {symbol}: {e}")
            return None
    
    def compare_models(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Сравнить версии модели
        
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
            # Используем ModelManager для сравнения (ИСПРАВЛЕНО: добавлен task='regression')
            comparison = self.model_manager.compare_models(symbol, task='regression')
            
            self.logger.info(f"✅ Сравнение моделей {symbol} выполнено")
            return comparison
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка сравнения моделей {symbol}: {e}")
            return None
    
    def get_model_history(self, symbol: str) -> Optional[List[Dict[str, Any]]]:
        """
        Получить историю версий модели
        
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
            # Используем ModelManager для получения истории (ИСПРАВЛЕНО: добавлен task='regression')
            history = self.model_manager.get_model_history(symbol, task='regression')
            
            self.logger.info(f"✅ История моделей {symbol} получена")
            return history
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка получения истории моделей {symbol}: {e}")
            return None 