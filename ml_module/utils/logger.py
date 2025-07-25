#!/usr/bin/env python3
"""
📝 Система логирования для ML Trading System

Централизованное логирование с красивым форматированием.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

class ColoredFormatter(logging.Formatter):
    """Форматтер с цветным выводом"""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record):
        # Добавляем цвет к уровню логирования
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.COLORS['RESET']}"
        
        # Добавляем эмодзи к сообщениям
        if record.levelname == 'INFO':
            record.msg = f"ℹ️  {record.msg}"
        elif record.levelname == 'WARNING':
            record.msg = f"⚠️  {record.msg}"
        elif record.levelname == 'ERROR':
            record.msg = f"❌ {record.msg}"
        elif record.levelname == 'CRITICAL':
            record.msg = f"🚨 {record.msg}"
        elif record.levelname == 'DEBUG':
            record.msg = f"🔍 {record.msg}"
        
        return super().format(record)

class Logger:
    """Централизованный логгер для ML системы"""
    
    def __init__(self, name: str = 'MLSystem', level: str = 'INFO', 
                 log_file: Optional[str] = None, log_format: Optional[str] = None):
        """
        Инициализация логгера
        
        Args:
            name: Имя логгера
            level: Уровень логирования
            log_file: Путь к файлу логов (опционально)
            log_format: Формат логов (опционально)
        """
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # Очищаем существующие обработчики
        self.logger.handlers.clear()
        
        # Формат по умолчанию
        if log_format is None:
            log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        # Создаем форматтер
        formatter = ColoredFormatter(log_format)
        
        # Консольный обработчик
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # Файловый обработчик (если указан)
        if log_file:
            # Создаем директорию для логов
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter(log_format))
            self.logger.addHandler(file_handler)
    
    def debug(self, message: str):
        """Логирование отладочной информации"""
        self.logger.debug(message)
    
    def info(self, message: str):
        """Логирование информационных сообщений"""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Логирование предупреждений"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Логирование ошибок"""
        self.logger.error(message)
    
    def critical(self, message: str):
        """Логирование критических ошибок"""
        self.logger.critical(message)
    
    def log_experiment(self, experiment_name: str, metrics: dict):
        """Логирование результатов эксперимента"""
        self.info(f"🧪 Эксперимент: {experiment_name}")
        for metric, value in metrics.items():
            if isinstance(value, float):
                self.info(f"   📊 {metric}: {value:.6f}")
            else:
                self.info(f"   📊 {metric}: {value}")
    
    def log_training_start(self, symbol: str, config: dict):
        """Логирование начала обучения"""
        self.info(f"🚀 Начинаем обучение модели для {symbol}")
        self.info(f"⚙️  Конфигурация:")
        for key, value in config.items():
            self.info(f"   🔧 {key}: {value}")
    
    def log_training_end(self, symbol: str, metrics: dict):
        """Логирование завершения обучения"""
        self.info(f"✅ Обучение завершено для {symbol}")
        self.info(f"📈 Результаты:")
        for metric, value in metrics.items():
            if isinstance(value, float):
                self.info(f"   📊 {metric}: {value:.6f}")
            else:
                self.info(f"   📊 {metric}: {value}")
    
    def log_prediction(self, symbol: str, prediction: float, confidence: float):
        """Логирование предсказания"""
        self.info(f"🔮 Предсказание для {symbol}: {prediction:.6f} (уверенность: {confidence:.1f}%)")
    
    def log_data_loading(self, symbol: str, timeframes: list, rows: int):
        """Логирование загрузки данных"""
        self.info(f"📊 Загружены данные для {symbol}")
        self.info(f"   ⏰ Таймфреймы: {', '.join(timeframes)}")
        self.info(f"   📈 Строк: {rows:,}")
    
    def log_feature_generation(self, features_count: int, rows: int):
        """Логирование генерации признаков"""
        self.info(f"🔬 Сгенерированы признаки")
        self.info(f"   🔧 Количество: {features_count}")
        self.info(f"   📈 Строк: {rows:,}")
    
    def log_model_saving(self, symbol: str, model_path: str):
        """Логирование сохранения модели"""
        self.info(f"💾 Модель сохранена для {symbol}")
        self.info(f"   📁 Путь: {model_path}")
    
    def log_error(self, operation: str, error: Exception):
        """Логирование ошибок с контекстом"""
        self.error(f"❌ Ошибка в операции '{operation}': {str(error)}")
        self.debug(f"🔍 Детали ошибки: {type(error).__name__}")
    
    def log_performance(self, operation: str, duration: float):
        """Логирование производительности"""
        if duration < 1:
            self.info(f"⚡ {operation}: {duration*1000:.1f}ms")
        elif duration < 60:
            self.info(f"⚡ {operation}: {duration:.1f}s")
        else:
            minutes = int(duration // 60)
            seconds = duration % 60
            self.info(f"⚡ {operation}: {minutes}m {seconds:.1f}s")

# Глобальный логгер по умолчанию
default_logger = Logger('MLSystem')

def get_logger(name: str = None) -> Logger:
    """Получить логгер по имени"""
    if name is None:
        return default_logger
    return Logger(name) 