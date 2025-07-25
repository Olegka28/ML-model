#!/usr/bin/env python3
"""
📊 DataManager - управление данными

Загрузка, сохранение, кэширование и валидация данных.
"""

import os
import pickle
import hashlib
from typing import Dict, List, Optional, Any
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from ..utils.config import Config
from ..utils.logger import Logger
from ..utils.validators import DataValidator, ValidationError
from .data_collector import DataCollector

class DataManager:
    """
    Менеджер данных
    
    Отвечает за:
    - Загрузку данных из файлов
    - Скачивание данных с бирж
    - Кэширование данных
    - Валидацию данных
    """
    
    def __init__(self, config: Config):
        """
        Инициализация менеджера данных
        
        Args:
            config: Конфигурация системы
        """
        self.config = config
        self.collector = DataCollector()
        self.logger = Logger('DataManager', level=config.log_level)
        
        # Создаем директории
        self.data_root = Path(config.data_root)
        self.cache_root = Path(config.cache_root)
        self.data_root.mkdir(parents=True, exist_ok=True)
        self.cache_root.mkdir(parents=True, exist_ok=True)
        
        # Кэш в памяти
        self._memory_cache = {}
        self._cache_metadata = {}
    
    def normalize_symbol(self, symbol: str) -> str:
        """
        Нормализация символа для работы с данными
        
        Args:
            symbol: Символ в любом формате (SOLUSDT, SOL_USDT, SOL/USDT)
            
        Returns:
            Нормализованный символ для API (например, SOL/USDT)
        """
        # Если уже содержит /, возвращаем как есть
        if '/' in symbol:
            return symbol
        
        # Если содержит _, заменяем на /
        if '_' in symbol:
            return symbol.replace('_', '/')
        
        # Иначе добавляем / перед USDT
        return symbol.replace('USDT', '/USDT')
    
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
    
    def get_data_path(self, symbol: str, timeframe: str, period_info: Optional[str] = None) -> Path:
        """
        Получение пути к файлу данных
        
        Args:
            symbol: Символ монеты
            timeframe: Таймфрейм
            period_info: Информация о периоде (например, "2years", "4years")
            
        Returns:
            Путь к файлу данных
        """
        symbol_file = self.symbol_to_filename(symbol)
        
        if period_info:
            return self.data_root / f"{symbol_file}_{timeframe}_{period_info}.csv"
        else:
            # Для обратной совместимости
            years_back = self.config.data.years_back
            return self.data_root / f"{symbol_file}_{timeframe}_{years_back}years.csv"
    
    def get_cache_path(self, symbol: str, timeframe: str) -> Path:
        """
        Получение пути к кэшу данных
        
        Args:
            symbol: Символ монеты
            timeframe: Таймфрейм
            
        Returns:
            Путь к кэшу
        """
        symbol_file = self.symbol_to_filename(symbol)
        cache_key = f"{symbol_file}_{timeframe}"
        return self.cache_root / f"{cache_key}.pkl"
    
    def _generate_cache_key(self, symbol: str, timeframe: str) -> str:
        """
        Генерация ключа кэша
        
        Args:
            symbol: Символ монеты
            timeframe: Таймфрейм
            
        Returns:
            Ключ кэша
        """
        symbol_file = self.symbol_to_filename(symbol)
        return f"{symbol_file}_{timeframe}"
    
    def is_data_fresh(self, file_path: Path, max_days_old: int = 7) -> bool:
        """
        Проверка свежести данных
        
        Args:
            file_path: Путь к файлу данных
            max_days_old: Максимальный возраст данных в днях
            
        Returns:
            True если данные свежие
        """
        if not file_path.exists():
            return False
        
        # Проверяем время модификации файла
        mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
        age = datetime.now() - mtime
        
        return age.days <= max_days_old
    
    def load_data(self, symbol: str, timeframes: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Загрузка данных для символа и таймфреймов
        
        Args:
            symbol: Символ монеты
            timeframes: Список таймфреймов
            
        Returns:
            Словарь с данными по таймфреймам
        """
        self.logger.info(f"📊 Загрузка данных для {symbol}")
        
        data = {}
        
        for timeframe in timeframes:
            self.logger.info(f"   ⏰ Загрузка {timeframe}")
            
            # Пытаемся загрузить из кэша
            cached_data = self._load_from_cache(symbol, timeframe)
            if cached_data is not None:
                data[timeframe] = cached_data
                self.logger.info(f"   ✅ Загружено из кэша: {timeframe}")
                continue
            
            # Пытаемся загрузить из файла
            file_data = self._load_from_file(symbol, timeframe)
            if file_data is not None:
                data[timeframe] = file_data
                self._save_to_cache(symbol, timeframe, file_data)
                self.logger.info(f"   ✅ Загружено из файла: {timeframe}")
                continue
            
            # Скачиваем данные
            if self.config.data.force_download or not self._file_exists(symbol, timeframe):
                downloaded_data = self._download_data(symbol, timeframe)
                if downloaded_data is not None:
                    data[timeframe] = downloaded_data
                    self._save_to_file(symbol, timeframe, downloaded_data)
                    self._save_to_cache(symbol, timeframe, downloaded_data)
                    self.logger.info(f"   ✅ Скачано: {timeframe}")
                else:
                    self.logger.error(f"   ❌ Не удалось скачать: {timeframe}")
            else:
                self.logger.error(f"   ❌ Данные не найдены: {timeframe}")
        
        if not data:
            raise ValueError(f"Не удалось загрузить данные для {symbol}")
        
        self.logger.info(f"✅ Загружено {len(data)} таймфреймов для {symbol}")
        return data
    
    def _load_from_cache(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """
        Загрузка данных из кэша
        
        Args:
            symbol: Символ монеты
            timeframe: Таймфрейм
            
        Returns:
            DataFrame с данными или None
        """
        if not self.config.data.cache_data:
            return None
        
        cache_key = self._generate_cache_key(symbol, timeframe)
        
        # Проверяем кэш в памяти
        if cache_key in self._memory_cache:
            metadata = self._cache_metadata.get(cache_key, {})
            if self._is_cache_valid(metadata):
                return self._memory_cache[cache_key]
        
        # Проверяем кэш на диске
        cache_path = self.get_cache_path(symbol, timeframe)
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                
                # Проверяем валидность кэша
                if isinstance(cached_data, dict) and 'data' in cached_data:
                    metadata = cached_data.get('metadata', {})
                    if self._is_cache_valid(metadata):
                        # Сохраняем в память
                        self._memory_cache[cache_key] = cached_data['data']
                        self._cache_metadata[cache_key] = metadata
                        return cached_data['data']
            except Exception as e:
                self.logger.warning(f"⚠️ Ошибка загрузки кэша: {e}")
        
        return None
    
    def _load_from_file(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """
        Загрузка данных из файла с поддержкой обратной совместимости
        
        Args:
            symbol: Символ монеты
            timeframe: Таймфрейм
            
        Returns:
            DataFrame с данными или None
        """
        # Пытаемся загрузить с разными вариантами имен файлов
        possible_paths = []
        
        # Новый формат без суффикса
        possible_paths.append(self.get_data_path(symbol, timeframe))
        
        # Старый формат с years_back
        years_back = self.config.data.years_back
        possible_paths.append(self.get_data_path(symbol, timeframe, f"{years_back}years"))
        
        # Для обратной совместимости
        possible_paths.append(self.get_data_path(symbol, timeframe, "2years"))
        possible_paths.append(self.get_data_path(symbol, timeframe, "4years"))
        
        for file_path in possible_paths:
            if file_path.exists():
                try:
                    # Загружаем данные
                    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                    
                    # Проверяем свежесть
                    if not self.is_data_fresh(file_path, max_days_old=7):
                        self.logger.warning(f"⚠️ Данные устарели: {file_path}")
                    
                    # Валидируем данные
                    if self.config.data.validate_data:
                        DataValidator.validate_ohlcv_data(df)
                        DataValidator.validate_data_completeness(df)
                    
                    self.logger.info(f"✅ Загружено из файла: {file_path}")
                    return df
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ Ошибка загрузки файла {file_path}: {e}")
                    continue
        
        return None
    
    def _download_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """
        Скачивание данных с биржи
        
        Args:
            symbol: Символ монеты
            timeframe: Таймфрейм
            
        Returns:
            DataFrame с данными или None
        """
        try:
            symbol_normalized = self.normalize_symbol(symbol)
            
            # Используем новый метод с периодом
            years_back = self.config.data.years_back
            df = self.collector.get_data_for_period(symbol_normalized, timeframe, years_back)
            
            if df is not None:
                # Валидируем скачанные данные
                if self.config.data.validate_data:
                    DataValidator.validate_ohlcv_data(df)
                    DataValidator.validate_data_completeness(df)
                
                # Сохраняем скачанные данные
                self._save_to_file(symbol, timeframe, df)
                
                return df
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Ошибка скачивания данных {symbol} {timeframe}: {e}")
            return None
    
    def _save_to_file(self, symbol: str, timeframe: str, df: pd.DataFrame, period_info: Optional[str] = None):
        """
        Сохранение данных в файл
        
        Args:
            symbol: Символ монеты
            timeframe: Таймфрейм
            df: DataFrame с данными
            period_info: Информация о периоде
        """
        file_path = self.get_data_path(symbol, timeframe, period_info)
        
        try:
            # Создаем директорию если нужно
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Сохраняем данные
            df.to_csv(file_path)
            self.logger.info(f"💾 Данные сохранены: {file_path}")
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка сохранения файла {file_path}: {e}")
    
    def _save_to_cache(self, symbol: str, timeframe: str, df: pd.DataFrame):
        """
        Сохранение данных в кэш
        
        Args:
            symbol: Символ монеты
            timeframe: Таймфрейм
            df: DataFrame с данными
        """
        if not self.config.data.cache_data:
            return
        
        cache_key = self._generate_cache_key(symbol, timeframe)
        cache_path = self.get_cache_path(symbol, timeframe)
        
        try:
            # Создаем метаданные кэша
            metadata = {
                'created_at': datetime.now().isoformat(),
                'symbol': symbol,
                'timeframe': timeframe,
                'rows': len(df),
                'columns': list(df.columns),
                'data_hash': self._calculate_data_hash(df)
            }
            
            # Сохраняем в память
            self._memory_cache[cache_key] = df
            self._cache_metadata[cache_key] = metadata
            
            # Сохраняем на диск
            cache_data = {
                'data': df,
                'metadata': metadata
            }
            
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Ошибка сохранения кэша: {e}")
    
    def _is_cache_valid(self, metadata: Dict[str, Any]) -> bool:
        """
        Проверка валидности кэша
        
        Args:
            metadata: Метаданные кэша
            
        Returns:
            True если кэш валиден
        """
        if not metadata:
            return False
        
        # Проверяем время создания
        created_at = metadata.get('created_at')
        if created_at:
            try:
                created_time = datetime.fromisoformat(created_at)
                age = datetime.now() - created_time
                if age.total_seconds() > self.config.cache_ttl:
                    return False
            except:
                return False
        
        return True
    
    def _calculate_data_hash(self, df: pd.DataFrame) -> str:
        """
        Расчет хеша данных для проверки целостности
        
        Args:
            df: DataFrame с данными
            
        Returns:
            Хеш данных
        """
        # Используем первые и последние строки для быстрого хеша
        sample_data = pd.concat([df.head(10), df.tail(10)])
        data_str = sample_data.to_string()
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def _file_exists(self, symbol: str, timeframe: str) -> bool:
        """
        Проверка существования файла данных
        
        Args:
            symbol: Символ монеты
            timeframe: Таймфрейм
            
        Returns:
            True если файл существует
        """
        # Проверяем все возможные варианты имен файлов
        possible_paths = [
            self.get_data_path(symbol, timeframe),
            self.get_data_path(symbol, timeframe, "2years"),
            self.get_data_path(symbol, timeframe, "4years"),
            self.get_data_path(symbol, timeframe, f"{self.config.data.years_back}years")
        ]
        
        return any(path.exists() for path in possible_paths)
    
    def clear_cache(self, symbol: Optional[str] = None):
        """
        Очистка кэша
        
        Args:
            symbol: Символ монеты (если None, очищается весь кэш)
        """
        if symbol:
            # Очищаем кэш для конкретного символа
            for timeframe in self.config.data.timeframes:
                cache_key = self._generate_cache_key(symbol, timeframe)
                if cache_key in self._memory_cache:
                    del self._memory_cache[cache_key]
                if cache_key in self._cache_metadata:
                    del self._cache_metadata[cache_key]
                
                cache_path = self.get_cache_path(symbol, timeframe)
                if cache_path.exists():
                    cache_path.unlink()
            
            self.logger.info(f"🗑️ Кэш очищен для {symbol}")
        else:
            # Очищаем весь кэш
            self._memory_cache.clear()
            self._cache_metadata.clear()
            
            for cache_file in self.cache_root.glob("*.pkl"):
                cache_file.unlink()
            
            self.logger.info("🗑️ Весь кэш очищен")
    
    def get_data_info(self, symbol: str, timeframe: str) -> Optional[Dict[str, Any]]:
        """
        Получение информации о данных
        
        Args:
            symbol: Символ монеты
            timeframe: Таймфрейм
            
        Returns:
            Словарь с информацией о данных или None
        """
        # Пытаемся найти файл данных
        possible_paths = [
            self.get_data_path(symbol, timeframe),
            self.get_data_path(symbol, timeframe, "2years"),
            self.get_data_path(symbol, timeframe, "4years"),
            self.get_data_path(symbol, timeframe, f"{self.config.data.years_back}years")
        ]
        
        file_path = None
        for path in possible_paths:
            if path.exists():
                file_path = path
                break
        
        if not file_path:
            return None
        
        try:
            # Загружаем данные для анализа
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            
            info = {
                'symbol': symbol,
                'timeframe': timeframe,
                'file_path': str(file_path),
                'rows': len(df),
                'columns': list(df.columns),
                'date_range': {
                    'start': df.index.min().isoformat(),
                    'end': df.index.max().isoformat()
                },
                'file_size': file_path.stat().st_size,
                'last_modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                'is_fresh': self.is_data_fresh(file_path)
            }
            
            return info
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка получения информации о данных: {e}")
            return None
    
    def get_info(self) -> Dict[str, Any]:
        """
        Получение информации о менеджере данных
        
        Returns:
            Словарь с информацией
        """
        return {
            'data_root': str(self.data_root),
            'cache_root': str(self.cache_root),
            'cache_enabled': self.config.data.cache_data,
            'validation_enabled': self.config.data.validate_data,
            'memory_cache_size': len(self._memory_cache),
            'timeframes': self.config.data.timeframes,
            'years_back': self.config.data.years_back
        } 