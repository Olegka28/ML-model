#!/usr/bin/env python3
"""
🔬 FeatureManager - управление признаками

Генерация, кэширование и валидация признаков.
"""

import pickle
import hashlib
from typing import Dict, List, Optional, Any
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

from ..utils.config import Config
from ..utils.logger import Logger
from ..utils.validators import FeatureValidator, ValidationError
from .feature_engineer import FeatureEngineer

class FeatureManager:
    """
    Менеджер признаков
    
    Отвечает за:
    - Генерацию технических индикаторов
    - Создание multi-timeframe признаков
    - Кэширование признаков
    - Валидацию признаков
    """
    
    def __init__(self, config: Config):
        """
        Инициализация менеджера признаков
        
        Args:
            config: Конфигурация системы
        """
        self.config = config
        self.engineer = FeatureEngineer()
        self.logger = Logger('FeatureManager', level=config.log_level)
        
        # Создаем директории
        self.cache_root = Path(config.cache_root) / 'features'
        self.cache_root.mkdir(parents=True, exist_ok=True)
        
        # Кэш в памяти
        self._memory_cache = {}
        self._cache_metadata = {}
    
    def _generate_cache_key(self, data_hash: str, feature_config: Dict[str, Any]) -> str:
        """
        Генерация ключа кэша для признаков
        
        Args:
            data_hash: Хеш данных
            feature_config: Конфигурация признаков
            
        Returns:
            Ключ кэша
        """
        # Создаем строку конфигурации
        config_str = str(sorted(feature_config.items()))
        
        # Объединяем хеш данных и конфигурацию
        combined = f"{data_hash}_{config_str}"
        
        # Создаем финальный хеш
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _calculate_data_hash(self, data: Dict[str, pd.DataFrame]) -> str:
        """
        Расчет хеша данных для кэширования
        
        Args:
            data: Словарь с данными по таймфреймам
            
        Returns:
            Хеш данных
        """
        # Создаем строку с информацией о данных
        data_info = []
        for timeframe, df in data.items():
            data_info.append(f"{timeframe}:{len(df)}:{df.index.min()}:{df.index.max()}")
        
        data_str = "|".join(sorted(data_info))
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def generate_features(self, data: Dict[str, pd.DataFrame], 
                         feature_config: Optional[Dict] = None) -> pd.DataFrame:
        """
        Генерация признаков
        
        Args:
            data: Словарь с данными по таймфреймам
            feature_config: Конфигурация признаков (опционально)
            
        Returns:
            DataFrame с признаками
        """
        self.logger.info("🔬 Генерация признаков")
        
        # Используем конфигурацию по умолчанию если не указана
        if feature_config is None:
            feature_config = {
                'include_technical_indicators': self.config.features.include_technical_indicators,
                'include_multi_timeframe': self.config.features.include_multi_timeframe,
                'include_lag_features': self.config.features.include_lag_features,
                'include_rolling_features': self.config.features.include_rolling_features,
                'lag_windows': self.config.features.lag_windows,
                'roll_windows': self.config.features.roll_windows
            }
        
        # Проверяем кэш
        data_hash = self._calculate_data_hash(data)
        cache_key = self._generate_cache_key(data_hash, feature_config)
        
        cached_features = self._load_from_cache(cache_key)
        if cached_features is not None:
            self.logger.info("✅ Признаки загружены из кэша")
            return cached_features
        
        # Генерируем признаки
        self.logger.info("🔧 Создание признаков...")
        
        try:
            # Получаем основные данные (15m)
            df_15m = data.get('15m')
            if df_15m is None:
                raise ValueError("Отсутствуют данные 15m")
            
            # Генерируем базовые признаки 15m
            if feature_config.get('include_technical_indicators', True):
                self.logger.info("   📊 Технические индикаторы")
                df_15m_all = self.engineer.create_all_features(df_15m)
            else:
                df_15m_all = df_15m.copy()
            
            # Генерируем multi-timeframe признаки
            if feature_config.get('include_multi_timeframe', True):
                self.logger.info("   🔄 Multi-timeframe признаки")
                df_multi = self._create_multi_timeframe_features(data)
                
                # Объединяем признаки
                df_full = pd.concat([df_15m_all, df_multi], axis=1)
            else:
                df_full = df_15m_all
            
            # Обрабатываем inf/NaN значения
            if self.config.features.handle_inf_nan:
                self.logger.info("   🧹 Очистка inf/NaN значений")
                df_full = self._clean_features(df_full)
            
            # Валидируем признаки
            if self.config.features.validate_features:
                self.logger.info("   ✅ Валидация признаков")
                FeatureValidator.validate_feature_quality(df_full)
                FeatureValidator.validate_feature_types(df_full)
            
            # Сохраняем в кэш
            self._save_to_cache(cache_key, df_full, data_hash, feature_config)
            
            self.logger.info(f"✅ Признаки сгенерированы: {df_full.shape}")
            return df_full
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка генерации признаков: {e}")
            raise
    
    def _create_multi_timeframe_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Создание multi-timeframe признаков
        
        Args:
            data: Словарь с данными по таймфреймам
            
        Returns:
            DataFrame с multi-timeframe признаками
        """
        df_15m = data.get('15m')
        df_1h = data.get('1h')
        df_4h = data.get('4h')
        df_1d = data.get('1d')
        
        # Создаем multi-timeframe признаки
        df_multi = self.engineer.create_multi_timeframe_features(df_15m, df_1h, df_4h, df_1d)
        
        # Оставляем только multi-TF признаки
        multi_cols = [
            col for col in df_multi.columns 
            if any(prefix in col for prefix in ['rsi_1h', 'trend_1h', 'macd_4h', 'adx_4h', 'trend_1d'])
        ]
        
        if multi_cols:
            df_multi = df_multi[multi_cols]
        else:
            # Если нет multi-TF признаков, создаем пустой DataFrame
            df_multi = pd.DataFrame(index=df_15m.index)
        
        return df_multi
    
    def _clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Очистка признаков от inf/NaN значений и константных признаков
        
        Args:
            df: DataFrame с признаками
            
        Returns:
            Очищенный DataFrame
        """
        # Удаляем константные признаки
        constant_features = []
        for col in df.columns:
            if df[col].nunique() <= 1:
                constant_features.append(col)
        
        if constant_features:
            self.logger.info(f"   🗑️ Удаляем {len(constant_features)} константных признаков")
            df = df.drop(columns=constant_features)
        
        # Заменяем inf на NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Заполняем NaN значения
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Если остались NaN, заполняем нулями
        df = df.fillna(0)
        
        return df
    
    def _load_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """
        Загрузка признаков из кэша
        
        Args:
            cache_key: Ключ кэша
            
        Returns:
            DataFrame с признаками или None
        """
        if not self.config.features.cache_features:
            return None
        
        # Проверяем кэш в памяти
        if cache_key in self._memory_cache:
            metadata = self._cache_metadata.get(cache_key, {})
            if self._is_cache_valid(metadata):
                return self._memory_cache[cache_key]
        
        # Проверяем кэш на диске
        cache_path = self.cache_root / f"{cache_key}.pkl"
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                
                # Проверяем валидность кэша
                if isinstance(cached_data, dict) and 'features' in cached_data:
                    metadata = cached_data.get('metadata', {})
                    if self._is_cache_valid(metadata):
                        # Сохраняем в память
                        self._memory_cache[cache_key] = cached_data['features']
                        self._cache_metadata[cache_key] = metadata
                        return cached_data['features']
            except Exception as e:
                self.logger.warning(f"⚠️ Ошибка загрузки кэша признаков: {e}")
        
        return None
    
    def _save_to_cache(self, cache_key: str, features: pd.DataFrame, 
                      data_hash: str, feature_config: Dict[str, Any]):
        """
        Сохранение признаков в кэш
        
        Args:
            cache_key: Ключ кэша
            features: DataFrame с признаками
            data_hash: Хеш данных
            feature_config: Конфигурация признаков
        """
        if not self.config.features.cache_features:
            return
        
        cache_path = self.cache_root / f"{cache_key}.pkl"
        
        try:
            # Создаем метаданные кэша
            metadata = {
                'created_at': datetime.now().isoformat(),
                'data_hash': data_hash,
                'feature_config': feature_config,
                'shape': features.shape,
                'columns': list(features.columns),
                'features_hash': self._calculate_features_hash(features)
            }
            
            # Сохраняем в память
            self._memory_cache[cache_key] = features
            self._cache_metadata[cache_key] = metadata
            
            # Сохраняем на диск
            cache_data = {
                'features': features,
                'metadata': metadata
            }
            
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Ошибка сохранения кэша признаков: {e}")
    
    def _is_cache_valid(self, metadata: Dict[str, Any]) -> bool:
        """
        Проверка валидности кэша признаков
        
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
    
    def _calculate_features_hash(self, features: pd.DataFrame) -> str:
        """
        Расчет хеша признаков для проверки целостности
        
        Args:
            features: DataFrame с признаками
            
        Returns:
            Хеш признаков
        """
        # Используем первые и последние строки для быстрого хеша
        sample_features = pd.concat([features.head(10), features.tail(10)])
        features_str = sample_features.to_string()
        return hashlib.md5(features_str.encode()).hexdigest()
    
    def validate_features_for_model(self, features: pd.DataFrame, 
                                  expected_features: List[str]) -> bool:
        """
        Валидация признаков для конкретной модели
        
        Args:
            features: DataFrame с признаками
            expected_features: Список ожидаемых признаков
            
        Returns:
            True если признаки валидны
        """
        try:
            # Проверяем наличие всех признаков
            FeatureValidator.validate_features(features, expected_features)
            
            # Проверяем качество признаков
            FeatureValidator.validate_feature_quality(features)
            
            # Проверяем типы признаков
            FeatureValidator.validate_feature_types(features)
            
            return True
            
        except ValidationError as e:
            self.logger.error(f"❌ Ошибка валидации признаков: {e}")
            return False
    
    def get_feature_info(self, features: pd.DataFrame) -> Dict[str, Any]:
        """
        Получение информации о признаках
        
        Args:
            features: DataFrame с признаками
            
        Returns:
            Словарь с информацией о признаках
        """
        info = {
            'shape': features.shape,
            'total_features': len(features.columns),
            'numeric_features': len(features.select_dtypes(include=[np.number]).columns),
            'memory_usage': features.memory_usage(deep=True).sum(),
            'missing_values': features.isnull().sum().sum(),
            'inf_values': np.isinf(features.select_dtypes(include=[np.number])).sum().sum(),
            'feature_types': features.dtypes.value_counts().to_dict()
        }
        
        # Анализ признаков по категориям
        feature_categories = {
            'technical_indicators': [col for col in features.columns if any(indicator in col.lower() for indicator in ['rsi', 'macd', 'ema', 'sma', 'bollinger', 'stochastic'])],
            'lag_features': [col for col in features.columns if 'lag' in col.lower()],
            'rolling_features': [col for col in features.columns if 'rolling' in col.lower() or 'roll' in col.lower()],
            'multi_timeframe': [col for col in features.columns if any(tf in col for tf in ['_1h', '_4h', '_1d'])],
            'price_features': [col for col in features.columns if any(price in col.lower() for price in ['open', 'high', 'low', 'close', 'volume'])],
            'other': []
        }
        
        # Определяем "другие" признаки
        all_categorized = set()
        for category_features in feature_categories.values():
            all_categorized.update(category_features)
        
        feature_categories['other'] = [col for col in features.columns if col not in all_categorized]
        
        # Добавляем количество признаков по категориям
        for category, category_features in feature_categories.items():
            info[f'{category}_count'] = len(category_features)
        
        return info
    
    def select_features(self, features, target: pd.Series, 
                       method: str = 'permutation', threshold: float = 0.01, 
                       remove_correlated: bool = True, correlation_threshold: float = 0.95,
                       n_features: Optional[int] = None, feature_names: Optional[List[str]] = None) -> List[str]:
        """
        Отбор признаков
        
        Args:
            features: DataFrame с признаками
            target: Series с таргетом
            method: Метод отбора ('permutation', 'correlation', 'mutual_info', 'combined', 'recursive')
            threshold: Порог для отбора
            remove_correlated: Удалять ли коррелированные признаки
            correlation_threshold: Порог корреляции для удаления
            n_features: Количество признаков для отбора (для recursive метода)
            
        Returns:
            Список отобранных признаков
        """
        self.logger.info(f"🎯 Отбор признаков методом: {method}")
        
        # Преобразуем numpy массив в DataFrame если нужно
        if isinstance(features, np.ndarray):
            if feature_names is None:
                feature_names = [f'feature_{i}' for i in range(features.shape[1])]
            features_df = pd.DataFrame(features, columns=feature_names)
        else:
            features_df = features
            feature_names = list(features.columns)
        
        # Сначала удаляем сильно коррелированные признаки
        if remove_correlated:
            self.logger.info("🔗 Удаление коррелированных признаков")
            features_clean = self._remove_correlated_features(features_df, correlation_threshold)
            self.logger.info(f"   Осталось {len(features_clean.columns)} признаков из {len(features_df.columns)}")
        else:
            features_clean = features_df
        
        # Выбираем метод отбора
        if method == 'permutation':
            selected = self._select_features_permutation(features_clean, target, threshold)
        elif method == 'correlation':
            selected = self._select_features_correlation(features_clean, target, threshold)
        elif method == 'mutual_info':
            selected = self._select_features_mutual_info(features_clean, target, threshold)
        elif method == 'combined':
            selected = self._select_features_combined(features_clean, target, threshold)
        elif method == 'recursive':
            if n_features is None:
                n_features = min(50, len(features_clean.columns) // 2)
            selected = self._select_features_recursive(features_clean, target, n_features)
        elif method == 'stability':
            selected = self._select_features_stability(features_clean, target, threshold)
        else:
            raise ValueError(f"Неизвестный метод отбора признаков: {method}")
        
        # Возвращаем понятные названия признаков
        if isinstance(features, np.ndarray):
            # Если исходные данные были numpy массивом, возвращаем реальные названия
            selected_real_names = []
            for feat in selected:
                if feat in feature_names:
                    selected_real_names.append(feat)
                elif feat.startswith('feature_'):
                    # Извлекаем индекс из feature_X
                    try:
                        idx = int(feat.split('_')[1])
                        if idx < len(feature_names):
                            selected_real_names.append(feature_names[idx])
                    except (ValueError, IndexError):
                        selected_real_names.append(feat)
            return selected_real_names
        else:
            return selected
        
        # Логируем результат с понятными названиями
        if selected:
            # Показываем первые 5 признаков как примеры
            examples = selected[:5]
            if len(selected) > 5:
                examples_str = ", ".join(examples) + f" ... и еще {len(selected)-5}"
            else:
                examples_str = ", ".join(examples)
            
            self.logger.info(f"✅ Отобрано {len(selected)} признаков: {examples_str}")
        else:
            self.logger.info(f"✅ Отобрано {len(selected)} признаков")
        
        return selected
    
    def _select_features_permutation(self, features: pd.DataFrame, target: pd.Series, 
                                   threshold: float) -> List[str]:
        """
        Отбор признаков методом permutation importance
        
        Args:
            features: DataFrame с признаками
            target: Series с таргетом
            threshold: Порог важности
            
        Returns:
            Список отобранных признаков
        """
        from sklearn.inspection import permutation_importance
        from sklearn.ensemble import RandomForestRegressor
        
        # Обучаем модель с лучшими параметрами для стабильности
        model = RandomForestRegressor(
            n_estimators=200,  # Больше деревьев для стабильности
            max_depth=10,      # Ограничиваем глубину
            min_samples_split=10,  # Минимум сэмплов для разделения
            min_samples_leaf=5,    # Минимум сэмплов в листе
            random_state=42,
            n_jobs=-1  # Используем все ядра
        )
        model.fit(features, target)
        
        # Рассчитываем permutation importance с оптимизированными параметрами
        perm_importance = permutation_importance(
            model, features, target, 
            n_repeats=5,   # Уменьшаем для скорости
            random_state=42,
            n_jobs=-1
        )
        
        # Отбираем признаки выше порога с учетом стандартного отклонения
        selected_features = []
        for i, col in enumerate(features.columns):
            mean_importance = perm_importance.importances_mean[i]
            std_importance = perm_importance.importances_std[i]
            
            # Признак важен если средняя важность выше порога И стандартное отклонение не слишком большое
            if (mean_importance > threshold and 
                std_importance < mean_importance * 0.5):  # Стабильность > 50%
                selected_features.append(col)
        
        # Сортируем по важности
        selected_features.sort(key=lambda x: perm_importance.importances_mean[features.columns.get_loc(x)], reverse=True)
        
        self.logger.info(f"✅ Отобрано {len(selected_features)} признаков из {len(features.columns)}")
        return selected_features
    
    def _select_features_correlation(self, features: pd.DataFrame, target: pd.Series, 
                                   threshold: float) -> List[str]:
        """
        Отбор признаков по корреляции с таргетом
        
        Args:
            features: DataFrame с признаками
            target: Series с таргетом
            threshold: Порог корреляции
            
        Returns:
            Список отобранных признаков
        """
        # Преобразуем target в Series если это numpy массив
        if isinstance(target, np.ndarray):
            target_series = pd.Series(target, index=features.index)
        else:
            target_series = target
        
        # Рассчитываем корреляции
        correlations = features.corrwith(target_series).abs()
        
        # Отбираем признаки выше порога
        selected_features = correlations[correlations > threshold].index.tolist()
        
        self.logger.info(f"✅ Отобрано {len(selected_features)} признаков из {len(features.columns)}")
        return selected_features
    
    def _select_features_mutual_info(self, features: pd.DataFrame, target: pd.Series, 
                                   threshold: float) -> List[str]:
        """
        Отбор признаков по mutual information
        
        Args:
            features: DataFrame с признаками
            target: Series с таргетом
            threshold: Порог mutual information
            
        Returns:
            Список отобранных признаков
        """
        from sklearn.feature_selection import mutual_info_regression
        
        # Рассчитываем mutual information
        mi_scores = mutual_info_regression(features, target, random_state=42)
        
        # Отбираем признаки выше порога и сортируем по важности
        feature_scores = list(zip(features.columns, mi_scores))
        selected_features = [feat for feat, score in feature_scores if score > threshold]
        
        # Сортируем по важности (mutual information)
        if selected_features:
            selected_scores = [(feat, score) for feat, score in feature_scores if feat in selected_features]
            selected_scores.sort(key=lambda x: x[1], reverse=True)
            selected_features = [feat for feat, score in selected_scores]
        
        self.logger.info(f"✅ Отобрано {len(selected_features)} признаков из {len(features.columns)}")
        return selected_features
    
    def _remove_correlated_features(self, features: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
        """
        Удаление сильно коррелированных признаков
        
        Args:
            features: DataFrame с признаками
            threshold: Порог корреляции для удаления
            
        Returns:
            DataFrame без коррелированных признаков
        """
        # Рассчитываем корреляционную матрицу
        corr_matrix = features.corr().abs()
        
        # Находим верхний треугольник матрицы
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Находим признаки для удаления
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        
        if to_drop:
            self.logger.info(f"   Удаляем {len(to_drop)} коррелированных признаков")
            return features.drop(columns=to_drop)
        
        return features
    
    def _select_features_combined(self, features: pd.DataFrame, target: pd.Series, 
                                threshold: float) -> List[str]:
        """
        Комбинированный отбор признаков (пересечение методов)
        
        Args:
            features: DataFrame с признаками
            target: Series с таргетом
            threshold: Порог для отбора
            
        Returns:
            Список отобранных признаков
        """
        self.logger.info("🔄 Комбинированный отбор признаков")
        
        # Получаем результаты всех методов с таймаутом для медленных
        methods = ['correlation', 'mutual_info', 'permutation']  # correlation самый быстрый
        selected_sets = []
        
        for method in methods:
            try:
                self.logger.info(f"   Запуск метода {method}...")
                
                if method == 'permutation':
                    # Для permutation используем только если данных не слишком много
                    if len(features) > 50000:  # Если больше 50k строк, пропускаем permutation
                        self.logger.info(f"   Пропускаем {method} (слишком много данных)")
                        continue
                    selected = self._select_features_permutation(features, target, threshold)
                elif method == 'correlation':
                    selected = self._select_features_correlation(features, target, threshold)
                elif method == 'mutual_info':
                    selected = self._select_features_mutual_info(features, target, threshold)
                
                selected_sets.append(set(selected))
                self.logger.info(f"   {method}: {len(selected)} признаков")
                
            except Exception as e:
                self.logger.warning(f"   Ошибка в методе {method}: {e}")
                continue
        
        if not selected_sets:
            return []
        
        # Находим пересечение всех методов
        final_selected = set.intersection(*selected_sets)
        
        # Если пересечение слишком маленькое, берем объединение
        if len(final_selected) < 5:
            self.logger.info("   Пересечение слишком маленькое, используем объединение")
            final_selected = set.union(*selected_sets)
        
        return list(final_selected)
    
    def _select_features_recursive(self, features: pd.DataFrame, target: pd.Series, 
                                 n_features: int) -> List[str]:
        """
        Рекурсивный отбор признаков
        
        Args:
            features: DataFrame с признаками
            target: Series с таргетом
            n_features: Количество признаков для отбора
            
        Returns:
            Список отобранных признаков
        """
        from sklearn.feature_selection import RFE
        from sklearn.ensemble import RandomForestRegressor
        
        self.logger.info(f"🔄 Рекурсивный отбор {n_features} признаков")
        
        # Создаем модель для RFE
        estimator = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        
        # Создаем селектор
        selector = RFE(estimator, n_features_to_select=n_features, step=0.1)
        
        # Выполняем отбор
        selector.fit(features, target)
        
        # Получаем отобранные признаки
        selected_features = features.columns[selector.support_].tolist()
        
        return selected_features
    
    def _select_features_stability(self, features: pd.DataFrame, target: pd.Series, 
                                 threshold: float) -> List[str]:
        """
        Отбор признаков с анализом стабильности
        
        Args:
            features: DataFrame с признаками
            target: Series с таргетом
            threshold: Порог для отбора
            
        Returns:
            Список стабильно важных признаков
        """
        from sklearn.model_selection import KFold
        from sklearn.inspection import permutation_importance
        from sklearn.ensemble import RandomForestRegressor
        
        self.logger.info("🔄 Анализ стабильности важности признаков")
        
        # Настройки для анализа стабильности
        n_splits = 5
        n_repeats = 5
        
        # Словарь для хранения важности признаков
        feature_importance = {col: [] for col in features.columns}
        
        # K-fold кросс-валидация
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(features)):
            X_train, X_val = features.iloc[train_idx], features.iloc[val_idx]
            y_train, y_val = target.iloc[train_idx], target.iloc[val_idx]
            
            # Обучаем модель
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Рассчитываем permutation importance
            perm_importance = permutation_importance(model, X_val, y_val, 
                                                   n_repeats=n_repeats, random_state=42)
            
            # Сохраняем важность для каждого признака
            for i, col in enumerate(features.columns):
                feature_importance[col].append(perm_importance.importances_mean[i])
        
        # Рассчитываем среднюю важность и стабильность
        feature_stats = {}
        for col, importances in feature_importance.items():
            mean_importance = np.mean(importances)
            std_importance = np.std(importances)
            stability = 1 - (std_importance / (mean_importance + 1e-8))  # Коэффициент вариации
            
            feature_stats[col] = {
                'mean_importance': mean_importance,
                'std_importance': std_importance,
                'stability': stability
            }
        
        # Отбираем стабильно важные признаки
        selected_features = []
        for col, stats in feature_stats.items():
            if (stats['mean_importance'] > threshold and 
                stats['stability'] > 0.5):  # Стабильность > 50%
                selected_features.append(col)
        
        # Сортируем по важности
        selected_features.sort(key=lambda x: feature_stats[x]['mean_importance'], reverse=True)
        
        self.logger.info(f"   Найдено {len(selected_features)} стабильно важных признаков")
        
        return selected_features
    
    def clear_cache(self):
        """
        Очистка кэша признаков
        """
        # Очищаем кэш в памяти
        self._memory_cache.clear()
        self._cache_metadata.clear()
        
        # Очищаем кэш на диске
        for cache_file in self.cache_root.glob("*.pkl"):
            cache_file.unlink()
        
        self.logger.info("🗑️ Кэш признаков очищен")
    
    def get_info(self) -> Dict[str, Any]:
        """
        Получение информации о менеджере признаков
        
        Returns:
            Словарь с информацией
        """
        return {
            'cache_root': str(self.cache_root),
            'cache_enabled': self.config.features.cache_features,
            'validation_enabled': self.config.features.validate_features,
            'handle_inf_nan': self.config.features.handle_inf_nan,
            'memory_cache_size': len(self._memory_cache),
            'include_technical_indicators': self.config.features.include_technical_indicators,
            'include_multi_timeframe': self.config.features.include_multi_timeframe,
            'include_lag_features': self.config.features.include_lag_features,
            'include_rolling_features': self.config.features.include_rolling_features
        } 