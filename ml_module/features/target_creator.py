#!/usr/bin/env python3
"""
🎯 TargetCreator - создание целевых переменных для криптовалют

Оптимизированный модуль для создания реалистичных таргетов для ML моделей криптовалют.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

class TargetCreator:
    """
    Создание различных типов целевых переменных для ML моделей криптовалют
    """
    
    def __init__(self):
        pass
    
    def create_crypto_clipped_target(self, df: pd.DataFrame, horizon: int, clip_percentile: float = 0.90) -> pd.Series:
        """
        🥇 Крипто-оптимизированное обрезание - лучший выбор для регрессии
        
        Args:
            df: DataFrame с OHLCV данными
            horizon: Горизонт предсказания в барах
            clip_percentile: Процент обрезания (0.90 = 90% перцентиль)
            
        Returns:
            Series с обрезанным таргетом
        """
        future_price = df['close'].shift(-horizon)
        target = (future_price - df['close']) / df['close']
        
        # Агрессивное обрезание для крипты (90% вместо 95%)
        lower_bound = target.quantile(1 - clip_percentile)
        upper_bound = target.quantile(clip_percentile)
        clipped_target = np.clip(target, lower_bound, upper_bound)
        
        return pd.Series(clipped_target, index=df.index, name=f'target_crypto_clipped_{horizon}')
    
    def create_volume_weighted_target(self, df: pd.DataFrame, horizon: int, volume_window: int = 20) -> pd.Series:
        """
        🥈 Таргет с учетом объема - учитывает ликвидность
        
        Args:
            df: DataFrame с OHLCV данными
            horizon: Горизонт предсказания в барах
            volume_window: Окно для скользящего среднего объема
            
        Returns:
            Series с объемно-взвешенным таргетом
        """
        future_price = df['close'].shift(-horizon)
        price_change = (future_price - df['close']) / df['close']
        
        # Нормализуем объем относительно скользящего среднего
        volume_ma = df['volume'].rolling(window=volume_window).mean()
        volume_ratio = df['volume'] / volume_ma
        
        # Корректируем таргет на объем (больше объем = больше доверия)
        # Используем квадратный корень для сглаживания эффекта
        volume_weighted = price_change * np.sqrt(volume_ratio)
        
        return pd.Series(volume_weighted, index=df.index, name=f'target_volume_weighted_{horizon}')
    
    def create_volatility_regime_target(self, df: pd.DataFrame, horizon: int, window: int = 50) -> pd.Series:
        """
        🥉 Таргет с учетом режима волатильности - адаптивный
        
        Args:
            df: DataFrame с OHLCV данными
            horizon: Горизонт предсказания в барах
            window: Окно для вычисления волатильности
            
        Returns:
            Series с таргетом, скорректированным на волатильность
        """
        future_price = df['close'].shift(-horizon)
        price_change = (future_price - df['close']) / df['close']
        
        # Определяем режим волатильности
        returns = df['close'].pct_change()
        volatility = returns.rolling(window=window).std()
        vol_ma = volatility.rolling(window=window).mean()
        
        # Режим волатильности: 0 (низкая), 1 (средняя), 2 (высокая)
        vol_regime = np.where(volatility < vol_ma * 0.7, 0,
                             np.where(volatility > vol_ma * 1.3, 2, 1))
        
        # Корректируем таргет на режим волатильности
        # В высокой волатильности уменьшаем таргет
        regime_adjusted = price_change / (1 + vol_regime * 0.5)
        
        return pd.Series(regime_adjusted, index=df.index, name=f'target_vol_regime_{horizon}')
    
    def create_market_regime_target(self, df: pd.DataFrame, horizon: int, short_window: int = 20, long_window: int = 50) -> pd.Series:
        """
        🎯 Таргет с учетом рыночного режима (бычий/медвежий/боковик)
        
        Args:
            df: DataFrame с OHLCV данными
            horizon: Горизонт предсказания в барах
            short_window: Короткое окно для SMA
            long_window: Длинное окно для SMA
            
        Returns:
            Series с таргетом, скорректированным на рыночный режим
        """
        future_price = df['close'].shift(-horizon)
        price_change = (future_price - df['close']) / df['close']
        
        # Определяем рыночный режим
        sma_short = df['close'].rolling(window=short_window).mean()
        sma_long = df['close'].rolling(window=long_window).mean()
        
        # Режим: 1 (бычий), -1 (медвежий), 0 (боковик)
        market_regime = np.where(sma_short > sma_long * 1.02, 1,
                                np.where(sma_short < sma_long * 0.98, -1, 0))
        
        # Корректируем таргет на рыночный режим
        # В бычьем рынке увеличиваем положительные таргеты
        regime_adjusted = price_change * (1 + market_regime * 0.3)
        
        return pd.Series(regime_adjusted, index=df.index, name=f'target_market_regime_{horizon}')
    
    def create_crypto_binary_target(self, df: pd.DataFrame, horizon: int, threshold: float = 0.02) -> pd.Series:
        """
        🎯 Бинарный таргет оптимизированный для крипты
        
        Args:
            df: DataFrame с OHLCV данными
            horizon: Горизонт предсказания в барах
            threshold: Порог для бинарной классификации (2% по умолчанию)
            
        Returns:
            Series с бинарным таргетом (0 или 1)
        """
        future_price = df['close'].shift(-horizon)
        price_change = (future_price - df['close']) / df['close']
        
        # Более высокий порог для крипты (2% вместо 1%)
        binary_target = (price_change >= threshold).astype(int)
        
        return pd.Series(binary_target, index=df.index, name=f'target_crypto_binary_{horizon}')
    
    def create_adaptive_threshold_target(self, df: pd.DataFrame, horizon: int, window: int = 100) -> pd.Series:
        """
        🆕 Адаптивный порог на основе исторической волатильности
        
        Args:
            df: DataFrame с OHLCV данными
            horizon: Горизонт предсказания в барах
            window: Окно для вычисления адаптивного порога
            
        Returns:
            Series с бинарным таргетом с адаптивным порогом
        """
        future_price = df['close'].shift(-horizon)
        price_change = (future_price - df['close']) / df['close']
        
        # Вычисляем адаптивный порог на основе исторической волатильности
        returns = df['close'].pct_change()
        volatility = returns.rolling(window=window).std()
        
        # Адаптивный порог: 1.5 * историческая волатильность
        adaptive_threshold = volatility * 1.5
        
        # Создаем бинарный таргет с адаптивным порогом
        binary_target = (price_change >= adaptive_threshold).astype(int)
        
        return pd.Series(binary_target, index=df.index, name=f'target_adaptive_threshold_{horizon}')
    
    def create_momentum_enhanced_target(self, df: pd.DataFrame, horizon: int, momentum_window: int = 20) -> pd.Series:
        """
        🆕 Улучшенный моментум-таргет с учетом силы тренда
        
        Args:
            df: DataFrame с OHLCV данными
            horizon: Горизонт предсказания в барах
            momentum_window: Окно для вычисления моментума
            
        Returns:
            Series с таргетом, скорректированным на моментум
        """
        future_price = df['close'].shift(-horizon)
        price_change = (future_price - df['close']) / df['close']
        
        # Вычисляем моментум (текущий тренд)
        momentum = df['close'].pct_change(momentum_window)
        
        # Вычисляем силу тренда (R-squared от линейной регрессии)
        def calculate_trend_strength(prices, window):
            strength = pd.Series(index=prices.index, dtype=float)
            for i in range(window, len(prices)):
                y = prices.iloc[i-window:i+1].values
                x = np.arange(len(y))
                if len(y) > 1:
                    correlation = np.corrcoef(x, y)[0, 1]
                    strength.iloc[i] = correlation ** 2 if not np.isnan(correlation) else 0
            return strength
        
        trend_strength = calculate_trend_strength(df['close'], momentum_window)
        
        # Корректируем таргет на моментум и силу тренда
        momentum_adjusted = price_change * (1 + momentum * trend_strength)
        
        return pd.Series(momentum_adjusted, index=df.index, name=f'target_momentum_enhanced_{horizon}')
    
    def create_volume_volatility_target(self, df: pd.DataFrame, horizon: int, window: int = 20) -> pd.Series:
        """
        🆕 Таргет с учетом объема и волатильности одновременно
        
        Args:
            df: DataFrame с OHLCV данными
            horizon: Горизонт предсказания в барах
            window: Окно для вычислений
            
        Returns:
            Series с таргетом, учитывающим объем и волатильность
        """
        future_price = df['close'].shift(-horizon)
        price_change = (future_price - df['close']) / df['close']
        
        # Нормализуем объем
        volume_ma = df['volume'].rolling(window=window).mean()
        volume_ratio = df['volume'] / volume_ma
        
        # Вычисляем волатильность
        returns = df['close'].pct_change()
        volatility = returns.rolling(window=window).std()
        vol_ma = volatility.rolling(window=window).mean()
        vol_ratio = volatility / vol_ma
        
        # Комбинируем объем и волатильность
        # Высокий объем + низкая волатильность = сильный сигнал
        combined_factor = np.sqrt(volume_ratio) / (1 + vol_ratio)
        
        # Корректируем таргет
        adjusted_target = price_change * combined_factor
        
        return pd.Series(adjusted_target, index=df.index, name=f'target_volume_volatility_{horizon}')
    
    def create_target(self, df: pd.DataFrame, target_type: str, horizon: int, **kwargs) -> pd.Series:
        """
        Создание таргета по типу
        
        Args:
            df: DataFrame с данными
            target_type: Тип таргета
            horizon: Горизонт предсказания
            **kwargs: Дополнительные параметры
            
        Returns:
            Series с таргетом
        """
        target_methods = {
            'crypto_clipped': self.create_crypto_clipped_target,
            'volume_weighted': self.create_volume_weighted_target,
            'vol_regime': self.create_volatility_regime_target,
            'market_regime': self.create_market_regime_target,
            'crypto_binary': self.create_crypto_binary_target,
            'adaptive_threshold': self.create_adaptive_threshold_target,
            'momentum_enhanced': self.create_momentum_enhanced_target,
            'volume_volatility': self.create_volume_volatility_target
        }
        
        if target_type not in target_methods:
            available = list(target_methods.keys())
            raise ValueError(f"Неизвестный тип таргета: {target_type}. Доступные: {available}")
        
        return target_methods[target_type](df, horizon, **kwargs)
    
    def create_all_targets(self, df: pd.DataFrame, horizon: int) -> Dict[str, pd.Series]:
        """
        Создает все варианты таргетов для сравнения
        
        Args:
            df: DataFrame с данными
            horizon: Горизонт предсказания
            
        Returns:
            Dict с различными таргетами
        """
        targets = {}
        
        # Основные крипто-таргеты
        targets['crypto_clipped'] = self.create_crypto_clipped_target(df, horizon)
        targets['volume_weighted'] = self.create_volume_weighted_target(df, horizon)
        targets['vol_regime'] = self.create_volatility_regime_target(df, horizon)
        targets['market_regime'] = self.create_market_regime_target(df, horizon)
        
        # Классификационные таргеты
        targets['crypto_binary'] = self.create_crypto_binary_target(df, horizon)
        targets['adaptive_threshold'] = self.create_adaptive_threshold_target(df, horizon)
        
        # Специальные таргеты
        targets['momentum_enhanced'] = self.create_momentum_enhanced_target(df, horizon)
        targets['volume_volatility'] = self.create_volume_volatility_target(df, horizon)
        
        return targets
    
    def get_target_info(self, df: pd.DataFrame, horizon: int) -> Dict[str, Dict[str, Any]]:
        """
        Получение информации о всех таргетах
        
        Args:
            df: DataFrame с данными
            horizon: Горизонт предсказания
            
        Returns:
            Dict с информацией о каждом таргете
        """
        targets = self.create_all_targets(df, horizon)
        info = {}
        
        for name, target in targets.items():
            if target is not None:
                clean_target = target.dropna()
                
                info[name] = {
                    'count': len(clean_target),
                    'mean': float(clean_target.mean()),
                    'std': float(clean_target.std()),
                    'min': float(clean_target.min()),
                    'max': float(clean_target.max()),
                    'type': 'binary' if clean_target.nunique() <= 2 else 'regression'
                }
                
                if info[name]['type'] == 'binary':
                    info[name]['positive_ratio'] = float(clean_target.mean())
                else:
                    info[name]['quantiles'] = {
                        '1%': float(clean_target.quantile(0.01)),
                        '5%': float(clean_target.quantile(0.05)),
                        '25%': float(clean_target.quantile(0.25)),
                        '50%': float(clean_target.quantile(0.50)),
                        '75%': float(clean_target.quantile(0.75)),
                        '95%': float(clean_target.quantile(0.95)),
                        '99%': float(clean_target.quantile(0.99))
                    }
        
        return info 