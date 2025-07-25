#!/usr/bin/env python3
"""
🔬 FeatureEngineer - инженерия признаков

Автономный модуль для создания технических индикаторов и признаков без зависимостей от старых файлов.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    """
    Инженерия признаков для ML моделей
    """
    
    def __init__(self):
        pass
    
    def add_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Добавление базовых признаков цены
        
        Args:
            df: DataFrame с OHLCV данными
            
        Returns:
            DataFrame с добавленными признаками
        """
        df_features = df.copy()
        
        # Базовые признаки цены
        df_features['price_range'] = df_features['high'] - df_features['low']
        df_features['body_size'] = abs(df_features['close'] - df_features['open'])
        df_features['upper_shadow'] = df_features['high'] - np.maximum(df_features['open'], df_features['close'])
        df_features['lower_shadow'] = np.minimum(df_features['open'], df_features['close']) - df_features['low']
        
        # Процентные изменения
        df_features['returns'] = df_features['close'].pct_change()
        df_features['log_returns'] = np.log(df_features['close'] / df_features['close'].shift(1))
        
        # Волатильность
        df_features['volatility'] = df_features['returns'].rolling(window=20).std()
        
        # Объемные признаки
        df_features['volume_ma'] = df_features['volume'].rolling(window=20).mean()
        df_features['volume_ratio'] = df_features['volume'] / df_features['volume_ma']
        
        return df_features
    
    def add_moving_averages(self, df: pd.DataFrame, windows: List[int] = None) -> pd.DataFrame:
        """
        Добавление скользящих средних
        
        Args:
            df: DataFrame с данными
            windows: Список окон для SMA и EMA
            
        Returns:
            DataFrame с добавленными признаками
        """
        if windows is None:
            windows = [5, 10, 20, 50, 100, 200]
        
        df_features = df.copy()
        
        for window in windows:
            # SMA
            df_features[f'sma_{window}'] = df_features['close'].rolling(window=window).mean()
            df_features[f'sma_{window}_slope'] = df_features[f'sma_{window}'].pct_change()
            
            # EMA
            df_features[f'ema_{window}'] = df_features['close'].ewm(span=window).mean()
            df_features[f'ema_{window}_slope'] = df_features[f'ema_{window}'].pct_change()
            
            # Отношение цены к средним
            df_features[f'price_sma_{window}_ratio'] = df_features['close'] / df_features[f'sma_{window}']
            df_features[f'price_ema_{window}_ratio'] = df_features['close'] / df_features[f'ema_{window}']
        
        return df_features
    
    def add_rsi_features(self, df: pd.DataFrame, windows: List[int] = None) -> pd.DataFrame:
        """
        Добавление RSI индикаторов
        
        Args:
            df: DataFrame с данными
            windows: Список окон для RSI
            
        Returns:
            DataFrame с добавленными признаками
        """
        if windows is None:
            windows = [7, 14, 21]
        
        df_features = df.copy()
        
        for window in windows:
            # Вычисляем RSI
            delta = df_features['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            df_features[f'rsi_{window}'] = rsi
            
            # Дополнительные признаки RSI
            df_features[f'rsi_{window}_overbought'] = (rsi > 70).astype(int)
            df_features[f'rsi_{window}_oversold'] = (rsi < 30).astype(int)
            df_features[f'rsi_{window}_slope'] = rsi.pct_change()
        
        return df_features
    
    def add_macd_features(self, df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """
        Добавление MACD индикаторов
        
        Args:
            df: DataFrame с данными
            fast: Быстрое EMA
            slow: Медленное EMA
            signal: Сигнальная линия
            
        Returns:
            DataFrame с добавленными признаками
        """
        df_features = df.copy()
        
        # Вычисляем MACD
        ema_fast = df_features['close'].ewm(span=fast).mean()
        ema_slow = df_features['close'].ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        
        df_features[f'macd_{fast}_{slow}'] = macd_line
        df_features[f'macd_signal_{fast}_{slow}'] = signal_line
        df_features[f'macd_histogram_{fast}_{slow}'] = histogram
        
        # Дополнительные признаки MACD
        df_features[f'macd_cross_above'] = ((macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))).astype(int)
        df_features[f'macd_cross_below'] = ((macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))).astype(int)
        df_features[f'macd_positive'] = (macd_line > 0).astype(int)
        
        return df_features
    
    def add_bollinger_bands(self, df: pd.DataFrame, window: int = 20, std_dev: float = 2) -> pd.DataFrame:
        """
        Добавление Bollinger Bands
        
        Args:
            df: DataFrame с данными
            window: Окно для SMA
            std_dev: Количество стандартных отклонений
            
        Returns:
            DataFrame с добавленными признаками
        """
        df_features = df.copy()
        
        # Вычисляем Bollinger Bands
        sma = df_features['close'].rolling(window=window).mean()
        std = df_features['close'].rolling(window=window).std()
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        df_features[f'bb_upper_{window}'] = upper_band
        df_features[f'bb_lower_{window}'] = lower_band
        df_features[f'bb_middle_{window}'] = sma
        df_features[f'bb_width_{window}'] = upper_band - lower_band
        df_features[f'bb_percent_{window}'] = (df_features['close'] - lower_band) / (upper_band - lower_band)
        
        # Дополнительные признаки
        df_features[f'bb_squeeze_{window}'] = (df_features[f'bb_width_{window}'] < df_features[f'bb_width_{window}'].rolling(20).mean()).astype(int)
        df_features[f'bb_breakout_up_{window}'] = (df_features['close'] > upper_band).astype(int)
        df_features[f'bb_breakout_down_{window}'] = (df_features['close'] < lower_band).astype(int)
        
        return df_features
    
    def add_stochastic_features(self, df: pd.DataFrame, window: int = 14, smooth_k: int = 3, smooth_d: int = 3) -> pd.DataFrame:
        """
        Добавление Stochastic Oscillator
        
        Args:
            df: DataFrame с данными
            window: Окно для %K
            smooth_k: Сглаживание %K
            smooth_d: Сглаживание %D
            
        Returns:
            DataFrame с добавленными признаками
        """
        df_features = df.copy()
        
        # Вычисляем Stochastic
        low_min = df_features['low'].rolling(window=window).min()
        high_max = df_features['high'].rolling(window=window).max()
        
        k_percent = 100 * ((df_features['close'] - low_min) / (high_max - low_min))
        k_smooth = k_percent.rolling(window=smooth_k).mean()
        d_smooth = k_smooth.rolling(window=smooth_d).mean()
        
        df_features[f'stoch_k_{window}'] = k_smooth
        df_features[f'stoch_d_{window}'] = d_smooth
        
        # Дополнительные признаки
        df_features[f'stoch_overbought_{window}'] = (k_smooth > 80).astype(int)
        df_features[f'stoch_oversold_{window}'] = (k_smooth < 20).astype(int)
        df_features[f'stoch_cross_above_{window}'] = ((k_smooth > d_smooth) & (k_smooth.shift(1) <= d_smooth.shift(1))).astype(int)
        df_features[f'stoch_cross_below_{window}'] = ((k_smooth < d_smooth) & (k_smooth.shift(1) >= d_smooth.shift(1))).astype(int)
        
        return df_features
    
    def add_atr_features(self, df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """
        Добавление Average True Range
        
        Args:
            df: DataFrame с данными
            window: Окно для ATR
            
        Returns:
            DataFrame с добавленными признаками
        """
        df_features = df.copy()
        
        # Вычисляем True Range
        high_low = df_features['high'] - df_features['low']
        high_close = np.abs(df_features['high'] - df_features['close'].shift())
        low_close = np.abs(df_features['low'] - df_features['close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(window=window).mean()
        
        df_features[f'atr_{window}'] = atr
        df_features[f'atr_percent_{window}'] = atr / df_features['close']
        
        return df_features
    
    def add_adx_features(self, df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """
        Добавление Average Directional Index
        
        Args:
            df: DataFrame с данными
            window: Окно для ADX
            
        Returns:
            DataFrame с добавленными признаками
        """
        df_features = df.copy()
        
        # Вычисляем +DM и -DM
        high_diff = df_features['high'] - df_features['high'].shift()
        low_diff = df_features['low'].shift() - df_features['low']
        
        plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
        minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
        
        # Сглаживаем с помощью Wilder's smoothing
        plus_dm_smooth = pd.Series(plus_dm, index=df_features.index).rolling(window=window).mean()
        minus_dm_smooth = pd.Series(minus_dm, index=df_features.index).rolling(window=window).mean()
        
        # Вычисляем True Range
        high_low = df_features['high'] - df_features['low']
        high_close = np.abs(df_features['high'] - df_features['close'].shift())
        low_close = np.abs(df_features['low'] - df_features['close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        tr_smooth = pd.Series(true_range, index=df_features.index).rolling(window=window).mean()
        
        # Вычисляем +DI и -DI (избегаем деления на ноль)
        plus_di = np.where(tr_smooth > 0, 100 * (plus_dm_smooth / tr_smooth), 0)
        minus_di = np.where(tr_smooth > 0, 100 * (minus_dm_smooth / tr_smooth), 0)
        
        # Вычисляем DX и ADX (избегаем деления на ноль)
        denominator = plus_di + minus_di
        dx = np.where(denominator > 0, 100 * np.abs(plus_di - minus_di) / denominator, 0)
        adx = pd.Series(dx).rolling(window=window).mean()
        
        # Заполняем оставшиеся NaN значения
        plus_di = pd.Series(plus_di).fillna(0)
        minus_di = pd.Series(minus_di).fillna(0)
        adx = adx.fillna(0)
        
        # Добавляем базовые ADX признаки
        df_features[f'adx_{window}'] = adx
        df_features[f'plus_di_{window}'] = plus_di
        df_features[f'minus_di_{window}'] = minus_di
        
        # Добавляем более разнообразные признаки
        df_features[f'adx_strong_trend_{window}'] = (adx > 25).astype(int)
        df_features[f'adx_very_strong_trend_{window}'] = (adx > 50).astype(int)
        df_features[f'adx_weak_trend_{window}'] = (adx < 20).astype(int)
        
        # Сигналы пересечений
        df_features[f'di_cross_above_{window}'] = ((plus_di > minus_di) & (plus_di.shift(1) <= minus_di.shift(1))).astype(int)
        df_features[f'di_cross_below_{window}'] = ((plus_di < minus_di) & (plus_di.shift(1) >= minus_di.shift(1))).astype(int)
        
        # Дополнительные признаки
        df_features[f'di_spread_{window}'] = plus_di - minus_di
        df_features[f'di_sum_{window}'] = plus_di + minus_di
        df_features[f'adx_slope_{window}'] = adx.pct_change()
        df_features[f'plus_di_slope_{window}'] = plus_di.pct_change()
        df_features[f'minus_di_slope_{window}'] = minus_di.pct_change()
        
        # Нормализованные признаки
        df_features[f'adx_normalized_{window}'] = adx / 100.0
        df_features[f'plus_di_normalized_{window}'] = plus_di / 100.0
        df_features[f'minus_di_normalized_{window}'] = minus_di / 100.0
        
        return df_features
    
    def add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Добавление объемных признаков
        
        Args:
            df: DataFrame с данными
            
        Returns:
            DataFrame с добавленными признаками
        """
        df_features = df.copy()
        
        # On Balance Volume
        obv = (np.sign(df_features['close'].diff()) * df_features['volume']).fillna(0).cumsum()
        df_features['obv'] = obv
        df_features['obv_ma'] = obv.rolling(window=20).mean()
        df_features['obv_ratio'] = obv / df_features['obv_ma']
        
        # Volume Price Trend
        vpt = (df_features['volume'] * df_features['close'].pct_change()).fillna(0).cumsum()
        df_features['vpt'] = vpt
        
        # Money Flow Index
        typical_price = (df_features['high'] + df_features['low'] + df_features['close']) / 3
        money_flow = typical_price * df_features['volume']
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(14).sum()
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(14).sum()
        
        mfi = 100 - (100 / (1 + positive_flow / negative_flow))
        df_features['mfi'] = mfi
        
        # Дополнительные объемные признаки
        df_features['volume_price_trend'] = df_features['volume'] * df_features['close'].pct_change()
        df_features['volume_sma_ratio'] = df_features['volume'] / df_features['volume'].rolling(20).mean()
        df_features['volume_ema_ratio'] = df_features['volume'] / df_features['volume'].ewm(span=20).mean()
        
        return df_features
    
    def add_lag_features(self, df: pd.DataFrame, windows: List[int] = None) -> pd.DataFrame:
        """
        Добавление лаговых признаков
        
        Args:
            df: DataFrame с данными
            windows: Список лагов
            
        Returns:
            DataFrame с добавленными признаками
        """
        if windows is None:
            windows = list(range(1, 11))
        
        df_features = df.copy()
        
        for lag in windows:
            df_features[f'close_lag_{lag}'] = df_features['close'].shift(lag)
            df_features[f'volume_lag_{lag}'] = df_features['volume'].shift(lag)
            df_features[f'returns_lag_{lag}'] = df_features['returns'].shift(lag)
        
        return df_features
    
    def add_rolling_features(self, df: pd.DataFrame, windows: List[int] = None) -> pd.DataFrame:
        """
        Добавление скользящих статистик
        
        Args:
            df: DataFrame с данными
            windows: Список окон
            
        Returns:
            DataFrame с добавленными признаками
        """
        if windows is None:
            windows = [5, 10, 20, 50]
        
        df_features = df.copy()
        
        for window in windows:
            # Статистики цены
            df_features[f'close_rolling_mean_{window}'] = df_features['close'].rolling(window=window).mean()
            df_features[f'close_rolling_std_{window}'] = df_features['close'].rolling(window=window).std()
            df_features[f'close_rolling_min_{window}'] = df_features['close'].rolling(window=window).min()
            df_features[f'close_rolling_max_{window}'] = df_features['close'].rolling(window=window).max()
            
            # Статистики объема
            df_features[f'volume_rolling_mean_{window}'] = df_features['volume'].rolling(window=window).mean()
            df_features[f'volume_rolling_std_{window}'] = df_features['volume'].rolling(window=window).std()
            
            # Статистики доходности
            df_features[f'returns_rolling_mean_{window}'] = df_features['returns'].rolling(window=window).mean()
            df_features[f'returns_rolling_std_{window}'] = df_features['returns'].rolling(window=window).std()
            df_features[f'returns_rolling_skew_{window}'] = df_features['returns'].rolling(window=window).skew()
            df_features[f'returns_rolling_kurt_{window}'] = df_features['returns'].rolling(window=window).kurt()
        
        return df_features
    
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Создание всех признаков
        
        Args:
            df: DataFrame с OHLCV данными
            
        Returns:
            DataFrame со всеми признаками
        """
        df_features = df.copy()
        
        # Добавляем все типы признаков
        df_features = self.add_basic_features(df_features)
        df_features = self.add_moving_averages(df_features)
        df_features = self.add_rsi_features(df_features)
        df_features = self.add_macd_features(df_features)
        df_features = self.add_bollinger_bands(df_features)
        df_features = self.add_stochastic_features(df_features)
        df_features = self.add_atr_features(df_features)
        df_features = self.add_adx_features(df_features)
        df_features = self.add_volume_features(df_features)
        df_features = self.add_lag_features(df_features)
        df_features = self.add_rolling_features(df_features)
        
        # Добавляем дополнительные полезные признаки
        df_features = self.add_momentum_features(df_features)
        df_features = self.add_support_resistance_features(df_features)
        
        # Удаляем inf и -inf значения
        df_features = df_features.replace([np.inf, -np.inf], np.nan)
        
        # Заполняем NaN значения
        df_features = df_features.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        return df_features
    
    def create_multi_timeframe_features(self, df_15m: pd.DataFrame, df_1h: pd.DataFrame = None, 
                                      df_4h: pd.DataFrame = None, df_1d: pd.DataFrame = None) -> pd.DataFrame:
        """
        Создание multi-timeframe признаков
        
        Args:
            df_15m: 15-минутные данные
            df_1h: 1-часовые данные (опционально)
            df_4h: 4-часовые данные (опционально)
            df_1d: Дневные данные (опционально)
            
        Returns:
            DataFrame с multi-timeframe признаками
        """
        df_features = df_15m.copy()
        
        # Добавляем признаки с других таймфреймов
        if df_1h is not None:
            df_features = self._add_timeframe_features(df_features, df_1h, '1h')
        
        if df_4h is not None:
            df_features = self._add_timeframe_features(df_features, df_4h, '4h')
        
        if df_1d is not None:
            df_features = self._add_timeframe_features(df_features, df_1d, '1d')
        
        return df_features
    
    def _add_timeframe_features(self, df_main: pd.DataFrame, df_tf: pd.DataFrame, suffix: str) -> pd.DataFrame:
        """
        Добавление признаков с другого таймфрейма
        
        Args:
            df_main: Основной DataFrame
            df_tf: DataFrame другого таймфрейма
            suffix: Суффикс для имен признаков
            
        Returns:
            DataFrame с добавленными признаками
        """
        # Ресемплируем данные другого таймфрейма к основному
        df_tf_resampled = df_tf.reindex(df_main.index, method='ffill')
        
        # Добавляем основные признаки
        df_main[f'close_{suffix}'] = df_tf_resampled['close']
        df_main[f'volume_{suffix}'] = df_tf_resampled['volume']
        
        # Добавляем технические индикаторы
        df_main[f'rsi_{suffix}'] = self._calculate_rsi(df_tf_resampled['close'])
        df_main[f'macd_{suffix}'] = self._calculate_macd(df_tf_resampled['close'])
        df_main[f'sma_20_{suffix}'] = df_tf_resampled['close'].rolling(20).mean()
        df_main[f'ema_20_{suffix}'] = df_tf_resampled['close'].ewm(span=20).mean()
        
        return df_main
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Вычисление RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
        """Вычисление MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        return ema_fast - ema_slow
    
    def add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Добавление импульсных признаков
        
        Args:
            df: DataFrame с данными
            
        Returns:
            DataFrame с добавленными признаками
        """
        df_features = df.copy()
        
        # Rate of Change (ROC)
        for period in [5, 10, 20]:
            df_features[f'roc_{period}'] = df_features['close'].pct_change(periods=period)
        
        # Williams %R
        for period in [14, 21]:
            highest_high = df_features['high'].rolling(window=period).max()
            lowest_low = df_features['low'].rolling(window=period).min()
            williams_r = -100 * (highest_high - df_features['close']) / (highest_high - lowest_low)
            df_features[f'williams_r_{period}'] = williams_r
        
        # Commodity Channel Index (CCI)
        for period in [14, 20]:
            typical_price = (df_features['high'] + df_features['low'] + df_features['close']) / 3
            sma_tp = typical_price.rolling(window=period).mean()
            mad = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
            cci = (typical_price - sma_tp) / (0.015 * mad)
            df_features[f'cci_{period}'] = cci
        
        # Momentum
        for period in [5, 10, 20]:
            df_features[f'momentum_{period}'] = df_features['close'] - df_features['close'].shift(period)
        
        return df_features
    
    def add_support_resistance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Добавление признаков поддержки и сопротивления
        
        Args:
            df: DataFrame с данными
            
        Returns:
            DataFrame с добавленными признаками
        """
        df_features = df.copy()
        
        # Pivot Points
        df_features['pivot'] = (df_features['high'] + df_features['low'] + df_features['close']) / 3
        df_features['r1'] = 2 * df_features['pivot'] - df_features['low']
        df_features['s1'] = 2 * df_features['pivot'] - df_features['high']
        df_features['r2'] = df_features['pivot'] + (df_features['high'] - df_features['low'])
        df_features['s2'] = df_features['pivot'] - (df_features['high'] - df_features['low'])
        
        # Расстояния до уровней
        df_features['distance_to_r1'] = (df_features['r1'] - df_features['close']) / df_features['close']
        df_features['distance_to_s1'] = (df_features['close'] - df_features['s1']) / df_features['close']
        df_features['distance_to_pivot'] = abs(df_features['close'] - df_features['pivot']) / df_features['close']
        
        # Сигналы пробития
        df_features['above_pivot'] = (df_features['close'] > df_features['pivot']).astype(int)
        df_features['above_r1'] = (df_features['close'] > df_features['r1']).astype(int)
        df_features['below_s1'] = (df_features['close'] < df_features['s1']).astype(int)
        
        # Волатильность на основе pivot
        df_features['pivot_volatility'] = (df_features['high'] - df_features['low']) / df_features['pivot']
        
        return df_features 