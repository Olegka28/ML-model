#!/usr/bin/env python3
"""
📈 Trading Strategy - торговые стратегии для бектестинга

Определение различных торговых стратегий на основе ML предсказаний.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class Signal:
    """Торговый сигнал"""
    timestamp: pd.Timestamp
    action: str  # 'buy', 'sell', 'hold'
    price: float
    confidence: float
    prediction: float
    volume: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

class TradingStrategy(ABC):
    """Абстрактный базовый класс для торговых стратегий"""
    
    def __init__(self, name: str = "Base Strategy"):
        self.name = name
        self.signals = []
    
    @abstractmethod
    def generate_signal(self, prediction: float, confidence: float, 
                       current_price: float, **kwargs) -> Signal:
        """Генерация торгового сигнала"""
        pass
    
    def get_signals(self) -> list:
        """Получить все сигналы"""
        return self.signals

class SimpleThresholdStrategy(TradingStrategy):
    """Простая стратегия на основе пороговых значений"""
    
    def __init__(self, buy_threshold: float = 0.01, sell_threshold: float = -0.01,
                 confidence_threshold: float = 0.5, name: str = "Simple Threshold"):
        super().__init__(name)
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.confidence_threshold = confidence_threshold
    
    def generate_signal(self, prediction: float, confidence: float, 
                       current_price: float, timestamp: pd.Timestamp, **kwargs) -> Signal:
        """Генерация сигнала на основе пороговых значений"""
        
        # Проверяем уверенность
        if confidence < self.confidence_threshold:
            action = 'hold'
        elif prediction > self.buy_threshold:
            action = 'buy'
        elif prediction < self.sell_threshold:
            action = 'sell'
        else:
            action = 'hold'
        
        signal = Signal(
            timestamp=timestamp,
            action=action,
            price=current_price,
            confidence=confidence,
            prediction=prediction
        )
        
        self.signals.append(signal)
        return signal

class DynamicThresholdStrategy(TradingStrategy):
    """Динамическая стратегия с адаптивными порогами"""
    
    def __init__(self, base_threshold: float = 0.01, volatility_multiplier: float = 1.0,
                 confidence_threshold: float = 0.5, name: str = "Dynamic Threshold"):
        super().__init__(name)
        self.base_threshold = base_threshold
        self.volatility_multiplier = volatility_multiplier
        self.confidence_threshold = confidence_threshold
        self.price_history = []
    
    def generate_signal(self, prediction: float, confidence: float, 
                       current_price: float, timestamp: pd.Timestamp, **kwargs) -> Signal:
        """Генерация сигнала с динамическими порогами"""
        
        # Обновляем историю цен
        self.price_history.append(current_price)
        if len(self.price_history) > 20:
            self.price_history.pop(0)
        
        # Рассчитываем волатильность
        if len(self.price_history) > 1:
            returns = np.diff(self.price_history) / self.price_history[:-1]
            volatility = np.std(returns) * np.sqrt(252)  # Годовая волатильность
        else:
            volatility = 0.1  # Значение по умолчанию
        
        # Адаптивные пороги
        dynamic_threshold = self.base_threshold * (1 + volatility * self.volatility_multiplier)
        
        # Проверяем уверенность
        if confidence < self.confidence_threshold:
            action = 'hold'
        elif prediction > dynamic_threshold:
            action = 'buy'
        elif prediction < -dynamic_threshold:
            action = 'sell'
        else:
            action = 'hold'
        
        signal = Signal(
            timestamp=timestamp,
            action=action,
            price=current_price,
            confidence=confidence,
            prediction=prediction
        )
        
        self.signals.append(signal)
        return signal

class MLConfidenceStrategy(TradingStrategy):
    """Стратегия на основе уверенности ML модели"""
    
    def __init__(self, min_confidence: float = 0.6, position_size_multiplier: float = 1.0,
                 name: str = "ML Confidence"):
        super().__init__(name)
        self.min_confidence = min_confidence
        self.position_size_multiplier = position_size_multiplier
    
    def generate_signal(self, prediction: float, confidence: float, 
                       current_price: float, timestamp: pd.Timestamp, **kwargs) -> Signal:
        """Генерация сигнала на основе уверенности модели"""
        
        # Определяем действие на основе предсказания
        if prediction > 0:
            action = 'buy'
        elif prediction < 0:
            action = 'sell'
        else:
            action = 'hold'
        
        # Проверяем минимальную уверенность
        if confidence < self.min_confidence:
            action = 'hold'
        
        # Рассчитываем размер позиции на основе уверенности
        position_size = confidence * self.position_size_multiplier if action != 'hold' else 0
        
        signal = Signal(
            timestamp=timestamp,
            action=action,
            price=current_price,
            confidence=confidence,
            prediction=prediction,
            volume=position_size
        )
        
        self.signals.append(signal)
        return signal

class RiskAdjustedStrategy(TradingStrategy):
    """Стратегия с управлением рисками"""
    
    def __init__(self, risk_per_trade: float = 0.02, max_position_size: float = 0.1,
                 stop_loss_pct: float = 0.05, take_profit_pct: float = 0.1,
                 name: str = "Risk Adjusted"):
        super().__init__(name)
        self.risk_per_trade = risk_per_trade
        self.max_position_size = max_position_size
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
    
    def generate_signal(self, prediction: float, confidence: float, 
                       current_price: float, timestamp: pd.Timestamp, **kwargs) -> Signal:
        """Генерация сигнала с управлением рисками"""
        
        # Определяем действие
        if prediction > 0.01 and confidence > 0.5:
            action = 'buy'
        elif prediction < -0.01 and confidence > 0.5:
            action = 'sell'
        else:
            action = 'hold'
        
        # Рассчитываем размер позиции на основе риска
        if action != 'hold':
            # Размер позиции = риск / (стоп-лосс * цена)
            position_size = min(
                self.risk_per_trade / (self.stop_loss_pct * current_price),
                self.max_position_size
            )
        else:
            position_size = 0
        
        # Рассчитываем стоп-лосс и тейк-профит
        stop_loss = None
        take_profit = None
        
        if action == 'buy':
            stop_loss = current_price * (1 - self.stop_loss_pct)
            take_profit = current_price * (1 + self.take_profit_pct)
        elif action == 'sell':
            stop_loss = current_price * (1 + self.stop_loss_pct)
            take_profit = current_price * (1 - self.take_profit_pct)
        
        signal = Signal(
            timestamp=timestamp,
            action=action,
            price=current_price,
            confidence=confidence,
            prediction=prediction,
            volume=position_size,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        self.signals.append(signal)
        return signal

def create_strategy(strategy_type: str, **kwargs) -> TradingStrategy:
    """Фабрика для создания торговых стратегий"""
    
    strategies = {
        'simple': SimpleThresholdStrategy,
        'dynamic': DynamicThresholdStrategy,
        'confidence': MLConfidenceStrategy,
        'risk_adjusted': RiskAdjustedStrategy
    }
    
    if strategy_type not in strategies:
        raise ValueError(f"Неизвестный тип стратегии: {strategy_type}")
    
    return strategies[strategy_type](**kwargs) 