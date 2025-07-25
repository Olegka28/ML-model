#!/usr/bin/env python3
"""
🔄 Backtesting Module - модуль для бектестинга торговых стратегий

Модуль для тестирования ML моделей на исторических данных с реалистичной симуляцией торговли.
"""

from .backtester import Backtester
from .strategy import TradingStrategy
from .portfolio import Portfolio
from .metrics import BacktestMetrics

__all__ = [
    'Backtester',
    'TradingStrategy', 
    'Portfolio',
    'BacktestMetrics'
] 