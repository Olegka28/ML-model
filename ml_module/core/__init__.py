"""
🏗️ Core модуль - базовые компоненты ML системы

Содержит основные классы для управления данными, признаками и моделями.
"""

from .base_system import BaseSystem
from ..data_collector import DataManager
from ..features import FeatureManager
from .model_manager import ModelManager

__all__ = [
    'BaseSystem',
    'DataManager', 
    'FeatureManager',
    'ModelManager'
] 