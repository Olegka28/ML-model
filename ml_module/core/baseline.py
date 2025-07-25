#!/usr/bin/env python3
"""
📊 Baseline Models - базовые модели для сравнения

Модуль с простыми базовыми моделями для сравнения производительности ML моделей.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import logging

class BaselineModels:
    """
    Класс для создания и оценки базовых моделей
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def create_regression_baselines(self, X_train: np.ndarray, y_train: np.ndarray, 
                                  X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Создает и оценивает базовые модели для регрессии
        
        Args:
            X_train, y_train: Тренировочные данные
            X_test, y_test: Тестовые данные
            
        Returns:
            Словарь с метриками для каждой базовой модели
        """
        baselines = {}
        
        # 1. Dummy Regressor (среднее значение)
        dummy_mean = DummyRegressor(strategy='mean')
        dummy_mean.fit(X_train, y_train)
        y_pred_mean = dummy_mean.predict(X_test)
        
        baselines['dummy_mean'] = {
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred_mean)),
            'mae': mean_absolute_error(y_test, y_pred_mean),
            'r2': r2_score(y_test, y_pred_mean)
        }
        
        # 2. Dummy Regressor (медиана)
        dummy_median = DummyRegressor(strategy='median')
        dummy_median.fit(X_train, y_train)
        y_pred_median = dummy_median.predict(X_test)
        
        baselines['dummy_median'] = {
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred_median)),
            'mae': mean_absolute_error(y_test, y_pred_median),
            'r2': r2_score(y_test, y_pred_median)
        }
        
        # 3. Linear Regression
        try:
            lr = LinearRegression()
            lr.fit(X_train, y_train)
            y_pred_lr = lr.predict(X_test)
            
            baselines['linear_regression'] = {
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred_lr)),
                'mae': mean_absolute_error(y_test, y_pred_lr),
                'r2': r2_score(y_test, y_pred_lr)
            }
        except Exception as e:
            self.logger.warning(f"Ошибка в Linear Regression: {e}")
            baselines['linear_regression'] = {'rmse': float('inf'), 'mae': float('inf'), 'r2': -float('inf')}
        
        # 4. Random Forest (простой)
        try:
            rf = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
            rf.fit(X_train, y_train)
            y_pred_rf = rf.predict(X_test)
            
            baselines['random_forest_simple'] = {
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred_rf)),
                'mae': mean_absolute_error(y_test, y_pred_rf),
                'r2': r2_score(y_test, y_pred_rf)
            }
        except Exception as e:
            self.logger.warning(f"Ошибка в Random Forest: {e}")
            baselines['random_forest_simple'] = {'rmse': float('inf'), 'mae': float('inf'), 'r2': -float('inf')}
        
        # 5. Persistence Model (предсказание = последнее значение)
        if len(y_test) > 1:
            y_pred_persist = np.roll(y_test, 1)
            y_pred_persist[0] = y_train[-1] if len(y_train) > 0 else 0
            
            baselines['persistence'] = {
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred_persist)),
                'mae': mean_absolute_error(y_test, y_pred_persist),
                'r2': r2_score(y_test, y_pred_persist)
            }
        
        return baselines
    
    def create_classification_baselines(self, X_train: np.ndarray, y_train: np.ndarray, 
                                      X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Создает и оценивает базовые модели для классификации
        
        Args:
            X_train, y_train: Тренировочные данные
            X_test, y_test: Тестовые данные
            
        Returns:
            Словарь с метриками для каждой базовой модели
        """
        baselines = {}
        
        # 1. Dummy Classifier (most frequent)
        dummy_freq = DummyClassifier(strategy='most_frequent', random_state=42)
        dummy_freq.fit(X_train, y_train)
        y_pred_freq = dummy_freq.predict(X_test)
        
        baselines['dummy_most_frequent'] = {
            'accuracy': accuracy_score(y_test, y_pred_freq),
            'f1': f1_score(y_test, y_pred_freq, average='weighted'),
            'precision': precision_score(y_test, y_pred_freq, average='weighted'),
            'recall': recall_score(y_test, y_pred_freq, average='weighted')
        }
        
        # 2. Dummy Classifier (stratified)
        dummy_strat = DummyClassifier(strategy='stratified', random_state=42)
        dummy_strat.fit(X_train, y_train)
        y_pred_strat = dummy_strat.predict(X_test)
        
        baselines['dummy_stratified'] = {
            'accuracy': accuracy_score(y_test, y_pred_strat),
            'f1': f1_score(y_test, y_pred_strat, average='weighted'),
            'precision': precision_score(y_test, y_pred_strat, average='weighted'),
            'recall': recall_score(y_test, y_pred_strat, average='weighted')
        }
        
        # 3. Logistic Regression
        try:
            lr = LogisticRegression(random_state=42, max_iter=1000)
            lr.fit(X_train, y_train)
            y_pred_lr = lr.predict(X_test)
            
            baselines['logistic_regression'] = {
                'accuracy': accuracy_score(y_test, y_pred_lr),
                'f1': f1_score(y_test, y_pred_lr, average='weighted'),
                'precision': precision_score(y_test, y_pred_lr, average='weighted'),
                'recall': recall_score(y_test, y_pred_lr, average='weighted')
            }
        except Exception as e:
            self.logger.warning(f"Ошибка в Logistic Regression: {e}")
            baselines['logistic_regression'] = {'accuracy': 0, 'f1': 0, 'precision': 0, 'recall': 0}
        
        # 4. Random Forest (простой)
        try:
            rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
            rf.fit(X_train, y_train)
            y_pred_rf = rf.predict(X_test)
            
            baselines['random_forest_simple'] = {
                'accuracy': accuracy_score(y_test, y_pred_rf),
                'f1': f1_score(y_test, y_pred_rf, average='weighted'),
                'precision': precision_score(y_test, y_pred_rf, average='weighted'),
                'recall': recall_score(y_test, y_pred_rf, average='weighted')
            }
        except Exception as e:
            self.logger.warning(f"Ошибка в Random Forest: {e}")
            baselines['random_forest_simple'] = {'accuracy': 0, 'f1': 0, 'precision': 0, 'recall': 0}
        
        return baselines
    
    def get_best_baseline(self, baselines: Dict[str, Dict[str, float]], 
                         task_type: str = 'regression') -> Tuple[str, Dict[str, float]]:
        """
        Находит лучшую базовую модель
        
        Args:
            baselines: Словарь с метриками базовых моделей
            task_type: Тип задачи ('regression' или 'classification')
            
        Returns:
            (best_model_name, best_metrics)
        """
        if not baselines:
            return None, {}
        
        if task_type == 'regression':
            # Для регрессии ищем модель с лучшим R²
            best_model = max(baselines.items(), key=lambda x: x[1].get('r2', -float('inf')))
        else:
            # Для классификации ищем модель с лучшей accuracy
            best_model = max(baselines.items(), key=lambda x: x[1].get('accuracy', 0))
        
        return best_model
    
    def compare_with_baseline(self, model_metrics: Dict[str, float], 
                            baseline_metrics: Dict[str, float], 
                            task_type: str = 'regression') -> Dict[str, Any]:
        """
        Сравнивает производительность модели с базовой моделью
        
        Args:
            model_metrics: Метрики основной модели
            baseline_metrics: Метрики базовой модели
            task_type: Тип задачи
            
        Returns:
            Словарь с результатами сравнения
        """
        comparison = {
            'model_metrics': model_metrics,
            'baseline_metrics': baseline_metrics,
            'improvements': {},
            'is_better_than_baseline': True
        }
        
        if task_type == 'regression':
            # Сравниваем RMSE, MAE, R²
            for metric in ['rmse', 'mae']:
                model_val = model_metrics.get(metric, float('inf'))
                baseline_val = baseline_metrics.get(metric, float('inf'))
                
                if baseline_val > 0 and model_val != float('inf'):
                    improvement = ((baseline_val - model_val) / baseline_val) * 100
                    comparison['improvements'][f'{metric}_improvement'] = improvement
                    comparison['improvements'][f'{metric}_better'] = model_val < baseline_val
                else:
                    comparison['improvements'][f'{metric}_improvement'] = 0
                    comparison['improvements'][f'{metric}_better'] = False
            
            # R² (чем больше, тем лучше)
            model_r2 = model_metrics.get('r2', -float('inf'))
            baseline_r2 = baseline_metrics.get('r2', -float('inf'))
            
            if baseline_r2 != -float('inf') and model_r2 != -float('inf'):
                improvement = ((model_r2 - baseline_r2) / abs(baseline_r2)) * 100 if baseline_r2 != 0 else 0
                comparison['improvements']['r2_improvement'] = improvement
                comparison['improvements']['r2_better'] = model_r2 > baseline_r2
            else:
                comparison['improvements']['r2_improvement'] = 0
                comparison['improvements']['r2_better'] = False
            
            # Общая оценка
            better_metrics = sum([
                comparison['improvements']['rmse_better'],
                comparison['improvements']['mae_better'],
                comparison['improvements']['r2_better']
            ])
            comparison['is_better_than_baseline'] = better_metrics >= 2
            
        else:  # classification
            # Сравниваем accuracy, f1, precision, recall
            for metric in ['accuracy', 'f1', 'precision', 'recall']:
                model_val = model_metrics.get(metric, 0)
                baseline_val = baseline_metrics.get(metric, 0)
                
                if baseline_val > 0:
                    improvement = ((model_val - baseline_val) / baseline_val) * 100
                    comparison['improvements'][f'{metric}_improvement'] = improvement
                    comparison['improvements'][f'{metric}_better'] = model_val > baseline_val
                else:
                    comparison['improvements'][f'{metric}_improvement'] = 0
                    comparison['improvements'][f'{metric}_better'] = False
            
            # Общая оценка
            better_metrics = sum([
                comparison['improvements']['accuracy_better'],
                comparison['improvements']['f1_better']
            ])
            comparison['is_better_than_baseline'] = better_metrics >= 1
        
        return comparison
    
    def create_baseline_report(self, X_train: np.ndarray, y_train: np.ndarray, 
                             X_test: np.ndarray, y_test: np.ndarray,
                             task_type: str = 'regression') -> Dict[str, Any]:
        """
        Создает полный отчет по базовым моделям
        
        Args:
            X_train, y_train: Тренировочные данные
            X_test, y_test: Тестовые данные
            task_type: Тип задачи
            
        Returns:
            Полный отчет с базовыми моделями
        """
        if task_type == 'regression':
            baselines = self.create_regression_baselines(X_train, y_train, X_test, y_test)
        else:
            baselines = self.create_classification_baselines(X_train, y_train, X_test, y_test)
        
        best_baseline_name, best_baseline_metrics = self.get_best_baseline(baselines, task_type)
        
        report = {
            'task_type': task_type,
            'all_baselines': baselines,
            'best_baseline': {
                'name': best_baseline_name,
                'metrics': best_baseline_metrics
            },
            'data_info': {
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'features': X_train.shape[1] if len(X_train.shape) > 1 else 1
            }
        }
        
        self.logger.info(f"📊 Создан отчет по базовым моделям для {task_type}")
        self.logger.info(f"🏆 Лучшая базовая модель: {best_baseline_name}")
        
        return report 