#!/usr/bin/env python3
"""
📚 ModelVersioning - улучшенная версификация моделей

Автономный модуль для управления версиями моделей с умным сравнением и валидацией.
"""

import os
import json
import shutil
import hashlib
import pickle
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple, List, Union
from pathlib import Path
import logging

class ModelVersioning:
    """
    Улучшенный класс для автоматической версификации моделей с умным сравнением метрик.
    Включает валидацию, статистическую значимость, автоматическую очистку и веса метрик.
    """
    
    def __init__(self, models_root: str = 'models', 
                 min_improvement: float = 0.001,
                 max_backups: int = 10,
                 cleanup_days: int = 30):
        """
        Инициализация системы версификации
        
        Args:
            models_root: Корневая папка для моделей
            min_improvement: Минимальное улучшение для замены модели (0.001 = 0.1%)
            max_backups: Максимальное количество резервных копий
            cleanup_days: Автоматически удалять версии старше N дней
        """
        self.models_root = Path(models_root)
        self.models_root.mkdir(parents=True, exist_ok=True)
        
        self.min_improvement = min_improvement
        self.max_backups = max_backups
        self.cleanup_days = cleanup_days
        
        # Настройка логирования
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        # Веса метрик для регрессии (сумма = 1.0)
        self.regression_weights = {
            'rmse': 0.4,    # RMSE - самый важный
            'mae': 0.3,     # MAE - средний вес
            'r2': 0.3       # R² - средний вес
        }
        
        # Веса метрик для классификации
        self.classification_weights = {
            'accuracy': 0.5,  # Accuracy - самый важный
            'f1': 0.5         # F1 - равный вес
        }
    
    def get_model_paths(self, symbol: str, modeltype: str = 'regression') -> Dict[str, Path]:
        """Получить пути к файлам модели"""
        symbol = symbol.upper().replace('/', '')
        base = self.models_root / symbol / modeltype
        
        return {
            'base': base,
            'model': base / 'model.pkl',
            'meta': base / 'meta.json',
            'backup_dir': base / 'backups',
            'validation': base / 'validation.json'
        }
    
    def validate_model_files(self, symbol: str, modeltype: str = 'regression') -> Tuple[bool, str]:
        """
        Валидация файлов модели
        
        Returns:
            (is_valid, error_message)
        """
        paths = self.get_model_paths(symbol, modeltype)
        
        # Проверяем существование файлов
        if not paths['model'].exists():
            return False, f"Файл модели не найден: {paths['model']}"
        
        if not paths['meta'].exists():
            return False, f"Файл метаданных не найден: {paths['meta']}"
        
        try:
            # Проверяем модель
            with open(paths['model'], 'rb') as f:
                model = pickle.load(f)
            
            # Проверяем метаданные
            with open(paths['meta'], 'r') as f:
                meta = json.load(f)
            
            # Валидация метаданных
            required_fields = ['metrics', 'model_type', 'features', 'saved_at']
            for field in required_fields:
                if field not in meta:
                    return False, f"Отсутствует обязательное поле в метаданных: {field}"
            
            # Проверяем целостность модели
            if hasattr(model, 'predict'):
                # Тестовая предсказание
                import numpy as np
                test_data = np.random.random((1, len(meta.get('features', []))))
                try:
                    _ = model.predict(test_data)
                except Exception as e:
                    return False, f"Ошибка тестового предсказания: {e}"
            else:
                return False, "Модель не имеет метода predict"
            
            return True, "Валидация прошла успешно"
            
        except Exception as e:
            return False, f"Ошибка валидации: {e}"
    
    def calculate_model_score(self, metrics: Dict[str, Any], task_type: str = 'regression') -> float:
        """
        Вычисляет общий скор модели на основе взвешенных метрик
        
        Args:
            metrics: Словарь с метриками
            task_type: Тип задачи ('regression' или 'classification')
            
        Returns:
            Общий скор модели (чем выше, тем лучше)
        """
        if task_type == 'regression':
            weights = self.regression_weights
            rmse = metrics.get('rmse', float('inf'))
            mae = metrics.get('mae', float('inf'))
            r2 = metrics.get('r2', -float('inf'))
            
            # Нормализуем метрики (RMSE и MAE - чем меньше, тем лучше)
            # R² - чем больше, тем лучше
            if rmse == float('inf'):
                rmse_score = 0
            else:
                rmse_score = 1.0 / (1.0 + rmse)  # Нормализация RMSE
            
            if mae == float('inf'):
                mae_score = 0
            else:
                mae_score = 1.0 / (1.0 + mae)    # Нормализация MAE
            
            r2_score = max(0, r2)  # R² не может быть меньше 0
            
            # Вычисляем взвешенный скор
            score = (weights['rmse'] * rmse_score + 
                    weights['mae'] * mae_score + 
                    weights['r2'] * r2_score)
            
        else:  # classification
            weights = self.classification_weights
            accuracy = metrics.get('accuracy', 0)
            f1 = metrics.get('f1', 0)
            
            score = (weights['accuracy'] * accuracy + 
                    weights['f1'] * f1)
        
        return score
    
    def compare_models_advanced(self, current_metrics: Optional[Dict[str, Any]], 
                              new_metrics: Dict[str, Any], 
                              task_type: str = 'regression') -> Tuple[bool, str, float]:
        """
        Улучшенное сравнение моделей с учетом минимального улучшения
        
        Args:
            current_metrics: Метрики текущей модели
            new_metrics: Метрики новой модели
            task_type: Тип задачи
            
        Returns:
            (is_better, reason, improvement_percent)
        """
        if current_metrics is None:
            return True, "Первая модель", 100.0
        
        # Вычисляем общие скоры
        current_score = self.calculate_model_score(current_metrics.get('metrics', {}), task_type)
        new_score = self.calculate_model_score(new_metrics.get('metrics', {}), task_type)
        
        # Вычисляем процент улучшения
        if current_score == 0:
            improvement_percent = 100.0 if new_score > 0 else 0.0
        else:
            improvement_percent = ((new_score - current_score) / current_score) * 100
        
        # Проверяем минимальное улучшение
        if improvement_percent < (self.min_improvement * 100):
            return False, f"Улучшение {improvement_percent:.2f}% меньше минимального {self.min_improvement * 100:.2f}%", improvement_percent
        
        # Детальное сравнение метрик
        if task_type == 'regression':
            current_rmse = current_metrics.get('metrics', {}).get('rmse', float('inf'))
            current_r2 = current_metrics.get('metrics', {}).get('r2', -float('inf'))
            current_mae = current_metrics.get('metrics', {}).get('mae', float('inf'))
            
            new_rmse = new_metrics.get('metrics', {}).get('rmse', float('inf'))
            new_r2 = new_metrics.get('metrics', {}).get('r2', -float('inf'))
            new_mae = new_metrics.get('metrics', {}).get('mae', float('inf'))
            
            improvements = []
            if new_rmse < current_rmse:
                rmse_improvement = ((current_rmse - new_rmse) / current_rmse) * 100
                improvements.append(f"RMSE: {current_rmse:.4f} → {new_rmse:.4f} (-{rmse_improvement:.2f}%)")
            if new_r2 > current_r2:
                r2_improvement = ((new_r2 - current_r2) / abs(current_r2)) * 100
                improvements.append(f"R²: {current_r2:.4f} → {new_r2:.4f} (+{r2_improvement:.2f}%)")
            if new_mae < current_mae:
                mae_improvement = ((current_mae - new_mae) / current_mae) * 100
                improvements.append(f"MAE: {current_mae:.4f} → {new_mae:.4f} (-{mae_improvement:.2f}%)")
            
            reason = f"Общее улучшение: {improvement_percent:.2f}%. " + ", ".join(improvements)
            
        else:  # classification
            current_accuracy = current_metrics.get('metrics', {}).get('accuracy', 0)
            current_f1 = current_metrics.get('metrics', {}).get('f1', 0)
            
            new_accuracy = new_metrics.get('metrics', {}).get('accuracy', 0)
            new_f1 = new_metrics.get('metrics', {}).get('f1', 0)
            
            improvements = []
            if new_accuracy > current_accuracy:
                acc_improvement = ((new_accuracy - current_accuracy) / current_accuracy) * 100
                improvements.append(f"Accuracy: {current_accuracy:.4f} → {new_accuracy:.4f} (+{acc_improvement:.2f}%)")
            if new_f1 > current_f1:
                f1_improvement = ((new_f1 - current_f1) / current_f1) * 100
                improvements.append(f"F1: {current_f1:.4f} → {new_f1:.4f} (+{f1_improvement:.2f}%)")
            
            reason = f"Общее улучшение: {improvement_percent:.2f}%. " + ", ".join(improvements)
        
        return True, reason, improvement_percent
    
    def cleanup_old_versions(self, symbol: str, modeltype: str = 'regression') -> int:
        """
        Автоматическая очистка старых версий моделей
        
        Returns:
            Количество удаленных версий
        """
        paths = self.get_model_paths(symbol, modeltype)
        
        if not paths['backup_dir'].exists():
            return 0
        
        cutoff_date = datetime.now() - timedelta(days=self.cleanup_days)
        deleted_count = 0
        
        for backup_dir in paths['backup_dir'].iterdir():
            if not backup_dir.is_dir():
                continue
            
            # Проверяем дату создания
            try:
                meta_path = backup_dir / 'meta.json'
                if meta_path.exists():
                    with open(meta_path, 'r') as f:
                        meta = json.load(f)
                    
                    saved_at = datetime.fromisoformat(meta.get('saved_at', '2000-01-01T00:00:00'))
                    
                    if saved_at < cutoff_date:
                        shutil.rmtree(backup_dir)
                        deleted_count += 1
                        self.logger.info(f"🗑️ Удалена старая версия: {backup_dir.name}")
                        
            except Exception as e:
                self.logger.warning(f"Ошибка при очистке {backup_dir}: {e}")
        
        # Ограничиваем количество версий
        if paths['backup_dir'].exists():
            backup_dirs = sorted([d for d in paths['backup_dir'].iterdir() if d.is_dir()], 
                               key=lambda x: x.name, reverse=True)
            
            if len(backup_dirs) > self.max_backups:
                for old_backup in backup_dirs[self.max_backups:]:
                    shutil.rmtree(old_backup)
                    deleted_count += 1
                    self.logger.info(f"🗑️ Удалена лишняя версия: {old_backup.name}")
        
        return deleted_count
    
    def load_current_model_metrics(self, symbol: str, modeltype: str = 'regression') -> Optional[Dict[str, Any]]:
        """Загрузить метрики текущей модели с валидацией"""
        paths = self.get_model_paths(symbol, modeltype)
        
        if not paths['meta'].exists():
            return None
        
        try:
            # Валидируем файлы перед загрузкой
            is_valid, error_msg = self.validate_model_files(symbol, modeltype)
            if not is_valid:
                self.logger.warning(f"⚠️ Валидация не прошла: {error_msg}")
                return None
            
            with open(paths['meta'], 'r') as f:
                meta = json.load(f)
            return meta
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка загрузки метаданных: {e}")
            return None
    
    def backup_current_model(self, symbol: str, modeltype: str = 'regression') -> bool:
        """Создать резервную копию текущей модели с валидацией"""
        paths = self.get_model_paths(symbol, modeltype)
        
        if not paths['model'].exists() or not paths['meta'].exists():
            return True  # Нет модели для резервного копирования
        
        # Валидируем модель перед backup
        is_valid, error_msg = self.validate_model_files(symbol, modeltype)
        if not is_valid:
            self.logger.error(f"❌ Не удалось создать backup - модель невалидна: {error_msg}")
            return False
        
        try:
            # Создаем директорию для резервных копий
            paths['backup_dir'].mkdir(parents=True, exist_ok=True)
            
            # Создаем имя резервной копии с временной меткой и хешем
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Создаем хеш модели для уникальности
            with open(paths['model'], 'rb') as f:
                model_hash = hashlib.md5(f.read()).hexdigest()[:8]
            
            backup_name = f"backup_{timestamp}_{model_hash}"
            backup_path = paths['backup_dir'] / backup_name
            
            # Создаем директорию для резервной копии
            backup_path.mkdir(parents=True, exist_ok=True)
            
            # Копируем файлы
            shutil.copy2(paths['model'], backup_path / 'model.pkl')
            shutil.copy2(paths['meta'], backup_path / 'meta.json')
            
            # Добавляем информацию о backup
            backup_info = {
                'backup_created_at': datetime.now().isoformat(),
                'original_path': str(paths['model']),
                'model_hash': model_hash
            }
            
            with open(backup_path / 'backup_info.json', 'w') as f:
                json.dump(backup_info, f, indent=2)
            
            self.logger.info(f"✅ Резервная копия создана: {backup_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка создания резервной копии: {e}")
            return False
    
    def save_model_with_versioning(self, symbol: str, model: Any, features: List[str], 
                                 meta: Dict[str, Any], modeltype: str = 'regression') -> bool:
        """
        Сохранить модель с улучшенной версификацией.
        Сравнивает с текущей моделью и заменяет только если новая лучше.
        """
        paths = self.get_model_paths(symbol, modeltype)
        
        # Определяем тип задачи
        task_type = 'classification' if 'accuracy' in meta.get('metrics', {}) else 'regression'
        
        # Загружаем метрики текущей модели
        current_metrics = self.load_current_model_metrics(symbol, modeltype)
        
        # Сравниваем модели с улучшенной логикой
        is_better, reason, improvement = self.compare_models_advanced(current_metrics, meta, task_type)
        
        self.logger.info(f"Сравнение моделей для {symbol}: {reason}")
        
        if not is_better:
            self.logger.warning(f"⚠️ Новая модель хуже текущей для {symbol}. Модель НЕ будет заменена.")
            return False
        
        # Создаём резервную копию текущей модели
        if current_metrics:
            if not self.backup_current_model(symbol, modeltype):
                self.logger.error("❌ Не удалось создать резервную копию. Модель НЕ будет заменена.")
                return False
        
        # Создаём директорию если её нет
        paths['base'].mkdir(parents=True, exist_ok=True)
        
        try:
            # Обновляем метаданные
            meta['features'] = features
            meta['saved_at'] = datetime.now().isoformat()
            meta['version'] = self._get_next_version(symbol, modeltype)
            meta['task_type'] = task_type
            meta['model_score'] = self.calculate_model_score(meta.get('metrics', {}), task_type)
            
            # Сохраняем модель
            with open(paths['model'], 'wb') as f:
                pickle.dump(model, f)
            
            # Сохраняем метаданные
            with open(paths['meta'], 'w') as f:
                json.dump(meta, f, indent=2)
            
            # Сохраняем информацию о валидации
            validation_info = {
                'last_validation': datetime.now().isoformat(),
                'validation_passed': True,
                'model_score': meta['model_score'],
                'improvement_percent': improvement
            }
            
            with open(paths['validation'], 'w') as f:
                json.dump(validation_info, f, indent=2)
            
            # Автоматическая очистка старых версий
            deleted_count = self.cleanup_old_versions(symbol, modeltype)
            if deleted_count > 0:
                self.logger.info(f"🧹 Удалено {deleted_count} старых версий")
            
            self.logger.info(f"✅ Модель сохранена: {paths['model']}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка сохранения модели: {e}")
            return False
    
    def _get_next_version(self, symbol: str, modeltype: str = 'regression') -> str:
        """Получить следующий номер версии"""
        paths = self.get_model_paths(symbol, modeltype)
        
        if not paths['backup_dir'].exists():
            return "1.0"
        
        # Подсчитываем существующие версии
        existing_versions = [d.name for d in paths['backup_dir'].iterdir() if d.is_dir()]
        version_number = len(existing_versions) + 1
        
        return f"{version_number}.0"
    
    def get_model_history(self, symbol: str, modeltype: str = 'regression') -> List[Dict[str, Any]]:
        """Получить улучшенную историю версий модели"""
        paths = self.get_model_paths(symbol, modeltype)
        
        history = []
        
        # Добавляем текущую модель
        current_metrics = self.load_current_model_metrics(symbol, modeltype)
        if current_metrics:
            history.append({
                'version': 'current',
                'saved_at': current_metrics.get('saved_at', 'unknown'),
                'metrics': current_metrics.get('metrics', {}),
                'model_type': current_metrics.get('model_type', 'unknown'),
                'model_score': current_metrics.get('model_score', 0),
                'task_type': current_metrics.get('task_type', 'unknown'),
                'features_count': len(current_metrics.get('features', []))
            })
        
        # Добавляем резервные копии
        if paths['backup_dir'].exists():
            for backup_dir in sorted(paths['backup_dir'].iterdir(), key=lambda x: x.name, reverse=True):
                if backup_dir.is_dir():
                    meta_path = backup_dir / 'meta.json'
                    if meta_path.exists():
                        try:
                            with open(meta_path, 'r') as f:
                                meta = json.load(f)
                            
                            history.append({
                                'version': backup_dir.name,
                                'saved_at': meta.get('saved_at', 'unknown'),
                                'metrics': meta.get('metrics', {}),
                                'model_type': meta.get('model_type', 'unknown'),
                                'model_score': meta.get('model_score', 0),
                                'task_type': meta.get('task_type', 'unknown'),
                                'features_count': len(meta.get('features', []))
                            })
                        except Exception as e:
                            self.logger.warning(f"Ошибка загрузки метаданных {meta_path}: {e}")
        
        return history
    
    def restore_model_version(self, symbol: str, version: str, modeltype: str = 'regression') -> bool:
        """Восстановить модель из резервной копии с валидацией"""
        paths = self.get_model_paths(symbol, modeltype)
        
        if version == 'current':
            self.logger.info("Восстановление не требуется - это текущая модель")
            return True
        
        backup_path = paths['backup_dir'] / version
        if not backup_path.exists():
            self.logger.error(f"❌ Резервная копия {version} не найдена")
            return False
        
        try:
            # Проверяем валидность backup
            backup_model_path = backup_path / 'model.pkl'
            backup_meta_path = backup_path / 'meta.json'
            
            if not backup_model_path.exists() or not backup_meta_path.exists():
                self.logger.error(f"❌ Backup {version} поврежден - отсутствуют файлы")
                return False
            
            # Создаём резервную копию текущей модели перед восстановлением
            self.backup_current_model(symbol, modeltype)
            
            # Восстанавливаем файлы
            shutil.copy2(backup_model_path, paths['model'])
            shutil.copy2(backup_meta_path, paths['meta'])
            
            # Валидируем восстановленную модель
            is_valid, error_msg = self.validate_model_files(symbol, modeltype)
            if not is_valid:
                self.logger.error(f"❌ Восстановленная модель невалидна: {error_msg}")
                return False
            
            self.logger.info(f"✅ Модель успешно восстановлена из {version}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка восстановления модели: {e}")
            return False
    
    def list_models(self) -> List[Dict[str, Any]]:
        """Получить улучшенный список всех моделей"""
        models = []
        
        for symbol_dir in self.models_root.iterdir():
            if symbol_dir.is_dir():
                symbol = symbol_dir.name
                
                for modeltype_dir in symbol_dir.iterdir():
                    if modeltype_dir.is_dir():
                        modeltype = modeltype_dir.name
                        
                        meta_path = modeltype_dir / 'meta.json'
                        if meta_path.exists():
                            try:
                                with open(meta_path, 'r') as f:
                                    meta = json.load(f)
                                
                                # Валидируем модель
                                is_valid, _ = self.validate_model_files(symbol, modeltype)
                                
                                models.append({
                                    'symbol': symbol,
                                    'type': modeltype,
                                    'model_type': meta.get('model_type', 'unknown'),
                                    'saved_at': meta.get('saved_at', 'unknown'),
                                    'metrics': meta.get('metrics', {}),
                                    'model_score': meta.get('model_score', 0),
                                    'task_type': meta.get('task_type', 'unknown'),
                                    'features_count': len(meta.get('features', [])),
                                    'is_valid': is_valid,
                                    'backup_count': len([d for d in (modeltype_dir / 'backups').iterdir() 
                                                       if d.is_dir()]) if (modeltype_dir / 'backups').exists() else 0
                                })
                            except Exception as e:
                                self.logger.warning(f"Ошибка загрузки метаданных {meta_path}: {e}")
        
        return models
    
    def get_model_statistics(self, symbol: str, modeltype: str = 'regression') -> Dict[str, Any]:
        """
        Получить статистику по модели
        
        Returns:
            Словарь со статистикой модели
        """
        history = self.get_model_history(symbol, modeltype)
        
        if not history:
            return {'error': 'Модель не найдена'}
        
        current_model = history[0] if history[0]['version'] == 'current' else None
        backup_models = [m for m in history if m['version'] != 'current']
        
        stats = {
            'symbol': symbol,
            'modeltype': modeltype,
            'current_model': current_model,
            'total_versions': len(history),
            'backup_count': len(backup_models),
            'first_version': backup_models[-1] if backup_models else None,
            'latest_backup': backup_models[0] if backup_models else None,
            'score_trend': [m['model_score'] for m in history],
            'improvement_history': []
        }
        
        # Вычисляем историю улучшений
        for i in range(1, len(history)):
            prev_score = history[i]['model_score']
            curr_score = history[i-1]['model_score']
            
            if prev_score > 0:
                improvement = ((curr_score - prev_score) / prev_score) * 100
                stats['improvement_history'].append({
                    'from_version': history[i]['version'],
                    'to_version': history[i-1]['version'],
                    'improvement_percent': improvement
                })
        
        return stats 