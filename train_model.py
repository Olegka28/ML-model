#!/usr/bin/env python3
"""
🚀 Простой интерфейс для обучения ML моделей

Быстрый запуск обучения с новой модульной системой:
- Регрессионные модели (crypto_clipped, volume_weighted, vol_regime, market_regime, momentum_enhanced, volume_volatility)
- Классификационные модели (binary classification)
- Автоматическая настройка гиперпараметров
- Сохранение и версификация моделей
"""

import argparse
import sys
import os
sys.path.append('.')

from ml_module.systems.regression_system import RegressionSystem
from ml_module.systems.classification_system import ClassificationSystem
from ml_module.utils.config import Config

def train_regression_model(symbol: str, target_type: str = 'crypto_clipped', horizon: int = 10, 
                          timeframes: list = None, n_trials: int = 50):
    """
    Обучение регрессионной модели
    
    Args:
        symbol: Символ монеты (например, SOL_USDT)
        target_type: Тип таргета (crypto_clipped, volume_weighted, vol_regime, market_regime, momentum_enhanced, volume_volatility)
        horizon: Горизонт предсказания в барах
        timeframes: Список таймфреймов для multi-timeframe признаков
        n_trials: Количество trials для Optuna
    """
    print(f"🤖 Обучение регрессионной модели для {symbol}")
    print(f"   Таргет: {target_type}, Горизонт: {horizon}, Таймфреймы: {timeframes}")
    
    # Конфигурация
    config = Config(
        models_root='models',
        data_root='data',
        log_level='INFO'
    )
    
    # Создаем регрессионную систему (ИСПРАВЛЕНО: используем RegressionSystem)
    system = RegressionSystem(config)
    
    try:
        # Запускаем полный эксперимент (ИСПРАВЛЕНО: используем run_experiment)
        print(f"\n🚀 Запуск полного эксперимента регрессии...")
        metadata = system.run_experiment(
            symbol=symbol,
            target_type=target_type,
            horizon=horizon
        )
        
        # Выводим результаты
        print(f"\n📊 Результаты обучения:")
        metrics = metadata.get('metrics', {})
        for metric, value in metrics.items():
            print(f"   {metric.upper()}: {value:.6f}")
        
        baseline = metadata.get('baseline', {})
        if baseline:
            print(f"\n📈 Baseline сравнение:")
            if isinstance(baseline, dict):
                for baseline_name, baseline_metrics in baseline.items():
                    if isinstance(baseline_metrics, dict):
                        print(f"   {baseline_name}: RMSE={baseline_metrics.get('rmse', 0):.6f}")
                    else:
                        print(f"   {baseline_name}: {baseline_metrics:.6f}")
            else:
                print(f"   Baseline RMSE: {baseline:.6f}")
        
        # Тестируем предсказание
        print(f"\n🔮 Тестирование предсказания...")
        # Получаем последние данные для тестирования через систему
        try:
            data = system.load_and_validate_data(symbol, ['15m'])
            latest_data = data['15m'].tail(100)
            
            if latest_data is not None and not latest_data.empty:
                prediction_result = system.predict_latest(symbol, latest_data, '15m')
                if prediction_result:
                    print(f"✅ Предсказание: {prediction_result['prediction']:.6f}")
                    print(f"   Уверенность: {prediction_result['confidence']:.1f}%")
                else:
                    print("⚠️ Не удалось получить предсказание")
            else:
                print("⚠️ Не удалось получить последние данные для тестирования")
        except Exception as e:
            print(f"⚠️ Ошибка при тестировании предсказания: {e}")
        
        print(f"\n🎉 Обучение регрессии завершено успешно!")
        return True
        
    except Exception as e:
        print(f"\n❌ Ошибка обучения регрессии: {e}")
        import traceback
        traceback.print_exc()
        return False

def train_classification_model(symbol: str, percent: float = 0.025, horizon: int = 20, n_trials: int = 50):
    """
    Обучение классификационной модели
    
    Args:
        symbol: Символ монеты
        percent: Процент роста для классификации (по умолчанию: 0.025 = 2.5%)
        horizon: Горизонт предсказания в барах
        n_trials: Количество trials для Optuna
    """
    print(f"🔀 Обучение классификационной модели для {symbol}")
    print(f"   Процент роста: {percent*100}%, Горизонт: {horizon}")
    
    # Конфигурация
    config = Config(
        models_root='models',
        data_root='data',
        log_level='INFO'
    )
    
    # Создаем классификационную систему
    system = ClassificationSystem(config)
    
    try:
        # Запускаем полный эксперимент (ИСПРАВЛЕНО: используем run_experiment)
        print(f"\n🚀 Запуск полного эксперимента классификации...")
        metadata = system.run_experiment(
            symbol=symbol,
            percent=percent,
            horizon=horizon
        )
        
        # Выводим результаты
        print(f"\n📊 Результаты обучения:")
        metrics = metadata.get('metrics', {})
        for metric, value in metrics.items():
            print(f"   {metric.upper()}: {value:.6f}")
        
        # Получаем информацию о распределении классов
        class_distribution = system.get_class_distribution(symbol)
        if class_distribution:
            print(f"\n📊 Распределение классов:")
            print(f"   Оригинальное: {class_distribution['original_class_counts']}")
            print(f"   Сбалансированное: {class_distribution['balanced_class_counts']}")
            print(f"   Соотношение: {class_distribution['class_balance_ratio']:.2f}")
        
        # Тестируем предсказание
        print(f"\n🔮 Тестирование предсказания...")
        # Получаем последние данные для тестирования через систему
        try:
            data = system.load_and_validate_data(symbol, ['15m'])
            latest_data = data['15m'].tail(100)
            
            if latest_data is not None and not latest_data.empty:
                prediction_result = system.predict_latest(symbol, latest_data, '15m')
                if prediction_result:
                    print(f"✅ Предсказание: {prediction_result['prediction_label']} (класс {prediction_result['prediction']})")
                    print(f"   Уверенность: {prediction_result['confidence']:.1f}%")
                else:
                    print("⚠️ Не удалось получить предсказание")
            else:
                print("⚠️ Не удалось получить последние данные для тестирования")
        except Exception as e:
            print(f"⚠️ Ошибка при тестировании предсказания: {e}")
        
        print(f"\n🎉 Обучение классификации завершено успешно!")
        return True
        
    except Exception as e:
        print(f"\n❌ Ошибка обучения классификации: {e}")
        import traceback
        traceback.print_exc()
        return False

def list_available_models():
    """Показать доступные модели"""
    import glob
    from pathlib import Path
    
    models_root = Path('models')
    if not models_root.exists():
        print("📁 Папка models не найдена")
        return
    
    print("📋 Доступные модели:")
    for model_dir in models_root.glob('*/*'):
        if model_dir.is_dir() and (model_dir / 'meta.json').exists():
            try:
                import json
                with open(model_dir / 'meta.json', 'r') as f:
                    meta = json.load(f)
                
                symbol = meta.get('symbol', model_dir.parent.name)
                task = meta.get('task', model_dir.name)
                saved_at = meta.get('saved_at', meta.get('train_date', 'unknown'))  # ИСПРАВЛЕНО: используем saved_at
                metrics = meta.get('metrics', {})
                model_type = meta.get('model_type', 'unknown')
                features_count = meta.get('features_count', 0)
                
                print(f"   {symbol} ({task}):")
                print(f"     Тип модели: {model_type}")
                print(f"     Признаков: {features_count}")
                print(f"     Дата: {saved_at}")
                
                # Выводим метрики в зависимости от типа задачи
                if task == 'regression':
                    if 'rmse' in metrics:
                        print(f"     RMSE: {metrics['rmse']:.6f}")
                    if 'mae' in metrics:
                        print(f"     MAE: {metrics['mae']:.6f}")
                    if 'r2' in metrics:
                        print(f"     R²: {metrics['r2']:.4f}")
                elif task == 'classification':
                    if 'accuracy' in metrics:
                        print(f"     Accuracy: {metrics['accuracy']:.4f}")
                    if 'f1_score' in metrics:
                        print(f"     F1: {metrics['f1_score']:.4f}")
                    if 'precision' in metrics:
                        print(f"     Precision: {metrics['precision']:.4f}")
                    if 'recall' in metrics:
                        print(f"     Recall: {metrics['recall']:.4f}")
                
                # Выводим дополнительную информацию
                target_type = meta.get('target_type', 'unknown')
                horizon = meta.get('horizon', 0)
                if target_type != 'unknown':
                    print(f"     Таргет: {target_type}, горизонт: {horizon}")
                
                model_score = meta.get('model_score', 0)
                if model_score > 0:
                    print(f"     Оценка модели: {model_score:.4f}")
                
                print()
                
            except Exception as e:
                print(f"   {model_dir}: ошибка чтения метаданных - {e}")

def get_model_info(symbol: str, task: str = 'regression'):
    """Получить подробную информацию о модели"""
    print(f"📊 Информация о модели {symbol} ({task})")
    print("=" * 50)
    
    # Конфигурация
    config = Config(
        models_root='models',
        data_root='data',
        log_level='INFO'
    )
    
    try:
        if task == 'regression':
            system = RegressionSystem(config)
        else:
            system = ClassificationSystem(config)
        
        # Получаем информацию о модели
        model_info = system.get_model_info(symbol)
        if model_info:
            print(f"✅ Модель найдена:")
            print(f"   Символ: {model_info['symbol']}")
            print(f"   Задача: {model_info['task']}")
            print(f"   Тип модели: {model_info['model_type']}")
            print(f"   Количество признаков: {model_info['features_count']}")
            print(f"   Дата обучения: {model_info['training_date']}")
            
            print(f"\n📈 Метрики:")
            for metric, value in model_info['metrics'].items():
                print(f"   {metric.upper()}: {value:.6f}")
            
            if task == 'regression':
                print(f"\n🎯 Параметры таргета:")
                print(f"   Тип: {model_info['target_type']}")
                print(f"   Горизонт: {model_info['horizon']}")
            else:
                print(f"\n🎯 Параметры классификации:")
                print(f"   Процент роста: {model_info['target_percent']*100}%")
                print(f"   Горизонт: {model_info['horizon']}")
            
            print(f"\n⭐ Оценка модели: {model_info['model_score']:.4f}")
            
            # Получаем историю версий
            history = system.get_model_history(symbol)
            if history and len(history) > 1:
                print(f"\n📚 История версий:")
                for i, version in enumerate(history[:5]):  # Показываем последние 5 версий
                    print(f"   v{version['version']}: {version['score']:.4f}")
        else:
            print(f"❌ Модель {symbol} ({task}) не найдена")
            
    except Exception as e:
        print(f"❌ Ошибка получения информации: {e}")

def main():
    parser = argparse.ArgumentParser(description='🚀 Обучение ML моделей для торговли')
    parser.add_argument('action', choices=['train', 'list', 'info'], 
                       help='Действие: train - обучение, list - список моделей, info - информация о модели')
    parser.add_argument('--symbol', '-s', default='SOL_USDT',
                       help='Символ монеты (по умолчанию: SOL_USDT)')
    parser.add_argument('--task', '-t', choices=['regression', 'classification'], 
                       default='regression', help='Тип задачи (по умолчанию: regression)')
    parser.add_argument('--target', choices=['crypto_clipped', 'volume_weighted', 'vol_regime', 'market_regime', 'momentum_enhanced', 'volume_volatility'],
                       default='crypto_clipped', help='Тип таргета для регрессии (по умолчанию: crypto_clipped)')
    parser.add_argument('--percent', type=float, default=0.025,
                       help='Процент роста для классификации (по умолчанию: 0.025 = 2.5%)')
    parser.add_argument('--horizon', type=int, default=10,
                       help='Горизонт предсказания в барах (по умолчанию: 10)')
    parser.add_argument('--timeframes', nargs='+', 
                       default=['15m', '1h', '4h', '1d'],
                       help='Таймфреймы для multi-timeframe признаков')
    parser.add_argument('--trials', type=int, default=50,
                       help='Количество Optuna trials (по умолчанию: 50)')
    
    args = parser.parse_args()
    
    if args.action == 'list':
        list_available_models()
        return
    
    if args.action == 'info':
        get_model_info(args.symbol, args.task)
        return
    
    if args.action == 'train':
        print("🚀 Запуск обучения ML модели")
        print("=" * 50)
        
        if args.task == 'regression':
            success = train_regression_model(
                symbol=args.symbol,
                target_type=args.target,
                horizon=args.horizon,
                timeframes=args.timeframes,
                n_trials=args.trials
            )
        else:  # classification
            success = train_classification_model(
                symbol=args.symbol,
                percent=args.percent,
                horizon=args.horizon,
                n_trials=args.trials
            )
        
        if success:
            print("\n✅ Обучение завершено успешно!")
            sys.exit(0)
        else:
            print("\n❌ Обучение завершено с ошибками")
            sys.exit(1)

if __name__ == "__main__":
    import numpy as np
    main() 