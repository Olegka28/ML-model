#!/usr/bin/env python3
"""
🔮 Простой интерфейс для предсказаний

Быстрое получение предсказаний от обученных моделей:
- Загрузка модели и метаданных
- Генерация признаков для последних данных
- Получение предсказания с уверенностью
- Использование улучшенных систем (RegressionSystem, ClassificationSystem)
"""

import argparse
import sys
import os
sys.path.append('.')

from ml_module.systems.regression_system import RegressionSystem
from ml_module.systems.classification_system import ClassificationSystem
from ml_module.utils.config import Config
import numpy as np

def get_prediction(symbol: str, task: str = 'regression', timeframes: list = None):
    """
    Получение предсказания от обученной модели
    
    Args:
        symbol: Символ монеты
        task: Тип задачи (regression или classification)
        timeframes: Таймфреймы для признаков
    """
    # Валидация входных данных
    if not symbol or not isinstance(symbol, str):
        print("❌ Ошибка: Symbol должен быть непустой строкой")
        return None
    
    if task not in ['regression', 'classification']:
        print("❌ Ошибка: Task должен быть 'regression' или 'classification'")
        return None
    
    print(f"🔮 Получение предсказания для {symbol} ({task})")
    
    # Конфигурация
    config = Config(
        models_root='models',
        data_root='data',
        log_level='WARNING'  # Уменьшаем логирование
    )
    
    try:
        # Выбираем правильную систему (ИСПРАВЛЕНО)
        if task == 'classification':
            system = ClassificationSystem(config)
            if timeframes is None:
                timeframes = ['15m']
        else:  # regression
            system = RegressionSystem(config)  # ИСПРАВЛЕНО: используем RegressionSystem
            if timeframes is None:
                timeframes = ['15m', '1h', '4h']
        
        # 1. Загружаем модель (ИСПРАВЛЕНО: добавляем task параметр)
        print("📥 Загрузка модели...")
        model, metadata = system.load_model(symbol, task=task)
        print(f"✅ Модель загружена: {metadata.get('model_type', 'unknown')}")
        
        # 2. Загружаем последние данные
        print("📊 Загрузка данных...")
        data = system.load_and_validate_data(symbol, timeframes)
        print(f"✅ Данные загружены: {len(data)} таймфреймов")
        
        # 3. Генерируем признаки
        print("🔧 Генерация признаков...")
        features = system.generate_and_validate_features(data)
        print(f"✅ Признаки сгенерированы: {features.shape[1]} признаков")
        
        # 4. Подготавливаем данные для предсказания
        print("⚙️ Подготовка данных...")
        # Берем последние данные
        X_pred = features.iloc[[-1]].values
        
        # 5. Делаем предсказание (ИСПРАВЛЕНО: используем правильный метод)
        print("🔮 Получение предсказания...")
        if task == 'classification':
            # Для классификации используем predict_latest
            latest_data = data['15m'].tail(100)  # Последние 100 свечей
            prediction_result = system.predict_latest(symbol, latest_data, '15m')
            
            if prediction_result is None:
                print("❌ Не удалось получить предсказание классификации")
                return None
            
            pred_value = prediction_result['prediction']
            confidence = prediction_result['confidence']
            prediction_label = prediction_result['prediction_label']
            
        else:
            # Для регрессии используем predict_latest
            latest_data = data['15m'].tail(100)  # Последние 100 свечей
            prediction_result = system.predict_latest(symbol, latest_data, '15m')
            
            if prediction_result is None:
                print("❌ Не удалось получить предсказание регрессии")
                return None
            
            pred_value = prediction_result['prediction']
            confidence = prediction_result['confidence']
        
        # 6. Выводим результат
        print("\n" + "="*50)
        print("📊 РЕЗУЛЬТАТ ПРЕДСКАЗАНИЯ")
        print("="*50)
        
        if task == 'classification':
            print(f"🎯 Направление: {prediction_label}")
            print(f"🔢 Класс: {pred_value}")
            print(f"💪 Уверенность: {confidence:.1f}%")
        else:
            # Регрессия
            print(f"🎯 Предсказание: {pred_value:.6f}")
            print(f"💪 Уверенность: {confidence:.1f}%")
            
            # Интерпретируем результат
            if pred_value > 0.01:
                direction = "РОСТ 📈"
            elif pred_value < -0.01:
                direction = "ПАДЕНИЕ 📉"
            else:
                direction = "БОКОВИК ↔️"
            
            print(f"📈 Ожидаемое направление: {direction}")
            
            # Рассчитываем примерное изменение цены
            current_price = data['15m']['close'].iloc[-1]
            expected_change = current_price * pred_value
            print(f"💰 Ожидаемое изменение: {expected_change:.4f} USDT")
        
        # 7. Дополнительная информация (ИСПРАВЛЕНО: используем saved_at)
        print(f"\n📋 Информация о модели:")
        print(f"   Символ: {metadata.get('symbol', 'N/A')}")
        print(f"   Тип: {metadata.get('task', 'N/A')}")
        print(f"   Дата обучения: {metadata.get('saved_at', metadata.get('train_date', 'N/A'))}")
        print(f"   Горизонт: {metadata.get('horizon', 'N/A')} баров")
        
        metrics = metadata.get('metrics', {})
        if metrics:
            print(f"   Метрики модели:")
            for metric, value in metrics.items():
                print(f"     {metric.upper()}: {value:.6f}")
        
        # 8. Получаем дополнительную информацию о модели
        print(f"\n🔍 Дополнительная информация:")
        model_info = system.get_model_info(symbol)
        if model_info:
            print(f"   Тип модели: {model_info.get('model_type', 'N/A')}")
            print(f"   Количество признаков: {model_info.get('features_count', 'N/A')}")
            print(f"   Оценка модели: {model_info.get('model_score', 0):.4f}")
            
            if task == 'regression':
                print(f"   Тип таргета: {model_info.get('target_type', 'N/A')}")
            else:
                print(f"   Процент роста: {model_info.get('target_percent', 0)*100:.1f}%")
        
        return pred_value
        
    except Exception as e:
        print(f"\n❌ Ошибка получения предсказания: {e}")
        import traceback
        traceback.print_exc()
        return None

def compare_models(symbol: str):
    """Сравнение регрессионной и классификационной моделей"""
    print(f"🔄 Сравнение моделей для {symbol}")
    print("="*50)
    
    # Регрессионная модель
    print("\n📊 РЕГРЕССИОННАЯ МОДЕЛЬ:")
    reg_pred = get_prediction(symbol, 'regression')
    
    print("\n" + "="*50)
    
    # Классификационная модель
    print("\n🔀 КЛАССИФИКАЦИОННАЯ МОДЕЛЬ:")
    cls_pred = get_prediction(symbol, 'classification')
    
    # Сравнение
    print("\n" + "="*50)
    print("🔄 СРАВНЕНИЕ РЕЗУЛЬТАТОВ:")
    print("="*50)
    
    if reg_pred is not None and cls_pred is not None:
        # Для регрессии
        if isinstance(reg_pred, (int, float)):
            reg_direction = "РОСТ" if reg_pred > 0 else "ПАДЕНИЕ"
        else:
            reg_direction = "НЕИЗВЕСТНО"
        
        # Для классификации
        if isinstance(cls_pred, int):
            cls_direction = "РОСТ" if cls_pred == 1 else "ПАДЕНИЕ"
        else:
            cls_direction = "НЕИЗВЕСТНО"
        
        print(f"📊 Регрессия: {reg_pred} → {reg_direction}")
        print(f"🔀 Классификация: {cls_pred} → {cls_direction}")
        
        if reg_direction == cls_direction and reg_direction != "НЕИЗВЕСТНО":
            print("✅ Модели согласны!")
        elif reg_direction != "НЕИЗВЕСТНО" and cls_direction != "НЕИЗВЕСТНО":
            print("⚠️ Модели расходятся во мнении")
        else:
            print("❓ Не удалось сравнить результаты")

def get_model_info(symbol: str, task: str = 'regression'):
    """Получить подробную информацию о модели"""
    print(f"📊 Информация о модели {symbol} ({task})")
    print("="*50)
    
    # Конфигурация
    config = Config(
        models_root='models',
        data_root='data',
        log_level='WARNING'
    )
    
    try:
        # Выбираем систему
        if task == 'classification':
            system = ClassificationSystem(config)
        else:
            system = RegressionSystem(config)
        
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
    parser = argparse.ArgumentParser(description='🔮 Получение предсказаний от ML моделей')
    parser.add_argument('symbol', help='Символ монеты (например, SOL_USDT)')
    parser.add_argument('--task', '-t', choices=['regression', 'classification', 'compare', 'info'], 
                       default='regression', help='Тип задачи (по умолчанию: regression)')
    parser.add_argument('--timeframes', nargs='+', 
                       help='Таймфреймы для признаков (по умолчанию: 15m,1h,4h для регрессии, 15m для классификации)')
    
    args = parser.parse_args()
    
    if args.task == 'compare':
        compare_models(args.symbol)
    elif args.task == 'info':
        get_model_info(args.symbol, 'regression')  # По умолчанию показываем регрессию
    else:
        get_prediction(args.symbol, args.task, args.timeframes)

if __name__ == "__main__":
    main() 