#!/usr/bin/env python3
"""
🔄 Backtesting Script - скрипт для запуска бектестинга

Использование:
    python backtest.py --symbol SOL_USDT --strategy simple
    python backtest.py --symbol BTCUSDT --strategy risk_adjusted --capital 50000
"""

import argparse
import sys
from pathlib import Path

# Добавляем путь к модулям
sys.path.append(str(Path(__file__).parent))

import pandas as pd
from ml_module.utils.config import Config
from ml_module.backtesting import Backtester
from ml_module.core.model_manager import ModelManager
from ml_module.data_collector.data_manager import DataManager
from ml_module.features.feature_manager import FeatureManager

def load_model_and_data(symbol: str, task: str = 'regression'):
    """Загрузить модель и данные"""
    config = Config()
    
    # Загружаем модель
    model_manager = ModelManager(config)
    model, metadata = model_manager.load_model(symbol, task)
    
    if model is None:
        raise ValueError(f"Модель для {symbol} не найдена")
    
    # Загружаем данные
    data_manager = DataManager(config)
    data = data_manager.load_data(symbol, ['15m'])
    
    if not data or '15m' not in data:
        raise ValueError(f"Данные для {symbol} не найдены")
    
    # Генерируем признаки
    feature_manager = FeatureManager(config)
    features = feature_manager.generate_features(data)
    
    # Объединяем данные и признаки
    df_15m = data['15m']
    combined_data = pd.concat([df_15m, features], axis=1)
    
    # Получаем список признаков
    feature_columns = [col for col in features.columns if col not in df_15m.columns]
    
    return model, combined_data, feature_columns, metadata

def run_backtest(symbol: str, strategy_type: str = 'simple', 
                initial_capital: float = 10000.0, 
                commission_rate: float = 0.001,
                slippage_rate: float = 0.0005,
                start_date: str = None,
                end_date: str = None,
                save_results: bool = True):
    """Запустить бектестинг"""
    
    print(f"🔄 Запуск бектестинга для {symbol}")
    print(f"   Стратегия: {strategy_type}")
    print(f"   Капитал: ${initial_capital:,.2f}")
    print(f"   Комиссия: {commission_rate:.3%}")
    print(f"   Проскальзывание: {slippage_rate:.3%}")
    
    # Загружаем модель и данные
    print("\n📊 Загрузка модели и данных...")
    model, data, feature_columns, metadata = load_model_and_data(symbol)
    
    print(f"✅ Данные загружены: {len(data)} баров")
    print(f"✅ Признаков: {len(feature_columns)}")
    
    # Настраиваем бектестинг
    config = Config()
    backtester = Backtester(config)
    
    # Параметры стратегии
    strategy_params = {}
    if strategy_type == 'simple':
        strategy_params = {
            'buy_threshold': 0.01,
            'sell_threshold': -0.01,
            'confidence_threshold': 0.5
        }
    elif strategy_type == 'dynamic':
        strategy_params = {
            'base_threshold': 0.01,
            'volatility_multiplier': 1.0,
            'confidence_threshold': 0.5
        }
    elif strategy_type == 'confidence':
        strategy_params = {
            'min_confidence': 0.6,
            'position_size_multiplier': 1.0
        }
    elif strategy_type == 'risk_adjusted':
        strategy_params = {
            'risk_per_trade': 0.02,
            'max_position_size': 0.1,
            'stop_loss_pct': 0.05,
            'take_profit_pct': 0.1
        }
    
    # Настраиваем бектестинг
    backtester.setup_backtest(
        strategy_type=strategy_type,
        strategy_params=strategy_params,
        initial_capital=initial_capital,
        commission_rate=commission_rate,
        slippage_rate=slippage_rate
    )
    
    # Запускаем бектестинг
    print(f"\n🚀 Запуск бектестинга...")
    results = backtester.run_backtest(
        data=data,
        model=model,
        feature_columns=feature_columns,
        symbol=symbol,
        start_date=start_date,
        end_date=end_date
    )
    
    # Выводим результаты
    print("\n" + "="*60)
    print(backtester.get_summary_report())
    
    # Детальный отчет
    print("\n" + "="*60)
    print("📊 ДЕТАЛЬНЫЙ ОТЧЕТ:")
    print(backtester.get_detailed_report())
    
    # Сохраняем результаты
    if save_results:
        results_file = f"backtest_results/{symbol}_{strategy_type}_backtest.json"
        backtester.save_results(results_file)
    
    return results

def compare_strategies(symbol: str, strategies: dict, 
                      initial_capital: float = 10000.0):
    """Сравнить несколько стратегий"""
    
    print(f"🔄 Сравнение стратегий для {symbol}")
    print(f"   Капитал: ${initial_capital:,.2f}")
    
    # Загружаем модель и данные
    print("\n📊 Загрузка модели и данных...")
    model, data, feature_columns, metadata = load_model_and_data(symbol)
    
    # Настраиваем бектестинг
    config = Config()
    backtester = Backtester(config)
    
    # Сравниваем стратегии
    comparison_results = backtester.compare_strategies(
        strategies=strategies,
        data=data,
        model=model,
        feature_columns=feature_columns,
        symbol=symbol
    )
    
    # Выводим результаты сравнения
    print("\n" + "="*80)
    print("📊 СРАВНЕНИЕ СТРАТЕГИЙ")
    print("="*80)
    
    print(f"{'Стратегия':<20} {'Доходность':<12} {'Годовая':<10} {'Шарп':<8} {'Макс.ДД':<10} {'Сделки':<8} {'Винрейт':<10}")
    print("-" * 80)
    
    for strategy_name, results in comparison_results.items():
        print(f"{strategy_name:<20} "
              f"{results['total_return']:<11.2%} "
              f"{results['annual_return']:<9.2%} "
              f"{results['sharpe_ratio']:<7.3f} "
              f"{results['max_drawdown']:<9.2%} "
              f"{results['total_trades']:<7} "
              f"{results['win_rate']:<9.2%}")
    
    return comparison_results

def main():
    parser = argparse.ArgumentParser(description='Backtesting ML Trading Strategy')
    
    # Основные параметры
    parser.add_argument('--symbol', type=str, required=True, 
                       help='Торговый символ (например, SOL_USDT)')
    parser.add_argument('--strategy', type=str, default='simple',
                       choices=['simple', 'dynamic', 'confidence', 'risk_adjusted'],
                       help='Тип торговой стратегии')
    parser.add_argument('--capital', type=float, default=10000.0,
                       help='Начальный капитал')
    parser.add_argument('--commission', type=float, default=0.001,
                       help='Комиссия за сделку (0.001 = 0.1%)')
    parser.add_argument('--slippage', type=float, default=0.0005,
                       help='Проскальзывание (0.0005 = 0.05%)')
    
    # Даты
    parser.add_argument('--start-date', type=str, default=None,
                       help='Дата начала (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None,
                       help='Дата окончания (YYYY-MM-DD)')
    
    # Дополнительные опции
    parser.add_argument('--compare', action='store_true',
                       help='Сравнить все стратегии')
    parser.add_argument('--no-save', action='store_true',
                       help='Не сохранять результаты')
    parser.add_argument('--task', type=str, default='regression',
                       choices=['regression', 'classification'],
                       help='Тип задачи ML модели')
    
    args = parser.parse_args()
    
    try:
        if args.compare:
            # Сравнение стратегий
            strategies = {
                'Simple': {
                    'strategy_type': 'simple',
                    'initial_capital': args.capital,
                    'commission_rate': args.commission,
                    'slippage_rate': args.slippage
                },
                'Dynamic': {
                    'strategy_type': 'dynamic',
                    'initial_capital': args.capital,
                    'commission_rate': args.commission,
                    'slippage_rate': args.slippage
                },
                'Confidence': {
                    'strategy_type': 'confidence',
                    'initial_capital': args.capital,
                    'commission_rate': args.commission,
                    'slippage_rate': args.slippage
                },
                'Risk Adjusted': {
                    'strategy_type': 'risk_adjusted',
                    'initial_capital': args.capital,
                    'commission_rate': args.commission,
                    'slippage_rate': args.slippage
                }
            }
            
            compare_strategies(args.symbol, strategies, args.capital)
            
        else:
            # Одиночный бектестинг
            run_backtest(
                symbol=args.symbol,
                strategy_type=args.strategy,
                initial_capital=args.capital,
                commission_rate=args.commission,
                slippage_rate=args.slippage,
                start_date=args.start_date,
                end_date=args.end_date,
                save_results=not args.no_save
            )
    
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 