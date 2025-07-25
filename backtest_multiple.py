#!/usr/bin/env python3
"""
🔄 Multiple Backtesting - бектестинг множества моделей

Использование:
    python backtest_multiple.py --top 10 --strategy simple
    python backtest_multiple.py --symbols BTCUSDT,ETHUSDT --strategy risk_adjusted
"""

import argparse
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import json

# Добавляем путь к модулям
sys.path.append(str(Path(__file__).parent))

from ml_module.utils.config import Config
from ml_module.backtesting import Backtester
from ml_module.core.model_manager import ModelManager
from ml_module.data_collector.data_manager import DataManager
from ml_module.features.feature_manager import FeatureManager

# Топ-20 монет (такой же список как в train_multiple.py)
TOP_COINS = [
    'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT',
    'ADAUSDT', 'AVAXUSDT', 'DOGEUSDT', 'DOTUSDT', 'MATICUSDT',
    'LINKUSDT', 'UNIUSDT', 'ATOMUSDT', 'LTCUSDT', 'ETCUSDT',
    'XLMUSDT', 'BCHUSDT', 'FILUSDT', 'TRXUSDT', 'NEARUSDT'
]

def get_symbols_list(top_n: int = None, symbols: str = None) -> list:
    """Получить список символов для бектестинга"""
    if symbols:
        return [s.strip().upper() for s in symbols.split(',')]
    elif top_n:
        return TOP_COINS[:top_n]
    else:
        return TOP_COINS[:10]

def load_model_and_data(symbol: str, task: str = 'regression'):
    """Загрузить модель и данные для одной монеты"""
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

def backtest_single_model(symbol: str, strategy_type: str, strategy_params: dict,
                         initial_capital: float, commission_rate: float,
                         slippage_rate: float, start_date: str = None,
                         end_date: str = None) -> dict:
    """Провести бектестинг для одной монеты"""
    start_time = time.time()
    
    try:
        print(f"\n🔄 Бектестинг {symbol} с стратегией {strategy_type}...")
        
        # Загружаем модель и данные
        model, data, feature_columns, metadata = load_model_and_data(symbol)
        
        # Настраиваем бектестинг
        config = Config()
        backtester = Backtester(config)
        
        backtester.setup_backtest(
            strategy_type=strategy_type,
            strategy_params=strategy_params,
            initial_capital=initial_capital,
            commission_rate=commission_rate,
            slippage_rate=slippage_rate
        )
        
        # Запускаем бектестинг
        results = backtester.run_backtest(
            data=data,
            model=model,
            feature_columns=feature_columns,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date
        )
        
        # Извлекаем ключевые метрики
        portfolio_stats = results['portfolio_stats']
        metrics = results['metrics']
        
        return {
            'symbol': symbol,
            'status': 'success',
            'total_return': portfolio_stats['total_return'],
            'annual_return': metrics.get('annual_return', 0),
            'sharpe_ratio': metrics.get('sharpe_ratio', 0),
            'max_drawdown': metrics.get('max_drawdown', 0),
            'volatility': metrics.get('volatility', 0),
            'total_trades': portfolio_stats['total_trades'],
            'win_rate': portfolio_stats['win_rate'],
            'total_commission': portfolio_stats['total_commission'],
            'total_slippage': portfolio_stats['total_slippage'],
            'final_equity': portfolio_stats['total_equity'],
            'time': time.time() - start_time,
            'strategy': strategy_type
        }
        
    except Exception as e:
        return {
            'symbol': symbol,
            'status': 'failed',
            'error': str(e),
            'time': time.time() - start_time,
            'strategy': strategy_type
        }

def get_strategy_params(strategy_type: str) -> dict:
    """Получить параметры стратегии"""
    if strategy_type == 'simple':
        return {
            'buy_threshold': 0.01,
            'sell_threshold': -0.01,
            'confidence_threshold': 0.5
        }
    elif strategy_type == 'dynamic':
        return {
            'base_threshold': 0.01,
            'volatility_multiplier': 1.0,
            'confidence_threshold': 0.5
        }
    elif strategy_type == 'confidence':
        return {
            'min_confidence': 0.6,
            'position_size_multiplier': 1.0
        }
    elif strategy_type == 'risk_adjusted':
        return {
            'risk_per_trade': 0.02,
            'max_position_size': 0.1,
            'stop_loss_pct': 0.05,
            'take_profit_pct': 0.1
        }
    else:
        return {}

def backtest_multiple_models(symbols: list, strategy_type: str,
                           initial_capital: float = 10000.0,
                           commission_rate: float = 0.001,
                           slippage_rate: float = 0.0005,
                           start_date: str = None,
                           end_date: str = None,
                           parallel: bool = False,
                           max_workers: int = 4) -> pd.DataFrame:
    """Провести бектестинг для множества монет"""
    
    results = []
    strategy_params = get_strategy_params(strategy_type)
    
    print(f"🎯 Бектестинг {len(symbols)} монет")
    print(f"   Стратегия: {strategy_type}")
    print(f"   Капитал: ${initial_capital:,.2f}")
    print(f"   Комиссия: {commission_rate:.3%}")
    print(f"   Проскальзывание: {slippage_rate:.3%}")
    print(f"   Параллельное выполнение: {'Да' if parallel else 'Нет'}")
    
    if parallel and max_workers > 1:
        # Параллельный бектестинг
        print(f"\n🔄 Запуск параллельного бектестинга...")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Запускаем задачи
            future_to_symbol = {
                executor.submit(backtest_single_model, symbol, strategy_type, 
                              strategy_params, initial_capital, commission_rate,
                              slippage_rate, start_date, end_date): symbol 
                for symbol in symbols
            }
            
            # Собираем результаты
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    if result['status'] == 'success':
                        print(f"✅ {symbol}: Доходность={result['total_return']:.2%}, "
                              f"Шарп={result['sharpe_ratio']:.3f}, Время={result['time']:.1f}с")
                    else:
                        print(f"❌ {symbol}: {result['error']}")
                        
                except Exception as e:
                    print(f"❌ {symbol}: Ошибка выполнения - {e}")
                    results.append({
                        'symbol': symbol,
                        'status': 'failed',
                        'error': str(e),
                        'time': 0,
                        'strategy': strategy_type
                    })
    else:
        # Последовательный бектестинг
        print(f"\n🔄 Запуск последовательного бектестинга...")
        
        for i, symbol in enumerate(symbols, 1):
            print(f"\n📊 Прогресс: {i}/{len(symbols)}")
            
            result = backtest_single_model(symbol, strategy_type, strategy_params,
                                         initial_capital, commission_rate,
                                         slippage_rate, start_date, end_date)
            results.append(result)
            
            if result['status'] == 'success':
                print(f"✅ {symbol}: Доходность={result['total_return']:.2%}, "
                      f"Шарп={result['sharpe_ratio']:.3f}, Время={result['time']:.1f}с")
            else:
                print(f"❌ {symbol}: {result['error']}")
    
    # Создаем DataFrame с результатами
    df_results = pd.DataFrame(results)
    
    # Сортируем по доходности (лучшие результаты вверху)
    if not df_results.empty and 'total_return' in df_results.columns:
        df_results = df_results.sort_values('total_return', ascending=False)
    
    return df_results

def compare_strategies_for_symbols(symbols: list, strategies: dict,
                                  initial_capital: float = 10000.0,
                                  commission_rate: float = 0.001,
                                  slippage_rate: float = 0.0005,
                                  start_date: str = None,
                                  end_date: str = None) -> dict:
    """Сравнить стратегии для множества монет"""
    
    all_results = {}
    
    for strategy_name, strategy_config in strategies.items():
        print(f"\n🔄 Тестируем стратегию: {strategy_name}")
        
        strategy_type = strategy_config['strategy_type']
        strategy_params = get_strategy_params(strategy_type)
        
        results = []
        
        for symbol in symbols:
            try:
                result = backtest_single_model(
                    symbol, strategy_type, strategy_params,
                    initial_capital, commission_rate, slippage_rate,
                    start_date, end_date
                )
                results.append(result)
                
                if result['status'] == 'success':
                    print(f"✅ {symbol}: {result['total_return']:.2%}")
                else:
                    print(f"❌ {symbol}: {result['error']}")
                    
            except Exception as e:
                print(f"❌ {symbol}: Ошибка - {e}")
                results.append({
                    'symbol': symbol,
                    'status': 'failed',
                    'error': str(e),
                    'strategy': strategy_type
                })
        
        all_results[strategy_name] = pd.DataFrame(results)
    
    return all_results

def print_backtest_summary(df_results: pd.DataFrame):
    """Вывести сводный отчет по бектестингу"""
    
    if df_results.empty:
        print("❌ Нет результатов для отчета")
        return
    
    print("\n" + "="*100)
    print("📊 СВОДНЫЙ ОТЧЕТ ПО БЕКТЕСТИНГУ")
    print("="*100)
    
    # Общая статистика
    total_models = len(df_results)
    successful_models = len(df_results[df_results['status'] == 'success'])
    failed_models = total_models - successful_models
    
    print(f"\n📈 ОБЩАЯ СТАТИСТИКА:")
    print(f"   Всего монет: {total_models}")
    print(f"   Успешно протестировано: {successful_models}")
    print(f"   Ошибок: {failed_models}")
    print(f"   Успешность: {successful_models/total_models*100:.1f}%")
    
    if successful_models > 0:
        # Статистика по успешным моделям
        successful_df = df_results[df_results['status'] == 'success']
        
        print(f"\n🏆 ЛУЧШИЕ РЕЗУЛЬТАТЫ (по доходности):")
        print(f"{'Монета':<12} {'Доходность':<12} {'Годовая':<10} {'Шарп':<8} {'Макс.ДД':<10} {'Сделки':<8} {'Винрейт':<10}")
        print("-" * 80)
        
        for _, row in successful_df.head(10).iterrows():
            print(f"{row['symbol']:<12} {row['total_return']:<11.2%} "
                  f"{row['annual_return']:<9.2%} {row['sharpe_ratio']:<7.3f} "
                  f"{row['max_drawdown']:<9.2%} {row['total_trades']:<7} "
                  f"{row['win_rate']:<9.2%}")
        
        print(f"\n📊 СТАТИСТИКА ПО УСПЕШНЫМ МОДЕЛЯМ:")
        print(f"   Средняя доходность: {successful_df['total_return'].mean():.2%}")
        print(f"   Медианная доходность: {successful_df['total_return'].median():.2%}")
        print(f"   Лучшая доходность: {successful_df['total_return'].max():.2%}")
        print(f"   Худшая доходность: {successful_df['total_return'].min():.2%}")
        print(f"   Средний коэффициент Шарпа: {successful_df['sharpe_ratio'].mean():.3f}")
        print(f"   Средняя максимальная просадка: {successful_df['max_drawdown'].mean():.2%}")
        print(f"   Средний винрейт: {successful_df['win_rate'].mean():.2%}")
        print(f"   Общее время бектестинга: {successful_df['time'].sum():.1f}с")
        
        # Анализ риска
        profitable_models = successful_df[successful_df['total_return'] > 0]
        if len(profitable_models) > 0:
            print(f"\n💰 ПРИБЫЛЬНЫЕ МОДЕЛИ: {len(profitable_models)}/{len(successful_df)} ({len(profitable_models)/len(successful_df)*100:.1f}%)")
            print(f"   Средняя доходность прибыльных: {profitable_models['total_return'].mean():.2%}")
            print(f"   Средний коэффициент Шарпа прибыльных: {profitable_models['sharpe_ratio'].mean():.3f}")
    
    if failed_models > 0:
        print(f"\n❌ ОШИБКИ:")
        failed_df = df_results[df_results['status'] == 'failed']
        for _, row in failed_df.iterrows():
            print(f"   {row['symbol']}: {row['error']}")

def save_backtest_results(df_results: pd.DataFrame, filename: str = None):
    """Сохранить результаты бектестинга"""
    if df_results.empty:
        print("❌ Нет результатов для сохранения")
        return
    
    if filename is None:
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        filename = f"backtest_results_{timestamp}.csv"
    
    # Создаем директорию если нужно
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    
    # Сохраняем результаты
    df_results.to_csv(filename, index=False)
    print(f"✅ Результаты сохранены: {filename}")

def main():
    parser = argparse.ArgumentParser(description='Multiple Model Backtesting')
    
    # Параметры выбора монет
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--top', type=int, 
                      help='Бектестинг топ-N монет (например, --top 10)')
    group.add_argument('--symbols', type=str,
                      help='Список конкретных монет через запятую (например, BTCUSDT,ETHUSDT)')
    
    # Параметры бектестинга
    parser.add_argument('--strategy', type=str, default='simple',
                       choices=['simple', 'dynamic', 'confidence', 'risk_adjusted'],
                       help='Тип торговой стратегии')
    parser.add_argument('--capital', type=float, default=10000.0,
                       help='Начальный капитал')
    parser.add_argument('--commission', type=float, default=0.001,
                       help='Комиссия за сделку')
    parser.add_argument('--slippage', type=float, default=0.0005,
                       help='Проскальзывание')
    
    # Даты
    parser.add_argument('--start-date', type=str, default=None,
                       help='Дата начала (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None,
                       help='Дата окончания (YYYY-MM-DD)')
    
    # Дополнительные опции
    parser.add_argument('--compare', action='store_true',
                       help='Сравнить все стратегии')
    parser.add_argument('--parallel', action='store_true',
                       help='Параллельное выполнение')
    parser.add_argument('--max-workers', type=int, default=4,
                       help='Максимум воркеров для параллельного выполнения')
    parser.add_argument('--output', type=str, default=None,
                       help='Файл для сохранения результатов')
    
    args = parser.parse_args()
    
    try:
        # Получаем список символов
        symbols = get_symbols_list(args.top, args.symbols)
        
        print(f"🎯 Бектестинг {len(symbols)} монет:")
        for i, symbol in enumerate(symbols, 1):
            print(f"   {i:2d}. {symbol}")
        
        if args.compare:
            # Сравнение стратегий
            strategies = {
                'Simple': {'strategy_type': 'simple'},
                'Dynamic': {'strategy_type': 'dynamic'},
                'Confidence': {'strategy_type': 'confidence'},
                'Risk Adjusted': {'strategy_type': 'risk_adjusted'}
            }
            
            all_results = compare_strategies_for_symbols(
                symbols, strategies, args.capital, args.commission, args.slippage,
                args.start_date, args.end_date
            )
            
            # Выводим сравнение
            print("\n" + "="*120)
            print("📊 СРАВНЕНИЕ СТРАТЕГИЙ")
            print("="*120)
            
            for strategy_name, df_results in all_results.items():
                if not df_results.empty:
                    successful = df_results[df_results['status'] == 'success']
                    if len(successful) > 0:
                        avg_return = successful['total_return'].mean()
                        avg_sharpe = successful['sharpe_ratio'].mean()
                        print(f"{strategy_name:<15}: Доходность={avg_return:.2%}, Шарп={avg_sharpe:.3f}")
            
            # Сохраняем результаты
            if args.output:
                for strategy_name, df_results in all_results.items():
                    filename = args.output.replace('.csv', f'_{strategy_name.lower().replace(" ", "_")}.csv')
                    save_backtest_results(df_results, filename)
        else:
            # Одиночный бектестинг
            results_df = backtest_multiple_models(
                symbols=symbols,
                strategy_type=args.strategy,
                initial_capital=args.capital,
                commission_rate=args.commission,
                slippage_rate=args.slippage,
                start_date=args.start_date,
                end_date=args.end_date,
                parallel=args.parallel,
                max_workers=args.max_workers
            )
            
            # Выводим отчет
            print_backtest_summary(results_df)
            
            # Сохраняем результаты
            save_backtest_results(results_df, args.output)
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 