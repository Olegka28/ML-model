#!/usr/bin/env python3
"""
🚀 Multiple Model Training - обучение моделей на множестве монет

Использование:
    python train_multiple.py --top 10 --target crypto_clipped
    python train_multiple.py --symbols BTCUSDT,ETHUSDT,SOLUSDT --target crypto_clipped
"""

import argparse
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

# Добавляем путь к модулям
sys.path.append(str(Path(__file__).parent))

from ml_module.utils.config import Config
from ml_module.core.model_manager import ModelManager
from ml_module.data_collector.data_manager import DataManager
from ml_module.features.feature_manager import FeatureManager
from ml_module.systems.regression_system import RegressionSystem

# Топ-20 монет по капитализации (обновляется периодически)
TOP_COINS = [
    'BTCUSDT',   # Bitcoin
    'ETHUSDT',   # Ethereum
    'BNBUSDT',   # BNB
    'SOLUSDT',   # Solana
    'XRPUSDT',   # XRP
    'ADAUSDT',   # Cardano
    'AVAXUSDT',  # Avalanche
    'DOGEUSDT',  # Dogecoin
    'DOTUSDT',   # Polkadot
    'MATICUSDT', # Polygon
    'LINKUSDT',  # Chainlink
    'UNIUSDT',   # Uniswap
    'ATOMUSDT',  # Cosmos
    'LTCUSDT',   # Litecoin
    'ETCUSDT',   # Ethereum Classic
    'XLMUSDT',   # Stellar
    'BCHUSDT',   # Bitcoin Cash
    'FILUSDT',   # Filecoin
    'TRXUSDT',   # TRON
    'NEARUSDT'   # NEAR Protocol
]

def get_symbols_list(top_n: int = None, symbols: str = None) -> list:
    """Получить список символов для обучения"""
    if symbols:
        # Пользователь указал конкретные символы
        return [s.strip().upper() for s in symbols.split(',')]
    elif top_n:
        # Берем топ-N монет
        return TOP_COINS[:top_n]
    else:
        # По умолчанию топ-10
        return TOP_COINS[:10]

def train_single_model(symbol: str, target: str, config: Config, 
                      max_workers: int = 1) -> dict:
    """Обучить модель для одной монеты"""
    start_time = time.time()
    
    try:
        print(f"\n🚀 Обучение модели для {symbol}...")
        
        # Создаем систему обучения
        system = RegressionSystem(config)
        
        # Обучаем модель через run_experiment
        metadata = system.run_experiment(
            symbol=symbol,
            target_type=target
        )
        
        # Загружаем обученную модель
        model, _ = system.load_model(symbol, 'regression')
        
        if model is None:
            return {
                'symbol': symbol,
                'status': 'failed',
                'error': 'Модель не была создана',
                'time': time.time() - start_time
            }
        
        # Получаем метрики
        metrics = metadata.get('metrics', {})
        baseline = metadata.get('baseline', {})
        
        # Определяем лучшую baseline модель
        best_baseline = None
        if isinstance(baseline, dict):
            for name, metrics_dict in baseline.items():
                if isinstance(metrics_dict, dict):
                    if best_baseline is None or metrics_dict.get('rmse', float('inf')) < best_baseline[1]:
                        best_baseline = (name, metrics_dict.get('rmse', float('inf')))
        
        return {
            'symbol': symbol,
            'status': 'success',
            'rmse': metrics.get('rmse', 0),
            'mae': metrics.get('mae', 0),
            'r2': metrics.get('r2', 0),
            'best_baseline': best_baseline[0] if best_baseline else 'N/A',
            'baseline_rmse': best_baseline[1] if best_baseline else 0,
            'improvement': ((best_baseline[1] - metrics.get('rmse', 0)) / best_baseline[1] * 100) if best_baseline and best_baseline[1] > 0 else 0,
            'time': time.time() - start_time,
            'model_path': metadata.get('model_path', 'N/A')
        }
        
    except Exception as e:
        return {
            'symbol': symbol,
            'status': 'failed',
            'error': str(e),
            'time': time.time() - start_time
        }

def train_multiple_models(symbols: list, target: str, 
                         max_workers: int = 1, 
                         parallel: bool = False) -> pd.DataFrame:
    """Обучить модели для множества монет"""
    
    config = Config()
    results = []
    
    print(f"🎯 Обучение моделей для {len(symbols)} монет")
    print(f"   Целевая переменная: {target}")
    print(f"   Параллельное выполнение: {'Да' if parallel else 'Нет'}")
    print(f"   Максимум воркеров: {max_workers}")
    
    if parallel and max_workers > 1:
        # Параллельное обучение
        print(f"\n🔄 Запуск параллельного обучения...")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Запускаем задачи
            future_to_symbol = {
                executor.submit(train_single_model, symbol, target, config, 1): symbol 
                for symbol in symbols
            }
            
            # Собираем результаты
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    if result['status'] == 'success':
                        print(f"✅ {symbol}: RMSE={result['rmse']:.6f}, R²={result['r2']:.3f}, Время={result['time']:.1f}с")
                    else:
                        print(f"❌ {symbol}: {result['error']}")
                        
                except Exception as e:
                    print(f"❌ {symbol}: Ошибка выполнения - {e}")
                    results.append({
                        'symbol': symbol,
                        'status': 'failed',
                        'error': str(e),
                        'time': 0
                    })
    else:
        # Последовательное обучение
        print(f"\n🔄 Запуск последовательного обучения...")
        
        for i, symbol in enumerate(symbols, 1):
            print(f"\n📊 Прогресс: {i}/{len(symbols)}")
            
            result = train_single_model(symbol, target, config, max_workers)
            results.append(result)
            
            if result['status'] == 'success':
                print(f"✅ {symbol}: RMSE={result['rmse']:.6f}, R²={result['r2']:.3f}, Время={result['time']:.1f}с")
            else:
                print(f"❌ {symbol}: {result['error']}")
    
    # Создаем DataFrame с результатами
    df_results = pd.DataFrame(results)
    
    # Сортируем по RMSE (лучшие модели вверху)
    if not df_results.empty and 'rmse' in df_results.columns:
        df_results = df_results.sort_values('rmse')
    
    return df_results

def print_summary_report(df_results: pd.DataFrame):
    """Вывести сводный отчет"""
    
    if df_results.empty:
        print("❌ Нет результатов для отчета")
        return
    
    print("\n" + "="*100)
    print("📊 СВОДНЫЙ ОТЧЕТ ОБ ОБУЧЕНИИ")
    print("="*100)
    
    # Общая статистика
    total_models = len(df_results)
    successful_models = len(df_results[df_results['status'] == 'success'])
    failed_models = total_models - successful_models
    
    print(f"\n📈 ОБЩАЯ СТАТИСТИКА:")
    print(f"   Всего монет: {total_models}")
    print(f"   Успешно обучено: {successful_models}")
    print(f"   Ошибок: {failed_models}")
    print(f"   Успешность: {successful_models/total_models*100:.1f}%")
    
    if successful_models > 0:
        # Статистика по успешным моделям
        successful_df = df_results[df_results['status'] == 'success']
        
        print(f"\n🏆 ЛУЧШИЕ МОДЕЛИ (по RMSE):")
        print(f"{'Монета':<12} {'RMSE':<12} {'R²':<8} {'Улучшение':<12} {'Время':<8}")
        print("-" * 60)
        
        for _, row in successful_df.head(5).iterrows():
            print(f"{row['symbol']:<12} {row['rmse']:<11.6f} {row['r2']:<7.3f} "
                  f"{row['improvement']:<11.1f}% {row['time']:<7.1f}с")
        
        print(f"\n📊 СТАТИСТИКА ПО УСПЕШНЫМ МОДЕЛЯМ:")
        print(f"   Средний RMSE: {successful_df['rmse'].mean():.6f}")
        print(f"   Медианный RMSE: {successful_df['rmse'].median():.6f}")
        print(f"   Лучший RMSE: {successful_df['rmse'].min():.6f}")
        print(f"   Худший RMSE: {successful_df['rmse'].max():.6f}")
        print(f"   Средний R²: {successful_df['r2'].mean():.3f}")
        print(f"   Среднее улучшение: {successful_df['improvement'].mean():.1f}%")
        print(f"   Общее время обучения: {successful_df['time'].sum():.1f}с")
    
    if failed_models > 0:
        print(f"\n❌ ОШИБКИ:")
        failed_df = df_results[df_results['status'] == 'failed']
        for _, row in failed_df.iterrows():
            print(f"   {row['symbol']}: {row['error']}")

def save_results(df_results: pd.DataFrame, filename: str = None):
    """Сохранить результаты в файл"""
    if df_results.empty:
        print("❌ Нет результатов для сохранения")
        return
    
    if filename is None:
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        filename = f"training_results_{timestamp}.csv"
    
    # Создаем директорию если нужно
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    
    # Сохраняем результаты
    df_results.to_csv(filename, index=False)
    print(f"✅ Результаты сохранены: {filename}")

def main():
    parser = argparse.ArgumentParser(description='Multiple Model Training')
    
    # Параметры выбора монет
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--top', type=int, 
                      help='Обучить топ-N монет (например, --top 10)')
    group.add_argument('--symbols', type=str,
                      help='Список конкретных монет через запятую (например, BTCUSDT,ETHUSDT)')
    
    # Параметры обучения
    parser.add_argument('--target', type=str, default='crypto_clipped',
                       help='Целевая переменная (по умолчанию: crypto_clipped)')
    parser.add_argument('--max-workers', type=int, default=1,
                       help='Максимум воркеров для Optuna (по умолчанию: 1)')
    parser.add_argument('--parallel', action='store_true',
                       help='Параллельное обучение моделей')
    parser.add_argument('--output', type=str, default=None,
                       help='Файл для сохранения результатов')
    
    args = parser.parse_args()
    
    try:
        # Получаем список символов
        symbols = get_symbols_list(args.top, args.symbols)
        
        print(f"🎯 Обучение моделей для {len(symbols)} монет:")
        for i, symbol in enumerate(symbols, 1):
            print(f"   {i:2d}. {symbol}")
        
        # Обучаем модели
        results_df = train_multiple_models(
            symbols=symbols,
            target=args.target,
            max_workers=args.max_workers,
            parallel=args.parallel
        )
        
        # Выводим отчет
        print_summary_report(results_df)
        
        # Сохраняем результаты
        save_results(results_df, args.output)
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 