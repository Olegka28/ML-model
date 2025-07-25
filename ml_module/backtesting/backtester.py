#!/usr/bin/env python3
"""
🔄 Backtester - основной класс для бектестинга

Проведение полного бектестинга торговых стратегий на основе ML моделей.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
from datetime import datetime

from ..utils.config import Config
from ..utils.logger import Logger
from .strategy import TradingStrategy, create_strategy
from .portfolio import Portfolio
from .metrics import BacktestMetrics

class Backtester:
    """Основной класс для проведения бектестинга"""
    
    def __init__(self, config: Config):
        """
        Инициализация бектестера
        
        Args:
            config: Конфигурация системы
        """
        self.config = config
        self.logger = Logger('Backtester', level=config.log_level)
        
        # Компоненты бектестинга
        self.strategy = None
        self.portfolio = None
        self.metrics_calculator = BacktestMetrics()
        
        # Результаты
        self.results = {}
        self.equity_curve = pd.DataFrame()
        self.trades_history = []
        
    def setup_backtest(self, 
                      strategy_type: str = 'simple',
                      strategy_params: Optional[Dict] = None,
                      initial_capital: float = 10000.0,
                      commission_rate: float = 0.001,
                      slippage_rate: float = 0.0005,
                      max_position_size: float = 0.2):
        """
        Настройка параметров бектестинга
        
        Args:
            strategy_type: Тип стратегии ('simple', 'dynamic', 'confidence', 'risk_adjusted')
            strategy_params: Параметры стратегии
            initial_capital: Начальный капитал
            commission_rate: Комиссия за сделку
            slippage_rate: Проскальзывание
            max_position_size: Максимальный размер позиции
        """
        # Создаем стратегию
        if strategy_params is None:
            strategy_params = {}
        
        self.strategy = create_strategy(strategy_type, **strategy_params)
        
        # Создаем портфель
        self.portfolio = Portfolio(
            initial_capital=initial_capital,
            commission_rate=commission_rate,
            slippage_rate=slippage_rate,
            max_position_size=max_position_size
        )
        
        self.logger.info(f"✅ Бектестинг настроен:")
        self.logger.info(f"   Стратегия: {self.strategy.name}")
        self.logger.info(f"   Начальный капитал: ${initial_capital:,.2f}")
        self.logger.info(f"   Комиссия: {commission_rate:.3%}")
        self.logger.info(f"   Проскальзывание: {slippage_rate:.3%}")
    
    def run_backtest(self, 
                    data: pd.DataFrame,
                    model,
                    feature_columns: List[str],
                    symbol: str = 'UNKNOWN',
                    start_date: Optional[str] = None,
                    end_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Запуск бектестинга
        
        Args:
            data: DataFrame с данными (должен содержать 'close' и feature_columns)
            model: Обученная ML модель
            feature_columns: Список колонок с признаками
            symbol: Символ торгового инструмента
            start_date: Дата начала (опционально)
            end_date: Дата окончания (опционально)
            
        Returns:
            Словарь с результатами бектестинга
        """
        if self.strategy is None or self.portfolio is None:
            raise ValueError("Бектестинг не настроен. Вызовите setup_backtest()")
        
        self.logger.info(f"🚀 Запуск бектестинга для {symbol}")
        
        # Фильтруем данные по датам
        if start_date:
            data = data[data.index >= start_date]
        if end_date:
            data = data[data.index <= end_date]
        
        if len(data) == 0:
            raise ValueError("Нет данных для бектестинга")
        
        self.logger.info(f"📊 Период: {data.index[0]} - {data.index[-1]}")
        self.logger.info(f"📈 Количество баров: {len(data)}")
        
        # Проверяем наличие необходимых колонок
        required_columns = ['close'] + feature_columns
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Отсутствуют колонки: {missing_columns}")
        
        # Основной цикл бектестинга
        for i, (timestamp, row) in enumerate(data.iterrows()):
            try:
                # Получаем признаки
                features = row[feature_columns].values.reshape(1, -1)
                
                # Получаем предсказание модели
                prediction = model.predict(features)[0]
                
                # Рассчитываем уверенность (упрощенная версия)
                confidence = self._calculate_confidence(prediction, model)
                
                # Генерируем сигнал
                signal = self.strategy.generate_signal(
                    prediction=prediction,
                    confidence=confidence,
                    current_price=row['close'],
                    timestamp=timestamp,
                    symbol=symbol
                )
                
                # Обновляем цены в портфеле
                self.portfolio.update_prices({symbol: row['close']}, timestamp)
                
                # Выполняем сигнал
                if signal.action != 'hold':
                    success = self.portfolio.execute_signal(signal, timestamp)
                    if success:
                        self.logger.debug(f"✅ {timestamp}: {signal.action.upper()} {symbol} @ {row['close']:.4f}")
                    else:
                        self.logger.debug(f"❌ {timestamp}: Не удалось выполнить {signal.action}")
                
                # Логируем прогресс
                if i % 1000 == 0:
                    equity = self.portfolio.get_total_equity()
                    self.logger.info(f"📊 Прогресс: {i}/{len(data)} баров, Equity: ${equity:,.2f}")
            
            except Exception as e:
                self.logger.error(f"❌ Ошибка на баре {timestamp}: {e}")
                continue
        
        # Закрываем все открытые позиции в конце
        self._close_all_positions(data.iloc[-1]['close'], data.index[-1])
        
        # Рассчитываем результаты
        results = self._calculate_results()
        
        self.logger.info("✅ Бектестинг завершен")
        return results
    
    def _calculate_confidence(self, prediction: float, model) -> float:
        """Рассчитать уверенность в предсказании"""
        # Упрощенная версия - используем абсолютное значение предсказания
        # В реальном приложении здесь можно использовать predict_proba или другие методы
        abs_pred = abs(prediction)
        
        # Нормализуем к диапазону [0, 1]
        confidence = min(abs_pred * 10, 0.95)  # Максимум 95%
        
        return confidence
    
    def _close_all_positions(self, final_price: float, timestamp: pd.Timestamp):
        """Закрыть все открытые позиции"""
        if not self.portfolio.positions:
            return
        
        for symbol in list(self.portfolio.positions.keys()):
            commission = self.portfolio.positions[symbol].size * final_price * self.portfolio.commission_rate
            slippage = self.portfolio.positions[symbol].size * final_price * self.portfolio.slippage_rate
            self.portfolio._close_position(symbol, final_price, timestamp, commission, slippage)
    
    def _calculate_results(self) -> Dict[str, Any]:
        """Рассчитать итоговые результаты"""
        # Получаем статистику портфеля
        portfolio_stats = self.portfolio.get_statistics()
        
        # Получаем кривую доходности
        equity_curve = self.portfolio.get_equity_curve()
        
        # Рассчитываем метрики
        metrics = self.metrics_calculator.calculate_metrics(equity_curve)
        
        # Формируем результаты
        results = {
            'portfolio_stats': portfolio_stats,
            'metrics': metrics,
            'equity_curve': equity_curve,
            'trades': self.portfolio.trades,
            'signals': self.strategy.get_signals(),
            'strategy_name': self.strategy.name,
            'backtest_date': datetime.now().isoformat()
        }
        
        self.results = results
        self.equity_curve = equity_curve
        self.trades_history = self.portfolio.trades
        
        return results
    
    def get_summary_report(self) -> str:
        """Получить краткий отчет"""
        if not self.results:
            return "❌ Нет результатов бектестинга"
        
        portfolio_stats = self.results['portfolio_stats']
        metrics = self.results['metrics']
        
        report = f"📊 ОТЧЕТ О БЕКТЕСТИНГЕ - {self.results['strategy_name']}\n"
        report += "=" * 60 + "\n\n"
        
        # Основные результаты
        report += "💰 РЕЗУЛЬТАТЫ:\n"
        report += f"   Начальный капитал: ${portfolio_stats['initial_capital']:,.2f}\n"
        report += f"   Конечный капитал: ${portfolio_stats['total_equity']:,.2f}\n"
        report += f"   Общая доходность: {portfolio_stats['total_return']:.2%}\n"
        report += f"   Годовая доходность: {metrics.get('annual_return', 0):.2%}\n\n"
        
        # Торговля
        report += "📈 ТОРГОВЛЯ:\n"
        report += f"   Всего сделок: {portfolio_stats['total_trades']}\n"
        report += f"   Прибыльных: {portfolio_stats['winning_trades']}\n"
        report += f"   Убыточных: {portfolio_stats['losing_trades']}\n"
        report += f"   Винрейт: {portfolio_stats['win_rate']:.2%}\n\n"
        
        # Риск
        report += "⚠️ РИСК:\n"
        report += f"   Максимальная просадка: {metrics.get('max_drawdown', 0):.2%}\n"
        report += f"   Волатильность: {metrics.get('volatility', 0):.2%}\n"
        report += f"   Коэффициент Шарпа: {metrics.get('sharpe_ratio', 0):.3f}\n\n"
        
        # Комиссии
        report += "💸 КОМИССИИ:\n"
        report += f"   Общие комиссии: ${portfolio_stats['total_commission']:,.2f}\n"
        report += f"   Общее проскальзывание: ${portfolio_stats['total_slippage']:,.2f}\n"
        
        return report
    
    def get_detailed_report(self) -> str:
        """Получить детальный отчет"""
        if not self.results:
            return "❌ Нет результатов бектестинга"
        
        return self.metrics_calculator.generate_report(self.results['metrics'])
    
    def save_results(self, filepath: str):
        """Сохранить результаты в файл"""
        if not self.results:
            self.logger.warning("Нет результатов для сохранения")
            return
        
        try:
            # Создаем директорию если нужно
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            # Сохраняем результаты
            with open(filepath, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            
            # Сохраняем кривую доходности
            equity_file = filepath.replace('.json', '_equity.csv')
            self.equity_curve.to_csv(equity_file)
            
            self.logger.info(f"✅ Результаты сохранены: {filepath}")
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка сохранения результатов: {e}")
    
    def load_results(self, filepath: str) -> bool:
        """Загрузить результаты из файла"""
        try:
            with open(filepath, 'r') as f:
                self.results = json.load(f)
            
            # Загружаем кривую доходности
            equity_file = filepath.replace('.json', '_equity.csv')
            if Path(equity_file).exists():
                self.equity_curve = pd.read_csv(equity_file, index_col=0, parse_dates=True)
            
            self.logger.info(f"✅ Результаты загружены: {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка загрузки результатов: {e}")
            return False
    
    def compare_strategies(self, strategies: Dict[str, Dict], 
                          data: pd.DataFrame, model, feature_columns: List[str],
                          symbol: str = 'UNKNOWN') -> Dict[str, Any]:
        """
        Сравнить несколько стратегий
        
        Args:
            strategies: Словарь с параметрами стратегий
            data: Данные для бектестинга
            model: ML модель
            feature_columns: Признаки
            symbol: Символ
            
        Returns:
            Результаты сравнения
        """
        comparison_results = {}
        
        for strategy_name, strategy_config in strategies.items():
            self.logger.info(f"🔄 Тестируем стратегию: {strategy_name}")
            
            # Настраиваем стратегию
            self.setup_backtest(**strategy_config)
            
            # Запускаем бектестинг
            results = self.run_backtest(data, model, feature_columns, symbol)
            
            comparison_results[strategy_name] = {
                'total_return': results['portfolio_stats']['total_return'],
                'annual_return': results['metrics'].get('annual_return', 0),
                'sharpe_ratio': results['metrics'].get('sharpe_ratio', 0),
                'max_drawdown': results['metrics'].get('max_drawdown', 0),
                'total_trades': results['portfolio_stats']['total_trades'],
                'win_rate': results['portfolio_stats']['win_rate']
            }
        
        return comparison_results 