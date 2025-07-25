#!/usr/bin/env python3
"""
📊 Backtest Metrics - метрики для оценки результатов бектестинга

Расчет различных метрик производительности торговой стратегии.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy import stats
import warnings

class BacktestMetrics:
    """Класс для расчета метрик бектестинга"""
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Инициализация
        
        Args:
            risk_free_rate: Безрисковая ставка (годовая)
        """
        self.risk_free_rate = risk_free_rate
    
    def calculate_metrics(self, equity_curve: pd.DataFrame, 
                         benchmark_curve: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """
        Рассчитать все метрики бектестинга
        
        Args:
            equity_curve: DataFrame с колонками ['timestamp', 'equity', 'returns']
            benchmark_curve: DataFrame с бенчмарком (опционально)
            
        Returns:
            Словарь с метриками
        """
        if equity_curve.empty:
            return {}
        
        # Базовые метрики
        metrics = {}
        
        # Общая доходность
        total_return = self._calculate_total_return(equity_curve)
        metrics['total_return'] = total_return
        
        # Годовая доходность
        annual_return = self._calculate_annual_return(equity_curve)
        metrics['annual_return'] = annual_return
        
        # Волатильность
        volatility = self._calculate_volatility(equity_curve)
        metrics['volatility'] = volatility
        
        # Коэффициент Шарпа
        sharpe_ratio = self._calculate_sharpe_ratio(equity_curve)
        metrics['sharpe_ratio'] = sharpe_ratio
        
        # Максимальная просадка
        max_drawdown = self._calculate_max_drawdown(equity_curve)
        metrics['max_drawdown'] = max_drawdown
        
        # Коэффициент Кальмара
        calmar_ratio = self._calculate_calmar_ratio(annual_return, max_drawdown)
        metrics['calmar_ratio'] = calmar_ratio
        
        # Коэффициент Сортино
        sortino_ratio = self._calculate_sortino_ratio(equity_curve)
        metrics['sortino_ratio'] = sortino_ratio
        
        # Коэффициент Трейнора
        treynor_ratio = self._calculate_treynor_ratio(equity_curve)
        metrics['treynor_ratio'] = treynor_ratio
        
        # Коэффициент информации
        if benchmark_curve is not None:
            information_ratio = self._calculate_information_ratio(equity_curve, benchmark_curve)
            metrics['information_ratio'] = information_ratio
        
        # Бета и Альфа
        if benchmark_curve is not None:
            beta = self._calculate_beta(equity_curve, benchmark_curve)
            alpha = self._calculate_alpha(annual_return, beta, benchmark_curve)
            metrics['beta'] = beta
            metrics['alpha'] = alpha
        
        # Дополнительные метрики
        metrics['var_95'] = self._calculate_var(equity_curve, 0.95)
        metrics['cvar_95'] = self._calculate_cvar(equity_curve, 0.95)
        metrics['skewness'] = self._calculate_skewness(equity_curve)
        metrics['kurtosis'] = self._calculate_kurtosis(equity_curve)
        
        # Метрики риска
        metrics['downside_deviation'] = self._calculate_downside_deviation(equity_curve)
        metrics['gain_loss_ratio'] = self._calculate_gain_loss_ratio(equity_curve)
        
        return metrics
    
    def _calculate_total_return(self, equity_curve: pd.DataFrame) -> float:
        """Рассчитать общую доходность"""
        if len(equity_curve) < 2:
            return 0.0
        
        initial_equity = equity_curve['equity'].iloc[0]
        final_equity = equity_curve['equity'].iloc[-1]
        
        return (final_equity - initial_equity) / initial_equity
    
    def _calculate_annual_return(self, equity_curve: pd.DataFrame) -> float:
        """Рассчитать годовую доходность"""
        if len(equity_curve) < 2:
            return 0.0
        
        total_return = self._calculate_total_return(equity_curve)
        
        # Рассчитываем количество лет
        start_date = equity_curve['timestamp'].iloc[0]
        end_date = equity_curve['timestamp'].iloc[-1]
        years = (end_date - start_date).days / 365.25
        
        if years <= 0:
            return 0.0
        
        return (1 + total_return) ** (1 / years) - 1
    
    def _calculate_volatility(self, equity_curve: pd.DataFrame) -> float:
        """Рассчитать волатильность (годовую)"""
        if 'returns' not in equity_curve.columns or len(equity_curve) < 2:
            return 0.0
        
        returns = equity_curve['returns'].dropna()
        if len(returns) == 0:
            return 0.0
        
        # Годовая волатильность
        daily_vol = returns.std()
        annual_vol = daily_vol * np.sqrt(252)  # 252 торговых дня
        
        return annual_vol
    
    def _calculate_sharpe_ratio(self, equity_curve: pd.DataFrame) -> float:
        """Рассчитать коэффициент Шарпа"""
        if 'returns' not in equity_curve.columns or len(equity_curve) < 2:
            return 0.0
        
        returns = equity_curve['returns'].dropna()
        if len(returns) == 0:
            return 0.0
        
        annual_return = self._calculate_annual_return(equity_curve)
        annual_vol = self._calculate_volatility(equity_curve)
        
        if annual_vol == 0:
            return 0.0
        
        # Годовая безрисковая ставка
        daily_rf = (1 + self.risk_free_rate) ** (1 / 252) - 1
        excess_return = annual_return - self.risk_free_rate
        
        return excess_return / annual_vol
    
    def _calculate_max_drawdown(self, equity_curve: pd.DataFrame) -> float:
        """Рассчитать максимальную просадку"""
        if len(equity_curve) < 2:
            return 0.0
        
        equity = equity_curve['equity'].values
        peak = equity[0]
        max_dd = 0.0
        
        for value in equity:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
        
        return max_dd
    
    def _calculate_calmar_ratio(self, annual_return: float, max_drawdown: float) -> float:
        """Рассчитать коэффициент Кальмара"""
        if max_drawdown == 0:
            return 0.0
        
        return annual_return / max_drawdown
    
    def _calculate_sortino_ratio(self, equity_curve: pd.DataFrame) -> float:
        """Рассчитать коэффициент Сортино"""
        if 'returns' not in equity_curve.columns or len(equity_curve) < 2:
            return 0.0
        
        returns = equity_curve['returns'].dropna()
        if len(returns) == 0:
            return 0.0
        
        annual_return = self._calculate_annual_return(equity_curve)
        downside_deviation = self._calculate_downside_deviation(equity_curve)
        
        if downside_deviation == 0:
            return 0.0
        
        excess_return = annual_return - self.risk_free_rate
        return excess_return / downside_deviation
    
    def _calculate_treynor_ratio(self, equity_curve: pd.DataFrame) -> float:
        """Рассчитать коэффициент Трейнора"""
        # Упрощенная версия без бенчмарка
        annual_return = self._calculate_annual_return(equity_curve)
        volatility = self._calculate_volatility(equity_curve)
        
        if volatility == 0:
            return 0.0
        
        excess_return = annual_return - self.risk_free_rate
        return excess_return / volatility
    
    def _calculate_information_ratio(self, equity_curve: pd.DataFrame, 
                                   benchmark_curve: pd.DataFrame) -> float:
        """Рассчитать коэффициент информации"""
        if 'returns' not in equity_curve.columns or 'returns' not in benchmark_curve.columns:
            return 0.0
        
        strategy_returns = equity_curve['returns'].dropna()
        benchmark_returns = benchmark_curve['returns'].dropna()
        
        # Выравниваем индексы
        common_index = strategy_returns.index.intersection(benchmark_returns.index)
        if len(common_index) == 0:
            return 0.0
        
        strategy_returns = strategy_returns.loc[common_index]
        benchmark_returns = benchmark_returns.loc[common_index]
        
        excess_returns = strategy_returns - benchmark_returns
        tracking_error = excess_returns.std() * np.sqrt(252)
        
        if tracking_error == 0:
            return 0.0
        
        return excess_returns.mean() * 252 / tracking_error
    
    def _calculate_beta(self, equity_curve: pd.DataFrame, 
                       benchmark_curve: pd.DataFrame) -> float:
        """Рассчитать бета"""
        if 'returns' not in equity_curve.columns or 'returns' not in benchmark_curve.columns:
            return 1.0
        
        strategy_returns = equity_curve['returns'].dropna()
        benchmark_returns = benchmark_curve['returns'].dropna()
        
        # Выравниваем индексы
        common_index = strategy_returns.index.intersection(benchmark_returns.index)
        if len(common_index) < 2:
            return 1.0
        
        strategy_returns = strategy_returns.loc[common_index]
        benchmark_returns = benchmark_returns.loc[common_index]
        
        # Рассчитываем ковариацию и дисперсию
        covariance = np.cov(strategy_returns, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns)
        
        if benchmark_variance == 0:
            return 1.0
        
        return covariance / benchmark_variance
    
    def _calculate_alpha(self, strategy_return: float, beta: float, 
                        benchmark_curve: pd.DataFrame) -> float:
        """Рассчитать альфа"""
        if 'returns' not in benchmark_curve.columns:
            return 0.0
        
        benchmark_returns = benchmark_curve['returns'].dropna()
        if len(benchmark_returns) == 0:
            return 0.0
        
        benchmark_return = benchmark_returns.mean() * 252
        alpha = strategy_return - (self.risk_free_rate + beta * (benchmark_return - self.risk_free_rate))
        
        return alpha
    
    def _calculate_var(self, equity_curve: pd.DataFrame, confidence: float) -> float:
        """Рассчитать Value at Risk"""
        if 'returns' not in equity_curve.columns:
            return 0.0
        
        returns = equity_curve['returns'].dropna()
        if len(returns) == 0:
            return 0.0
        
        return np.percentile(returns, (1 - confidence) * 100)
    
    def _calculate_cvar(self, equity_curve: pd.DataFrame, confidence: float) -> float:
        """Рассчитать Conditional Value at Risk (Expected Shortfall)"""
        if 'returns' not in equity_curve.columns:
            return 0.0
        
        returns = equity_curve['returns'].dropna()
        if len(returns) == 0:
            return 0.0
        
        var = self._calculate_var(equity_curve, confidence)
        return returns[returns <= var].mean()
    
    def _calculate_skewness(self, equity_curve: pd.DataFrame) -> float:
        """Рассчитать асимметрию"""
        if 'returns' not in equity_curve.columns:
            return 0.0
        
        returns = equity_curve['returns'].dropna()
        if len(returns) == 0:
            return 0.0
        
        return stats.skew(returns)
    
    def _calculate_kurtosis(self, equity_curve: pd.DataFrame) -> float:
        """Рассчитать эксцесс"""
        if 'returns' not in equity_curve.columns:
            return 0.0
        
        returns = equity_curve['returns'].dropna()
        if len(returns) == 0:
            return 0.0
        
        return stats.kurtosis(returns)
    
    def _calculate_downside_deviation(self, equity_curve: pd.DataFrame) -> float:
        """Рассчитать downside deviation"""
        if 'returns' not in equity_curve.columns:
            return 0.0
        
        returns = equity_curve['returns'].dropna()
        if len(returns) == 0:
            return 0.0
        
        # Только отрицательные доходности
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0:
            return 0.0
        
        return downside_returns.std() * np.sqrt(252)
    
    def _calculate_gain_loss_ratio(self, equity_curve: pd.DataFrame) -> float:
        """Рассчитать соотношение прибыли/убытков"""
        if 'returns' not in equity_curve.columns:
            return 0.0
        
        returns = equity_curve['returns'].dropna()
        if len(returns) == 0:
            return 0.0
        
        gains = returns[returns > 0]
        losses = returns[returns < 0]
        
        if len(losses) == 0:
            return float('inf') if len(gains) > 0 else 0.0
        
        avg_gain = gains.mean() if len(gains) > 0 else 0.0
        avg_loss = abs(losses.mean())
        
        return avg_gain / avg_loss if avg_loss != 0 else 0.0
    
    def generate_report(self, metrics: Dict[str, float]) -> str:
        """Сгенерировать текстовый отчет"""
        report = "📊 ОТЧЕТ О БЕКТЕСТИНГЕ\n"
        report += "=" * 50 + "\n\n"
        
        # Основные метрики
        report += "💰 ДОХОДНОСТЬ:\n"
        report += f"   Общая доходность: {metrics.get('total_return', 0):.2%}\n"
        report += f"   Годовая доходность: {metrics.get('annual_return', 0):.2%}\n"
        report += f"   Волатильность: {metrics.get('volatility', 0):.2%}\n\n"
        
        # Метрики риска
        report += "⚠️ РИСК:\n"
        report += f"   Максимальная просадка: {metrics.get('max_drawdown', 0):.2%}\n"
        report += f"   VaR (95%): {metrics.get('var_95', 0):.2%}\n"
        report += f"   CVaR (95%): {metrics.get('cvar_95', 0):.2%}\n\n"
        
        # Коэффициенты
        report += "📈 КОЭФФИЦИЕНТЫ:\n"
        report += f"   Коэффициент Шарпа: {metrics.get('sharpe_ratio', 0):.3f}\n"
        report += f"   Коэффициент Сортино: {metrics.get('sortino_ratio', 0):.3f}\n"
        report += f"   Коэффициент Кальмара: {metrics.get('calmar_ratio', 0):.3f}\n"
        report += f"   Коэффициент Трейнора: {metrics.get('treynor_ratio', 0):.3f}\n\n"
        
        # Дополнительные метрики
        if 'information_ratio' in metrics:
            report += f"   Коэффициент информации: {metrics['information_ratio']:.3f}\n"
        if 'beta' in metrics:
            report += f"   Бета: {metrics['beta']:.3f}\n"
        if 'alpha' in metrics:
            report += f"   Альфа: {metrics['alpha']:.2%}\n"
        
        report += f"   Соотношение прибыль/убыток: {metrics.get('gain_loss_ratio', 0):.3f}\n"
        report += f"   Асимметрия: {metrics.get('skewness', 0):.3f}\n"
        report += f"   Эксцесс: {metrics.get('kurtosis', 0):.3f}\n"
        
        return report 