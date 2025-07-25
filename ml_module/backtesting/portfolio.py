#!/usr/bin/env python3
"""
💰 Portfolio - управление портфелем в бектестинге

Симуляция торгового портфеля с учетом комиссий, проскальзываний и ограничений.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Position:
    """Позиция в портфеле"""
    symbol: str
    side: str  # 'long' или 'short'
    size: float
    entry_price: float
    entry_time: pd.Timestamp
    current_price: float
    pnl: float = 0.0
    unrealized_pnl: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

@dataclass
class Trade:
    """Сделка"""
    timestamp: pd.Timestamp
    symbol: str
    side: str  # 'buy' или 'sell'
    size: float
    price: float
    commission: float
    slippage: float
    total_cost: float

class Portfolio:
    """Класс для управления торговым портфелем"""
    
    def __init__(self, initial_capital: float = 10000.0, 
                 commission_rate: float = 0.001,  # 0.1%
                 slippage_rate: float = 0.0005,   # 0.05%
                 max_position_size: float = 0.2,  # 20% от капитала
                 enable_stop_loss: bool = True,
                 enable_take_profit: bool = True):
        
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.max_position_size = max_position_size
        self.enable_stop_loss = enable_stop_loss
        self.enable_take_profit = enable_take_profit
        
        # Состояние портфеля
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.cash = initial_capital
        self.equity_history = []
        
        # Статистика
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_commission = 0.0
        self.total_slippage = 0.0
    
    def get_available_capital(self) -> float:
        """Получить доступный капитал"""
        return self.cash
    
    def get_position_value(self, symbol: str) -> float:
        """Получить стоимость позиции"""
        if symbol in self.positions:
            position = self.positions[symbol]
            return position.size * position.current_price
        return 0.0
    
    def get_total_equity(self) -> float:
        """Получить общую стоимость портфеля"""
        total_equity = self.cash
        for position in self.positions.values():
            total_equity += position.size * position.current_price
        return total_equity
    
    def update_prices(self, price_data: Dict[str, float], timestamp: pd.Timestamp):
        """Обновить цены позиций"""
        for symbol, price in price_data.items():
            if symbol in self.positions:
                position = self.positions[symbol]
                old_price = position.current_price
                position.current_price = price
                
                # Обновляем P&L
                if position.side == 'long':
                    position.unrealized_pnl = (price - position.entry_price) * position.size
                else:  # short
                    position.unrealized_pnl = (position.entry_price - price) * position.size
        
        # Проверяем стоп-лоссы и тейк-профиты
        self._check_stop_orders(price_data, timestamp)
        
        # Обновляем историю
        self.equity_history.append({
            'timestamp': timestamp,
            'equity': self.get_total_equity(),
            'cash': self.cash,
            'positions_value': sum(self.get_position_value(sym) for sym in self.positions.keys())
        })
    
    def execute_signal(self, signal, timestamp: pd.Timestamp) -> bool:
        """Выполнить торговый сигнал"""
        symbol = getattr(signal, 'symbol', 'UNKNOWN')
        action = signal.action
        price = signal.price
        volume = getattr(signal, 'volume', 1.0)
        
        # Рассчитываем размер позиции
        if volume <= 0:
            volume = 1.0
        
        position_size = min(
            volume * self.max_position_size * self.current_capital / price,
            self.cash / price
        )
        
        if position_size <= 0:
            return False
        
        # Рассчитываем комиссии и проскальзывания
        commission = position_size * price * self.commission_rate
        slippage = position_size * price * self.slippage_rate
        total_cost = position_size * price + commission + slippage
        
        # Проверяем достаточность средств
        if total_cost > self.cash:
            return False
        
        # Выполняем сделку
        if action == 'buy':
            success = self._open_long_position(symbol, position_size, price, timestamp, 
                                             commission, slippage, total_cost)
        elif action == 'sell':
            success = self._open_short_position(symbol, position_size, price, timestamp,
                                              commission, slippage, total_cost)
        else:  # hold
            success = True
        
        return success
    
    def _open_long_position(self, symbol: str, size: float, price: float, 
                           timestamp: pd.Timestamp, commission: float, 
                           slippage: float, total_cost: float) -> bool:
        """Открыть длинную позицию"""
        
        # Закрываем существующую короткую позицию если есть
        if symbol in self.positions and self.positions[symbol].side == 'short':
            self._close_position(symbol, price, timestamp, commission, slippage)
        
        # Открываем новую позицию
        self.positions[symbol] = Position(
            symbol=symbol,
            side='long',
            size=size,
            entry_price=price,
            entry_time=timestamp,
            current_price=price
        )
        
        # Обновляем наличные
        self.cash -= total_cost
        
        # Записываем сделку
        self._record_trade(timestamp, symbol, 'buy', size, price, commission, slippage, total_cost)
        
        return True
    
    def _open_short_position(self, symbol: str, size: float, price: float,
                            timestamp: pd.Timestamp, commission: float,
                            slippage: float, total_cost: float) -> bool:
        """Открыть короткую позицию"""
        
        # Закрываем существующую длинную позицию если есть
        if symbol in self.positions and self.positions[symbol].side == 'long':
            self._close_position(symbol, price, timestamp, commission, slippage)
        
        # Открываем новую позицию
        self.positions[symbol] = Position(
            symbol=symbol,
            side='short',
            size=size,
            entry_price=price,
            entry_time=timestamp,
            current_price=price
        )
        
        # Обновляем наличные
        self.cash -= total_cost
        
        # Записываем сделку
        self._record_trade(timestamp, symbol, 'sell', size, price, commission, slippage, total_cost)
        
        return True
    
    def _close_position(self, symbol: str, price: float, timestamp: pd.Timestamp,
                       commission: float, slippage: float):
        """Закрыть позицию"""
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        
        # Рассчитываем P&L
        if position.side == 'long':
            pnl = (price - position.entry_price) * position.size
        else:  # short
            pnl = (position.entry_price - price) * position.size
        
        # Вычитаем комиссии
        pnl -= commission + slippage
        
        # Обновляем статистику
        self.total_trades += 1
        if pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        
        # Обновляем наличные
        self.cash += position.size * price - commission - slippage
        
        # Удаляем позицию
        del self.positions[symbol]
    
    def _check_stop_orders(self, price_data: Dict[str, float], timestamp: pd.Timestamp):
        """Проверить стоп-лоссы и тейк-профиты"""
        positions_to_close = []
        
        for symbol, position in self.positions.items():
            if symbol not in price_data:
                continue
            
            current_price = price_data[symbol]
            
            # Проверяем стоп-лосс
            if self.enable_stop_loss and position.stop_loss is not None:
                if (position.side == 'long' and current_price <= position.stop_loss) or \
                   (position.side == 'short' and current_price >= position.stop_loss):
                    positions_to_close.append(symbol)
                    continue
            
            # Проверяем тейк-профит
            if self.enable_take_profit and position.take_profit is not None:
                if (position.side == 'long' and current_price >= position.take_profit) or \
                   (position.side == 'short' and current_price <= position.take_profit):
                    positions_to_close.append(symbol)
                    continue
        
        # Закрываем позиции
        for symbol in positions_to_close:
            commission = self.positions[symbol].size * price_data[symbol] * self.commission_rate
            slippage = self.positions[symbol].size * price_data[symbol] * self.slippage_rate
            self._close_position(symbol, price_data[symbol], timestamp, commission, slippage)
    
    def _record_trade(self, timestamp: pd.Timestamp, symbol: str, side: str,
                     size: float, price: float, commission: float,
                     slippage: float, total_cost: float):
        """Записать сделку"""
        trade = Trade(
            timestamp=timestamp,
            symbol=symbol,
            side=side,
            size=size,
            price=price,
            commission=commission,
            slippage=slippage,
            total_cost=total_cost
        )
        
        self.trades.append(trade)
        self.total_commission += commission
        self.total_slippage += slippage
    
    def get_statistics(self) -> Dict[str, float]:
        """Получить статистику портфеля"""
        total_equity = self.get_total_equity()
        total_return = (total_equity - self.initial_capital) / self.initial_capital
        
        win_rate = self.winning_trades / max(self.total_trades, 1)
        
        return {
            'initial_capital': self.initial_capital,
            'current_capital': self.current_capital,
            'total_equity': total_equity,
            'total_return': total_return,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': win_rate,
            'total_commission': self.total_commission,
            'total_slippage': self.total_slippage,
            'open_positions': len(self.positions)
        }
    
    def get_equity_curve(self) -> pd.DataFrame:
        """Получить кривую доходности"""
        if not self.equity_history:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.equity_history)
        df['returns'] = df['equity'].pct_change()
        df['cumulative_returns'] = (1 + df['returns']).cumprod() - 1
        
        return df 