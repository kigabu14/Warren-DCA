"""
DCA Strategies Module
Implementation of various Dollar Cost Averaging strategies
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum


class DCAStrategy(Enum):
    """Enumeration of available DCA strategies"""
    FIXED_AMOUNT = "fixed_amount"
    VALUE_AVERAGING = "value_averaging" 
    DRAWDOWN_TRIGGER = "drawdown_trigger"
    MOMENTUM_GATED = "momentum_gated"
    ADAPTIVE_HYBRID = "adaptive_hybrid"


class BaseDCAStrategy(ABC):
    """Base class for all DCA strategies"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    @abstractmethod
    def execute(self, price_data: pd.DataFrame, dividend_data: pd.DataFrame, 
                params: Dict) -> pd.DataFrame:
        """
        Execute the DCA strategy
        
        Args:
            price_data: DataFrame with Date, Close, Volume, High, Low, Open
            dividend_data: DataFrame with Date, Dividend
            params: Strategy-specific parameters
            
        Returns:
            DataFrame with DCA execution results
        """
        pass
    
    def _prepare_result_dataframe(self) -> pd.DataFrame:
        """Prepare empty result DataFrame with standard columns"""
        return pd.DataFrame(columns=[
            'date', 'action', 'units', 'price', 'invested', 
            'cumulative_units', 'cost_basis', 'equity', 'pnl', 
            'peak_equity', 'drawdown', 'dividend_received'
        ])
    
    def _calculate_metrics(self, result_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate cumulative metrics for result DataFrame"""
        if result_df.empty:
            return result_df
        
        result_df = result_df.copy()
        
        # Ensure proper data types
        result_df['invested'] = result_df['invested'].fillna(0)
        result_df['units'] = result_df['units'].fillna(0)
        result_df['dividend_received'] = result_df['dividend_received'].fillna(0)
        
        # Calculate cumulative metrics
        result_df['cumulative_invested'] = result_df['invested'].cumsum()
        result_df['cumulative_units'] = result_df['units'].cumsum()
        result_df['cumulative_dividends'] = result_df['dividend_received'].cumsum()
        
        # Calculate cost basis
        valid_invested = result_df['cumulative_invested'] > 0
        result_df.loc[valid_invested, 'cost_basis'] = (
            result_df.loc[valid_invested, 'cumulative_invested'] / 
            result_df.loc[valid_invested, 'cumulative_units']
        )
        
        # Calculate equity (market value)
        result_df['equity'] = result_df['cumulative_units'] * result_df['price']
        
        # Calculate P&L
        result_df['pnl'] = result_df['equity'] - result_df['cumulative_invested']
        
        # Calculate drawdown
        result_df['peak_equity'] = result_df['equity'].expanding().max()
        result_df['drawdown'] = (result_df['equity'] - result_df['peak_equity']) / result_df['peak_equity']
        result_df['drawdown'] = result_df['drawdown'].fillna(0)
        
        return result_df


class FixedAmountStrategy(BaseDCAStrategy):
    """Traditional fixed amount DCA strategy"""
    
    def __init__(self):
        super().__init__(
            name="Fixed Amount DCA",
            description="Invest fixed amount at regular intervals"
        )
    
    def execute(self, price_data: pd.DataFrame, dividend_data: pd.DataFrame, 
                params: Dict) -> pd.DataFrame:
        """
        Execute fixed amount DCA strategy
        
        Parameters:
            - amount: Fixed amount to invest per period
            - frequency: 'weekly', 'biweekly', 'monthly' (default: 'monthly')
        """
        amount = params.get('amount', 1000)
        frequency = params.get('frequency', 'monthly')
        
        # Determine sampling frequency
        freq_map = {'weekly': 'W', 'biweekly': '2W', 'monthly': 'M'}
        freq = freq_map.get(frequency, 'M')
        
        # Resample price data to DCA frequency
        price_data = price_data.set_index('Date')
        dca_prices = price_data['Close'].resample(freq).first().dropna()
        
        # Prepare dividend data
        if not dividend_data.empty:
            dividend_data = dividend_data.set_index('Date')
            dividends = dividend_data['Dividend'].resample(freq).sum()
        else:
            dividends = pd.Series(index=dca_prices.index, data=0)
        
        # Execute DCA
        results = []
        cumulative_units = 0
        
        for date, price in dca_prices.items():
            units_bought = amount / price
            cumulative_units += units_bought
            
            # Get dividends received this period
            div_received = dividends.get(date, 0) * cumulative_units
            
            results.append({
                'date': date,
                'action': 'buy',
                'units': units_bought,
                'price': price,
                'invested': amount,
                'cumulative_units': cumulative_units,
                'dividend_received': div_received
            })
        
        result_df = pd.DataFrame(results)
        return self._calculate_metrics(result_df)


class ValueAveragingStrategy(BaseDCAStrategy):
    """Value Averaging strategy - target portfolio value growth"""
    
    def __init__(self):
        super().__init__(
            name="Value Averaging",
            description="Target specific portfolio value growth rate"
        )
    
    def execute(self, price_data: pd.DataFrame, dividend_data: pd.DataFrame, 
                params: Dict) -> pd.DataFrame:
        """
        Execute value averaging strategy
        
        Parameters:
            - target_growth_rate: Annual target growth rate (default: 0.08 = 8%)
            - initial_amount: Initial investment amount
            - max_investment_per_period: Maximum allowed investment per period
            - frequency: Investment frequency
        """
        target_growth_rate = params.get('target_growth_rate', 0.08)
        initial_amount = params.get('initial_amount', 1000)
        max_per_period = params.get('max_investment_per_period', 5000)
        frequency = params.get('frequency', 'monthly')
        
        # Convert annual growth rate to period growth rate
        periods_per_year = {'weekly': 52, 'biweekly': 26, 'monthly': 12}
        period_growth_rate = target_growth_rate / periods_per_year.get(frequency, 12)
        
        # Prepare data
        freq_map = {'weekly': 'W', 'biweekly': '2W', 'monthly': 'M'}
        freq = freq_map.get(frequency, 'M')
        
        price_data = price_data.set_index('Date')
        dca_prices = price_data['Close'].resample(freq).first().dropna()
        
        if not dividend_data.empty:
            dividend_data = dividend_data.set_index('Date')
            dividends = dividend_data['Dividend'].resample(freq).sum()
        else:
            dividends = pd.Series(index=dca_prices.index, data=0)
        
        # Execute Value Averaging
        results = []
        cumulative_units = 0
        period = 0
        
        for date, price in dca_prices.items():
            period += 1
            
            # Calculate target portfolio value
            target_value = initial_amount * ((1 + period_growth_rate) ** period)
            
            # Current portfolio value
            current_value = cumulative_units * price
            
            # Calculate required investment
            required_investment = target_value - current_value
            
            # Apply constraints
            actual_investment = max(0, min(required_investment, max_per_period))
            
            if actual_investment > 0:
                units_bought = actual_investment / price
                cumulative_units += units_bought
                action = 'buy'
            else:
                units_bought = 0
                action = 'hold'
            
            # Get dividends
            div_received = dividends.get(date, 0) * cumulative_units
            
            results.append({
                'date': date,
                'action': action,
                'units': units_bought,
                'price': price,
                'invested': actual_investment,
                'cumulative_units': cumulative_units,
                'dividend_received': div_received,
                'target_value': target_value,
                'required_investment': required_investment
            })
        
        result_df = pd.DataFrame(results)
        return self._calculate_metrics(result_df)


class DrawdownTriggerStrategy(BaseDCAStrategy):
    """DCA with additional purchases during price drawdowns"""
    
    def __init__(self):
        super().__init__(
            name="Drawdown Trigger DCA",
            description="Regular DCA with extra purchases during price drops"
        )
    
    def execute(self, price_data: pd.DataFrame, dividend_data: pd.DataFrame, 
                params: Dict) -> pd.DataFrame:
        """
        Execute drawdown trigger strategy
        
        Parameters:
            - base_amount: Regular DCA amount
            - drawdown_threshold: Trigger additional purchase when price drops X% (default: 0.1 = 10%)
            - extra_multiplier: Multiplier for extra investment during drawdowns (default: 2.0)
            - lookback_periods: Number of periods to look back for high price (default: 4)
            - frequency: Investment frequency
        """
        base_amount = params.get('base_amount', 1000)
        drawdown_threshold = params.get('drawdown_threshold', 0.1)
        extra_multiplier = params.get('extra_multiplier', 2.0)
        lookback_periods = params.get('lookback_periods', 4)
        frequency = params.get('frequency', 'monthly')
        
        # Prepare data
        freq_map = {'weekly': 'W', 'biweekly': '2W', 'monthly': 'M'}
        freq = freq_map.get(frequency, 'M')
        
        price_data = price_data.set_index('Date')
        dca_prices = price_data['Close'].resample(freq).first().dropna()
        
        if not dividend_data.empty:
            dividend_data = dividend_data.set_index('Date')
            dividends = dividend_data['Dividend'].resample(freq).sum()
        else:
            dividends = pd.Series(index=dca_prices.index, data=0)
        
        # Execute strategy
        results = []
        cumulative_units = 0
        price_history = []
        
        for i, (date, price) in enumerate(dca_prices.items()):
            price_history.append(price)
            
            # Base investment
            investment = base_amount
            action = 'buy'
            
            # Check for drawdown trigger
            if len(price_history) >= lookback_periods:
                recent_high = max(price_history[-lookback_periods:])
                drawdown = (recent_high - price) / recent_high
                
                if drawdown >= drawdown_threshold:
                    investment *= extra_multiplier
                    action = 'buy_extra'
            
            units_bought = investment / price
            cumulative_units += units_bought
            
            # Get dividends
            div_received = dividends.get(date, 0) * cumulative_units
            
            results.append({
                'date': date,
                'action': action,
                'units': units_bought,
                'price': price,
                'invested': investment,
                'cumulative_units': cumulative_units,
                'dividend_received': div_received,
                'drawdown_triggered': action == 'buy_extra'
            })
        
        result_df = pd.DataFrame(results)
        return self._calculate_metrics(result_df)


class MomentumGatedStrategy(BaseDCAStrategy):
    """DCA with momentum filtering - skip purchases during negative momentum"""
    
    def __init__(self):
        super().__init__(
            name="Momentum Gated DCA",
            description="Skip DCA purchases during negative momentum periods"
        )
    
    def execute(self, price_data: pd.DataFrame, dividend_data: pd.DataFrame, 
                params: Dict) -> pd.DataFrame:
        """
        Execute momentum gated strategy
        
        Parameters:
            - base_amount: Regular DCA amount
            - momentum_lookback: Periods for momentum calculation (default: 3)
            - momentum_threshold: Skip purchase if momentum below this (default: -0.05 = -5%)
            - frequency: Investment frequency
        """
        base_amount = params.get('base_amount', 1000)
        momentum_lookback = params.get('momentum_lookback', 3)
        momentum_threshold = params.get('momentum_threshold', -0.05)
        frequency = params.get('frequency', 'monthly')
        
        # Prepare data
        freq_map = {'weekly': 'W', 'biweekly': '2W', 'monthly': 'M'}
        freq = freq_map.get(frequency, 'M')
        
        price_data = price_data.set_index('Date')
        dca_prices = price_data['Close'].resample(freq).first().dropna()
        
        if not dividend_data.empty:
            dividend_data = dividend_data.set_index('Date')
            dividends = dividend_data['Dividend'].resample(freq).sum()
        else:
            dividends = pd.Series(index=dca_prices.index, data=0)
        
        # Execute strategy
        results = []
        cumulative_units = 0
        price_history = []
        
        for i, (date, price) in enumerate(dca_prices.items()):
            price_history.append(price)
            
            # Calculate momentum
            momentum = 0
            if len(price_history) >= momentum_lookback + 1:
                old_price = price_history[-(momentum_lookback + 1)]
                momentum = (price - old_price) / old_price
            
            # Decide whether to invest
            if momentum >= momentum_threshold or len(price_history) <= momentum_lookback:
                # Invest normally
                investment = base_amount
                units_bought = investment / price
                cumulative_units += units_bought
                action = 'buy'
            else:
                # Skip investment due to negative momentum
                investment = 0
                units_bought = 0
                action = 'skip_momentum'
            
            # Get dividends
            div_received = dividends.get(date, 0) * cumulative_units
            
            results.append({
                'date': date,
                'action': action,
                'units': units_bought,
                'price': price,
                'invested': investment,
                'cumulative_units': cumulative_units,
                'dividend_received': div_received,
                'momentum': momentum,
                'momentum_triggered': action == 'skip_momentum'
            })
        
        result_df = pd.DataFrame(results)
        return self._calculate_metrics(result_df)


class AdaptiveHybridStrategy(BaseDCAStrategy):
    """Hybrid strategy combining multiple approaches"""
    
    def __init__(self):
        super().__init__(
            name="Adaptive Hybrid DCA",
            description="Combines base schedule with opportunistic extra allocation"
        )
    
    def execute(self, price_data: pd.DataFrame, dividend_data: pd.DataFrame, 
                params: Dict) -> pd.DataFrame:
        """
        Execute adaptive hybrid strategy
        
        Parameters:
            - base_amount: Regular DCA amount
            - max_total_per_period: Maximum total investment per period
            - drawdown_threshold: Trigger extra investment on drawdown
            - momentum_threshold: Reduce investment on negative momentum
            - value_averaging_weight: Weight for value averaging component (0-1)
            - frequency: Investment frequency
        """
        base_amount = params.get('base_amount', 1000)
        max_total = params.get('max_total_per_period', 3000)
        drawdown_threshold = params.get('drawdown_threshold', 0.15)
        momentum_threshold = params.get('momentum_threshold', -0.1)
        va_weight = params.get('value_averaging_weight', 0.3)
        frequency = params.get('frequency', 'monthly')
        
        # Prepare data
        freq_map = {'weekly': 'W', 'biweekly': '2W', 'monthly': 'M'}
        freq = freq_map.get(frequency, 'M')
        
        price_data = price_data.set_index('Date')
        dca_prices = price_data['Close'].resample(freq).first().dropna()
        
        if not dividend_data.empty:
            dividend_data = dividend_data.set_index('Date')
            dividends = dividend_data['Dividend'].resample(freq).sum()
        else:
            dividends = pd.Series(index=dca_prices.index, data=0)
        
        # Execute hybrid strategy
        results = []
        cumulative_units = 0
        price_history = []
        period = 0
        
        for i, (date, price) in enumerate(dca_prices.items()):
            period += 1
            price_history.append(price)
            
            # Start with base investment
            investment = base_amount
            
            # Calculate momentum
            momentum = 0
            if len(price_history) >= 4:
                momentum = (price - price_history[-4]) / price_history[-4]
            
            # Apply momentum adjustment
            if momentum < momentum_threshold:
                investment *= 0.5  # Reduce investment during bad momentum
            
            # Check for drawdown opportunity
            if len(price_history) >= 6:
                recent_high = max(price_history[-6:])
                drawdown = (recent_high - price) / recent_high
                
                if drawdown >= drawdown_threshold:
                    # Add extra investment for drawdown
                    extra = base_amount * (drawdown / drawdown_threshold)
                    investment += extra
            
            # Value averaging component
            if va_weight > 0:
                target_growth = 0.08 / 12  # 8% annual growth
                target_value = base_amount * period * (1 + target_growth) ** period
                current_value = cumulative_units * price
                va_adjustment = (target_value - current_value) * va_weight
                investment += max(0, va_adjustment)
            
            # Apply maximum constraint
            investment = min(investment, max_total)
            investment = max(0, investment)
            
            # Execute investment
            if investment > 0:
                units_bought = investment / price
                cumulative_units += units_bought
                action = 'buy_adaptive'
            else:
                units_bought = 0
                action = 'hold'
            
            # Get dividends
            div_received = dividends.get(date, 0) * cumulative_units
            
            results.append({
                'date': date,
                'action': action,
                'units': units_bought,
                'price': price,
                'invested': investment,
                'cumulative_units': cumulative_units,
                'dividend_received': div_received,
                'momentum': momentum,
                'base_component': base_amount,
                'final_investment': investment
            })
        
        result_df = pd.DataFrame(results)
        return self._calculate_metrics(result_df)


class DCAStrategyFactory:
    """Factory class to create and manage DCA strategies"""
    
    _strategies = {
        DCAStrategy.FIXED_AMOUNT: FixedAmountStrategy,
        DCAStrategy.VALUE_AVERAGING: ValueAveragingStrategy,
        DCAStrategy.DRAWDOWN_TRIGGER: DrawdownTriggerStrategy,
        DCAStrategy.MOMENTUM_GATED: MomentumGatedStrategy,
        DCAStrategy.ADAPTIVE_HYBRID: AdaptiveHybridStrategy
    }
    
    @classmethod
    def create_strategy(cls, strategy_type: DCAStrategy) -> BaseDCAStrategy:
        """Create a strategy instance"""
        if strategy_type not in cls._strategies:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
        
        return cls._strategies[strategy_type]()
    
    @classmethod
    def get_available_strategies(cls) -> List[Tuple[DCAStrategy, str]]:
        """Get list of available strategies with descriptions"""
        strategies = []
        for strategy_type in cls._strategies:
            instance = cls.create_strategy(strategy_type)
            strategies.append((strategy_type, instance.name, instance.description))
        
        return strategies
    
    @classmethod
    def get_default_parameters(cls, strategy_type: DCAStrategy) -> Dict:
        """Get default parameters for a strategy"""
        defaults = {
            DCAStrategy.FIXED_AMOUNT: {
                'amount': 1000,
                'frequency': 'monthly'
            },
            DCAStrategy.VALUE_AVERAGING: {
                'target_growth_rate': 0.08,
                'initial_amount': 1000,
                'max_investment_per_period': 5000,
                'frequency': 'monthly'
            },
            DCAStrategy.DRAWDOWN_TRIGGER: {
                'base_amount': 1000,
                'drawdown_threshold': 0.1,
                'extra_multiplier': 2.0,
                'lookback_periods': 4,
                'frequency': 'monthly'
            },
            DCAStrategy.MOMENTUM_GATED: {
                'base_amount': 1000,
                'momentum_lookback': 3,
                'momentum_threshold': -0.05,
                'frequency': 'monthly'
            },
            DCAStrategy.ADAPTIVE_HYBRID: {
                'base_amount': 1000,
                'max_total_per_period': 3000,
                'drawdown_threshold': 0.15,
                'momentum_threshold': -0.1,
                'value_averaging_weight': 0.3,
                'frequency': 'monthly'
            }
        }
        
        return defaults.get(strategy_type, {})