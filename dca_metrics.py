"""
DCA Metrics Module
Calculates comprehensive metrics and break-even forecasts for DCA strategies
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class DCAMetrics:
    """Calculate comprehensive metrics for DCA strategy results"""
    
    def __init__(self):
        self.risk_free_rate = 0.0  # Assume 0% risk-free rate if not provided
    
    def calculate_comprehensive_metrics(self, result_df: pd.DataFrame, 
                                      ticker: str = "", strategy_name: str = "") -> Dict:
        """
        Calculate comprehensive metrics for DCA strategy results
        
        Args:
            result_df: DataFrame from DCA strategy execution
            ticker: Stock ticker symbol
            strategy_name: Name of the strategy used
            
        Returns:
            Dict with comprehensive metrics
        """
        if result_df.empty:
            return self._empty_metrics_dict(ticker, strategy_name)
        
        try:
            # Basic metrics
            total_invested = result_df['cumulative_invested'].iloc[-1] if 'cumulative_invested' in result_df.columns else result_df['invested'].sum()
            total_units = result_df['cumulative_units'].iloc[-1] if 'cumulative_units' in result_df.columns else 0
            final_price = result_df['price'].iloc[-1]
            current_value = total_units * final_price
            total_return = current_value - total_invested
            total_return_pct = (total_return / total_invested * 100) if total_invested > 0 else 0
            
            # Cost basis
            cost_basis = (total_invested / total_units) if total_units > 0 else 0
            
            # Dividend metrics
            total_dividends = result_df['cumulative_dividends'].iloc[-1] if 'cumulative_dividends' in result_df.columns else result_df['dividend_received'].sum()
            
            # Time metrics
            start_date = result_df['date'].iloc[0]
            end_date = result_df['date'].iloc[-1]
            total_days = (end_date - start_date).days
            total_years = total_days / 365.25
            
            # CAGR calculation
            if total_years > 0 and total_invested > 0:
                final_value = current_value + total_dividends
                cagr = ((final_value / total_invested) ** (1 / total_years) - 1) * 100
            else:
                cagr = 0
            
            # Volatility and risk metrics
            equity_series = result_df['equity'] if 'equity' in result_df.columns else result_df['cumulative_units'] * result_df['price']
            returns = equity_series.pct_change().dropna()
            
            volatility = returns.std() * np.sqrt(12) * 100 if len(returns) > 1 else 0  # Annualized
            
            # Sharpe ratio (using 0% risk-free rate)
            if volatility > 0:
                excess_return = (cagr / 100) - self.risk_free_rate
                sharpe_ratio = excess_return / (volatility / 100)
            else:
                sharpe_ratio = 0
            
            # Maximum drawdown
            if 'drawdown' in result_df.columns:
                max_drawdown = result_df['drawdown'].min() * 100
            else:
                peak_values = equity_series.expanding().max()
                drawdowns = (equity_series - peak_values) / peak_values
                max_drawdown = drawdowns.min() * 100
            
            # Risk-adjusted return
            risk_adjusted_return = (total_return_pct / abs(max_drawdown)) if max_drawdown != 0 else 0
            
            # Break-even analysis
            break_even_info = self._calculate_break_even(result_df, cost_basis, final_price)
            
            # Time in profit analysis
            profit_periods = (equity_series > result_df['cumulative_invested']).sum() if 'cumulative_invested' in result_df.columns else 0
            total_periods = len(result_df)
            time_in_profit_pct = (profit_periods / total_periods * 100) if total_periods > 0 else 0
            
            # Worst underwater duration
            underwater_duration = self._calculate_worst_underwater_duration(result_df)
            
            # Purchase efficiency
            purchase_count = len(result_df[result_df['invested'] > 0])
            avg_purchase_amount = total_invested / purchase_count if purchase_count > 0 else 0
            
            # Compile metrics
            metrics = {
                # Basic Info
                'ticker': ticker,
                'strategy': strategy_name,
                'start_date': start_date,
                'end_date': end_date,
                'total_days': total_days,
                'total_years': round(total_years, 2),
                
                # Investment Metrics
                'total_invested': round(total_invested, 2),
                'total_units': round(total_units, 4),
                'final_price': round(final_price, 2),
                'cost_basis': round(cost_basis, 2),
                'current_value': round(current_value, 2),
                
                # Return Metrics
                'total_return': round(total_return, 2),
                'total_return_pct': round(total_return_pct, 2),
                'cagr': round(cagr, 2),
                'total_dividends': round(total_dividends, 2),
                'total_return_including_dividends': round(total_return + total_dividends, 2),
                
                # Risk Metrics
                'volatility_pct': round(volatility, 2),
                'max_drawdown_pct': round(max_drawdown, 2),
                'sharpe_ratio': round(sharpe_ratio, 3),
                'risk_adjusted_return': round(risk_adjusted_return, 3),
                
                # Break-even
                'break_even_price': break_even_info['break_even_price'],
                'break_even_achieved': break_even_info['achieved'],
                'break_even_date': break_even_info['break_even_date'],
                'break_even_forecast': break_even_info['forecast'],
                
                # Time Analysis
                'time_in_profit_pct': round(time_in_profit_pct, 2),
                'worst_underwater_days': underwater_duration,
                
                # Purchase Analysis
                'purchase_count': purchase_count,
                'avg_purchase_amount': round(avg_purchase_amount, 2),
                
                # Score for ranking
                'ranking_score': self._calculate_ranking_score(total_return_pct, max_drawdown, sharpe_ratio, time_in_profit_pct)
            }
            
            return metrics
            
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            return self._empty_metrics_dict(ticker, strategy_name)
    
    def _empty_metrics_dict(self, ticker: str, strategy_name: str) -> Dict:
        """Return empty metrics dict for error cases"""
        return {
            'ticker': ticker,
            'strategy': strategy_name,
            'error': 'Failed to calculate metrics',
            'total_invested': 0,
            'total_return': 0,
            'total_return_pct': 0,
            'ranking_score': 0
        }
    
    def _calculate_break_even(self, result_df: pd.DataFrame, cost_basis: float, 
                            current_price: float) -> Dict:
        """Calculate break-even analysis"""
        break_even_price = cost_basis
        
        # Check if already at break-even
        if current_price >= cost_basis:
            # Find when break-even was first achieved
            if 'equity' in result_df.columns and 'cumulative_invested' in result_df.columns:
                profit_mask = result_df['equity'] >= result_df['cumulative_invested']
                if profit_mask.any():
                    break_even_date = result_df.loc[profit_mask, 'date'].iloc[0]
                    return {
                        'break_even_price': round(break_even_price, 2),
                        'achieved': True,
                        'break_even_date': break_even_date,
                        'forecast': None
                    }
            
            return {
                'break_even_price': round(break_even_price, 2),
                'achieved': True,
                'break_even_date': result_df['date'].iloc[-1],
                'forecast': None
            }
        
        # Forecast break-even date
        forecast = self._forecast_break_even_date(result_df, cost_basis, current_price)
        
        return {
            'break_even_price': round(break_even_price, 2),
            'achieved': False,
            'break_even_date': None,
            'forecast': forecast
        }
    
    def _forecast_break_even_date(self, result_df: pd.DataFrame, cost_basis: float, 
                                current_price: float) -> Dict:
        """Forecast when break-even might be achieved"""
        try:
            # Method 1: Linear regression on price trend
            prices = result_df['price'].values
            days = np.arange(len(prices))
            
            if len(prices) >= 3:
                # Fit linear trend
                slope, intercept, r_value, p_value, std_err = stats.linregress(days, prices)
                
                if slope > 0:  # Positive trend
                    # Calculate days needed to reach break-even price
                    current_day = len(prices) - 1
                    days_to_breakeven = (cost_basis - (slope * current_day + intercept)) / slope
                    
                    if days_to_breakeven > 0:
                        forecast_date = result_df['date'].iloc[-1] + timedelta(days=int(days_to_breakeven))
                        confidence = min(abs(r_value) * 100, 95)  # Cap confidence at 95%
                        
                        return {
                            'method': 'linear_regression',
                            'forecast_date': forecast_date,
                            'confidence_pct': round(confidence, 1),
                            'price_slope_daily': round(slope, 4),
                            'r_squared': round(r_value**2, 3)
                        }
            
            # Method 2: Historical return-based projection
            if len(result_df) >= 6:
                recent_periods = min(12, len(result_df) // 2)
                recent_prices = prices[-recent_periods:]
                
                if len(recent_prices) >= 2:
                    # Calculate average return over recent periods
                    recent_returns = np.diff(recent_prices) / recent_prices[:-1]
                    avg_return = np.mean(recent_returns)
                    
                    if avg_return > 0:
                        # Project based on average return
                        periods_needed = np.log(cost_basis / current_price) / np.log(1 + avg_return)
                        
                        if periods_needed > 0 and periods_needed < 1000:  # Reasonable limit
                            # Estimate time between periods
                            time_between = (result_df['date'].iloc[-1] - result_df['date'].iloc[0]) / len(result_df)
                            forecast_date = result_df['date'].iloc[-1] + timedelta(days=int(periods_needed * time_between.days))
                            
                            return {
                                'method': 'historical_returns',
                                'forecast_date': forecast_date,
                                'confidence_pct': 60,  # Lower confidence for this method
                                'avg_return_per_period': round(avg_return * 100, 2),
                                'periods_needed': round(periods_needed, 1)
                            }
            
            # Fallback: Conservative estimate based on long-term market average
            market_annual_return = 0.08  # 8% long-term average
            required_gain = (cost_basis - current_price) / current_price
            years_needed = np.log(1 + required_gain) / np.log(1 + market_annual_return)
            
            if years_needed > 0 and years_needed < 20:  # Reasonable limit
                forecast_date = result_df['date'].iloc[-1] + timedelta(days=int(years_needed * 365))
                
                return {
                    'method': 'market_average',
                    'forecast_date': forecast_date,
                    'confidence_pct': 30,  # Low confidence
                    'years_needed': round(years_needed, 1),
                    'required_gain_pct': round(required_gain * 100, 1)
                }
            
        except Exception as e:
            print(f"Error in break-even forecast: {e}")
        
        return {
            'method': 'unable_to_forecast',
            'forecast_date': None,
            'confidence_pct': 0,
            'note': 'Insufficient data or negative trend'
        }
    
    def _calculate_worst_underwater_duration(self, result_df: pd.DataFrame) -> int:
        """Calculate the worst (longest) underwater period in days"""
        if 'equity' not in result_df.columns or 'cumulative_invested' not in result_df.columns:
            return 0
        
        try:
            underwater = result_df['equity'] < result_df['cumulative_invested']
            
            if not underwater.any():
                return 0
            
            # Find consecutive underwater periods
            underwater_periods = []
            current_period = 0
            
            for is_underwater in underwater:
                if is_underwater:
                    current_period += 1
                else:
                    if current_period > 0:
                        underwater_periods.append(current_period)
                        current_period = 0
            
            # Add final period if still underwater
            if current_period > 0:
                underwater_periods.append(current_period)
            
            if underwater_periods:
                max_periods = max(underwater_periods)
                # Convert periods to approximate days
                total_days = (result_df['date'].iloc[-1] - result_df['date'].iloc[0]).days
                days_per_period = total_days / len(result_df) if len(result_df) > 1 else 30
                return int(max_periods * days_per_period)
            
        except Exception as e:
            print(f"Error calculating underwater duration: {e}")
        
        return 0
    
    def _calculate_ranking_score(self, return_pct: float, max_drawdown: float, 
                               sharpe_ratio: float, time_in_profit_pct: float) -> float:
        """Calculate a composite ranking score for strategy comparison"""
        try:
            # Normalize metrics (higher is better for score)
            return_score = max(0, return_pct) / 100  # 0-1+ scale
            drawdown_score = max(0, (20 + max_drawdown) / 20)  # Better with lower drawdown
            sharpe_score = max(0, min(2, sharpe_ratio)) / 2  # 0-1 scale, cap at 2
            profit_time_score = time_in_profit_pct / 100  # 0-1 scale
            
            # Weighted combination
            score = (return_score * 0.4 + 
                    drawdown_score * 0.3 + 
                    sharpe_score * 0.2 + 
                    profit_time_score * 0.1)
            
            return round(score, 4)
            
        except:
            return 0.0
    
    def run_monte_carlo_break_even(self, result_df: pd.DataFrame, cost_basis: float,
                                  current_price: float, num_simulations: int = 1000,
                                  forecast_years: int = 5) -> Dict:
        """
        Run Monte Carlo simulation to estimate break-even probability distribution
        
        Args:
            result_df: Historical DCA results
            cost_basis: Current cost basis
            current_price: Current stock price
            num_simulations: Number of Monte Carlo runs
            forecast_years: Years to simulate forward
            
        Returns:
            Dict with Monte Carlo results
        """
        if current_price >= cost_basis:
            return {
                'already_profitable': True,
                'break_even_probability': 1.0,
                'median_days_to_breakeven': 0,
                'percentile_75_days': 0
            }
        
        try:
            # Calculate historical daily returns
            prices = result_df['price'].values
            
            if len(prices) < 10:
                return {'error': 'Insufficient historical data for Monte Carlo'}
            
            daily_returns = np.diff(prices) / prices[:-1]
            
            # Remove extreme outliers (beyond 3 standard deviations)
            mean_return = np.mean(daily_returns)
            std_return = np.std(daily_returns)
            filtered_returns = daily_returns[np.abs(daily_returns - mean_return) <= 3 * std_return]
            
            # Monte Carlo simulation
            forecast_days = forecast_years * 365
            break_even_days = []
            
            for _ in range(num_simulations):
                # Simulate price path
                price_path = [current_price]
                
                for day in range(forecast_days):
                    # Sample random return from historical distribution
                    random_return = np.random.choice(filtered_returns)
                    new_price = price_path[-1] * (1 + random_return)
                    price_path.append(new_price)
                    
                    # Check if break-even reached
                    if new_price >= cost_basis:
                        break_even_days.append(day + 1)
                        break
            
            # Calculate statistics
            if break_even_days:
                break_even_probability = len(break_even_days) / num_simulations
                median_days = np.median(break_even_days)
                percentile_75 = np.percentile(break_even_days, 75)
                
                return {
                    'break_even_probability': round(break_even_probability, 3),
                    'median_days_to_breakeven': int(median_days),
                    'percentile_75_days': int(percentile_75),
                    'simulations_successful': len(break_even_days),
                    'total_simulations': num_simulations,
                    'median_date': result_df['date'].iloc[-1] + timedelta(days=int(median_days)),
                    'percentile_75_date': result_df['date'].iloc[-1] + timedelta(days=int(percentile_75))
                }
            else:
                return {
                    'break_even_probability': 0.0,
                    'median_days_to_breakeven': None,
                    'percentile_75_days': None,
                    'note': f'No break-even achieved in {forecast_years} year forecast'
                }
                
        except Exception as e:
            return {'error': f'Monte Carlo simulation failed: {str(e)}'}
    
    def rank_strategies(self, metrics_list: List[Dict], 
                       ranking_criteria: str = 'total_return') -> List[Dict]:
        """
        Rank strategies based on specified criteria
        
        Args:
            metrics_list: List of metrics dicts from different strategies
            ranking_criteria: 'total_return', 'cost_basis', 'sharpe_ratio', 'break_even_speed'
            
        Returns:
            Sorted list of metrics (best first)
        """
        if not metrics_list:
            return []
        
        try:
            # Filter out error entries
            valid_metrics = [m for m in metrics_list if 'error' not in m]
            
            if not valid_metrics:
                return metrics_list
            
            # Define ranking logic
            if ranking_criteria == 'total_return':
                # Highest total return percentage
                valid_metrics.sort(key=lambda x: x.get('total_return_pct', 0), reverse=True)
                
            elif ranking_criteria == 'cost_basis':
                # Lowest cost basis (better average price)
                valid_metrics.sort(key=lambda x: x.get('cost_basis', float('inf')))
                
            elif ranking_criteria == 'sharpe_ratio':
                # Highest risk-adjusted return
                valid_metrics.sort(key=lambda x: x.get('sharpe_ratio', 0), reverse=True)
                
            elif ranking_criteria == 'break_even_speed':
                # Fastest to break-even (if achieved) or best forecast
                def break_even_score(m):
                    if m.get('break_even_achieved', False):
                        # Already profitable - score by time taken
                        days = (m['end_date'] - m['start_date']).days
                        return -days  # Negative so shorter time ranks higher
                    else:
                        # Not profitable - score by forecast confidence and time
                        forecast = m.get('break_even_forecast', {})
                        if forecast and forecast.get('forecast_date'):
                            confidence = forecast.get('confidence_pct', 0)
                            return confidence / 100  # 0-1 scale
                        return -1000  # No forecast available
                
                valid_metrics.sort(key=break_even_score, reverse=True)
                
            else:
                # Default to composite ranking score
                valid_metrics.sort(key=lambda x: x.get('ranking_score', 0), reverse=True)
            
            # Add rank numbers
            for i, metric in enumerate(valid_metrics):
                metric['rank'] = i + 1
            
            return valid_metrics
            
        except Exception as e:
            print(f"Error ranking strategies: {e}")
            return metrics_list