"""
DCA Optimizer Module
Implements parameter optimization and grid search for DCA strategies
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Iterator
from itertools import product
import random
from dca_strategies import DCAStrategy, DCAStrategyFactory
from dca_metrics import DCAMetrics


class DCAOptimizer:
    """Optimize DCA strategy parameters using grid search and random search"""
    
    def __init__(self):
        self.metrics_calculator = DCAMetrics()
        self.strategy_factory = DCAStrategyFactory()
    
    def generate_parameter_grid(self, strategy_type: DCAStrategy, 
                               custom_ranges: Optional[Dict] = None) -> List[Dict]:
        """
        Generate parameter grid for optimization
        
        Args:
            strategy_type: Type of DCA strategy
            custom_ranges: Custom parameter ranges (overrides defaults)
            
        Returns:
            List of parameter combinations
        """
        # Default parameter ranges for each strategy
        default_ranges = {
            DCAStrategy.FIXED_AMOUNT: {
                'amount': [500, 1000, 1500, 2000, 3000],
                'frequency': ['monthly', 'biweekly']
            },
            DCAStrategy.VALUE_AVERAGING: {
                'target_growth_rate': [0.06, 0.08, 0.10, 0.12],
                'initial_amount': [1000, 1500, 2000],
                'max_investment_per_period': [3000, 5000, 8000],
                'frequency': ['monthly']
            },
            DCAStrategy.DRAWDOWN_TRIGGER: {
                'base_amount': [500, 1000, 1500],
                'drawdown_threshold': [0.05, 0.10, 0.15, 0.20],
                'extra_multiplier': [1.5, 2.0, 3.0],
                'lookback_periods': [3, 4, 6],
                'frequency': ['monthly']
            },
            DCAStrategy.MOMENTUM_GATED: {
                'base_amount': [500, 1000, 1500, 2000],
                'momentum_lookback': [2, 3, 4, 6],
                'momentum_threshold': [-0.03, -0.05, -0.08, -0.10],
                'frequency': ['monthly']
            },
            DCAStrategy.ADAPTIVE_HYBRID: {
                'base_amount': [500, 1000, 1500],
                'max_total_per_period': [2000, 3000, 5000],
                'drawdown_threshold': [0.10, 0.15, 0.20],
                'momentum_threshold': [-0.05, -0.10, -0.15],
                'value_averaging_weight': [0.0, 0.2, 0.3, 0.5],
                'frequency': ['monthly']
            }
        }
        
        # Use custom ranges if provided
        ranges = custom_ranges if custom_ranges else default_ranges.get(strategy_type, {})
        
        if not ranges:
            # Return default parameters if no ranges defined
            return [self.strategy_factory.get_default_parameters(strategy_type)]
        
        # Generate all combinations
        param_names = list(ranges.keys())
        param_values = list(ranges.values())
        
        combinations = []
        for combo in product(*param_values):
            param_dict = dict(zip(param_names, combo))
            combinations.append(param_dict)
        
        return combinations
    
    def random_parameter_search(self, strategy_type: DCAStrategy, 
                               num_samples: int = 50,
                               custom_ranges: Optional[Dict] = None) -> List[Dict]:
        """
        Generate random parameter combinations for optimization
        
        Args:
            strategy_type: Type of DCA strategy
            num_samples: Number of random samples to generate
            custom_ranges: Custom parameter ranges
            
        Returns:
            List of random parameter combinations
        """
        # Define parameter ranges for random sampling
        random_ranges = {
            DCAStrategy.FIXED_AMOUNT: {
                'amount': (200, 5000, 'uniform'),
                'frequency': (['weekly', 'biweekly', 'monthly'], None, 'choice')
            },
            DCAStrategy.VALUE_AVERAGING: {
                'target_growth_rate': (0.04, 0.15, 'uniform'),
                'initial_amount': (500, 3000, 'uniform'),
                'max_investment_per_period': (2000, 10000, 'uniform'),
                'frequency': (['monthly'], None, 'choice')
            },
            DCAStrategy.DRAWDOWN_TRIGGER: {
                'base_amount': (300, 2000, 'uniform'),
                'drawdown_threshold': (0.03, 0.25, 'uniform'),
                'extra_multiplier': (1.2, 4.0, 'uniform'),
                'lookback_periods': (2, 8, 'randint'),
                'frequency': (['monthly'], None, 'choice')
            },
            DCAStrategy.MOMENTUM_GATED: {
                'base_amount': (300, 3000, 'uniform'),
                'momentum_lookback': (2, 8, 'randint'),
                'momentum_threshold': (-0.15, -0.01, 'uniform'),
                'frequency': (['monthly'], None, 'choice')
            },
            DCAStrategy.ADAPTIVE_HYBRID: {
                'base_amount': (300, 2000, 'uniform'),
                'max_total_per_period': (1500, 8000, 'uniform'),
                'drawdown_threshold': (0.05, 0.30, 'uniform'),
                'momentum_threshold': (-0.20, -0.02, 'uniform'),
                'value_averaging_weight': (0.0, 0.8, 'uniform'),
                'frequency': (['monthly'], None, 'choice')
            }
        }
        
        # Use custom ranges if provided
        ranges = custom_ranges if custom_ranges else random_ranges.get(strategy_type, {})
        
        if not ranges:
            return [self.strategy_factory.get_default_parameters(strategy_type)] * num_samples
        
        combinations = []
        for _ in range(num_samples):
            param_dict = {}
            for param_name, (range_def, _, sample_type) in ranges.items():
                if sample_type == 'uniform':
                    min_val, max_val = range_def
                    param_dict[param_name] = random.uniform(min_val, max_val)
                elif sample_type == 'randint':
                    min_val, max_val = range_def
                    param_dict[param_name] = random.randint(min_val, max_val)
                elif sample_type == 'choice':
                    choices = range_def
                    param_dict[param_name] = random.choice(choices)
                    
            combinations.append(param_dict)
        
        return combinations
    
    def optimize_single_ticker(self, ticker: str, price_data: pd.DataFrame, 
                              dividend_data: pd.DataFrame, strategy_type: DCAStrategy,
                              optimization_type: str = 'grid',
                              num_random_samples: int = 50,
                              custom_ranges: Optional[Dict] = None,
                              ranking_criteria: str = 'total_return') -> Dict:
        """
        Optimize parameters for a single ticker and strategy
        
        Args:
            ticker: Stock ticker symbol
            price_data: Historical price data
            dividend_data: Dividend data
            strategy_type: DCA strategy to optimize
            optimization_type: 'grid' or 'random'
            num_random_samples: Number of samples for random search
            custom_ranges: Custom parameter ranges
            ranking_criteria: Criteria for ranking results
            
        Returns:
            Dict with optimization results
        """
        try:
            # Generate parameter combinations
            if optimization_type == 'grid':
                param_combinations = self.generate_parameter_grid(strategy_type, custom_ranges)
            else:
                param_combinations = self.random_parameter_search(
                    strategy_type, num_random_samples, custom_ranges
                )
            
            if not param_combinations:
                return {'error': 'No parameter combinations generated'}
            
            # Create strategy instance
            strategy = self.strategy_factory.create_strategy(strategy_type)
            
            # Test all parameter combinations
            results = []
            errors = 0
            
            for i, params in enumerate(param_combinations):
                try:
                    # Execute strategy with these parameters
                    result_df = strategy.execute(price_data, dividend_data, params)
                    
                    if result_df.empty:
                        errors += 1
                        continue
                    
                    # Calculate metrics
                    metrics = self.metrics_calculator.calculate_comprehensive_metrics(
                        result_df, ticker, strategy.name
                    )
                    
                    # Add parameter info
                    metrics['parameters'] = params
                    metrics['parameter_set_id'] = i + 1
                    
                    results.append(metrics)
                    
                except Exception as e:
                    errors += 1
                    print(f"Error with parameter set {i+1}: {e}")
            
            if not results:
                return {'error': f'All {len(param_combinations)} parameter combinations failed'}
            
            # Rank results
            ranked_results = self.metrics_calculator.rank_strategies(results, ranking_criteria)
            
            # Prepare summary
            best_result = ranked_results[0] if ranked_results else None
            
            summary = {
                'ticker': ticker,
                'strategy': strategy.name,
                'optimization_type': optimization_type,
                'total_combinations_tested': len(param_combinations),
                'successful_combinations': len(results),
                'failed_combinations': errors,
                'ranking_criteria': ranking_criteria,
                'best_parameters': best_result['parameters'] if best_result else None,
                'best_metrics': {k: v for k, v in best_result.items() if k != 'parameters'} if best_result else None,
                'all_results': ranked_results,
                'optimization_completed_at': datetime.now()
            }
            
            return summary
            
        except Exception as e:
            return {'error': f'Optimization failed: {str(e)}'}
    
    def optimize_multiple_tickers(self, ticker_data_dict: Dict[str, Dict], 
                                 strategy_configs: List[Dict],
                                 ranking_criteria: str = 'total_return') -> Dict:
        """
        Optimize multiple tickers with multiple strategies
        
        Args:
            ticker_data_dict: Dict mapping ticker -> {price_data, dividend_data}
            strategy_configs: List of strategy configurations
            ranking_criteria: Ranking criteria
            
        Returns:
            Dict with comprehensive optimization results
        """
        all_results = {}
        summary_by_ticker = {}
        
        for ticker, data in ticker_data_dict.items():
            logger.info(f"Optimizing {ticker}...")
            
            ticker_results = {}
            best_overall = None
            best_score = -float('inf')
            
            for config in strategy_configs:
                strategy_type = config['strategy']
                optimization_params = config.get('optimization', {})
                
                # Run optimization for this strategy
                result = self.optimize_single_ticker(
                    ticker=ticker,
                    price_data=data['price_data'],
                    dividend_data=data['dividend_data'],
                    strategy_type=strategy_type,
                    optimization_type=optimization_params.get('type', 'grid'),
                    num_random_samples=optimization_params.get('num_samples', 50),
                    custom_ranges=optimization_params.get('custom_ranges'),
                    ranking_criteria=ranking_criteria
                )
                
                if 'error' not in result:
                    ticker_results[strategy_type.value] = result
                    
                    # Track best overall for this ticker
                    if result['best_metrics']:
                        score = result['best_metrics'].get('ranking_score', 0)
                        if score > best_score:
                            best_score = score
                            best_overall = {
                                'strategy': strategy_type.value,
                                'parameters': result['best_parameters'],
                                'metrics': result['best_metrics']
                            }
                else:
                    print(f"Failed to optimize {ticker} with {strategy_type.value}: {result.get('error')}")
            
            all_results[ticker] = ticker_results
            summary_by_ticker[ticker] = best_overall
        
        # Create overall summary
        overall_summary = {
            'optimization_completed_at': datetime.now(),
            'tickers_optimized': list(ticker_data_dict.keys()),
            'strategies_tested': [config['strategy'].value for config in strategy_configs],
            'ranking_criteria': ranking_criteria,
            'summary_by_ticker': summary_by_ticker,
            'detailed_results': all_results
        }
        
        return overall_summary
    
    def bayesian_like_optimization(self, ticker: str, price_data: pd.DataFrame, 
                                  dividend_data: pd.DataFrame, strategy_type: DCAStrategy,
                                  num_iterations: int = 20,
                                  initial_samples: int = 10,
                                  ranking_criteria: str = 'total_return') -> Dict:
        """
        Simple Bayesian-like optimization using iterative narrowing
        
        Args:
            ticker: Stock ticker
            price_data: Price data
            dividend_data: Dividend data
            strategy_type: Strategy type
            num_iterations: Number of optimization iterations
            initial_samples: Initial random samples
            ranking_criteria: Ranking criteria
            
        Returns:
            Optimization results with iterative improvements
        """
        try:
            # Start with random search
            print(f"Starting Bayesian-like optimization for {ticker} with {strategy_type.value}")
            
            # Initial random exploration
            initial_results = self.optimize_single_ticker(
                ticker, price_data, dividend_data, strategy_type,
                optimization_type='random',
                num_random_samples=initial_samples,
                ranking_criteria=ranking_criteria
            )
            
            if 'error' in initial_results:
                return initial_results
            
            all_results = initial_results['all_results']
            iteration_history = [{'iteration': 0, 'best_score': all_results[0]['ranking_score'], 'num_results': len(all_results)}]
            
            # Iterative narrowing
            for iteration in range(1, num_iterations + 1):
                # Take top 20% of current results
                top_results = all_results[:max(1, len(all_results) // 5)]
                
                # Analyze parameter ranges from top performers
                narrowed_ranges = self._narrow_parameter_ranges(strategy_type, top_results)
                
                # Generate new samples around promising areas
                new_samples = initial_samples // 2
                iteration_results = self.optimize_single_ticker(
                    ticker, price_data, dividend_data, strategy_type,
                    optimization_type='random',
                    num_random_samples=new_samples,
                    custom_ranges=narrowed_ranges,
                    ranking_criteria=ranking_criteria
                )
                
                if 'error' not in iteration_results:
                    # Merge results and re-rank
                    all_results.extend(iteration_results['all_results'])
                    all_results = self.metrics_calculator.rank_strategies(all_results, ranking_criteria)
                    
                    # Track progress
                    best_score = all_results[0]['ranking_score']
                    iteration_history.append({
                        'iteration': iteration,
                        'best_score': best_score,
                        'num_results': len(all_results),
                        'improvement': best_score - iteration_history[-1]['best_score']
                    })
                    
                    print(f"Iteration {iteration}: Best score = {best_score:.4f}")
                    
                    # Early stopping if no improvement
                    if iteration >= 3:
                        recent_improvements = [h['improvement'] for h in iteration_history[-3:]]
                        if all(imp <= 0.001 for imp in recent_improvements):
                            print(f"Early stopping at iteration {iteration} - no significant improvement")
                            break
            
            # Final results
            best_result = all_results[0]
            
            return {
                'ticker': ticker,
                'strategy': strategy_type.value,
                'optimization_type': 'bayesian_like',
                'total_iterations': iteration,
                'total_evaluations': len(all_results),
                'iteration_history': iteration_history,
                'best_parameters': best_result['parameters'],
                'best_metrics': {k: v for k, v in best_result.items() if k != 'parameters'},
                'top_10_results': all_results[:10],
                'optimization_completed_at': datetime.now()
            }
            
        except Exception as e:
            return {'error': f'Bayesian optimization failed: {str(e)}'}
    
    def _narrow_parameter_ranges(self, strategy_type: DCAStrategy, 
                                top_results: List[Dict]) -> Dict:
        """Narrow parameter ranges based on top-performing results"""
        if not top_results:
            return {}
        
        narrowed_ranges = {}
        
        # Extract parameters from top results
        all_params = [result['parameters'] for result in top_results]
        
        for param_name in all_params[0].keys():
            param_values = [params[param_name] for params in all_params]
            
            if isinstance(param_values[0], (int, float)):
                # Numeric parameter - create range around top values
                min_val = min(param_values)
                max_val = max(param_values)
                
                # Expand range by 20% on each side
                range_size = max_val - min_val
                expansion = range_size * 0.2
                
                narrowed_ranges[param_name] = (
                    max(0, min_val - expansion),
                    max_val + expansion,
                    'uniform'
                )
            else:
                # Categorical parameter - keep unique values
                unique_values = list(set(param_values))
                narrowed_ranges[param_name] = (unique_values, None, 'choice')
        
        return narrowed_ranges
    
    def compare_strategies_for_ticker(self, ticker: str, price_data: pd.DataFrame, 
                                    dividend_data: pd.DataFrame,
                                    strategies_to_test: Optional[List[DCAStrategy]] = None) -> Dict:
        """
        Compare all strategies for a single ticker with default parameters
        
        Args:
            ticker: Stock ticker
            price_data: Price data  
            dividend_data: Dividend data
            strategies_to_test: List of strategies to test (default: all)
            
        Returns:
            Strategy comparison results
        """
        if strategies_to_test is None:
            strategies_to_test = list(DCAStrategy)
        
        results = []
        
        for strategy_type in strategies_to_test:
            try:
                strategy = self.strategy_factory.create_strategy(strategy_type)
                default_params = self.strategy_factory.get_default_parameters(strategy_type)
                
                # Execute strategy
                result_df = strategy.execute(price_data, dividend_data, default_params)
                
                if not result_df.empty:
                    # Calculate metrics
                    metrics = self.metrics_calculator.calculate_comprehensive_metrics(
                        result_df, ticker, strategy.name
                    )
                    metrics['parameters'] = default_params
                    metrics['strategy_type'] = strategy_type.value
                    results.append(metrics)
                    
            except Exception as e:
                print(f"Error testing {strategy_type.value} for {ticker}: {e}")
        
        if results:
            # Rank strategies
            ranked_results = self.metrics_calculator.rank_strategies(results, 'total_return')
            
            return {
                'ticker': ticker,
                'strategies_compared': len(results),
                'best_strategy': ranked_results[0]['strategy_type'] if ranked_results else None,
                'all_results': ranked_results,
                'comparison_completed_at': datetime.now()
            }
        else:
            return {'error': f'All strategy tests failed for {ticker}'}