"""
Simple test file for Bollinger Bands strategy
This can be run manually: python test_bb_strategy.py
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from streamlit_app import calculate_bollinger_bands, bollinger_bands_strategy


class TestBollingerBandsStrategy(unittest.TestCase):
    
    def setUp(self):
        """Set up test data"""
        # Create sample data for testing
        np.random.seed(42)
        dates = pd.date_range(start=datetime(2023, 1, 1), periods=100, freq='D')
        prices = [100 + np.random.normal(0, 2) * i * 0.1 for i in range(100)]
        
        self.sample_data = pd.DataFrame({
            'Open': prices,
            'High': [p * 1.02 for p in prices],
            'Low': [p * 0.98 for p in prices],
            'Close': prices,
            'Volume': [1000000] * 100
        }, index=dates)
    
    def test_bollinger_bands_calculation(self):
        """Test Bollinger Bands calculation"""
        prices = self.sample_data['Close']
        upper, middle, lower = calculate_bollinger_bands(prices, period=20, multiplier=2)
        
        # Check that we get the right number of values
        self.assertEqual(len(upper), len(prices))
        self.assertEqual(len(middle), len(prices))
        self.assertEqual(len(lower), len(prices))
        
        # Check that upper > middle > lower (where not NaN)
        valid_idx = ~np.isnan(upper)
        self.assertTrue(all(upper[valid_idx] >= middle[valid_idx]))
        self.assertTrue(all(middle[valid_idx] >= lower[valid_idx]))
    
    def test_insufficient_data_error(self):
        """Test error handling with insufficient data"""
        small_data = self.sample_data.head(10)
        
        # Should raise ValueError for insufficient data
        with self.assertRaises(ValueError):
            calculate_bollinger_bands(small_data['Close'], period=20)
    
    def test_strategy_execution(self):
        """Test strategy execution"""
        result = bollinger_bands_strategy(
            hist_data=self.sample_data,
            period=20,
            multiplier=2,
            initial_capital=10000
        )
        
        # Should not have error
        self.assertNotIn('error', result)
        
        # Should have required fields
        required_fields = [
            'initial_capital', 'final_value', 'total_return',
            'total_trades', 'win_rate', 'max_drawdown'
        ]
        for field in required_fields:
            self.assertIn(field, result)
        
        # Values should be reasonable
        self.assertGreaterEqual(result['final_value'], 0)
        self.assertGreaterEqual(result['total_trades'], 0)
        self.assertGreaterEqual(result['win_rate'], 0)
        self.assertLessEqual(result['win_rate'], 100)
    
    def test_parameter_validation(self):
        """Test parameter validation"""
        # Test with invalid stop loss
        result = bollinger_bands_strategy(
            self.sample_data,
            period=20,
            multiplier=2,
            stop_loss_pct=-5  # Invalid negative value
        )
        
        # Should still work (parameters are handled by UI validation)
        self.assertIn('success', result)
    
    def test_empty_data_handling(self):
        """Test handling of empty data"""
        empty_data = pd.DataFrame()
        result = bollinger_bands_strategy(empty_data)
        
        # Should return error
        self.assertIn('error', result)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)