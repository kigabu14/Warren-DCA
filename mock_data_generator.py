"""
Test data generator for demonstration purposes.
Generates mock stock data when external APIs are not accessible.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any
import random


class MockDataGenerator:
    """Generate mock stock data for testing and demonstration."""
    
    def __init__(self):
        """Initialize mock data generator."""
        self.stock_info = {
            'AAPL': {
                'longName': 'Apple Inc.',
                'sector': 'Technology',
                'industry': 'Consumer Electronics',
                'currentPrice': 175.50,
                'marketCap': 2800000000000,
                'trailingPE': 25.5,
                'trailingEps': 6.87,
                'returnOnEquity': 0.1234,
                'dividendYield': 0.0165,
                'dividendRate': 0.92,
                'bookValue': 4.25,
                'priceToBook': 41.3
            },
            'MSFT': {
                'longName': 'Microsoft Corporation',
                'sector': 'Technology',
                'industry': 'Software',
                'currentPrice': 385.75,
                'marketCap': 2850000000000,
                'trailingPE': 28.2,
                'trailingEps': 13.68,
                'returnOnEquity': 0.2145,
                'dividendYield': 0.0089,
                'dividendRate': 3.44,
                'bookValue': 19.12,
                'priceToBook': 20.2
            },
            'TSLA': {
                'longName': 'Tesla, Inc.',
                'sector': 'Consumer Cyclical',
                'industry': 'Auto Manufacturers',
                'currentPrice': 245.20,
                'marketCap': 780000000000,
                'trailingPE': 65.8,
                'trailingEps': 3.73,
                'returnOnEquity': 0.1789,
                'dividendYield': None,
                'dividendRate': None,
                'bookValue': 28.84,
                'priceToBook': 8.5
            }
        }
    
    def generate_historical_data(self, ticker: str, months: int) -> pd.DataFrame:
        """Generate mock historical price data."""
        # Start price based on current price with some variation
        base_price = self.stock_info.get(ticker, {}).get('currentPrice', 100.0)
        start_price = base_price * random.uniform(0.8, 1.2)
        
        # Generate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=months * 30)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Generate realistic price movements using random walk
        prices = []
        current_price = start_price
        
        for i in range(len(dates)):
            # Random daily change between -5% and +5%
            daily_change = random.uniform(-0.05, 0.05)
            current_price *= (1 + daily_change)
            
            # Add some volatility
            volatility = random.uniform(0.995, 1.005)
            current_price *= volatility
            
            prices.append(current_price)
        
        # Generate OHLCV data
        data = []
        for i, (date, close) in enumerate(zip(dates, prices)):
            high = close * random.uniform(1.0, 1.03)
            low = close * random.uniform(0.97, 1.0)
            open_price = prices[i-1] * random.uniform(0.99, 1.01) if i > 0 else close
            volume = random.randint(50000000, 200000000)
            
            data.append({
                'Open': open_price,
                'High': high,
                'Low': low,
                'Close': close,
                'Adj Close': close * random.uniform(0.998, 1.002),
                'Volume': volume,
                'Dividends': 0.23 if date.month % 3 == 0 and date.day < 5 and ticker in ['AAPL', 'MSFT'] else 0,
                'Stock Splits': 0
            })
        
        df = pd.DataFrame(data, index=dates)
        return df
    
    def generate_stock_info(self, ticker: str) -> Dict[str, Any]:
        """Generate mock stock info."""
        info = self.stock_info.get(ticker, {
            'longName': f'{ticker} Corporation',
            'sector': 'Technology',
            'industry': 'Software',
            'currentPrice': random.uniform(50, 500),
            'marketCap': random.randint(100000000000, 3000000000000),
            'trailingPE': random.uniform(15, 50),
            'trailingEps': random.uniform(2, 15),
            'returnOnEquity': random.uniform(0.05, 0.25),
            'dividendYield': random.uniform(0.01, 0.05),
            'dividendRate': random.uniform(0.5, 5.0),
            'bookValue': random.uniform(5, 50),
            'priceToBook': random.uniform(5, 45)
        })
        
        # Add current market data
        current_price = info['currentPrice']
        info.update({
            'regularMarketPrice': current_price,
            'previousClose': current_price * random.uniform(0.98, 1.02),
            'regularMarketOpen': current_price * random.uniform(0.99, 1.01),
            'dayHigh': current_price * random.uniform(1.01, 1.05),
            'dayLow': current_price * random.uniform(0.95, 0.99),
            'volume': random.randint(10000000, 100000000)
        })
        
        return info
    
    def generate_financials(self, ticker: str) -> pd.DataFrame:
        """Generate mock financial statements."""
        dates = [
            datetime.now() - timedelta(days=365*i) for i in range(4)
        ]
        dates.reverse()
        
        # Generate mock financial data
        base_revenue = random.randint(100000000000, 500000000000)
        financial_data = {}
        
        for i, date in enumerate(dates):
            revenue = base_revenue * (1.1 ** i) * random.uniform(0.9, 1.1)
            net_income = revenue * random.uniform(0.15, 0.25)
            
            financial_data[date] = {
                'Total Revenue': revenue,
                'Net Income': net_income,
                'Operating Income': revenue * random.uniform(0.2, 0.3),
                'Gross Profit': revenue * random.uniform(0.35, 0.45)
            }
        
        df = pd.DataFrame(financial_data)
        return df
    
    def generate_balance_sheet(self, ticker: str) -> pd.DataFrame:
        """Generate mock balance sheet."""
        dates = [
            datetime.now() - timedelta(days=365*i) for i in range(4)
        ]
        dates.reverse()
        
        base_assets = random.randint(200000000000, 800000000000)
        balance_data = {}
        
        for i, date in enumerate(dates):
            total_assets = base_assets * (1.08 ** i) * random.uniform(0.9, 1.1)
            total_liabilities = total_assets * random.uniform(0.3, 0.6)
            equity = total_assets - total_liabilities
            
            balance_data[date] = {
                'Total Assets': total_assets,
                'Total Liabilities Net Minority Interest': total_liabilities,
                'Stockholders Equity': equity,
                'Cash And Cash Equivalents': total_assets * random.uniform(0.05, 0.15),
                'Inventory': total_assets * random.uniform(0.02, 0.08)
            }
        
        df = pd.DataFrame(balance_data)
        return df
    
    def generate_cashflow(self, ticker: str) -> pd.DataFrame:
        """Generate mock cash flow statement."""
        dates = [
            datetime.now() - timedelta(days=365*i) for i in range(4)
        ]
        dates.reverse()
        
        base_cash_flow = random.randint(50000000000, 200000000000)
        cashflow_data = {}
        
        for i, date in enumerate(dates):
            operating_cf = base_cash_flow * (1.07 ** i) * random.uniform(0.9, 1.1)
            free_cf = operating_cf * random.uniform(0.7, 0.9)
            
            cashflow_data[date] = {
                'Operating Cash Flow': operating_cf,
                'Free Cash Flow': free_cf,
                'Capital Expenditure': operating_cf * random.uniform(-0.3, -0.1)
            }
        
        df = pd.DataFrame(cashflow_data)
        return df