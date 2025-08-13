"""
Enhanced data loader with database integration.
Extends the existing DCADataLoader to support database storage.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import yfinance as yf
import logging

from dca_data_loader import DCADataLoader
from database import StockDatabase


class EnhancedDCADataLoader(DCADataLoader):
    """Enhanced data loader with database integration for stock data storage."""
    
    def __init__(self, cache_dir: str = "data/cache", db_path: str = "data/stocks.db"):
        """Initialize enhanced data loader with database support."""
        super().__init__(cache_dir)
        self.db = StockDatabase(db_path)
    
    def fetch_and_store_ticker_data(self, ticker: str, period: str = "1y", store_in_db: bool = True) -> Dict[str, Any]:
        """Fetch ticker data and optionally store in database."""
        # Fetch data using parent method
        data = self.fetch_ticker_data(ticker, period)
        
        if store_in_db and data:
            try:
                # Get ticker info for basic stock information
                ticker_obj = yf.Ticker(ticker)
                info = ticker_obj.info
                
                # Add stock to database
                self.db.add_stock(
                    ticker=ticker,
                    company_name=info.get('longName', info.get('shortName')),
                    sector=info.get('sector'),
                    industry=info.get('industry')
                )
                
                # Store historical data
                if 'historical_prices' in data and not data['historical_prices'].empty:
                    self.db.save_historical_data(ticker, data['historical_prices'])
                
                # Store current stock data with enhanced fields
                current_data = self._extract_enhanced_stock_data(ticker_obj, info, data)
                if current_data:
                    self.db.save_stock_data(ticker, current_data)
                
                # Store financial statements if available
                self._store_financial_statements(ticker, ticker_obj)
                
                logging.info(f"Successfully stored data for {ticker} in database")
                
            except Exception as e:
                logging.error(f"Error storing data for {ticker}: {e}")
        
        return data
    
    def _extract_enhanced_stock_data(self, ticker_obj: yf.Ticker, info: Dict, data: Dict) -> Dict[str, Any]:
        """Extract enhanced stock data including new required fields."""
        try:
            hist = data.get('historical_prices')
            current_data = {
                'date': datetime.now().strftime('%Y-%m-%d'),
                'current_price': info.get('currentPrice') or info.get('regularMarketPrice'),
                'open_price': info.get('regularMarketOpen') or info.get('open'),
                'close_price': info.get('previousClose'),
                'high_price': info.get('dayHigh') or info.get('regularMarketDayHigh'),
                'low_price': info.get('dayLow') or info.get('regularMarketDayLow'),
                'volume': info.get('volume') or info.get('regularMarketVolume'),
                'market_cap': info.get('marketCap'),
                'pe_ratio': info.get('trailingPE') or info.get('forwardPE'),
                'eps': info.get('trailingEps') or info.get('forwardEps'),
                'roe': info.get('returnOnEquity'),
                'dividend_yield': info.get('dividendYield'),
                'ex_dividend_date': info.get('exDividendDate'),
                'dividend_amount': info.get('dividendRate'),
                'book_value': info.get('bookValue'),
                'price_to_book': info.get('priceToBook')
            }
            
            # Get latest prices from historical data if current prices not available
            if not current_data['current_price'] and hist is not None and not hist.empty:
                latest = hist.iloc[-1]
                current_data.update({
                    'current_price': latest.get('Close'),
                    'open_price': latest.get('Open'),
                    'high_price': latest.get('High'),
                    'low_price': latest.get('Low'),
                    'volume': latest.get('Volume')
                })
            
            return current_data
            
        except Exception as e:
            logging.error(f"Error extracting enhanced stock data: {e}")
            return {}
    
    def _store_financial_statements(self, ticker: str, ticker_obj: yf.Ticker):
        """Store financial statements in database."""
        try:
            # Get financial statements
            financials = ticker_obj.financials
            balance_sheet = ticker_obj.balance_sheet
            cash_flow = ticker_obj.cashflow
            
            # Store income statements
            if not financials.empty:
                for period in financials.columns:
                    period_str = period.strftime('%Y-%m-%d') if hasattr(period, 'strftime') else str(period)
                    statement_data = {
                        'total_revenue': financials.loc[financials.index.str.contains('Total Revenue', case=False, na=False)].iloc[0, financials.columns.get_loc(period)] if not financials.loc[financials.index.str.contains('Total Revenue', case=False, na=False)].empty else None,
                        'net_income': financials.loc[financials.index.str.contains('Net Income', case=False, na=False)].iloc[0, financials.columns.get_loc(period)] if not financials.loc[financials.index.str.contains('Net Income', case=False, na=False)].empty else None,
                        'statement_data': financials[period].to_dict()
                    }
                    self.db.save_financial_statement(ticker, period_str, 'income', statement_data)
            
            # Store balance sheets
            if not balance_sheet.empty:
                for period in balance_sheet.columns:
                    period_str = period.strftime('%Y-%m-%d') if hasattr(period, 'strftime') else str(period)
                    statement_data = {
                        'total_assets': balance_sheet.loc[balance_sheet.index.str.contains('Total Assets', case=False, na=False)].iloc[0, balance_sheet.columns.get_loc(period)] if not balance_sheet.loc[balance_sheet.index.str.contains('Total Assets', case=False, na=False)].empty else None,
                        'total_liabilities': balance_sheet.loc[balance_sheet.index.str.contains('Total Liab', case=False, na=False)].iloc[0, balance_sheet.columns.get_loc(period)] if not balance_sheet.loc[balance_sheet.index.str.contains('Total Liab', case=False, na=False)].empty else None,
                        'shareholders_equity': balance_sheet.loc[balance_sheet.index.str.contains('Stockholders Equity', case=False, na=False)].iloc[0, balance_sheet.columns.get_loc(period)] if not balance_sheet.loc[balance_sheet.index.str.contains('Stockholders Equity', case=False, na=False)].empty else None,
                        'statement_data': balance_sheet[period].to_dict()
                    }
                    self.db.save_financial_statement(ticker, period_str, 'balance_sheet', statement_data)
            
            # Store cash flow statements
            if not cash_flow.empty:
                for period in cash_flow.columns:
                    period_str = period.strftime('%Y-%m-%d') if hasattr(period, 'strftime') else str(period)
                    statement_data = {
                        'operating_cash_flow': cash_flow.loc[cash_flow.index.str.contains('Operating Cash Flow', case=False, na=False)].iloc[0, cash_flow.columns.get_loc(period)] if not cash_flow.loc[cash_flow.index.str.contains('Operating Cash Flow', case=False, na=False)].empty else None,
                        'free_cash_flow': cash_flow.loc[cash_flow.index.str.contains('Free Cash Flow', case=False, na=False)].iloc[0, cash_flow.columns.get_loc(period)] if not cash_flow.loc[cash_flow.index.str.contains('Free Cash Flow', case=False, na=False)].empty else None,
                        'statement_data': cash_flow[period].to_dict()
                    }
                    self.db.save_financial_statement(ticker, period_str, 'cash_flow', statement_data)
                    
        except Exception as e:
            logging.error(f"Error storing financial statements for {ticker}: {e}")
    
    def fetch_historical_data_for_period(self, ticker: str, period_months: int) -> Dict[str, Any]:
        """Fetch historical data for specific time period and store in database."""
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=period_months * 30)
            
            # Convert to yfinance period format
            if period_months == 1:
                yf_period = "1mo"
            elif period_months == 3:
                yf_period = "3mo"
            elif period_months == 12:
                yf_period = "1y"
            else:
                yf_period = f"{period_months}mo"
            
            # Fetch and store data
            data = self.fetch_and_store_ticker_data(ticker, yf_period, store_in_db=True)
            
            return data
            
        except Exception as e:
            logging.error(f"Error fetching historical data for {ticker}: {e}")
            return {}
    
    def get_stored_stock_data(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Get stored stock data from database."""
        try:
            # Get latest stock data
            stock_data = self.db.get_stock_data(ticker, limit=1)
            if not stock_data:
                return None
            
            # Get historical data
            historical_data = self.db.get_historical_data(ticker)
            
            # Get financial statements
            financial_statements = self.db.get_financial_statements(ticker)
            
            return {
                'stock_data': stock_data[0] if stock_data else None,
                'historical_data': historical_data,
                'financial_statements': financial_statements
            }
            
        except Exception as e:
            logging.error(f"Error getting stored data for {ticker}: {e}")
            return None
    
    def get_available_periods(self, ticker: str) -> List[str]:
        """Get available time periods for stored historical data."""
        try:
            historical_data = self.db.get_historical_data(ticker)
            if historical_data is None or historical_data.empty:
                return []
            
            # Calculate available periods based on data range
            data_range = (historical_data.index.max() - historical_data.index.min()).days
            
            periods = []
            if data_range >= 30:
                periods.append("1 เดือน")
            if data_range >= 90:
                periods.append("3 เดือน")
            if data_range >= 365:
                periods.append("1 ปี")
            
            return periods
            
        except Exception as e:
            logging.error(f"Error getting available periods for {ticker}: {e}")
            return []
    
    def list_stored_stocks(self) -> List[Dict[str, Any]]:
        """List all stocks stored in database."""
        return self.db.list_stocks()
    
    def delete_stored_stock(self, ticker: str) -> bool:
        """Delete stored stock data from database."""
        return self.db.delete_stock(ticker)