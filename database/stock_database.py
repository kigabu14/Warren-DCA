"""
SQLite database manager for stock data storage.
Supports extended stock data including financial metrics and historical data.
"""

import sqlite3
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import json


class StockDatabase:
    """SQLite database manager for stock data storage and retrieval."""
    
    def __init__(self, db_path: str = "data/stocks.db"):
        """Initialize database connection and create tables if needed."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create stocks table for basic stock information
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS stocks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL UNIQUE,
                    company_name TEXT,
                    sector TEXT,
                    industry TEXT,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(ticker)
                )
            """)
            
            # Create stock_data table for detailed financial data
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS stock_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    date TEXT NOT NULL,
                    current_price REAL,
                    open_price REAL,
                    close_price REAL,
                    high_price REAL,
                    low_price REAL,
                    volume INTEGER,
                    market_cap REAL,
                    pe_ratio REAL,
                    eps REAL,
                    roe REAL,
                    dividend_yield REAL,
                    ex_dividend_date TEXT,
                    dividend_amount REAL,
                    book_value REAL,
                    price_to_book REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (ticker) REFERENCES stocks (ticker),
                    UNIQUE(ticker, date)
                )
            """)
            
            # Create financial_statements table for income statement and balance sheet data
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS financial_statements (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    period_ending TEXT NOT NULL,
                    statement_type TEXT NOT NULL, -- 'income', 'balance_sheet', 'cash_flow'
                    total_revenue REAL,
                    net_income REAL,
                    total_assets REAL,
                    total_liabilities REAL,
                    shareholders_equity REAL,
                    operating_cash_flow REAL,
                    free_cash_flow REAL,
                    statement_data TEXT, -- JSON of complete statement data
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (ticker) REFERENCES stocks (ticker),
                    UNIQUE(ticker, period_ending, statement_type)
                )
            """)
            
            # Create historical_data table for price history
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS historical_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    date TEXT NOT NULL,
                    open_price REAL,
                    high_price REAL,
                    low_price REAL,
                    close_price REAL,
                    adj_close REAL,
                    volume INTEGER,
                    dividends REAL DEFAULT 0,
                    splits REAL DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (ticker) REFERENCES stocks (ticker),
                    UNIQUE(ticker, date)
                )
            """)
            
            conn.commit()
    
    def add_stock(self, ticker: str, company_name: str = None, sector: str = None, industry: str = None) -> bool:
        """Add a stock to the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO stocks (ticker, company_name, sector, industry, last_updated)
                    VALUES (?, ?, ?, ?, ?)
                """, (ticker, company_name, sector, industry, datetime.now()))
                conn.commit()
                return True
        except Exception as e:
            logging.error(f"Error adding stock {ticker}: {e}")
            return False
    
    def save_stock_data(self, ticker: str, data: Dict[str, Any]) -> bool:
        """Save stock financial data to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                date_str = data.get('date', datetime.now().strftime('%Y-%m-%d'))
                
                cursor.execute("""
                    INSERT OR REPLACE INTO stock_data (
                        ticker, date, current_price, open_price, close_price, high_price, low_price,
                        volume, market_cap, pe_ratio, eps, roe, dividend_yield, ex_dividend_date,
                        dividend_amount, book_value, price_to_book
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    ticker, date_str, data.get('current_price'), data.get('open_price'),
                    data.get('close_price'), data.get('high_price'), data.get('low_price'),
                    data.get('volume'), data.get('market_cap'), data.get('pe_ratio'),
                    data.get('eps'), data.get('roe'), data.get('dividend_yield'),
                    data.get('ex_dividend_date'), data.get('dividend_amount'),
                    data.get('book_value'), data.get('price_to_book')
                ))
                conn.commit()
                return True
        except Exception as e:
            logging.error(f"Error saving stock data for {ticker}: {e}")
            return False
    
    def save_historical_data(self, ticker: str, df: pd.DataFrame) -> bool:
        """Save historical price data to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                for date, row in df.iterrows():
                    date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)
                    cursor.execute("""
                        INSERT OR REPLACE INTO historical_data (
                            ticker, date, open_price, high_price, low_price, close_price,
                            adj_close, volume, dividends, splits
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        ticker, date_str, row.get('Open'), row.get('High'),
                        row.get('Low'), row.get('Close'), row.get('Adj Close'),
                        row.get('Volume'), row.get('Dividends', 0), row.get('Stock Splits', 0)
                    ))
                
                conn.commit()
                return True
        except Exception as e:
            logging.error(f"Error saving historical data for {ticker}: {e}")
            return False
    
    def save_financial_statement(self, ticker: str, period_ending: str, statement_type: str, statement_data: Dict[str, Any]) -> bool:
        """Save financial statement data to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO financial_statements (
                        ticker, period_ending, statement_type, total_revenue, net_income,
                        total_assets, total_liabilities, shareholders_equity,
                        operating_cash_flow, free_cash_flow, statement_data
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    ticker, period_ending, statement_type,
                    statement_data.get('total_revenue'), statement_data.get('net_income'),
                    statement_data.get('total_assets'), statement_data.get('total_liabilities'),
                    statement_data.get('shareholders_equity'), statement_data.get('operating_cash_flow'),
                    statement_data.get('free_cash_flow'), json.dumps(statement_data)
                ))
                conn.commit()
                return True
        except Exception as e:
            logging.error(f"Error saving financial statement for {ticker}: {e}")
            return False
    
    def get_stock_data(self, ticker: str, limit: int = 1) -> Optional[List[Dict[str, Any]]]:
        """Get recent stock data from database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM stock_data 
                    WHERE ticker = ? 
                    ORDER BY date DESC 
                    LIMIT ?
                """, (ticker, limit))
                
                columns = [description[0] for description in cursor.description]
                results = []
                for row in cursor.fetchall():
                    results.append(dict(zip(columns, row)))
                return results
        except Exception as e:
            logging.error(f"Error getting stock data for {ticker}: {e}")
            return None
    
    def get_historical_data(self, ticker: str, start_date: str = None, end_date: str = None) -> Optional[pd.DataFrame]:
        """Get historical price data from database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = "SELECT * FROM historical_data WHERE ticker = ?"
                params = [ticker]
                
                if start_date:
                    query += " AND date >= ?"
                    params.append(start_date)
                if end_date:
                    query += " AND date <= ?"
                    params.append(end_date)
                
                query += " ORDER BY date"
                
                df = pd.read_sql_query(query, conn, params=params)
                if not df.empty:
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                return df
        except Exception as e:
            logging.error(f"Error getting historical data for {ticker}: {e}")
            return None
    
    def get_financial_statements(self, ticker: str, statement_type: str = None) -> Optional[List[Dict[str, Any]]]:
        """Get financial statements from database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                if statement_type:
                    cursor.execute("""
                        SELECT * FROM financial_statements 
                        WHERE ticker = ? AND statement_type = ? 
                        ORDER BY period_ending DESC
                    """, (ticker, statement_type))
                else:
                    cursor.execute("""
                        SELECT * FROM financial_statements 
                        WHERE ticker = ? 
                        ORDER BY period_ending DESC, statement_type
                    """, (ticker,))
                
                columns = [description[0] for description in cursor.description]
                results = []
                for row in cursor.fetchall():
                    result = dict(zip(columns, row))
                    if result['statement_data']:
                        result['statement_data'] = json.loads(result['statement_data'])
                    results.append(result)
                return results
        except Exception as e:
            logging.error(f"Error getting financial statements for {ticker}: {e}")
            return None
    
    def list_stocks(self) -> List[Dict[str, Any]]:
        """List all stocks in database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM stocks ORDER BY ticker")
                
                columns = [description[0] for description in cursor.description]
                results = []
                for row in cursor.fetchall():
                    results.append(dict(zip(columns, row)))
                return results
        except Exception as e:
            logging.error(f"Error listing stocks: {e}")
            return []
    
    def delete_stock(self, ticker: str) -> bool:
        """Delete a stock and all its related data."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Delete in order due to foreign key constraints
                cursor.execute("DELETE FROM historical_data WHERE ticker = ?", (ticker,))
                cursor.execute("DELETE FROM financial_statements WHERE ticker = ?", (ticker,))
                cursor.execute("DELETE FROM stock_data WHERE ticker = ?", (ticker,))
                cursor.execute("DELETE FROM stocks WHERE ticker = ?", (ticker,))
                
                conn.commit()
                return True
        except Exception as e:
            logging.error(f"Error deleting stock {ticker}: {e}")
            return False