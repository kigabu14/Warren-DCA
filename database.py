"""
Database module for Warren DCA application.
Handles storage and retrieval of stock analysis data using SQLite.
"""

import sqlite3
import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path


class StockDatabase:
    """SQLite database handler for stock analysis data."""
    
    def __init__(self, db_path: str = "data/warren_dca.db"):
        """Initialize database connection and create tables if needed."""
        self.db_path = Path(db_path)
        # Ensure data directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Create database tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS stocks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    company_name TEXT,
                    sector TEXT,
                    currency TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(ticker)
                );
                
                CREATE TABLE IF NOT EXISTS analyses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    stock_id INTEGER NOT NULL,
                    analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    period TEXT NOT NULL,
                    monthly_investment REAL,
                    
                    -- DCA Simulation Results
                    total_investment REAL,
                    total_units REAL,
                    avg_buy_price REAL,
                    current_value REAL,
                    profit_loss REAL,
                    profit_percentage REAL,
                    total_dividends REAL,
                    
                    -- Buffett 11 Checklist Results
                    buffett_score INTEGER,
                    buffett_total_checks INTEGER,
                    buffett_percentage INTEGER,
                    buffett_details TEXT, -- JSON string
                    
                    -- Additional stock metrics
                    dividend_yield REAL,
                    ex_dividend_date TEXT,
                    week_52_high REAL,
                    week_52_low REAL,
                    last_close REAL,
                    last_open REAL,
                    dividend_yield_1y REAL,
                    
                    FOREIGN KEY (stock_id) REFERENCES stocks (id)
                );
                
                CREATE INDEX IF NOT EXISTS idx_analyses_stock_date 
                ON analyses(stock_id, analysis_date);
                
                CREATE INDEX IF NOT EXISTS idx_stocks_ticker 
                ON stocks(ticker);
            """)
    
    def store_stock_info(self, ticker: str, company_name: str = None, 
                        sector: str = None, currency: str = None) -> int:
        """Store or update stock basic information. Returns stock_id."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO stocks 
                (ticker, company_name, sector, currency, updated_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (ticker, company_name, sector, currency))
            
            # Get the stock_id
            cursor = conn.execute("SELECT id FROM stocks WHERE ticker = ?", (ticker,))
            return cursor.fetchone()[0]
    
    def store_analysis(self, stock_id: int, analysis_data: Dict[str, Any]) -> int:
        """Store analysis results for a stock. Returns analysis_id."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO analyses (
                    stock_id, period, monthly_investment,
                    total_investment, total_units, avg_buy_price, current_value,
                    profit_loss, profit_percentage, total_dividends,
                    buffett_score, buffett_total_checks, buffett_percentage, buffett_details,
                    dividend_yield, ex_dividend_date, week_52_high, week_52_low,
                    last_close, last_open, dividend_yield_1y
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                stock_id,
                analysis_data.get('period', '5y'),
                analysis_data.get('monthly_investment', 1000),
                analysis_data.get('total_investment', 0),
                analysis_data.get('total_units', 0),
                analysis_data.get('avg_buy_price', 0),
                analysis_data.get('current_value', 0),
                analysis_data.get('profit_loss', 0),
                analysis_data.get('profit_percentage', 0),
                analysis_data.get('total_dividends', 0),
                analysis_data.get('buffett_score', 0),
                analysis_data.get('buffett_total_checks', 0),
                analysis_data.get('buffett_percentage', 0),
                json.dumps(analysis_data.get('buffett_details', {})),
                analysis_data.get('dividend_yield'),
                analysis_data.get('ex_dividend_date'),
                analysis_data.get('week_52_high'),
                analysis_data.get('week_52_low'),
                analysis_data.get('last_close'),
                analysis_data.get('last_open'),
                analysis_data.get('dividend_yield_1y')
            ))
            return cursor.lastrowid
    
    def get_stock_history(self, ticker: str, limit: int = 10) -> List[Dict]:
        """Get analysis history for a specific stock."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT 
                    s.ticker, s.company_name, s.sector,
                    a.analysis_date, a.period, a.monthly_investment,
                    a.total_investment, a.profit_loss, a.profit_percentage,
                    a.total_dividends, a.buffett_score, a.buffett_total_checks,
                    a.buffett_percentage, a.dividend_yield, a.dividend_yield_1y,
                    a.last_close
                FROM analyses a
                JOIN stocks s ON a.stock_id = s.id
                WHERE s.ticker = ?
                ORDER BY a.analysis_date DESC
                LIMIT ?
            """, (ticker, limit))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_all_analyzed_stocks(self, limit: int = 50) -> List[Dict]:
        """Get summary of all stocks that have been analyzed."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT 
                    s.ticker, s.company_name, s.sector,
                    COUNT(a.id) as analysis_count,
                    MAX(a.analysis_date) as last_analysis,
                    AVG(a.profit_percentage) as avg_profit_pct,
                    AVG(a.buffett_percentage) as avg_buffett_score,
                    SUM(a.total_investment) as total_invested,
                    SUM(a.profit_loss) as total_profit
                FROM stocks s
                LEFT JOIN analyses a ON s.id = a.stock_id
                GROUP BY s.id
                HAVING analysis_count > 0
                ORDER BY last_analysis DESC
                LIMIT ?
            """, (limit,))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_latest_analysis(self, ticker: str) -> Optional[Dict]:
        """Get the most recent analysis for a stock."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT 
                    s.ticker, s.company_name, s.sector,
                    a.analysis_date, a.period, a.monthly_investment,
                    a.total_investment, a.total_units, a.avg_buy_price,
                    a.current_value, a.profit_loss, a.profit_percentage,
                    a.total_dividends, a.buffett_score, a.buffett_total_checks,
                    a.buffett_percentage, a.buffett_details,
                    a.dividend_yield, a.ex_dividend_date, a.week_52_high,
                    a.week_52_low, a.last_close, a.last_open, a.dividend_yield_1y
                FROM analyses a
                JOIN stocks s ON a.stock_id = s.id
                WHERE s.ticker = ?
                ORDER BY a.analysis_date DESC
                LIMIT 1
            """, (ticker,))
            
            row = cursor.fetchone()
            if row:
                result = dict(row)
                # Parse JSON details
                if result['buffett_details']:
                    try:
                        result['buffett_details'] = json.loads(result['buffett_details'])
                    except:
                        result['buffett_details'] = {}
                return result
            return None
    
    def get_database_stats(self) -> Dict[str, int]:
        """Get basic statistics about the database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM stocks")
            stock_count = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT COUNT(*) FROM analyses")
            analysis_count = cursor.fetchone()[0]
            
            return {
                'total_stocks': stock_count,
                'total_analyses': analysis_count
            }