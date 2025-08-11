# Warren-DCA: Stock Analysis & Backtesting Tool

A comprehensive stock analysis and investment strategy testing application that combines Warren Buffett's investment principles with modern backtesting capabilities.

## Features

### 📊 Warren Buffett Analysis
- **18-Point Checklist**: Extended analysis based on Warren Buffett's 11 investment criteria
- **DCA Simulation**: Dollar Cost Averaging simulation with dividend calculations
- **Financial Analysis**: Comprehensive financial statement analysis
- **Multi-Market Support**: US, SET100, Europe, Asia, Australia markets

### 🔍 Backtesting Engine
- **Multiple Strategies**: 
  - Moving Average Crossover
  - RSI (Relative Strength Index)
  - Buy and Hold
- **Configurable Parameters**: Stop-loss, take-profit, strategy-specific settings
- **Performance Metrics**: Total return, Sharpe ratio, maximum drawdown, win rate
- **Visual Results**: Charts with buy/sell signals and performance analysis

### 🌐 Supported Markets
- **US Market**: AAPL, TSLA, NVDA, GOOG, MSFT, and more
- **Thai SET100**: PTT.BK, CPALL.BK, KBANK.BK, and more
- **European Markets**: ASML.AS, SAP.DE, NESN.SW, and more
- **Asian Markets**: Japanese, Korean, Hong Kong, Chinese stocks
- **Australian Market**: CBA.AX, WBC.AX, ANZ.AX, and more

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
streamlit run streamlit_app.py
```

## Technology Stack

- **Frontend**: Streamlit
- **Data Source**: Yahoo Finance (yfinance)
- **Backtesting**: Backtrader
- **Visualization**: Matplotlib
- **Data Processing**: Pandas, NumPy

## Goal

Our goal is to define exactly what it means to classify a company for having a highly durable competitive advantage in the market utilizing well-known billionaire Warren Buffet's strategies, while providing modern backtesting capabilities to validate investment strategies with historical data.