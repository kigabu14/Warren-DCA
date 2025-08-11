# Warren-DCA: Advanced Stock Analysis & Backtesting Tool

A comprehensive financial analysis application that combines Warren Buffett's fundamental analysis principles with modern technical analysis backtesting capabilities.

## Features

### 📊 **Fundamental Analysis (วิเคราะห์หุ้น)**
- **Warren Buffett's 11 Checklist** expanded to 18 detailed parameters
- **DCA (Dollar Cost Averaging) Simulation** with dividend analysis
- **Multi-market support**: US, SET100, Europe, Asia, Australia
- **Financial metrics evaluation** for investment decisions

### 🎯 **Backtesting Strategies**
- **Bollinger Bands Strategy**: Technical analysis backtesting with configurable parameters
- **Performance metrics**: Total return, win rate, max drawdown, trade statistics
- **Visual charts**: Price charts with Bollinger Bands overlay and trade signals
- **Risk management**: Configurable stop loss and take profit levels

### 📈 **Key Capabilities**
- Real-time stock data from Yahoo Finance
- Excel export functionality for all results
- Comprehensive error handling and validation
- Multi-language support (Thai/English)
- Responsive web interface built with Streamlit

## Bollinger Bands Strategy

### Strategy Logic
- **Buy Signal**: When price crosses below the lower Bollinger Band
- **Sell Signal**: When price crosses above the upper Bollinger Band
- **Risk Management**: Configurable stop loss and take profit percentages

### Configurable Parameters
- **Period**: Moving average period (default: 20 days)
- **Multiplier**: Standard deviation multiplier (default: 2.0)
- **Initial Capital**: Starting investment amount
- **Stop Loss**: Maximum loss percentage before exit
- **Take Profit**: Target profit percentage for exit

### Performance Metrics
- Total return and final portfolio value
- Number of trades and win rate
- Average win/loss per trade
- Maximum drawdown analysis
- Complete trade history with timestamps

## Installation & Usage

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application**:
   ```bash
   streamlit run streamlit_app.py
   ```

3. **Access the web interface** at `http://localhost:8501`

## Testing

Run the included tests to verify functionality:
```bash
python test_bb_strategy.py
```

## Navigation

- **วิเคราะห์หุ้น (Stock Analysis)**: Warren Buffett fundamental analysis
- **Backtesting**: Technical analysis strategy testing
- **คู่มือการใช้งาน (User Guide)**: Comprehensive documentation

## Goal

Our goal is to provide a comprehensive tool that combines:
1. **Fundamental Analysis**: Warren Buffett's proven investment strategies
2. **Technical Analysis**: Modern backtesting capabilities for trading strategies
3. **Risk Management**: Proper position sizing and risk controls
4. **Performance Tracking**: Detailed analytics and reporting

This allows investors to make informed decisions using both fundamental and technical analysis approaches.