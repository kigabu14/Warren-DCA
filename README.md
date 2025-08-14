# DCA Calculator with AI Optimization

Our goal is to define exactly what it means to classify a company for having a highly durable competitive advantage in the market utilizing well-known billionaire Warren Buffet's strategies. By researching his strategies, we will be able to create a concrete formula to find and invest in said companies.

## 🎯 New Feature: DCA AI Optimizer

### Advanced DCA Strategies with AI Analysis

**NEW in v2.0**: Intelligent Dollar Cost Averaging optimization with multiple strategies and AI-powered analysis in Thai language.

### 🚀 Core Features

- **🎯 Multi-ticker Analysis**: Analyze multiple stocks simultaneously
- **📊 5 Advanced DCA Strategies**:
  1. **Fixed Amount DCA** - Traditional constant investment
  2. **Value Averaging** - Target portfolio value growth  
  3. **Drawdown Trigger DCA** - Extra purchases during price drops
  4. **Momentum-Gated DCA** - Skip purchases during negative momentum
  5. **Adaptive Hybrid** - Combines multiple approaches intelligently

- **🔍 Parameter Optimization**:
  - Grid Search: Test all parameter combinations
  - Random Search: Sample parameter space efficiently  
  - Bayesian-like: Iterative improvement approach

- **🤖 AI Analysis** (Thai language):
  - Google Gemini integration
  - OpenAI GPT integration
  - Strategy recommendations
  - Risk analysis and warnings
  - Break-even forecasting

- **📈 Comprehensive Metrics**:
  - Total return and CAGR
  - Risk-adjusted returns (Sharpe ratio)
  - Maximum drawdown analysis
  - Break-even forecasting with multiple methods
  - Time-in-profit analysis
  - Monte Carlo simulations (optional)

- **📋 Advanced Export**:
  - Multi-sheet Excel workbooks
  - Summary and detailed results
  - AI analysis reports
  - Parameter configurations

### 🎮 How to Use

1. **Select Menu**: Choose "DCA AI Optimizer" from the sidebar
2. **Pick Stocks**: Select multiple tickers from global markets
3. **Configure Strategies**: Enable and tune DCA strategies
4. **Set AI Provider**: Optional - add Gemini or OpenAI API key
5. **Run Optimization**: Click "เริ่มการเพิ่มประสิทธิภาพ DCA"
6. **Review Results**: Get comprehensive analysis and AI insights
7. **Export Data**: Download detailed Excel reports

### 📊 Original Features (Preserved)

- **Warren Buffett Analysis**: 18-point checklist evaluation
- **Traditional DCA Simulation**: Monthly investment simulation
- **Dividend Analysis**: Historical dividend yield calculation
- **Multi-market Support**: US, SET100, Europe, Asia, Australia

### 🛠️ Technical Implementation

- **Data Layer**: `dca_data_loader.py` - Multi-ticker data fetching with caching
- **Strategy Layer**: `dca_strategies.py` - 5 DCA strategy implementations
- **Optimization**: `dca_optimizer.py` - Grid/Random/Bayesian parameter search
- **Analytics**: `dca_metrics.py` - Comprehensive metrics and forecasting
- **AI Integration**: `ai_dca_helper.py` - Provider abstraction for AI analysis

### 🔧 Dependencies

```
streamlit==1.28.0
yfinance
pandas
openpyxl
xlsxwriter
matplotlib
google-generativeai
scipy
```

### 📈 Example Workflow

1. **Quick Compare**: Test all strategies on single ticker with default parameters
2. **Full Optimization**: Multi-ticker, multi-strategy parameter optimization
3. **AI Analysis**: Get Thai-language insights on best strategies
4. **Export Results**: Download comprehensive Excel reports

### ⚠️ Important Notes

- **Network Required**: Real-time data fetching from Yahoo Finance
- **API Keys**: Optional for AI analysis (not stored permanently)
- **Performance**: Optimization can take several minutes for multiple tickers
- **Risk Warning**: Past performance doesn't guarantee future results

---

## Legacy Warren Buffett Analysis

### 18-Point Checklist (Expanded from 11 categories)

1. Inventory & Net Earnings เพิ่มขึ้นต่อเนื่อง
2. ไม่มี R&D
3. EBITDA > Current Liabilities ทุกปี  
4. PPE เพิ่มขึ้น (ไม่มี spike)
5. RTA ≥ 11%
6. RTA ≥ 17%  
7. LTD/Total Assets ≤ 0.5
8. EBITDA ปีล่าสุดจ่ายหนี้ LTD หมดใน ≤ 4 ปี
9. Equity ติดลบในปีใดหรือไม่
10. DSER ≤ 1.0
11. DSER ≤ 0.8
12. ไม่มี Preferred Stock
13. Retained Earnings เติบโต ≥ 7%
14. Retained Earnings เติบโต ≥ 13.5%  
15. Retained Earnings เติบโต ≥ 17%
16. มี Treasury Stock
17. ROE ≥ 23%
18. Goodwill เพิ่มขึ้น

### Data Sources
- **Financial Data**: Yahoo Finance API
- **Coverage**: Global markets with focus on US and SET100
- **Timeframe**: Typically 4 years of annual financial statements

### Running the Application

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Navigate to the local URL provided by Streamlit to access the web interface.