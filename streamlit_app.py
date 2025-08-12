import streamlit as st
import yfinance as yf
import pandas as pd
import io
import datetime
import numpy as np
import matplotlib.pyplot as plt
import backtrader as bt
import google.generativeai as genai
from textwrap import shorten

# ----------------- Helper Functions -----------------
def bollinger_bands_strategy(
    hist_data: pd.DataFrame,
    period: int = 20,
    multiplier: float = 2,
    initial_capital: float = 10000,
    stop_loss_pct: float = 5.0,
    take_profit_pct: float = 10.0,
    transaction_cost_rate: float = 0.0
) -> dict:
    """
    Bollinger Bands trading strategy with transaction costs
    
    Args:
        hist_data: Historical price data with 'Close' column
        period: Period for moving average and standard deviation
        multiplier: Standard deviation multiplier for bands
        initial_capital: Starting capital
        stop_loss_pct: Stop loss percentage
        take_profit_pct: Take profit percentage  
        transaction_cost_rate: Transaction cost rate (e.g., 0.001 = 0.1%)
    
    Returns:
        Dictionary with strategy results including trades and performance metrics
    """
    if hist_data.empty or len(hist_data) < period:
        return {"error": "Insufficient data for Bollinger Bands strategy"}
    
    # Calculate Bollinger Bands
    prices = hist_data['Close'].copy()
    rolling_mean = prices.rolling(window=period).mean()
    rolling_std = prices.rolling(window=period).std()
    upper_band = rolling_mean + (multiplier * rolling_std)
    lower_band = rolling_mean - (multiplier * rolling_std)
    
    # Initialize trading variables
    capital = float(initial_capital)
    shares = 0.0
    position = 0  # 0 = no position, 1 = long position
    entry_price = 0.0
    trades = []
    equity_curve = []
    
    # Trading loop
    for i in range(period, len(hist_data)):
        current_price = prices.iloc[i]
        current_date = hist_data.index[i]
        
        # BUY condition: price crosses below lower band
        if position == 0 and current_price < lower_band.iloc[i] and not pd.isna(lower_band.iloc[i]):
            effective_price = current_price * (1 + transaction_cost_rate)
            shares = capital / effective_price
            fee_buy = shares * current_price * transaction_cost_rate
            total_cost = shares * current_price + fee_buy
            entry_price = effective_price
            capital -= total_cost
            position = 1
            
            trades.append({
                'date': current_date,
                'action': 'BUY',
                'price': round(current_price, 4),
                'shares': round(shares, 4),
                'value': round(total_cost, 2),
                'fee': round(fee_buy, 2)
            })
            
        # SELL conditions
        elif position == 1:
            sell_signal = False
            sell_reason = ''
            
            # Check exit conditions
            if current_price > upper_band.iloc[i] and not pd.isna(upper_band.iloc[i]):
                sell_signal = True
                sell_reason = 'Upper band crossed'
            elif (current_price - entry_price) / entry_price * 100 >= take_profit_pct:
                sell_signal = True
                sell_reason = f'Take profit ({take_profit_pct}%)'
            elif (entry_price - current_price) / entry_price * 100 >= stop_loss_pct:
                sell_signal = True
                sell_reason = f'Stop loss ({stop_loss_pct}%)'
            
            if sell_signal:
                gross_value = shares * current_price
                fee_sell = gross_value * transaction_cost_rate
                net_value = gross_value - fee_sell
                profit = net_value - (shares * (entry_price - transaction_cost_rate * entry_price))
                capital += net_value
                
                trades.append({
                    'date': current_date,
                    'action': 'SELL',
                    'price': round(current_price, 4),
                    'shares': round(shares, 4),
                    'value': round(net_value, 2),
                    'fee': round(fee_sell, 2),
                    'profit': round(profit, 2),
                    'reason': sell_reason
                })
                
                shares = 0
                position = 0
        
        # Calculate current portfolio value
        current_value = capital + (shares * current_price if shares > 0 else 0)
        equity_curve.append(current_value)
    
    # Final portfolio value
    final_value = capital + (shares * prices.iloc[-1] if shares > 0 else 0)
    total_return = (final_value - initial_capital) / initial_capital * 100
    
    # Calculate additional metrics
    if len(equity_curve) > 0:
        equity_series = pd.Series(equity_curve)
        max_value = equity_series.cummax()
        drawdown = (equity_series - max_value) / max_value * 100
        max_drawdown = drawdown.min()
    else:
        max_drawdown = 0
    
    # Trade statistics
    winning_trades = [t for t in trades if t.get('profit', 0) > 0]
    losing_trades = [t for t in trades if t.get('profit', 0) < 0]
    total_trades = len([t for t in trades if t['action'] == 'SELL'])
    win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
    
    return {
        "initial_capital": initial_capital,
        "final_value": round(final_value, 2),
        "total_return": round(total_return, 2),
        "total_trades": total_trades,
        "winning_trades": len(winning_trades),
        "losing_trades": len(losing_trades),
        "win_rate": round(win_rate, 2),
        "max_drawdown": round(max_drawdown, 2),
        "trades": trades,
        "equity_curve": equity_curve,
        "parameters": {
            "period": period, 
            "multiplier": multiplier, 
            "stop_loss_pct": stop_loss_pct, 
            "take_profit_pct": take_profit_pct, 
            "transaction_cost_rate": transaction_cost_rate
        }
    }

def human_format(num):
    if pd.isna(num):
        return ""
    num = float(num)
    for unit in ['', 'K', 'M', 'B', 'T']:
        if abs(num) < 1000.0:
            return "%3.1f%s" % (num, unit)
        num /= 1000.0
    return "%.1fP" % num

def df_human_format(df):
    return df.applymap(lambda x: human_format(x) if isinstance(x, (int, float)) else x)

def calc_dividend_yield_manual(div, hist):
    """คำนวณ Dividend Yield จากเงินปันผลที่ได้รับจริงย้อนหลัง 1 ปี"""
    if not div.empty and not hist.empty:
        last_year = hist.index[-1] - pd.DateOffset(years=1)
        recent_div = div[div.index >= last_year]
        total_div = recent_div.sum()
        avg_price = hist['Close'][hist.index >= last_year].mean()
        manual_yield = (total_div / avg_price) * 100 if avg_price > 0 else np.nan
        return total_div, avg_price, manual_yield
    return 0, 0, np.nan

def dca_simulation(hist_prices: pd.DataFrame, monthly_invest: float = 1000, div=None):
    if hist_prices.empty:
        return {"error": "ไม่มีข้อมูลราคาหุ้น"}
    prices = hist_prices['Close'].resample('M').first().dropna()
    units = monthly_invest / prices
    total_units = units.sum()
    total_invested = monthly_invest * len(prices)
    avg_buy_price = total_invested / total_units if total_units != 0 else 0
    latest_price = prices.iloc[-1]
    current_value = total_units * latest_price
    profit = current_value - total_invested
    # คำนวณเงินปันผลรวมที่ได้รับตามจำนวนหุ้นที่ถือในแต่ละเดือน
    total_div = 0
    if div is not None and not div.empty:
        div_period = div[div.index >= prices.index[0]]
        if not div_period.empty:
            cum_units = units.cumsum()
            for i, dt in enumerate(prices.index):
                div_in_month = div_period[(div_period.index.month == dt.month) & (div_period.index.year == dt.year)].sum()
                if div_in_month > 0:
                    total_div += div_in_month * cum_units.iloc[i]
    return {
        "เงินลงทุนรวม": round(total_invested, 2),
        "จำนวนหุ้นสะสม": round(total_units, 4),
        "มูลค่าปัจจุบัน": round(current_value, 2),
        "กำไร/ขาดทุน": round(profit, 2),
        "กำไร(%)": round(profit/total_invested*100, 2) if total_invested != 0 else 0,
        "ราคาเฉลี่ยที่ซื้อ": round(avg_buy_price, 2),
        "ราคาปิดล่าสุด": round(latest_price, 2),
        "เงินปันผลรวม": round(total_div, 2)
    }

def gemini_insight(user_prompt: str, context: str) -> str:
    """
    Generate AI insights using Gemini with Thai prompt structure
    
    Args:
        user_prompt: User's question about the stock
        context: Context including stock data and analysis
    
    Returns:
        AI-generated response or error message
    """
    # Check if API key is available
    if 'gemini_api_key' not in st.session_state or not st.session_state['gemini_api_key']:
        return "กรุณาเพิ่ม Gemini API Key ในส่วน Settings ทางซ้ายก่อนใช้งาน"
    
    try:
        # Get model and temperature from session state
        model_name = st.session_state.get('gemini_model', 'gemini-1.5-flash')
        temperature = st.session_state.get('gemini_temperature', 0.7)
        
        # Configure Gemini
        genai.configure(api_key=st.session_state['gemini_api_key'])
        model = genai.GenerativeModel(model_name)
        
        # Truncate context to avoid token limits
        truncated_context = shorten(context, width=1600, placeholder="...")
        
        # Build Thai prompt structure
        thai_prompt = f"""
คุณเป็นนักวิเคราะห์หุ้นมืออาชีพ กรุณาวิเคราะห์ข้อมูลต่อไปนี้และตอบคำถาม:

**ข้อมูลบริบท:**
{truncated_context}

**คำถามจากผู้ใช้:**
{user_prompt}

กรุณาตอบตามโครงสร้างต่อไปนี้:

1) **สรุปสถานการณ์**: สรุปสถานะปัจจุบันของหุ้นตัวนี้
2) **จุดเด่น / โอกาส**: วิเคราะห์จุดแข็งและโอกาสในการลงทุน  
3) **ความเสี่ยง / ข้อควรระวัง**: ประเด็นที่ควรระวังและความเสี่ยงที่อาจเกิดขึ้น
4) **แนวคิดกลยุทธ์ต่อไป**: แนะนำกลยุทธ์การลงทุน (ไม่ใช่คำแนะนำการลงทุนโดยตรง)

**ข้อจำกัดความรับผิดชอบ:** การวิเคราะห์นี้เป็นเพียงข้อมูลเพื่อการศึกษาเท่านั้น ไม่ใช่คำแนะนำการลงทุน ผู้ลงทุนควรศึกษาข้อมูลเพิ่มเติมและประเมินความเสี่ยงด้วยตนเองก่อนตัดสินใจลงทุน
"""
        
        # Generate response
        response = model.generate_content(
            thai_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=1000,
            )
        )
        
        return response.text
        
    except Exception as e:
        return f"เกิดข้อผิดพลาดในการเชื่อมต่อ Gemini API: {str(e)}"

# ----------------- Backtesting Functions & Strategies -----------------
class MovingAverageCrossStrategy(bt.Strategy):
    """กลยุทธ์ Moving Average Crossover"""
    params = (
        ('fast_period', 10),
        ('slow_period', 30),
        ('stop_loss', 0.1),  # 10% stop loss
        ('take_profit', 0.2),  # 20% take profit
    )

    def __init__(self):
        self.fast_ma = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.fast_period)
        self.slow_ma = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.slow_period)
        self.crossover = bt.indicators.CrossOver(self.fast_ma, self.slow_ma)
        self.buy_price = None
        self.buy_comm = None

    def next(self):
        if not self.position:
            # Buy signal: fast MA crosses above slow MA
            if self.crossover > 0:
                self.buy()
                self.buy_price = self.data.close[0]
        else:
            # Already in position
            current_price = self.data.close[0]
            if self.buy_price:
                # Calculate profit/loss percentage
                pnl_pct = (current_price - self.buy_price) / self.buy_price
                
                # Sell conditions
                if (self.crossover < 0 or  # fast MA crosses below slow MA
                    pnl_pct <= -self.params.stop_loss or  # stop loss
                    pnl_pct >= self.params.take_profit):  # take profit
                    self.sell()
                    self.buy_price = None

class RSIStrategy(bt.Strategy):
    """กลยุทธ์ RSI (Relative Strength Index)"""
    params = (
        ('rsi_period', 14),
        ('rsi_overbought', 70),
        ('rsi_oversold', 30),
        ('stop_loss', 0.1),
        ('take_profit', 0.15),
    )

    def __init__(self):
        self.rsi = bt.indicators.RelativeStrengthIndex(period=self.params.rsi_period)
        self.buy_price = None

    def next(self):
        if not self.position:
            # Buy signal: RSI < oversold level
            if self.rsi < self.params.rsi_oversold:
                self.buy()
                self.buy_price = self.data.close[0]
        else:
            current_price = self.data.close[0]
            if self.buy_price:
                pnl_pct = (current_price - self.buy_price) / self.buy_price
                
                # Sell conditions
                if (self.rsi > self.params.rsi_overbought or
                    pnl_pct <= -self.params.stop_loss or
                    pnl_pct >= self.params.take_profit):
                    self.sell()
                    self.buy_price = None

class BuyAndHoldStrategy(bt.Strategy):
    """กลยุทธ์ Buy and Hold"""
    def __init__(self):
        self.bought = False

    def next(self):
        if not self.bought:
            self.buy()
            self.bought = True

def run_backtest(ticker, strategy_name, start_date, end_date, initial_cash=10000, **strategy_params):
    """รันการทดสอบย้อนหลัง"""
    try:
        # ดาวน์โหลดข้อมูล
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date)
        
        if data.empty:
            return None, "ไม่มีข้อมูลสำหรับช่วงเวลาที่เลือก"
        
        # เตรียม Backtrader
        cerebro = bt.Cerebro()
        
        # เลือกกลยุทธ์
        if strategy_name == "Moving Average Cross":
            cerebro.addstrategy(MovingAverageCrossStrategy, **strategy_params)
        elif strategy_name == "RSI":
            cerebro.addstrategy(RSIStrategy, **strategy_params)
        elif strategy_name == "Buy and Hold":
            cerebro.addstrategy(BuyAndHoldStrategy)
        
        # แปลงข้อมูลสำหรับ Backtrader
        data_bt = bt.feeds.PandasData(dataname=data, fromdate=start_date, todate=end_date)
        cerebro.adddata(data_bt)
        
        # ตั้งค่าเงินเริ่มต้น
        cerebro.broker.setcash(initial_cash)
        cerebro.broker.setcommission(commission=0.001)  # 0.1% commission
        
        # เพิ่ม analyzers
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
        cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
        
        # รันการทดสอบ
        start_value = cerebro.broker.getvalue()
        results = cerebro.run()
        end_value = cerebro.broker.getvalue()
        
        # ดึงผลลัพธ์
        result = results[0]
        
        # คำนวณ metrics
        total_return = ((end_value - start_value) / start_value) * 100
        sharpe_ratio = result.analyzers.sharpe.get_analysis().get('sharperatio', 0)
        max_drawdown = result.analyzers.drawdown.get_analysis().get('max', {}).get('drawdown', 0)
        
        trade_analysis = result.analyzers.trades.get_analysis()
        total_trades = trade_analysis.get('total', {}).get('total', 0)
        won_trades = trade_analysis.get('won', {}).get('total', 0)
        win_rate = (won_trades / total_trades * 100) if total_trades > 0 else 0
        
        return {
            'total_return': round(total_return, 2),
            'sharpe_ratio': round(sharpe_ratio, 2) if sharpe_ratio else 0,
            'max_drawdown': round(max_drawdown, 2),
            'total_trades': total_trades,
            'won_trades': won_trades,
            'win_rate': round(win_rate, 2),
            'start_value': round(start_value, 2),
            'end_value': round(end_value, 2),
            'cerebro': cerebro,
            'data': data
        }, None
        
    except Exception as e:
        return None, f"เกิดข้อผิดพลาด: {str(e)}"

def plot_backtest_results(data, strategy_name, ticker):
    """สร้างกราฟแสดงผลการ backtest"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # กราฟราคาหุ้น
    ax1.plot(data.index, data['Close'], label='ราคาปิด', color='blue')
    ax1.set_title(f'{ticker} - {strategy_name} Backtest Results')
    ax1.set_ylabel('ราคา')
    ax1.legend()
    ax1.grid(True)
    
    # กราฟ Volume
    ax2.bar(data.index, data['Volume'], alpha=0.3, color='gray')
    ax2.set_ylabel('Volume')
    ax2.set_xlabel('วันที่')
    ax2.grid(True)
    
    plt.tight_layout()
    return fig

# ----------------- Buffett 11 Checklist (ละเอียด) -----------------
def buffett_11_checks_detail(financials, balance_sheet, cashflow, dividends, hist_prices):
    results = []
    score = 0
    evaluated = 0

    # 1.1 Inventory & Net Income growth
    try:
        inv = []
        for col in balance_sheet.columns:
            v = balance_sheet.loc[balance_sheet.index.str.contains("Inventor", case=False), col]
            if not v.empty and v.iloc[0] is not None:
                inv.append(v.iloc[0])
        inv_growth = all([inv[i] < inv[i+1] for i in range(len(inv)-1)]) if len(inv) >= 2 else True

        ni = []
        for col in financials.columns:
            v = financials.loc[financials.index.str.contains("Net Income", case=False), col]
            if not v.empty and v.iloc[0] is not None:
                ni.append(v.iloc[0])
        ni_growth = all([ni[i] < ni[i+1] for i in range(len(ni)-1)]) if len(ni) >= 2 else True
        res = 1 if inv_growth and ni_growth else 0
    except:
        res = -1
    results.append({
        'title': '1.1 Inventory & Net Earnings เพิ่มขึ้นต่อเนื่อง',
        'result': res,
        'desc': 'Inventory และ Net Income ต้องโตต่อเนื่อง'
    })
    if res != -1:
        score += res
        evaluated += 1

    # 1.2 No R&D
    try:
        r_and_d = any(financials.index.str.contains('Research', case=False))
        res = 0 if r_and_d else 1
    except:
        res = -1
    results.append({'title': '1.2 ไม่มี Research & Development', 'result': res, 'desc': 'ไม่มีค่าใช้จ่าย R&D'})
    if res != -1:
        score += res
        evaluated += 1

    # 2 EBITDA > Current Liabilities every year
    try:
        ebitda = []
        cl = []
        for col in financials.columns:
            v = financials.loc[financials.index.str.contains("EBITDA", case=False), col]
            if not v.empty and v.iloc[0] is not None:
                ebitda.append(v.iloc[0])
        for col in balance_sheet.columns:
            v = balance_sheet.loc[balance_sheet.index.str.contains("Current Liab", case=False), col]
            if not v.empty and v.iloc[0] is not None:
                cl.append(v.iloc[0])
        res = 1 if all([ebitda[i] > cl[i] for i in range(min(len(ebitda), len(cl)))]) and len(ebitda) > 0 else 0
    except:
        res = -1
    results.append({'title': '2. EBITDA > Current Liabilities ทุกปี', 'result': res, 'desc': 'EBITDA มากกว่าหนี้สินหมุนเวียนทุกปี'})
    if res != -1:
        score += res
        evaluated += 1

    # 3 PPE increasing (no spike)
    try:
        ppe = []
        for col in balance_sheet.columns:
            v = balance_sheet.loc[balance_sheet.index.str.contains("Property, Plant", case=False), col]
            if not v.empty and v.iloc[0] is not None:
                ppe.append(v.iloc[0])
        if len(ppe) >= 2:
            growth = all([ppe[i] <= ppe[i+1] for i in range(len(ppe)-1)])
            spike = max([abs(ppe[i+1]-ppe[i]) / ppe[i] if ppe[i] != 0 else 0 for i in range(len(ppe)-1)]) < 1.0
            res = 1 if growth and spike else 0
        else:
            res = -1
    except:
        res = -1
    results.append({'title': '3. PPE เพิ่มขึ้น (ไม่มี spike)', 'result': res, 'desc': 'Property, Plant & Equipment โตต่อเนื่อง'})
    if res != -1:
        score += res
        evaluated += 1

    # 4.1 RTA >= 11%
    try:
        ebitda = []
        ta = []
        for col in financials.columns:
            v = financials.loc[financials.index.str.contains("EBITDA", case=False), col]
            if not v.empty and v.iloc[0] is not None:
                ebitda.append(v.iloc[0])
        for col in balance_sheet.columns:
            v = balance_sheet.loc[balance_sheet.index.str.contains("Total Assets", case=False), col]
            if not v.empty and v.iloc[0] is not None:
                ta.append(v.iloc[0])
        rtas = [ebitda[i] / ta[i] for i in range(min(len(ebitda), len(ta))) if ta[i] != 0]
        avg_rta = sum(rtas) / len(rtas) if rtas else 0
        res = 1 if avg_rta >= 0.11 else 0
    except:
        res = -1
        avg_rta = 0
    results.append({'title': '4.1 RTA ≥ 11%', 'result': res, 'desc': 'Return on Total Assets เฉลี่ย ≥ 11%'})
    if res != -1:
        score += res
        evaluated += 1

    # 4.2 RTA >= 17%
    try:
        res = 1 if avg_rta >= 0.17 else 0
    except:
        res = -1
    results.append({'title': '4.2 RTA ≥ 17%', 'result': res, 'desc': 'Return on Total Assets เฉลี่ย ≥ 17%'})
    if res != -1:
        score += res
        evaluated += 1

    # 5.1 LTD/Total Assets <= 0.5
    try:
        ltd = []
        for col in balance_sheet.columns:
            v = balance_sheet.loc[balance_sheet.index.str.contains("Long Term Debt", case=False), col]
            if not v.empty and v.iloc[0] is not None:
                ltd.append(v.iloc[0])
        ratios = [ltd[i] / ta[i] for i in range(min(len(ltd), len(ta))) if ta[i] != 0]
        avg_ratio = sum(ratios) / len(ratios) if ratios else 1
        res = 1 if avg_ratio <= 0.5 else 0
    except:
        res = -1
    results.append({'title': '5.1 LTD/Total Assets ≤ 0.5', 'result': res, 'desc': 'อัตราส่วนหนี้สินระยะยาว ≤ 0.5'})
    if res != -1:
        score += res
        evaluated += 1

    # 5.2 LTD can be repaid by EBITDA in <= 4 yrs
    try:
        last_ebitda = ebitda[-1] if ebitda else None
        last_ltd = ltd[-1] if ltd else None
        if last_ebitda and last_ltd and last_ebitda > 0:
            res = 1 if last_ltd / last_ebitda <= 4 else 0
        else:
            res = -1
    except:
        res = -1
    results.append({
        'title': '5.2 EBITDA จ่ายหนี้ LTD หมดใน ≤ 4 ปี',
        'result': res,
        'desc': 'EBITDA ล่าสุดชำระหนี้ระยะยาวหมดภายใน ≤ 4 ปี'
    })
    if res != -1:
        score += res
        evaluated += 1

    # 6.1 Negative equity any year?
    try:
        se = []
        for col in balance_sheet.columns:
            v = balance_sheet.loc[balance_sheet.index.str.contains("Total Stock", case=False) &
                                  balance_sheet.index.str.contains("Equity", case=False), col]
            if not v.empty and v.iloc[0] is not None:
                se.append(v.iloc[0])
        neg_se = any([x < 0 for x in se])
        res = 1 if neg_se else 0  # If equity negative => flag (pass=1 meaning "Yes, negative"?) - kept original logic
    except:
        res = -1
        neg_se = False
    results.append({'title': '6.1 Equity ติดลบในปีใดหรือไม่', 'result': res, 'desc': 'ถ้าติดลบ ข้าม 6.2-6.3'})
    if res != -1:
        evaluated += 1  # ไม่บวกคะแนน

    # 6.2 DSER <= 1.0
    try:
        if not neg_se:
            tl = []
            ts = []
            for col in balance_sheet.columns:
                v = balance_sheet.loc[balance_sheet.index.str.contains("Total Liab", case=False), col]
                if not v.empty and v.iloc[0] is not None:
                    tl.append(v.iloc[0])
                v = balance_sheet.loc[balance_sheet.index.str.contains("Treasury Stock", case=False), col]
                if not v.empty and v.iloc[0] is not None:
                    ts.append(abs(v.iloc[0]))
            adj_se = [se[i] + (ts[i] if i < len(ts) else 0) for i in range(min(len(se), len(ts)))] if ts else se
            dser = [tl[i] / adj_se[i] for i in range(min(len(tl), len(adj_se))) if adj_se[i] != 0]
            avg_dser = sum(dser) / len(dser) if dser else 0
            res = 1 if avg_dser <= 1.0 else 0
        else:
            res = -1
            avg_dser = 0
    except:
        res = -1
    results.append({'title': '6.2 DSER ≤ 1.0', 'result': res, 'desc': 'Debt to Shareholder Equity Ratio ≤ 1.0'})
    if res != -1:
        score += res
        evaluated += 1

    # 6.3 DSER <= 0.8
    try:
        res = 1 if not neg_se and avg_dser <= 0.8 else (-1 if neg_se else 0)
    except:
        res = -1
    results.append({'title': '6.3 DSER ≤ 0.8', 'result': res, 'desc': 'Debt to Shareholder Equity Ratio ≤ 0.8'})
    if res != -1:
        score += res
        evaluated += 1

    # 7 No preferred stock
    try:
        pref = any(balance_sheet.index.str.contains('Preferred', case=False))
        res = 0 if pref else 1
    except:
        res = -1
    results.append({'title': '7. ไม่มี Preferred Stock', 'result': res, 'desc': 'ไม่มีหุ้นบุริมสิทธิ'})
    if res != -1:
        score += res
        evaluated += 1

    # 8.x Retained earnings growth
    try:
        re = []
        for col in balance_sheet.columns:
            v = balance_sheet.loc[balance_sheet.index.str.contains("Retained Earnings", case=False), col]
            if not v.empty and v.iloc[0] is not None:
                re.append(v.iloc[0])
        re_growths = [(re[i+1]-re[i]) / re[i] if re[i] != 0 else 0 for i in range(len(re)-1)]
        avg_re_growth = sum(re_growths) / len(re_growths) if re_growths else 0
        res1 = 1 if avg_re_growth >= 0.07 else 0
        res2 = 1 if avg_re_growth >= 0.135 else 0
        res3 = 1 if avg_re_growth >= 0.17 else 0
    except:
        res1 = res2 = res3 = -1
        avg_re_growth = 0
    results.append({'title': '8.1 Retained Earnings เติบโต ≥ 7%', 'result': res1, 'desc': 'Retained Earnings เติบโตเฉลี่ย ≥ 7%'})
    if res1 != -1:
        score += res1
        evaluated += 1
    results.append({'title': '8.2 Retained Earnings เติบโต ≥ 13.5%', 'result': res2, 'desc': 'Retained Earnings เติบโตเฉลี่ย ≥ 13.5%'})
    if res2 != -1:
        score += res2
        evaluated += 1
    results.append({'title': '8.3 Retained Earnings เติบโต ≥ 17%', 'result': res3, 'desc': 'Retained Earnings เติบโตเฉลี่ย ≥ 17%'})
    if res3 != -1:
        score += res3
        evaluated += 1

    # 9 Treasury stock exists?
    try:
        ts = []
        for col in balance_sheet.columns:
            v = balance_sheet.loc[balance_sheet.index.str.contains("Treasury Stock", case=False), col]
            if not v.empty and v.iloc[0] is not None:
                ts.append(v.iloc[0])
        res = 1 if any([x != 0 for x in ts]) else 0
    except:
        res = -1
    results.append({'title': '9. มี Treasury Stock', 'result': res, 'desc': 'มี Treasury Stock หรือไม่'})
    if res != -1:
        score += res
        evaluated += 1

    # 10 ROE >= 23%
    try:
        roe = [ebitda[i] / se[i] for i in range(min(len(ebitda), len(se))) if se[i] != 0]
        avg_roe = sum(roe) / len(roe) if roe else 0
        res = 1 if avg_roe >= 0.23 else 0
    except:
        res = -1
    results.append({'title': '10. ROE ≥ 23%', 'result': res, 'desc': 'Return on Shareholders Equity เฉลี่ย ≥ 23%'})
    if res != -1:
        score += res
        evaluated += 1

    # 11 Goodwill increasing
    try:
        gw = []
        for col in balance_sheet.columns:
            v = balance_sheet.loc[balance_sheet.index.str.contains("Goodwill", case=False), col]
            if not v.empty and v.iloc[0] is not None:
                gw.append(v.iloc[0])
        res = 1 if all([gw[i] <= gw[i+1] for i in range(len(gw)-1)]) and len(gw) >= 2 else 0
    except:
        res = -1
    results.append({'title': '11. Goodwill เพิ่มขึ้น', 'result': res, 'desc': 'Goodwill โตต่อเนื่อง'})
    if res != -1:
        score += res
        evaluated += 1

    score_pct = int(score / evaluated * 100) if evaluated > 0 else 0
    return {'details': results, 'score': score, 'evaluated': evaluated, 'score_pct': score_pct}

def get_badge(score_pct):
    if score_pct >= 80:
        return "🟢 ดีเยี่ยม (Excellent)"
    elif score_pct >= 60:
        return "🟩 ดี (Good)"
    elif score_pct >= 40:
        return "🟨 ปานกลาง (Average)"
    else:
        return "🟥 ควรระวัง (Poor)"

# ----------------- STOCK LISTS BY MARKET -----------------
set100 = [
    "ADVANC.BK", "AOT.BK", "AP.BK", "AWC.BK", "BAM.BK", "BANPU.BK", "BBL.BK", "BCP.BK", "BDMS.BK", "BEC.BK",
    "BEM.BK", "BGRIM.BK", "BH.BK", "BJC.BK", "BLA.BK", "BPP.BK", "BTS.BK", "CBG.BK", "CENTEL.BK", "CHG.BK",
    "CK.BK", "COM7.BK", "CPALL.BK", "CPF.BK", "CPN.BK", "CRC.BK", "DELTA.BK", "DOHOME.BK", "DTAC.BK", "EGCO.BK",
    "EPG.BK", "ESSO.BK", "GLOBAL.BK", "GPSC.BK", "GULF.BK", "HANA.BK", "HMPRO.BK", "INTUCH.BK", "IRPC.BK",
    "IVL.BK", "JMART.BK", "JMT.BK", "KBANK.BK", "KCE.BK", "KEX.BK", "KTB.BK", "KTC.BK", "LH.BK", "M.BK",
    "MINT.BK", "MTC.BK", "OR.BK", "OSP.BK", "PLANB.BK", "PRM.BK", "PTG.BK", "PTT.BK", "PTTEP.BK", "PTTGC.BK",
    "QH.BK", "RATCH.BK", "RS.BK", "SAWAD.BK", "SCB.BK", "SCC.BK", "SCGP.BK", "SGP.BK", "SIRI.BK", "SPALI.BK",
    "STA.BK", "STARK.BK", "STEC.BK", "STGT.BK", "STPI.BK", "SUPER.BK", "TASCO.BK", "TCAP.BK", "THANI.BK",
    "THG.BK", "TISCO.BK", "TKN.BK", "TMB.BK", "TOA.BK", "TOP.BK", "TRUE.BK", "TTB.BK", "TU.BK", "TVO.BK",
    "VGI.BK", "WHA.BK"
]

us_stocks = [
    "AAPL", "TSLA", "NVDA", "GOOG", "MSFT", "SBUX", "AMD", "BABA", "T", "WMT",
    "SONY", "KO", "MCD", "MCO", "SNAP", "DIS", "NFLX", "GPRO", "CCL", "PLTR", "CBOE", "HD", "F", "COIN"
]

# European stocks
european_stocks = [
    "ASML.AS", "SAP.DE", "NESN.SW", "INGA.AS", "MC.PA", "OR.PA", "SAN.PA", "RDSA.AS", "NOVN.SW", "ROG.SW",
    "LONN.SW", "UNA.AS", "ADYEN.AS", "DSM.AS", "PHIA.AS", "DBK.DE", "EOAN.DE", "VOW3.DE", "SIE.DE", "ALV.DE",
    "AZN.L", "ULVR.L", "SHEL.L", "BP.L", "HSBA.L", "GSK.L", "DGE.L", "VODL.L", "BARC.L", "LLOY.L"
]

# Asian stocks (excluding Thailand)
asian_stocks = [
    "7203.T", "9984.T", "6098.T", "6758.T", "8035.T", "9434.T", "4063.T", "7974.T", "6501.T", "9432.T",  # Japan
    "005930.KS", "000660.KS", "035420.KS", "207940.KS", "035720.KS", "068270.KS", "012330.KS", "051910.KS",  # South Korea
    "0700.HK", "9988.HK", "0941.HK", "1299.HK", "0175.HK", "1398.HK", "3690.HK", "0388.HK", "2318.HK", "1810.HK",  # Hong Kong
    "000001.SS", "000002.SS", "000858.SS", "600036.SS", "600519.SS", "000725.SS", "600276.SS", "002415.SS"  # China
]

# Australian stocks
australian_stocks = [
    "CBA.AX", "WBC.AX", "ANZ.AX", "NAB.AX", "CSL.AX", "BHP.AX", "WOW.AX", "TLS.AX", "WES.AX", "MQG.AX",
    "COL.AX", "TCL.AX", "RIO.AX", "WDS.AX", "REA.AX", "QBE.AX", "IAG.AX", "SUN.AX", "QAN.AX", "ALL.AX"
]

# Market definitions
markets = {
    "US": us_stocks,
    "SET100": set100,
    "Europe": european_stocks,
    "Asia": asian_stocks,
    "Australia": australian_stocks,
    "Global": us_stocks + set100 + european_stocks + asian_stocks + australian_stocks
}

# ----------------- UI & Main -----------------
st.set_page_config(page_title="Warren-DCA วิเคราะห์หุ้น", layout="wide")
menu = st.sidebar.radio("เลือกหน้าที่ต้องการ", ["วิเคราะห์หุ้น", "Backtesting", "คู่มือการใช้งาน"])

if menu == "คู่มือการใช้งาน":
    st.header("คู่มือการใช้งาน (ภาษาไทย)")
    st.markdown("""
**Warren-DCA คืออะไร?**  
โปรแกรมนี้ช่วยวิเคราะห์หุ้นตามแนวทางของ Warren Buffett (Buffett 11 Checklist แบบขยาย 18 เงื่อนไข) พร้อมจำลองการลงทุนแบบ DCA และคำนวณผลตอบแทนเงินปันผลย้อนหลัง 1 ปี  
**แหล่งข้อมูล:** Yahoo Finance

### กฎ 18 ข้อ (Buffett Checklist ย่อยจาก 11 หัวข้อ)
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

### หมายเหตุ
- ข้อมูลหุ้น US มักครบถ้วนกว่าหุ้นไทย
- ถ้าข้อมูลสำคัญไม่ครบ บางข้อจะขึ้น N/A
- ใช้งบการเงินย้อนหลัง (Annual) ตามที่ Yahoo ให้ (ปกติ 4 ปี)
- รองรับหุ้นจากตลาดทั่วโลก: US, SET100, Europe, Asia, Australia

### Backtesting
- รองรับกลยุทธ์ต่างๆ เช่น Moving Average Cross, RSI, Buy and Hold
- แสดงผลการวิเคราะห์ Sharpe Ratio, Maximum Drawdown, Win Rate
- สามารถตั้งค่า Stop Loss และ Take Profit ได้
""")
    st.stop()

elif menu == "Backtesting":
    st.header("🔍 Backtesting - ทดสอบกลยุทธ์การลงทุน")
    st.markdown("ทดสอบกลยุทธ์การลงทุนของคุณกับข้อมูลในอดีต")
    
    # Input section
    col1, col2 = st.columns(2)
    
    with col1:
        # Market selection for backtesting
        selected_market_bt = st.selectbox(
            "เลือกตลาดหุ้น (Backtesting)",
            options=list(markets.keys()),
            index=0,
            help="เลือกตลาดหุ้นที่ต้องการทดสอบ"
        )
        
        available_tickers_bt = markets[selected_market_bt]
        
        # Single ticker selection for backtesting
        ticker_bt = st.selectbox(
            f"เลือกหุ้น ({selected_market_bt})",
            available_tickers_bt,
            help="เลือกหุ้นที่ต้องการทดสอบกลยุทธ์"
        )
        
        # Strategy selection
        strategy_name = st.selectbox(
            "เลือกกลยุทธ์",
            ["Moving Average Cross", "RSI", "Buy and Hold"],
            help="เลือกกลยุทธ์การลงทุนที่ต้องการทดสอบ"
        )
        
        # Date range
        col_start, col_end = st.columns(2)
        with col_start:
            start_date = st.date_input(
                "วันที่เริ่มต้น",
                value=datetime.date.today() - datetime.timedelta(days=365*2),
                help="วันที่เริ่มต้นการทดสอบ"
            )
        with col_end:
            end_date = st.date_input(
                "วันที่สิ้นสุด",
                value=datetime.date.today(),
                help="วันที่สิ้นสุดการทดสอบ"
            )
    
    with col2:
        # Strategy parameters
        st.subheader("ตั้งค่ากลยุทธ์")
        
        initial_cash = st.number_input(
            "เงินทุนเริ่มต้น",
            min_value=1000.0,
            max_value=1000000.0,
            value=10000.0,
            step=1000.0,
            help="จำนวนเงินทุนเริ่มต้นสำหรับการทดสอบ"
        )
        
        strategy_params = {}
        
        if strategy_name == "Moving Average Cross":
            strategy_params['fast_period'] = st.slider("Fast MA Period", 5, 50, 10)
            strategy_params['slow_period'] = st.slider("Slow MA Period", 20, 100, 30)
            strategy_params['stop_loss'] = st.slider("Stop Loss (%)", 1, 20, 10) / 100
            strategy_params['take_profit'] = st.slider("Take Profit (%)", 5, 50, 20) / 100
            
        elif strategy_name == "RSI":
            strategy_params['rsi_period'] = st.slider("RSI Period", 10, 30, 14)
            strategy_params['rsi_oversold'] = st.slider("RSI Oversold Level", 20, 40, 30)
            strategy_params['rsi_overbought'] = st.slider("RSI Overbought Level", 60, 80, 70)
            strategy_params['stop_loss'] = st.slider("Stop Loss (%)", 1, 20, 10) / 100
            strategy_params['take_profit'] = st.slider("Take Profit (%)", 5, 30, 15) / 100
    
    # Run backtest button
    if st.button("🚀 เริ่มทดสอบ Backtest", type="primary"):
        with st.spinner("กำลังทดสอบกลยุทธ์..."):
            # Run backtest
            result, error = run_backtest(
                ticker=ticker_bt,
                strategy_name=strategy_name,
                start_date=start_date,
                end_date=end_date,
                initial_cash=initial_cash,
                **strategy_params
            )
            
            if error:
                st.error(f"❌ {error}")
            elif result:
                st.success("✅ การทดสอบเสร็จสิ้น!")
                
                # Display results
                st.subheader("📊 ผลการทดสอบ")
                
                # Metrics in columns
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Total Return",
                        f"{result['total_return']}%",
                        delta=f"{result['total_return']}%"
                    )
                    st.metric("เงินทุนเริ่มต้น", f"${result['start_value']:,.2f}")
                
                with col2:
                    st.metric(
                        "Sharpe Ratio",
                        f"{result['sharpe_ratio']}",
                        help="อัตราส่วนความเสี่ยงต่อผลตอบแทน"
                    )
                    st.metric("เงินทุนสุดท้าย", f"${result['end_value']:,.2f}")
                
                with col3:
                    st.metric(
                        "Max Drawdown",
                        f"{result['max_drawdown']}%",
                        delta=f"-{result['max_drawdown']}%",
                        help="การลดลงสูงสุดของเงินทุน"
                    )
                    st.metric("จำนวนการซื้อขายทั้งหมด", f"{result['total_trades']}")
                
                with col4:
                    st.metric(
                        "Win Rate",
                        f"{result['win_rate']}%",
                        help="อัตราการชนะ"
                    )
                    st.metric("การซื้อขายที่ชนะ", f"{result['won_trades']}")
                
                # Chart
                st.subheader("📈 กราฟแสดงผลการทดสอบ")
                fig = plot_backtest_results(result['data'], strategy_name, ticker_bt)
                st.pyplot(fig)
                
                # Strategy details
                st.subheader("⚙️ รายละเอียดกลยุทธ์")
                strategy_info = {
                    "หุ้น": ticker_bt,
                    "กลยุทธ์": strategy_name,
                    "ช่วงเวลาทดสอบ": f"{start_date} ถึง {end_date}",
                    "เงินทุนเริ่มต้น": f"${initial_cash:,.2f}"
                }
                
                for key, value in strategy_params.items():
                    if isinstance(value, float) and value < 1:
                        strategy_info[key] = f"{value*100}%"
                    else:
                        strategy_info[key] = value
                
                df_strategy = pd.DataFrame(list(strategy_info.items()), columns=['พารามิเตอร์', 'ค่า'])
                st.dataframe(df_strategy, hide_index=True)
    
    # Information about backtesting
    st.info("""
    💡 **คำแนะนำการใช้งาน Backtesting:**
    - เลือกหุ้นและช่วงเวลาที่ต้องการทดสอบ
    - ปรับแต่งพารามิเตอร์ของกลยุทธ์ตามความเหมาะสม
    - Sharpe Ratio > 1 ถือว่าดี, > 2 ถือว่าดีเยี่ยม
    - Max Drawdown ยิ่งต่ำยิ่งดี (ความเสี่ยงต่ำ)
    - Win Rate > 50% แสดงว่ากลยุทธ์มีประสิทธิภาพ
    - ผลการทดสอบในอดีตไม่รับประกันผลลัพธ์ในอนาคต
    """)
    st.stop()

# Gemini AI Settings in Sidebar
with st.sidebar.expander("🤖 Gemini AI Settings (Optional)", expanded=False):
    gemini_api_key = st.text_input(
        "Gemini API Key",
        type="password",
        value=st.session_state.get('gemini_api_key', ''),
        help="Enter your Google Gemini API key to enable AI analysis",
        placeholder="Enter your API key here..."
    )
    
    if gemini_api_key:
        st.session_state['gemini_api_key'] = gemini_api_key
        
        # Test API key and show status
        try:
            genai.configure(api_key=gemini_api_key)
            # Try to create a model to test the API key
            test_model = genai.GenerativeModel('gemini-1.5-flash')
            st.success("✅ API Key configured successfully!")
        except Exception as e:
            st.error(f"❌ API Key error: {str(e)}")
    
    # Model selection
    gemini_model = st.selectbox(
        "Model",
        options=["gemini-1.5-flash", "gemini-1.5-pro"],
        index=0,
        help="Select the Gemini model to use"
    )
    st.session_state['gemini_model'] = gemini_model
    
    # Temperature slider
    gemini_temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.5,
        value=0.7,
        step=0.1,
        help="Controls randomness in responses (0.0 = more focused, 1.5 = more creative)"
    )
    st.session_state['gemini_temperature'] = gemini_temperature

# Market selection
selected_market = st.selectbox(
    "เลือกตลาดหุ้น",
    options=list(markets.keys()),
    index=0,  # Default to US
    help="เลือกตลาดหุ้นที่ต้องการวิเคราะห์"
)

# Get available tickers based on selected market
available_tickers = markets[selected_market]

# Default tickers based on market
default_tickers = []
if selected_market == "US":
    default_tickers = ["AAPL"]
elif selected_market == "SET100":
    default_tickers = ["PTT.BK"]
elif selected_market == "Global":
    default_tickers = ["AAPL", "PTT.BK"]
else:
    # For other markets, select first ticker as default
    default_tickers = [available_tickers[0]] if available_tickers else []

tickers_selected = st.multiselect(
    f"เลือกหุ้น ({selected_market}) - หลายตัวได้",
    available_tickers,
    default=default_tickers,
    help=f"เลือกหุ้นจากตลาด {selected_market} ที่ต้องการวิเคราะห์ (สามารถเลือกหลายตัวได้)"
)

# Strategy selection
strategy_choice = st.selectbox(
    "เลือกกลยุทธ์การวิเคราะห์",
    ["DCA + Buffett Analysis", "Bollinger Bands Backtest", "Both"],
    index=0,
    help="เลือกวิธีการวิเคราะห์: DCA+Buffett (แบบเดิม) หรือ Bollinger Bands (ใหม่) หรือทั้งคู่"
)

period = st.selectbox("เลือกช่วงเวลาราคาหุ้น", ["1y", "5y", "max"], index=1)

# DCA settings
if strategy_choice in ["DCA + Buffett Analysis", "Both"]:
    monthly_invest = st.number_input("จำนวนเงินลงทุน DCA ต่อเดือน (บาทหรือ USD)", min_value=100.0, max_value=10000.0, value=1000.0, step=100.0)

# Bollinger Bands settings
if strategy_choice in ["Bollinger Bands Backtest", "Both"]:
    st.subheader("Bollinger Bands Strategy Settings")
    col1, col2 = st.columns(2)
    
    with col1:
        bb_period = st.number_input("BB Period", min_value=5, max_value=50, value=20, help="ช่วงเวลาสำหรับ Moving Average")
        bb_multiplier = st.number_input("BB Multiplier", min_value=1.0, max_value=3.0, value=2.0, step=0.1, help="ตัวคูณสำหรับ Standard Deviation")
        initial_capital = st.number_input("Initial Capital", min_value=1000.0, max_value=100000.0, value=10000.0, step=1000.0, help="เงินทุนเริ่มต้น")
    
    with col2:
        stop_loss_pct = st.number_input("Stop Loss (%)", min_value=1.0, max_value=20.0, value=5.0, step=0.5, help="Stop Loss เป็น %")
        take_profit_pct = st.number_input("Take Profit (%)", min_value=5.0, max_value=50.0, value=10.0, step=1.0, help="Take Profit เป็น %")
        transaction_cost_pct = st.number_input("Transaction Cost per side (%)", min_value=0.0, max_value=5.0, value=0.10, step=0.01, help="ค่าธรรมเนียมต่อฝั่ง เช่น 0.10 = 0.10%")

show_financials = st.checkbox("แสดงงบการเงิน (Income Statement)", value=False)

if st.button("วิเคราะห์"):
    export_list = []
    results_table = []
    summary_rows = []  # For summary sheet
    all_trades_data = {}  # For individual ticker trade sheets
    total_invest = 0
    total_profit = 0
    total_div = 0

    for ticker in tickers_selected:
        stock = yf.Ticker(ticker)
        fin = stock.financials
        bs = stock.balance_sheet
        cf = stock.cashflow
        div = stock.dividends
        hist = stock.history(period=period)
        info = stock.info

        manual_yield = np.nan
        total_div1y = np.nan

        # Get company name
        company_name = info.get('longName', ticker)
        company_symbol = ticker

        with st.expander(f"ดูรายละเอียดหุ้น {ticker} - {company_name}", expanded=False):
            st.subheader(f"ข้อมูลบริษัท: {company_name}")
            st.write(f"**สัญลักษณ์:** {company_symbol}")
            
            # Initialize results containers
            dca_result = None
            bb_result = None
            
            # Run DCA + Buffett Analysis if selected
            if strategy_choice in ["DCA + Buffett Analysis", "Both"]:
                st.subheader("ข้อมูลราคาหุ้นและปันผลล่าสุด")

                # 1. Dividend Yield (% ต่อปี)
                div_yield = info.get('dividendYield', None)
                div_yield_pct = round(div_yield * 100, 2) if div_yield is not None else "N/A"

                # 2. วันที่ปันผลล่าสุด (Ex-Dividend Date)
                ex_div = info.get('exDividendDate', None)
                if ex_div:
                    try:
                        ex_div_date = datetime.datetime.fromtimestamp(ex_div).strftime('%Y-%m-%d')
                    except Exception:
                        ex_div_date = str(ex_div)
                else:
                    ex_div_date = "N/A"

                # 3. 52-Week High / Low
                w52_high = info.get('fiftyTwoWeekHigh', "N/A")
                w52_low = info.get('fiftyTwoWeekLow', "N/A")

                # 4. ราคาปิดล่าสุด, ราคาเปิดล่าสุด
                last_close = info.get('previousClose', "N/A")
                last_open = info.get('open', "N/A")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Dividend Yield (%)", div_yield_pct)
                    st.metric("Ex-Dividend Date", ex_div_date)
                with col2:
                    st.metric("52W High", w52_high)
                    st.metric("52W Low", w52_low)
                with col3:
                    st.metric("ราคาปิดล่าสุด", last_close)
                    st.metric("ราคาเปิดล่าสุด", last_open)

                # --------- สรุปปันผลย้อนหลัง 1 ปี ---------
                st.subheader("ผลตอบแทนเงินปันผลย้อนหลัง 1 ปี (คำนวณจากราคาจริง)")
                if not div.empty and not hist.empty:
                    last_year = hist.index[-1] - pd.DateOffset(years=1)
                    recent_div = div[div.index >= last_year]
                    total_div1y = recent_div.sum()
                    avg_price1y = hist['Close'][hist.index >= last_year].mean()
                    price_base = avg_price1y if (avg_price1y and avg_price1y > 0) else hist['Close'].iloc[-1]
                    manual_yield = (total_div1y / price_base) * 100 if price_base > 0 else np.nan

                    st.markdown(f"""
                    <div style='font-size:1.1em;'>
                    <b>เงินปันผลรวม 1 ปี:</b> <span style='color:green'>{total_div1y:.2f}</span><br>
                    <b>ราคาเฉลี่ย 1 ปี:</b> <span style='color:blue'>{price_base:.2f}</span><br>
                    <b>อัตราผลตอบแทน (Dividend Yield):</b> <span style='color:red;font-size:1.3em'>{manual_yield:.2f}%</span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("ไม่มีข้อมูลปันผลย้อนหลัง (dividends) สำหรับหุ้นนี้")

                st.subheader("Buffett 11 Checklist (แบบละเอียด)")
                detail = buffett_11_checks_detail(fin, bs, cf, div, hist)
                badge = get_badge(detail['score_pct'])
                st.markdown(f"**คะแนนภาพรวม:** {detail['score']} / {detail['evaluated']} ({detail['score_pct']}%) &nbsp;&nbsp;|&nbsp;&nbsp;**ป้ายคะแนน:** {badge}")

                # ตารางรายละเอียดแต่ละข้อ
                df_detail = pd.DataFrame([
                    {
                        'ข้อ': i + 1,
                        'รายการ': d['title'],
                        'ผลลัพธ์': "✅ ผ่าน" if d['result'] == 1 else ("❌ ไม่ผ่าน" if d['result'] == 0 else "⚪ N/A"),
                        'คำอธิบาย': d['desc']
                    }
                    for i, d in enumerate(detail['details'])
                ])
                st.dataframe(df_detail, hide_index=True)

                st.subheader("DCA Simulation (จำลองลงทุนรายเดือน)")
                dca_result = dca_simulation(hist, monthly_invest, div)
                st.write(pd.DataFrame(dca_result, index=['สรุปผล']).T)

                # สะสมผลรวม DCA
                total_invest += dca_result["เงินลงทุนรวม"]
                total_profit += dca_result["กำไร/ขาดทุน"]
                total_div += dca_result["เงินปันผลรวม"]

            # Run Bollinger Bands Backtest if selected
            if strategy_choice in ["Bollinger Bands Backtest", "Both"]:
                st.subheader("Bollinger Bands Strategy Backtest")
                
                if not hist.empty:
                    bb_result = bollinger_bands_strategy(
                        hist, 
                        period=bb_period,
                        multiplier=bb_multiplier,
                        initial_capital=initial_capital,
                        stop_loss_pct=stop_loss_pct,
                        take_profit_pct=take_profit_pct,
                        transaction_cost_rate=transaction_cost_pct/100.0
                    )
                    
                    if "error" not in bb_result:
                        # Display Bollinger Bands results
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Initial Capital", f"${bb_result['initial_capital']:,.0f}")
                            st.metric("Final Value", f"${bb_result['final_value']:,.0f}")
                        with col2:
                            st.metric("Total Return", f"{bb_result['total_return']:.2f}%")
                            st.metric("Max Drawdown", f"{bb_result['max_drawdown']:.2f}%")
                        with col3:
                            st.metric("Total Trades", bb_result['total_trades'])
                            st.metric("Win Rate", f"{bb_result['win_rate']:.2f}%")
                        with col4:
                            st.metric("Winning Trades", bb_result['winning_trades'])
                            st.metric("Losing Trades", bb_result['losing_trades'])
                        
                        # Show trades table if there are trades
                        if bb_result['trades']:
                            st.subheader("Trade History")
                            trades_df = pd.DataFrame(bb_result['trades'])
                            st.dataframe(trades_df, hide_index=True)
                            
                            # Store trades for Excel export
                            sanitized_ticker = ticker.replace(".", "_").replace("-", "_")[:30]  # Sanitize sheet name
                            all_trades_data[sanitized_ticker] = trades_df
                        
                        # Add to summary
                        summary_rows.append({
                            'Ticker': ticker,
                            'Company': company_name,
                            'Strategy': 'Bollinger Bands',
                            'Initial Capital': bb_result['initial_capital'],
                            'Final Value': bb_result['final_value'],
                            'Return (%)': bb_result['total_return'],
                            'Total Trades': bb_result['total_trades'],
                            'Win Rate (%)': bb_result['win_rate'],
                            'Max Drawdown (%)': bb_result['max_drawdown']
                        })
                    else:
                        st.error(f"Error in Bollinger Bands strategy: {bb_result['error']}")
                else:
                    st.warning("ไม่มีข้อมูลราคาหุ้นสำหรับ Bollinger Bands strategy")

            if not hist.empty:
                st.line_chart(hist['Close'])
            else:
                st.warning("ไม่มีข้อมูลราคาหุ้น")

            if show_financials and fin is not None and not fin.empty:
                st.subheader("งบกำไรขาดทุน (Income Statement)")
                st.dataframe(df_human_format(fin))

            # 🤖 Gemini AI Analysis Section
            st.subheader("🤖 Gemini วิเคราะห์ (AI)")
            
            # Assemble context for AI analysis
            context_parts = []
            
            # Add Buffett checklist score if available
            if 'detail' in locals():
                context_parts.append(f"Buffett Checklist Score: {detail['score']}/{detail['evaluated']} ({detail['score_pct']}%)")
            
            # Add DCA summary if available
            if dca_result:
                context_parts.append(f"""DCA Summary:
- เงินลงทุนรวม: {dca_result['เงินลงทุนรวม']}
- มูลค่าปัจจุบัน: {dca_result['มูลค่าปัจจุบัน']} 
- กำไร/ขาดทุน: {dca_result['กำไร/ขาดทุน']} ({dca_result['กำไร(%)']:.2f}%)
- เงินปันผลรวม: {dca_result['เงินปันผลรวม']}""")
            
            # Add price and period info
            context_parts.append(f"ราคาปิดล่าสุด: {last_close}")
            context_parts.append(f"ช่วงเวลาวิเคราะห์: {period}")
            
            context_text = "\n".join(context_parts)
            
            # User question input
            default_prompt = f"วิเคราะห์หุ้น {ticker} ({company_name}) และให้คำแนะนำกลยุทธ์การลงทุนตามข้อมูลที่มี"
            user_question = st.text_area(
                "คำถามสำหรับ AI:",
                value=default_prompt,
                height=100,
                help="พิมพ์คำถามที่ต้องการให้ AI วิเคราะห์"
            )
            
            # Analysis button
            if st.button(f"วิเคราะห์ด้วย Gemini - {ticker}", type="primary"):
                with st.spinner("กำลังวิเคราะห์..."):
                    ai_response = gemini_insight(user_question, context_text)
                    st.markdown("**การวิเคราะห์จาก AI:**")
                    st.markdown(ai_response)

            # Prepare export data
            export_data = {
                "หุ้น": ticker,
                "ชื่อบริษัท": company_name,
            }
            
            if dca_result:
                export_data.update({
                    "DCA_เงินลงทุนรวม": dca_result["เงินลงทุนรวม"],
                    "DCA_จำนวนหุ้นสะสม": dca_result["จำนวนหุ้นสะสม"],
                    "DCA_มูลค่าปัจจุบัน": dca_result["มูลค่าปัจจุบัน"],
                    "DCA_กำไร/ขาดทุน": dca_result["กำไร/ขาดทุน"],
                    "DCA_กำไร(%)": dca_result["กำไร(%)"],
                    "DCA_เงินปันผลรวม": dca_result["เงินปันผลรวม"],
                    "Dividend Yield ย้อนหลัง 1 ปี (%)": manual_yield if not np.isnan(manual_yield) else "N/A",
                    "เงินปันผลย้อนหลัง 1 ปี": total_div1y if not np.isnan(total_div1y) else "N/A",
                })
                
                if 'detail' in locals():
                    export_data.update({
                        "Buffett_คะแนนรวม": f"{detail['score']}/{detail['evaluated']}",
                        "Buffett_เปอร์เซ็นต์": detail['score_pct'],
                        "Buffett_ป้ายคะแนน": badge,
                    })
            
            if bb_result and "error" not in bb_result:
                export_data.update({
                    "BB_Initial_Capital": bb_result['initial_capital'],
                    "BB_Final_Value": bb_result['final_value'],
                    "BB_Return(%)": bb_result['total_return'],
                    "BB_Total_Trades": bb_result['total_trades'],
                    "BB_Win_Rate(%)": bb_result['win_rate'],
                    "BB_Max_Drawdown(%)": bb_result['max_drawdown'],
                })
            
            export_list.append(export_data)
            
            # Add to results table for DCA summary
            if dca_result:
                results_table.append({
                    "หุ้น": ticker,
                    "ชื่อบริษัท": company_name,
                    "เงินลงทุน": dca_result["เงินลงทุนรวม"],
                    "กำไร": dca_result["กำไร/ขาดทุน"],
                    "เงินปันผล": dca_result["เงินปันผลรวม"],
                    "มูลค่าปัจจุบัน": dca_result["มูลค่าปัจจุบัน"]
                })

    # --- Enhanced Excel Export with Multiple Sheets ---
    if len(export_list) > 0:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            # Main data sheet
            df_export = pd.DataFrame(export_list)
            df_export.to_excel(writer, index=False, sheet_name='WarrenDCA_All_Data')
            
            # Summary sheet for Bollinger Bands
            if summary_rows:
                summary_df = pd.DataFrame(summary_rows)
                summary_df.to_excel(writer, index=False, sheet_name='Summary')
            
            # Individual ticker trade sheets
            for ticker_name, trades_df in all_trades_data.items():
                trades_df.to_excel(writer, index=False, sheet_name=f'Trades_{ticker_name}')
        
        st.download_button(
            label="📥 ดาวน์โหลดผลลัพธ์เป็น Excel (หลาย Sheets)",
            data=output.getvalue(),
            file_name='WarrenDCA_MultiStrategy_Result.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )

    # Display summary table for Bollinger Bands
    if summary_rows:
        st.header("สรุปผล Bollinger Bands Strategy")
        summary_df = pd.DataFrame(summary_rows)
        st.dataframe(summary_df, hide_index=True)

    # --- สรุปผลหุ้นที่เลือก DCA Simulator รวม ---
    if strategy_choice in ["DCA + Buffett Analysis", "Both"] and results_table:
        st.header("สรุปผลรวมหุ้นที่เลือก (DCA Simulator)")
        st.write(f"💰 เงินลงทุนรวม: {total_invest:.2f}")
        st.write(f"📈 กำไรรวม: {total_profit:.2f}")
        st.write(f"💵 เงินปันผลรวมที่ได้รับ: {total_div:.2f}")

        # ตารางสรุปหุ้นที่เลือก
        if results_table:
            st.subheader("ตารางสรุปรวม (แต่ละหุ้น)")
            st.dataframe(pd.DataFrame(results_table))

        # --- Pie Chart ---
        pie_labels = ["TOTAL INVEST", "TOTAL PROFIT", "div"]
        pie_values = [total_invest, total_profit if total_profit > 0 else 0, total_div]
        fig, ax = plt.subplots()
        colors = ['#2196f3', '#4caf50', '#ffc107']
        ax.pie(pie_values, labels=pie_labels, autopct='%1.1f%%', startangle=90, colors=colors)
        ax.set_title("INVEST/Profit/DivyYield")
        st.pyplot(fig)

st.caption("Powered by Yahoo Finance | วิเคราะห์หุ้นด้วย Buffett Checklist (ขยาย 18 เงื่อนไข) + DCA + ปันผลย้อนหลัง 1 ปี")