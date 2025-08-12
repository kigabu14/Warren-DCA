import streamlit as st
import yfinance as yf
import pandas as pd
import io
import datetime
import numpy as np
import matplotlib.pyplot as plt
import re
from typing import Dict, List, Tuple, Optional
import time

# AI imports (optional)
try:
    import google.generativeai as genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

# Sentiment analysis imports (optional)
try:
    from nltk.sentiment import SentimentIntensityAnalyzer
    import nltk
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False

# ----------------- Helper Functions -----------------
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

# ----------------- Technical Indicators -----------------
@st.cache_data
def compute_indicators(df: pd.DataFrame, sma_short: int = 20, sma_long: int = 50, 
                      rsi_period: int = 14, bb_window: int = 20, bb_stddev: float = 2.0) -> pd.DataFrame:
    """Compute technical indicators for the given price data"""
    df = df.copy()
    
    # Simple Moving Averages
    df[f'SMA_{sma_short}'] = df['Close'].rolling(window=sma_short).mean()
    df[f'SMA_{sma_long}'] = df['Close'].rolling(window=sma_long).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=bb_window).mean()
    bb_std = df['Close'].rolling(window=bb_window).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * bb_stddev)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * bb_stddev)
    
    return df

# ----------------- Sentiment Analysis -----------------
def get_simple_sentiment(text: str) -> float:
    """Simple rule-based sentiment analysis as fallback"""
    if not text:
        return 0.0
    
    positive_words = ['good', 'great', 'excellent', 'positive', 'up', 'gain', 'profit', 'bull', 'rise', 'high', 'strong',
                     'ดี', 'ดีเยี่ยม', 'เติบโต', 'กำไร', 'ขึ้น', 'สูง', 'แข็งแกร่ง', 'บวก']
    negative_words = ['bad', 'terrible', 'negative', 'down', 'loss', 'bear', 'fall', 'low', 'weak', 'decline',
                     'แย่', 'ลด', 'ลง', 'ขาดทุน', 'ต่ำ', 'อ่อนแอ', 'ลบ']
    
    text_lower = text.lower()
    pos_count = sum(1 for word in positive_words if word in text_lower)
    neg_count = sum(1 for word in negative_words if word in text_lower)
    
    if pos_count == 0 and neg_count == 0:
        return 0.0
    return (pos_count - neg_count) / (pos_count + neg_count)

def analyze_sentiment(text: str) -> float:
    """Analyze sentiment using NLTK VADER if available, otherwise use simple fallback"""
    if HAS_NLTK:
        try:
            analyzer = SentimentIntensityAnalyzer()
            score = analyzer.polarity_scores(text)
            return score['compound']
        except:
            pass
    
    return get_simple_sentiment(text)

# ----------------- AI Integration -----------------
def test_gemini_connection(api_key: str) -> Tuple[bool, str]:
    """Test Gemini API connection"""
    if not HAS_GENAI:
        return False, "google-generativeai library ไม่พร้อมใช้งาน"
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content("Hello.")
        return True, "✅ การเชื่อมต่อสำเร็จ"
    except Exception as e:
        return False, f"❌ การเชื่อมต่อล้มเหลว: {str(e)}"

def get_ai_insights(api_key: str, prompt: str) -> str:
    """Get AI insights using Gemini"""
    if not HAS_GENAI or not api_key:
        return "โหมด AI ไม่พร้อมใช้งาน - กรุณาใส่ Gemini API Key"
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"เกิดข้อผิดพลาดใน AI: {str(e)}"

# ----------------- Backtesting Functions -----------------
def backtest_buy_hold(df: pd.DataFrame) -> pd.DataFrame:
    """Buy and hold strategy backtest"""
    if df.empty:
        return pd.DataFrame()
    
    trades = pd.DataFrame({
        'Date': [df.index[0], df.index[-1]],
        'Action': ['BUY', 'SELL'],
        'Price': [df['Close'].iloc[0], df['Close'].iloc[-1]],
        'Position': [1, 0]
    })
    trades.set_index('Date', inplace=True)
    return trades

def backtest_sma_crossover(df: pd.DataFrame, short_period: int = 20, long_period: int = 50) -> pd.DataFrame:
    """SMA crossover strategy backtest"""
    df_with_sma = compute_indicators(df, sma_short=short_period, sma_long=long_period)
    
    # Generate signals
    df_with_sma['Signal'] = 0
    df_with_sma.loc[df_with_sma[f'SMA_{short_period}'] > df_with_sma[f'SMA_{long_period}'], 'Signal'] = 1
    df_with_sma['Position'] = df_with_sma['Signal'].diff()
    
    # Extract trades
    buy_signals = df_with_sma[df_with_sma['Position'] == 1]
    sell_signals = df_with_sma[df_with_sma['Position'] == -1]
    
    trades = []
    for buy_date in buy_signals.index:
        trades.append({'Date': buy_date, 'Action': 'BUY', 'Price': df_with_sma.loc[buy_date, 'Close'], 'Position': 1})
        
        # Find next sell signal
        future_sells = sell_signals[sell_signals.index > buy_date]
        if not future_sells.empty:
            sell_date = future_sells.index[0]
            trades.append({'Date': sell_date, 'Action': 'SELL', 'Price': df_with_sma.loc[sell_date, 'Close'], 'Position': 0})
    
    if trades:
        trades_df = pd.DataFrame(trades)
        trades_df.set_index('Date', inplace=True)
        return trades_df
    else:
        return pd.DataFrame()

def backtest_rsi(df: pd.DataFrame, period: int = 14, oversold: float = 30, overbought: float = 70) -> pd.DataFrame:
    """RSI strategy backtest"""
    df_with_rsi = compute_indicators(df, rsi_period=period)
    
    trades = []
    position = 0
    
    for i in range(1, len(df_with_rsi)):
        current_rsi = df_with_rsi['RSI'].iloc[i]
        prev_rsi = df_with_rsi['RSI'].iloc[i-1]
        date = df_with_rsi.index[i]
        price = df_with_rsi['Close'].iloc[i]
        
        # Buy signal: RSI crosses above oversold level
        if position == 0 and prev_rsi <= oversold and current_rsi > oversold:
            trades.append({'Date': date, 'Action': 'BUY', 'Price': price, 'Position': 1})
            position = 1
        
        # Sell signal: RSI crosses below overbought level
        elif position == 1 and prev_rsi >= overbought and current_rsi < overbought:
            trades.append({'Date': date, 'Action': 'SELL', 'Price': price, 'Position': 0})
            position = 0
    
    if trades:
        trades_df = pd.DataFrame(trades)
        trades_df.set_index('Date', inplace=True)
        return trades_df
    else:
        return pd.DataFrame()

def backtest_bollinger(df: pd.DataFrame, window: int = 20, stddev: float = 2.0) -> pd.DataFrame:
    """Bollinger Bands mean reversion strategy backtest"""
    df_with_bb = compute_indicators(df, bb_window=window, bb_stddev=stddev)
    
    trades = []
    position = 0
    
    for i in range(1, len(df_with_bb)):
        price = df_with_bb['Close'].iloc[i]
        lower_band = df_with_bb['BB_Lower'].iloc[i]
        middle_band = df_with_bb['BB_Middle'].iloc[i]
        date = df_with_bb.index[i]
        
        # Buy signal: price touches lower band
        if position == 0 and price <= lower_band:
            trades.append({'Date': date, 'Action': 'BUY', 'Price': price, 'Position': 1})
            position = 1
        
        # Sell signal: price crosses middle band
        elif position == 1 and price >= middle_band:
            trades.append({'Date': date, 'Action': 'SELL', 'Price': price, 'Position': 0})
            position = 0
    
    if trades:
        trades_df = pd.DataFrame(trades)
        trades_df.set_index('Date', inplace=True)
        return trades_df
    else:
        return pd.DataFrame()

def calc_metrics(trades_df: pd.DataFrame, hist_df: pd.DataFrame, initial_capital: float = 10000) -> Dict:
    """Calculate trading strategy metrics"""
    if trades_df.empty or hist_df.empty:
        return {}
    
    capital = initial_capital
    equity_curve = []
    trade_returns = []
    position = 0
    buy_price = 0
    
    # Calculate equity curve
    for date in hist_df.index:
        if date in trades_df.index:
            trade = trades_df.loc[date]
            if trade['Action'] == 'BUY':
                buy_price = trade['Price']
                position = capital / buy_price
                capital = 0
            elif trade['Action'] == 'SELL':
                capital = position * trade['Price']
                trade_return = (trade['Price'] - buy_price) / buy_price
                trade_returns.append(trade_return)
                position = 0
        
        # Current equity
        if position > 0:
            current_equity = position * hist_df.loc[date, 'Close']
        else:
            current_equity = capital
        
        equity_curve.append(current_equity)
    
    if not equity_curve:
        return {}
    
    # Calculate metrics
    final_equity = equity_curve[-1]
    total_return = (final_equity - initial_capital) / initial_capital
    
    # Calculate daily returns for volatility and Sharpe
    equity_series = pd.Series(equity_curve, index=hist_df.index)
    daily_returns = equity_series.pct_change().dropna()
    
    if len(daily_returns) > 1:
        volatility = daily_returns.std() * np.sqrt(252)  # Annualized
        sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0
    else:
        volatility = 0
        sharpe = 0
    
    # Calculate max drawdown
    peak = equity_series.expanding().max()
    drawdown = (equity_series - peak) / peak
    max_drawdown = abs(drawdown.min())
    
    # CAGR calculation
    years = len(hist_df) / 252
    cagr = (final_equity / initial_capital) ** (1/years) - 1 if years > 0 else 0
    
    # Other metrics
    win_rate = len([r for r in trade_returns if r > 0]) / len(trade_returns) if trade_returns else 0
    avg_trade_return = np.mean(trade_returns) if trade_returns else 0
    
    return {
        'CAGR': cagr,
        'Total Return (%)': total_return * 100,
        'Max Drawdown (%)': max_drawdown * 100,
        'Volatility (%)': volatility * 100,
        'Sharpe Ratio': sharpe,
        'Win Rate (%)': win_rate * 100,
        'Number of Trades': len(trade_returns),
        'Avg Trade Return (%)': avg_trade_return * 100,
        'Final Equity': final_equity,
        'Equity Curve': equity_series
    }

# ----------------- News Fetching with Caching -----------------
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_stock_news(ticker: str) -> List[Dict]:
    """Get news for a specific ticker with caching"""
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        return news if news else []
    except:
        return []

@st.cache_data(ttl=1800)  # Cache for 30 minutes  
def get_stock_data_cached(ticker: str, period: str = "1y") -> Tuple[pd.DataFrame, dict]:
    """Get stock data with caching"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        info = stock.info
        return hist, info
    except:
        return pd.DataFrame(), {}
    """คำนวณ Dividend Yield จากเงินปันผลที่ได้รับจริงย้อนหลัง 1 ปี"""
    if not div.empty and not hist.empty:
        last_year = hist.index[-1] - pd.DateOffset(years=1)
        recent_div = div[div.index >= last_year]
        total_div = recent_div.sum()
        avg_price = hist['Close'][hist.index >= last_year].mean()
        manual_yield = (total_div / avg_price) * 100 if avg_price > 0 else np.nan
        return total_div, avg_price, manual_yield
    return 0, 0, np.nan

def calc_dividend_yield_manual(div, hist):
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

# Sidebar for AI mode and navigation
st.sidebar.header("⚙️ การตั้งค่า")

# Initialize session state for AI mode
if 'use_ai' not in st.session_state:
    st.session_state['use_ai'] = False

# AI Mode Toggle
use_ai = st.sidebar.checkbox("เปิดโหมด AI ขั้นสูง", value=st.session_state['use_ai'], help="เปิดใช้งานฟีเจอร์ AI สำหรับการวิเคราะห์ขั้นสูง")
st.session_state['use_ai'] = use_ai

gemini_api_key = None
if st.session_state['use_ai']:
    st.sidebar.subheader("🤖 การตั้งค่า AI")
    gemini_api_key = st.sidebar.text_input(
        "Gemini API Key", 
        type="password", 
        help="ใส่ Gemini API Key ของคุณ (จะไม่ถูกบันทึกถาวร)"
    )
    
    if gemini_api_key:
        if st.sidebar.button("ทดสอบการเชื่อมต่อ"):
            with st.sidebar.spinner("กำลังทดสอบการเชื่อมต่อ..."):
                success, message = test_gemini_connection(gemini_api_key)
                if success:
                    st.sidebar.success(message)
                else:
                    st.sidebar.error(message)

# Main navigation
menu = st.sidebar.radio(
    "เลือกหน้าที่ต้องการ", 
    ["วิเคราะห์หุ้น", "ข่าว & Sentiment", "กลยุทธ์ & Backtest", "คู่มือการใช้งาน"],
    help="เลือกหน้าการใช้งานที่ต้องการ"
)

if menu == "ข่าว & Sentiment":
    st.header("📰 ข่าวและการวิเคราะห์ Sentiment")
    
    # Market and ticker selection
    selected_market = st.selectbox(
        "เลือกตลาดหุ้น",
        options=list(markets.keys()),
        index=0
    )
    available_tickers = markets[selected_market]
    
    selected_ticker = st.selectbox(
        f"เลือกหุ้น ({selected_market})",
        available_tickers,
        index=0 if available_tickers else None
    )
    
    if selected_ticker and st.button("ดึงข่าวและวิเคราะห์"):
        with st.spinner("กำลังดึงข้อมูลข่าว..."):
            try:
                # Get news using yfinance
                stock = yf.Ticker(selected_ticker)
                news = stock.news
                
                if not news:
                    st.warning("ไม่พบข่าวสำหรับหุ้นนี้")
                else:
                    st.success(f"พบข่าว {len(news)} ข่าว สำหรับ {selected_ticker}")
                    
                    # Analyze sentiment for each news item
                    news_data = []
                    total_sentiment = 0
                    positive_count = 0
                    negative_count = 0
                    neutral_count = 0
                    
                    for article in news:
                        title = article.get('title', '')
                        publisher = article.get('publisher', '')
                        published_time = article.get('publishedAt', '')
                        link = article.get('link', '')
                        
                        # Convert timestamp if needed
                        if isinstance(published_time, int):
                            published_time = datetime.datetime.fromtimestamp(published_time).strftime('%Y-%m-%d %H:%M')
                        
                        # Analyze sentiment
                        sentiment_score = analyze_sentiment(title)
                        total_sentiment += sentiment_score
                        
                        if sentiment_score > 0.1:
                            positive_count += 1
                            sentiment_label = "😊 บวก"
                        elif sentiment_score < -0.1:
                            negative_count += 1
                            sentiment_label = "😟 ลบ"
                        else:
                            neutral_count += 1
                            sentiment_label = "😐 เป็นกลาง"
                        
                        news_data.append({
                            'วันที่': published_time,
                            'หัวข้อ': title,
                            'สำนักข่าว': publisher,
                            'Sentiment': sentiment_label,
                            'คะแนน': round(sentiment_score, 3),
                            'ลิงก์': link
                        })
                    
                    # Display news table
                    st.subheader("📋 รายการข่าว")
                    df_news = pd.DataFrame(news_data)
                    st.dataframe(df_news, use_container_width=True)
                    
                    # Sentiment summary
                    st.subheader("📊 สรุป Sentiment")
                    avg_sentiment = total_sentiment / len(news) if news else 0
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("😊 ข่าวบวก", positive_count)
                    with col2:
                        st.metric("😟 ข่าวลบ", negative_count)
                    with col3:
                        st.metric("😐 ข่าวเป็นกลาง", neutral_count)
                    with col4:
                        st.metric("📈 Sentiment เฉลี่ย", f"{avg_sentiment:.3f}")
                    
                    # Get current stock price movement
                    hist = stock.history(period="1d")
                    if not hist.empty:
                        current_price = hist['Close'].iloc[-1]
                        open_price = hist['Open'].iloc[-1]
                        price_change = ((current_price - open_price) / open_price) * 100
                        
                        st.subheader("💹 การเคลื่อนไหวราคาวันนี้")
                        price_color = "🟢" if price_change >= 0 else "🔴"
                        st.write(f"{price_color} เปลี่ยนแปลง: {price_change:.2f}%")
                    
                    # AI Analysis if enabled
                    if st.session_state['use_ai'] and gemini_api_key:
                        st.subheader("🤖 การวิเคราะห์ AI")
                        
                        # Get Buffett score for context
                        fin = stock.financials
                        bs = stock.balance_sheet
                        cf = stock.cashflow
                        div = stock.dividends
                        hist_full = stock.history(period="1y")
                        
                        buffett_detail = buffett_11_checks_detail(fin, bs, cf, div, hist_full)
                        buffett_score = buffett_detail['score_pct']
                        
                        ai_prompt = f"""
                        วิเคราะห์หุ้น {selected_ticker} ในภาษาไทยอย่างสั้นกระชับ:
                        
                        ข้อมูลพื้นฐาน:
                        - คะแนน Buffett: {buffett_score}%
                        - การเปลี่ยนแปลงราคาวันนี้: {price_change:.2f}%
                        
                        ข้อมูล Sentiment จากข่าว:
                        - ข่าวบวก: {positive_count} ข่าว
                        - ข่าวลบ: {negative_count} ข่าว  
                        - ข่าวเป็นกลาง: {neutral_count} ข่าว
                        - Sentiment เฉลี่ย: {avg_sentiment:.3f}
                        
                        กรุณาให้การวิเคราะห์สั้นๆ ภายใน 3-4 ประโยค เกี่ยวกับโอกาสและความเสี่ยง
                        """
                        
                        with st.spinner("AI กำลังวิเคราะห์..."):
                            ai_analysis = get_ai_insights(gemini_api_key, ai_prompt)
                            st.write(ai_analysis)
                    elif st.session_state['use_ai'] and not gemini_api_key:
                        st.info("💡 ใส่ Gemini API Key ในแถบด้านข้างเพื่อดูการวิเคราะห์ AI")
                    
                    # Export functionality
                    if st.button("📥 ส่งออกข้อมูลข่าว"):
                        # Prepare export data
                        export_news = pd.DataFrame([{
                            'Ticker': selected_ticker,
                            'Date': item['วันที่'],
                            'Title': item['หัวข้อ'],
                            'Publisher': item['สำนักข่าว'],
                            'SentimentScore': item['คะแนน'],
                            'URL': item['ลิงก์']
                        } for item in news_data])
                        
                        export_summary = pd.DataFrame([{
                            'Ticker': selected_ticker,
                            'Total_News': len(news),
                            'Positive_Count': positive_count,
                            'Negative_Count': negative_count,
                            'Neutral_Count': neutral_count,
                            'Avg_Sentiment': avg_sentiment,
                            'Analysis_Date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        }])
                        
                        output = io.BytesIO()
                        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                            export_news.to_excel(writer, sheet_name='News_Details', index=False)
                            export_summary.to_excel(writer, sheet_name='Sentiment_Summary', index=False)
                        
                        st.download_button(
                            label="📥 ดาวน์โหลดข้อมูลข่าว Excel",
                            data=output.getvalue(),
                            file_name=f'News_Sentiment_{selected_ticker}_{datetime.datetime.now().strftime("%Y%m%d")}.xlsx',
                            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                        )
                        
            except Exception as e:
                st.error(f"เกิดข้อผิดพลาด: {str(e)}")

elif menu == "กลยุทธ์ & Backtest":
    st.header("📊 กลยุทธ์การลงทุนและ Backtest")
    
    # Market and ticker selection
    selected_market = st.selectbox(
        "เลือกตลาดหุ้น",
        options=list(markets.keys()),
        index=0
    )
    available_tickers = markets[selected_market]
    
    selected_ticker = st.selectbox(
        f"เลือกหุ้น ({selected_market})",
        available_tickers,
        index=0 if available_tickers else None
    )
    
    period = st.selectbox("ช่วงเวลาข้อมูล", ["1y", "2y", "5y", "max"], index=1)
    
    if selected_ticker:
        # Strategy selection
        st.subheader("🎯 เลือกกลยุทธ์")
        strategy = st.selectbox(
            "กลยุทธ์การลงทุน",
            ["Buy & Hold", "SMA Crossover", "RSI Strategy", "Bollinger Band Mean Reversion", "DCA Monthly"],
            help="เลือกกลยุทธ์ที่ต้องการทดสอบ"
        )
        
        # Strategy parameters
        col1, col2 = st.columns(2)
        
        if strategy == "SMA Crossover":
            with col1:
                short_period = st.slider("SMA ระยะสั้น", 5, 50, 20)
            with col2:
                long_period = st.slider("SMA ระยะยาว", 20, 200, 50)
        
        elif strategy == "RSI Strategy":
            with col1:
                rsi_period = st.slider("RSI Period", 7, 30, 14)
                oversold = st.slider("Oversold Level", 10, 40, 30)
            with col2:
                overbought = st.slider("Overbought Level", 60, 90, 70)
        
        elif strategy == "Bollinger Band Mean Reversion":
            with col1:
                bb_window = st.slider("Bollinger Period", 10, 50, 20)
            with col2:
                bb_stddev = st.slider("Standard Deviation", 1.0, 3.0, 2.0, 0.1)
        
        elif strategy == "DCA Monthly":
            with col1:
                monthly_invest = st.number_input("เงินลงทุนรายเดือน", min_value=100.0, value=1000.0)
        
        if st.button("🚀 ทดสอบกลยุทธ์"):
            with st.spinner("กำลังดึงข้อมูลและทดสอบกลยุทธ์..."):
                try:
                    # Get stock data
                    stock = yf.Ticker(selected_ticker)
                    hist = stock.history(period=period)
                    
                    if hist.empty:
                        st.error("ไม่พบข้อมูลราคาหุ้น")
                    else:
                        # Run backtest based on selected strategy
                        if strategy == "Buy & Hold":
                            trades = backtest_buy_hold(hist)
                        elif strategy == "SMA Crossover":
                            trades = backtest_sma_crossover(hist, short_period, long_period)
                        elif strategy == "RSI Strategy":
                            trades = backtest_rsi(hist, rsi_period, oversold, overbought)
                        elif strategy == "Bollinger Band Mean Reversion":
                            trades = backtest_bollinger(hist, bb_window, bb_stddev)
                        elif strategy == "DCA Monthly":
                            # Use existing DCA simulation
                            div = stock.dividends
                            dca_result = dca_simulation(hist, monthly_invest, div)
                            
                            st.subheader("📈 ผลลัพธ์ DCA")
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("เงินลงทุนรวม", f"{dca_result['เงินลงทุนรวม']:,.0f}")
                            with col2:
                                st.metric("กำไร/ขาดทุน", f"{dca_result['กำไร/ขาดทุน']:,.0f}")
                            with col3:
                                st.metric("กำไร (%)", f"{dca_result['กำไร(%)']:.2f}%")
                            with col4:
                                st.metric("เงินปันผลรวม", f"{dca_result['เงินปันผลรวม']:,.0f}")
                            
                            # Show price chart
                            st.subheader("📊 กราฟราคาหุ้น")
                            st.line_chart(hist['Close'])
                            
                            trades = pd.DataFrame()  # Empty for DCA
                        
                        # Calculate and display metrics for non-DCA strategies
                        if not trades.empty and strategy != "DCA Monthly":
                            metrics = calc_metrics(trades, hist)
                            
                            if metrics:
                                st.subheader("📊 ผลลัพธ์การทดสอบ")
                                
                                # Display key metrics
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("CAGR", f"{metrics['CAGR']*100:.2f}%")
                                    st.metric("Total Return", f"{metrics['Total Return (%)']:.2f}%")
                                with col2:
                                    st.metric("Max Drawdown", f"{metrics['Max Drawdown (%)']:.2f}%")
                                    st.metric("Volatility", f"{metrics['Volatility (%)']:.2f}%")
                                with col3:
                                    st.metric("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.2f}")
                                    st.metric("Win Rate", f"{metrics['Win Rate (%)']:.1f}%")
                                with col4:
                                    st.metric("Number of Trades", metrics['Number of Trades'])
                                    st.metric("Avg Trade Return", f"{metrics['Avg Trade Return (%)']:.2f}%")
                                
                                # Equity curve chart
                                st.subheader("📈 เส้นทางมูลค่าพอร์ต")
                                if 'Equity Curve' in metrics:
                                    fig, ax = plt.subplots(figsize=(12, 6))
                                    ax.plot(metrics['Equity Curve'].index, metrics['Equity Curve'].values, label='Strategy Equity')
                                    ax.plot(hist.index, hist['Close'] / hist['Close'].iloc[0] * 10000, label='Buy & Hold', alpha=0.7)
                                    ax.set_ylabel('มูลค่า ($)')
                                    ax.set_title(f'Equity Curve: {strategy}')
                                    ax.legend()
                                    st.pyplot(fig)
                                
                                # Trades table
                                if not trades.empty:
                                    st.subheader("📋 รายการการซื้อขาย")
                                    trades_display = trades.copy()
                                    trades_display['Price'] = trades_display['Price'].round(2)
                                    st.dataframe(trades_display)
                                
                                # AI insights if enabled
                                if st.session_state['use_ai'] and gemini_api_key:
                                    st.subheader("🤖 การวิเคราะห์ AI")
                                    
                                    ai_prompt = f"""
                                    วิเคราะห์ผลลัพธ์การทดสอบกลยุทธ์ {strategy} สำหรับหุ้น {selected_ticker} ในภาษาไทย:
                                    
                                    ผลลัพธ์:
                                    - CAGR: {metrics['CAGR']*100:.2f}%
                                    - Total Return: {metrics['Total Return (%)']:.2f}%
                                    - Max Drawdown: {metrics['Max Drawdown (%)']:.2f}%
                                    - Sharpe Ratio: {metrics['Sharpe Ratio']:.2f}
                                    - Win Rate: {metrics['Win Rate (%)']:.1f}%
                                    - จำนวนการซื้อขาย: {metrics['Number of Trades']}
                                    
                                    กรุณาวิเคราะห์:
                                    1. จุดแข็งและจุดอ่อนของกลยุทธ์นี้
                                    2. ความเสี่ยงที่ควรระวัง
                                    3. ข้อเสนอแนะสำหรับการปรับปรุงพารามิเตอร์
                                    
                                    ตอบภายใน 4-5 ประโยค
                                    """
                                    
                                    with st.spinner("AI กำลังวิเคราะห์ผลลัพธ์..."):
                                        ai_analysis = get_ai_insights(gemini_api_key, ai_prompt)
                                        st.write(ai_analysis)
                                
                                # Export functionality
                                if st.button("📥 ส่งออกผลลัพธ์"):
                                    # Prepare export data
                                    metrics_df = pd.DataFrame([{
                                        'Ticker': selected_ticker,
                                        'Strategy': strategy,
                                        'Period': period,
                                        'CAGR': metrics['CAGR'],
                                        'Total_Return_Pct': metrics['Total Return (%)'],
                                        'Max_Drawdown_Pct': metrics['Max Drawdown (%)'],
                                        'Volatility_Pct': metrics['Volatility (%)'],
                                        'Sharpe_Ratio': metrics['Sharpe Ratio'],
                                        'Win_Rate_Pct': metrics['Win Rate (%)'],
                                        'Number_of_Trades': metrics['Number of Trades'],
                                        'Avg_Trade_Return_Pct': metrics['Avg Trade Return (%)'],
                                        'Final_Equity': metrics['Final Equity']
                                    }])
                                    
                                    output = io.BytesIO()
                                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                                        metrics_df.to_excel(writer, sheet_name='Backtest_Results', index=False)
                                        if not trades.empty:
                                            trades.to_excel(writer, sheet_name='Backtest_Trades')
                                    
                                    st.download_button(
                                        label="📥 ดาวน์โหลดผลลัพธ์ Excel",
                                        data=output.getvalue(),
                                        file_name=f'Backtest_{strategy}_{selected_ticker}_{datetime.datetime.now().strftime("%Y%m%d")}.xlsx',
                                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                                    )
                            else:
                                st.warning("ไม่สามารถคำนวณ metrics ได้")
                        elif strategy != "DCA Monthly":
                            st.warning("ไม่พบสัญญาณการซื้อขายจากกลยุทธ์นี้ในช่วงเวลาที่เลือก")
                            
                except Exception as e:
                    st.error(f"เกิดข้อผิดพลาด: {str(e)}")

elif menu == "คู่มือการใช้งาน":
    st.header("คู่มือการใช้งาน (ภาษาไทย)")
    st.markdown("""
**Warren-DCA คืออะไร?**  
โปรแกรมนี้ช่วยวิเคราะห์หุ้นตามแนวทางของ Warren Buffett (Buffett 11 Checklist แบบขยาย 18 เงื่อนไข) พร้อมจำลองการลงทุนแบบ DCA, วิเคราะห์ข่าว & sentiment, และทดสอบกลยุทธ์การลงทุนหลากหลาย  
**แหล่งข้อมูล:** Yahoo Finance

### ฟีเจอร์หลัก
1. **วิเคราะห์หุ้น** - การวิเคราะห์หุ้นตามหลัก Warren Buffett พร้อม DCA simulation
2. **ข่าว & Sentiment** - ดึงข่าวล่าสุดและวิเคราะห์ sentiment อัตโนมัติ  
3. **กลยุทธ์ & Backtest** - ทดสอบกลยุทธ์การลงทุนต่างๆ ย้อนหลัง
4. **โหมด AI ขั้นสูง** - วิเคราะห์ข้อมูลด้วย AI (ต้องใส่ Gemini API Key)

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

### กลยุทธ์การลงทุนที่รองรับ
1. **Buy & Hold** - ซื้อและถือระยะยาว
2. **SMA Crossover** - การใช้เส้น Moving Average ตัดกัน
3. **RSI Strategy** - ซื้อขายตาม RSI oversold/overbought
4. **Bollinger Band Mean Reversion** - เก็งกำไรจากการกลับตัวของราคา
5. **DCA Monthly** - การลงทุนสม่ำเสมอรายเดือน

### การใช้งานโหมด AI
1. เปิด "เปิดโหมด AI ขั้นสูง" ในแถบด้านข้าง
2. ใส่ Gemini API Key (ไม่ถูกบันทึกถาวร)
3. ทดสอบการเชื่อมต่อด้วยปุ่ม "ทดสอบการเชื่อมต่อ"
4. AI จะแสดงการวิเคราะห์ขั้นสูงในทุกหน้า

### การวิเคราะห์ Sentiment
- ดึงข่าวล่าสุดจาก Yahoo Finance
- วิเคราะห์ sentiment ด้วย NLTK VADER (ถ้ามี) หรือ rule-based fallback
- แสดงสรุป sentiment เป็นบวก/ลบ/เป็นกลาง
- รวมกับการเคลื่อนไหวราคาและคะแนน Buffett สำหรับการวิเคราะห์ AI

### หมายเหตุ
- ข้อมูลหุ้น US มักครบถ้วนกว่าหุ้นไทย
- ถ้าข้อมูลสำคัญไม่ครบ บางข้อจะขึ้น N/A
- ใช้งบการเงินย้อนหลัง (Annual) ตามที่ Yahoo ให้ (ปกติ 4 ปี)
- รองรับหุ้นจากตลาดทั่วโลก: US, SET100, Europe, Asia, Australia
- การทดสอบกลยุทธ์ใช้ราคา close รายวัน และไม่รวมค่าธรรมเนียม
- ข้อมูลข่าวขึ้นอยู่กับ Yahoo Finance API
""")
    st.stop()

if menu == "วิเคราะห์หุ้น":
    st.header("📊 วิเคราะห์หุ้นตามหลัก Warren Buffett")
    
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

    tickers = st.multiselect(
        f"เลือกหุ้น ({selected_market})",
        available_tickers,
        default=default_tickers,
        help=f"เลือกหุ้นจากตลาด {selected_market} ที่ต้องการวิเคราะห์"
    )
    period = st.selectbox("เลือกช่วงเวลาราคาหุ้น", ["1y", "5y", "max"], index=1)
    monthly_invest = st.number_input("จำนวนเงินลงทุน DCA ต่อเดือน (บาทหรือ USD)", min_value=100.0, max_value=10000.0, value=1000.0, step=100.0)
    show_financials = st.checkbox("แสดงงบการเงิน (Income Statement)", value=False)

    if st.button("วิเคราะห์"):
        export_list = []
        results_table = []
        total_invest = 0
        total_profit = 0
        total_div = 0

        for ticker in tickers:
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
            
            # AI Insights for Buffett Analysis (if enabled)
            if st.session_state['use_ai'] and gemini_api_key:
                st.subheader("🤖 การวิเคราะห์ AI สำหรับหลัก Buffett")
                
                # Calculate current price change
                if not hist.empty and len(hist) > 1:
                    current_price = hist['Close'].iloc[-1]
                    prev_price = hist['Close'].iloc[-2]
                    price_change = ((current_price - prev_price) / prev_price) * 100
                else:
                    price_change = 0
                
                ai_prompt = f"""
                วิเคราะห์หุ้น {ticker} ({company_name}) ตามหลัก Warren Buffett ในภาษาไทย:
                
                ข้อมูลพื้นฐาน:
                - คะแนน Buffett Checklist: {detail['score']}/{detail['evaluated']} ({detail['score_pct']}%)
                - การเปลี่ยนแปลงราคาล่าสุด: {price_change:.2f}%
                - Dividend Yield: {div_yield_pct}
                
                ข้อมูลการลงทุน DCA:
                - เงินลงทุนรวม: {dca_result['เงินลงทุนรวม']:,.0f}
                - กำไร/ขาดทุน: {dca_result['กำไร/ขาดทุน']:,.0f}
                - ผลตอบแทน: {dca_result['กำไร(%)']:.2f}%
                
                กรุณาให้การวิเคราะห์ขั้นสูงภายใน 4-5 ประโยค:
                1. ความเหมาะสมสำหรับการลงทุนระยะยาว
                2. จุดแข็งและจุดที่ควรกังวล
                3. ข้อเสนอแนะการลงทุน
                """
                
                with st.spinner("AI กำลังวิเคราะห์หลัก Buffett..."):
                    ai_analysis = get_ai_insights(gemini_api_key, ai_prompt)
                    st.write(ai_analysis)
            elif st.session_state['use_ai'] and not gemini_api_key:
                st.info("💡 ใส่ Gemini API Key ในแถบด้านข้างเพื่อดูการวิเคราะห์ AI ขั้นสูง")

            st.subheader("DCA Simulation (จำลองลงทุนรายเดือน)")
            dca_result = dca_simulation(hist, monthly_invest, div)
            st.write(pd.DataFrame(dca_result, index=['สรุปผล']).T)

            # สะสมผลรวม
            total_invest += dca_result["เงินลงทุนรวม"]
            total_profit += dca_result["กำไร/ขาดทุน"]
            total_div += dca_result["เงินปันผลรวม"]

            results_table.append({
                "หุ้น": ticker,
                "ชื่อบริษัท": company_name,
                "เงินลงทุน": dca_result["เงินลงทุนรวม"],
                "กำไร": dca_result["กำไร/ขาดทุน"],
                "เงินปันผล": dca_result["เงินปันผลรวม"],
                "มูลค่าปัจจุบัน": dca_result["มูลค่าปัจจุบัน"]
            })

            if not hist.empty:
                st.line_chart(hist['Close'])
            else:
                st.warning("ไม่มีข้อมูลราคาหุ้น")

            if show_financials and fin is not None and not fin.empty:
                st.subheader("งบกำไรขาดทุน (Income Statement)")
                st.dataframe(df_human_format(fin))

            export_list.append({
                "หุ้น": ticker,
                "ชื่อบริษัท": company_name,
                "เงินลงทุนรวม": dca_result["เงินลงทุนรวม"],
                "จำนวนหุ้นสะสม": dca_result["จำนวนหุ้นสะสม"],
                "มูลค่าปัจจุบัน": dca_result["มูลค่าปัจจุบัน"],
                "กำไร/ขาดทุน": dca_result["กำไร/ขาดทุน"],
                "กำไร(%)": dca_result["กำไร(%)"],
                "เงินปันผลรวม": dca_result["เงินปันผลรวม"],
                "Dividend Yield ย้อนหลัง 1 ปี (%)": manual_yield if not np.isnan(manual_yield) else "N/A",
                "เงินปันผลย้อนหลัง 1 ปี": total_div1y if not np.isnan(total_div1y) else "N/A",
                "Dividend Yield (%)": div_yield_pct,
                "Ex-Dividend Date": ex_div_date,
                "52W High": w52_high,
                "52W Low": w52_low,
                "ราคาปิดล่าสุด": last_close,
                "ราคาเปิดล่าสุด": last_open,
                "คะแนนรวม": f"{detail['score']}/{detail['evaluated']}",
                "เปอร์เซ็นต์": detail['score_pct'],
                "ป้ายคะแนน": badge,
            })

        # --- Export to Excel ---
        if len(export_list) > 0:
            df_export = pd.DataFrame(export_list)
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df_export.to_excel(writer, index=False, sheet_name='WarrenDCA')
            st.download_button(
                label="📥 ดาวน์โหลดผลลัพธ์เป็น Excel",
                data=output.getvalue(),
                file_name='WarrenDCA_Result.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )

        # --- สรุปผลหุ้นที่เลือก DCA Simulator รวม ---
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

st.caption("Powered by Yahoo Finance | วิเคราะห์หุ้นด้วย Buffett Checklist (ขยาย 18 เงื่อนไข) + DCA + ข่าว & Sentiment + กลยุทธ์ Backtest + AI ขั้นสูง")