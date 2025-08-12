import streamlit as st
import yfinance as yf
import pandas as pd
import io
import datetime
import numpy as np
import matplotlib.pyplot as plt
import backtrader as bt

# AI Analysis imports - handle gracefully if not installed
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# ----------------- AI Helper Functions -----------------
def ai_configure(api_key):
    """Configure Gemini AI with API key"""
    if GEMINI_AVAILABLE and api_key:
        try:
            genai.configure(api_key=api_key)
            return True
        except Exception:
            return False
    return False

def ai_generate(prompt, max_retries=3):
    """Generate AI response with error handling"""
    if not GEMINI_AVAILABLE:
        return "❌ Google Generative AI library not installed"
    
    for attempt in range(max_retries):
        try:
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(prompt)
            return response.text if response.text else "ไม่สามารถสร้างคำตอบได้"
        except Exception as e:
            if attempt == max_retries - 1:
                return f"❌ AI Error: {str(e)[:100]}..."
            continue

def compute_timing_analysis(hist_prices, div_data, buffett_score_pct):
    """Compute timing analysis classification based on multiple factors"""
    if hist_prices.empty:
        return None, "ไม่มีข้อมูลราคา"
    
    try:
        # Calculate daily drop %
        recent_prices = hist_prices['Close'].tail(2)
        if len(recent_prices) >= 2:
            daily_drop_pct = ((recent_prices.iloc[-1] - recent_prices.iloc[-2]) / recent_prices.iloc[-2]) * 100
        else:
            daily_drop_pct = 0
        
        # Calculate volume ratio (last day vs 20-day average)
        if 'Volume' in hist_prices.columns:
            recent_volume = hist_prices['Volume'].iloc[-1] if not hist_prices['Volume'].empty else 0
            avg_volume_20d = hist_prices['Volume'].tail(20).mean() if len(hist_prices) >= 20 else recent_volume
            volume_ratio = recent_volume / avg_volume_20d if avg_volume_20d > 0 else 1
        else:
            volume_ratio = 1
        
        # Calculate volatility z-score (simplified)
        if len(hist_prices) >= 20:
            returns = hist_prices['Close'].pct_change().dropna()
            recent_vol = returns.tail(5).std()
            avg_vol = returns.tail(60).std() if len(returns) >= 60 else recent_vol
            vol_zscore = (recent_vol - avg_vol) / avg_vol if avg_vol > 0 else 0
        else:
            vol_zscore = 0
        
        # Classification logic
        classification = "NORMAL"
        reason = ""
        
        if daily_drop_pct <= -5 and volume_ratio > 1.5 and buffett_score_pct >= 60:
            classification = "OPPORTUNITY"
            reason = f"ราคาปรับลง {daily_drop_pct:.1f}% พร้อมปริมาณการซื้อขายสูง และคะแนน Buffett ดี"
        elif daily_drop_pct <= -3 and buffett_score_pct >= 40:
            classification = "NORMAL"
            reason = f"ราคาปรับลงเล็กน้อย {daily_drop_pct:.1f}% คะแนน Buffett ปานกลาง"
        elif daily_drop_pct >= 3 or vol_zscore > 2 or buffett_score_pct < 30:
            classification = "CAUTION"
            reason = f"ราคาผันผวนสูง หรือคะแนน Buffett ต่ำ"
        elif daily_drop_pct >= 5 or buffett_score_pct < 20:
            classification = "DANGER"
            reason = f"ราคาขึ้นรวดเร็ว {daily_drop_pct:.1f}% หรือคะแนน Buffett ต่ำมาก"
        else:
            reason = f"ราคาเปลี่ยนแปลง {daily_drop_pct:.1f}% อยู่ในเกณฑ์ปกติ"
        
        metrics = {
            'drop_pct': daily_drop_pct,
            'volume_ratio': volume_ratio,
            'vol_zscore': vol_zscore
        }
        
        return classification, reason, metrics
        
    except Exception as e:
        return None, f"Error in timing analysis: {str(e)[:50]}..."

def project_target(hist_prices, horizon_days=15):
    """Project target price using linear regression on log prices"""
    if hist_prices.empty or len(hist_prices) < 10:
        return None
    
    try:
        # Use up to 60 days of data
        prices = hist_prices['Close'].tail(60)
        log_prices = np.log(prices)
        
        # Simple linear regression on log prices
        x = np.arange(len(log_prices))
        coeffs = np.polyfit(x, log_prices, 1)
        slope, intercept = coeffs
        
        # Project future price
        future_x = len(log_prices) + horizon_days - 1
        projected_log_price = slope * future_x + intercept
        projected_price = np.exp(projected_log_price)
        
        # Calculate residual standard deviation
        fitted_log_prices = slope * x + intercept
        residuals = log_prices - fitted_log_prices
        residual_std = np.std(residuals)
        
        # Calculate confidence bands
        adjustment = residual_std * np.sqrt(horizon_days)
        
        mid_price = projected_price
        low_price = np.exp(projected_log_price - adjustment)
        high_price = np.exp(projected_log_price + adjustment)
        
        current_price = prices.iloc[-1]
        mid_change_pct = ((mid_price - current_price) / current_price) * 100
        low_change_pct = ((low_price - current_price) / current_price) * 100
        high_change_pct = ((high_price - current_price) / current_price) * 100
        
        return {
            'mid': mid_price,
            'low': low_price,
            'high': high_price,
            'mid_change_pct': mid_change_pct,
            'low_change_pct': low_change_pct,
            'high_change_pct': high_change_pct
        }
        
    except Exception as e:
        return None

def classification_color(classification):
    """Return color for classification"""
    colors = {
        'OPPORTUNITY': '#c8e6c9',  # soft green
        'NORMAL': '#e3f2fd',      # light blue
        'CAUTION': '#fff3e0',     # soft amber
        'DANGER': '#ffebee'       # soft red
    }
    return colors.get(classification, '#f5f5f5')

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

# AI Analysis Configuration
st.sidebar.markdown("---")
use_ai = st.sidebar.checkbox("ใช้ AI วิเคราะห์ (Gemini)", value=False)

gemini_api_key = None
ai_configured = False

if use_ai:
    if not GEMINI_AVAILABLE:
        st.sidebar.warning("⚠️ ต้องติดตั้ง google-generativeai library ก่อน: `pip install google-generativeai`")
    else:
        gemini_api_key = st.sidebar.text_input(
            "Gemini API Key", 
            type="password",
            help="API Key ใช้เฉพาะใน session นี้เท่านั้น ไม่มีการบันทึกหรือส่งออก"
        )
        
        if gemini_api_key:
            if 'gemini_api_key' not in st.session_state or st.session_state.get('gemini_api_key') != gemini_api_key:
                ai_configured = ai_configure(gemini_api_key)
                st.session_state['gemini_api_key'] = gemini_api_key
                st.session_state['ai_configured'] = ai_configured
            else:
                ai_configured = st.session_state.get('ai_configured', False)
            
            if ai_configured:
                st.sidebar.success("✅ AI พร้อมใช้งาน")
                
                # Test connection button
                if st.sidebar.button("ทดสอบการเชื่อมต่อ"):
                    test_response = ai_generate("สวัสดี ตอบสั้นๆ ว่า 'AI พร้อมใช้งาน'")
                    if "AI พร้อมใช้งาน" in test_response or "พร้อม" in test_response:
                        st.sidebar.success("🟢 การเชื่อมต่อ AI สำเร็จ")
                    else:
                        st.sidebar.error(f"🔴 การเชื่อมต่อล้มเหลว: {test_response[:50]}...")
            else:
                st.sidebar.error("❌ API Key ไม่ถูกต้อง")
        else:
            st.sidebar.info("💡 กรุณาใส่ Gemini API Key เพื่อเปิดใช้งาน AI")

st.sidebar.markdown("---")

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

            # =============== AI ANALYSIS SECTION ===============
            if use_ai:
                st.subheader("🔍 AI Timing & Short-Term Target")
                
                # Initialize AI variables
                timing_classification = None
                timing_reason = "N/A"
                timing_metrics = {}
                target_15d = None
                target_30d = None
                ai_insight = "N/A"
                
                # Timing Analysis
                if GEMINI_AVAILABLE:
                    timing_result = compute_timing_analysis(hist, div, detail['score_pct'])
                    if timing_result and len(timing_result) >= 3:
                        timing_classification, timing_reason, timing_metrics = timing_result
                    else:
                        timing_classification = None
                        timing_reason = timing_result[1] if timing_result else "ไม่สามารถวิเคราะห์ได้"
                
                # Target Projections
                target_15d = project_target(hist, 15)
                target_30d = project_target(hist, 30)
                
                # Display Timing Analysis
                if timing_classification:
                    timing_color = classification_color(timing_classification)
                    st.markdown(f"""
                    <div style='background-color: {timing_color}; padding: 15px; border-radius: 8px; margin: 10px 0; color: #222;'>
                        <h4 style='margin: 0; color: #222;'>🎯 การวิเคราะห์จังหวะ: {timing_classification}</h4>
                        <p style='margin: 5px 0; color: #222;'><strong>เหตุผล:</strong> {timing_reason}</p>
                        <p style='margin: 5px 0; color: #222;'>
                            <strong>ข้อมูลสำคัญ:</strong> 
                            Drop: {timing_metrics.get('drop_pct', 0):.1f}% | 
                            Volume Ratio: {timing_metrics.get('volume_ratio', 1):.1f}x | 
                            Vol Z-score: {timing_metrics.get('vol_zscore', 0):.1f}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style='background-color: #f5f5f5; padding: 15px; border-radius: 8px; margin: 10px 0; color: #222;'>
                        <h4 style='margin: 0; color: #222;'>🎯 การวิเคราะห์จังหวะ: ไม่สามารถวิเคราะห์ได้</h4>
                        <p style='margin: 5px 0; color: #222;'>{timing_reason}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Display Target Projections
                col1, col2 = st.columns(2)
                
                with col1:
                    if target_15d:
                        st.markdown(f"""
                        <div style='background-color: #e8f5e8; padding: 12px; border-radius: 8px; margin: 5px 0; color: #222;'>
                            <h5 style='margin: 0; color: #222;'>📈 เป้าหมาย 15 วัน</h5>
                            <p style='margin: 3px 0; color: #222;'><strong>กลาง:</strong> {target_15d['mid']:.2f} ({target_15d['mid_change_pct']:+.1f}%)</p>
                            <p style='margin: 3px 0; color: #222;'><strong>ต่ำ:</strong> {target_15d['low']:.2f} ({target_15d['low_change_pct']:+.1f}%)</p>
                            <p style='margin: 3px 0; color: #222;'><strong>สูง:</strong> {target_15d['high']:.2f} ({target_15d['high_change_pct']:+.1f}%)</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div style='background-color: #f5f5f5; padding: 12px; border-radius: 8px; margin: 5px 0; color: #222;'>
                            <h5 style='margin: 0; color: #222;'>📈 เป้าหมาย 15 วัน</h5>
                            <p style='margin: 3px 0; color: #222;'>ไม่สามารถคำนวณได้</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    if target_30d:
                        st.markdown(f"""
                        <div style='background-color: #e8f4fd; padding: 12px; border-radius: 8px; margin: 5px 0; color: #222;'>
                            <h5 style='margin: 0; color: #222;'>📈 เป้าหมาย 30 วัน</h5>
                            <p style='margin: 3px 0; color: #222;'><strong>กลาง:</strong> {target_30d['mid']:.2f} ({target_30d['mid_change_pct']:+.1f}%)</p>
                            <p style='margin: 3px 0; color: #222;'><strong>ต่ำ:</strong> {target_30d['low']:.2f} ({target_30d['low_change_pct']:+.1f}%)</p>
                            <p style='margin: 3px 0; color: #222;'><strong>สูง:</strong> {target_30d['high']:.2f} ({target_30d['high_change_pct']:+.1f}%)</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div style='background-color: #f5f5f5; padding: 12px; border-radius: 8px; margin: 5px 0; color: #222;'>
                            <h5 style='margin: 0; color: #222;'>📈 เป้าหมาย 30 วัน</h5>
                            <p style='margin: 3px 0; color: #222;'>ไม่สามารถคำนวณได้</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # AI Insight
                if ai_configured and GEMINI_AVAILABLE:
                    # Create cache key for AI insight
                    current_date = datetime.datetime.now().strftime('%Y-%m-%d')
                    cache_key = f"ai_insight_{ticker}_{current_date}"
                    
                    if cache_key not in st.session_state:
                        # Generate AI insight
                        prompt = f"""วิเคราะห์หุ้น {ticker} ({company_name}) ในภาษาไทยแบบกระชับ:

คะแนน Buffett: {detail['score_pct']}%
การจัดประเภทจังหวะ: {timing_classification or 'ไม่ระบุ'}
เหตุผล: {timing_reason}
การเปลี่ยนแปลงราคา: {timing_metrics.get('drop_pct', 0):.1f}%
เป้าหมาย 15 วัน: {target_15d['mid_change_pct']:+.1f}% ถ้ามีข้อมูล
เป้าหมาย 30 วัน: {target_30d['mid_change_pct']:+.1f}% ถ้ามีข้อมูล

ให้สรุปในย่อหน้าเดียว ไม่เกิน 3 ประโยค เน้นข้อมูลสำคัญและคำแนะนำสำหรับนักลงทุน"""
                        
                        ai_insight = ai_generate(prompt)
                        st.session_state[cache_key] = ai_insight
                    else:
                        ai_insight = st.session_state[cache_key]
                    
                    st.markdown(f"""
                    <div style='background-color: #fff3cd; padding: 15px; border-radius: 8px; margin: 10px 0; border-left: 4px solid #ffc107; color: #222;'>
                        <h5 style='margin: 0 0 8px 0; color: #222;'>🤖 AI Insight</h5>
                        <p style='margin: 0; color: #222; line-height: 1.5;'>{ai_insight}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style='background-color: #f8f9fa; padding: 15px; border-radius: 8px; margin: 10px 0; color: #222;'>
                        <h5 style='margin: 0 0 8px 0; color: #222;'>🤖 AI Insight</h5>
                        <p style='margin: 0; color: #666;'>ต้องการ Gemini API Key เพื่อแสดงการวิเคราะห์ AI</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Store AI data for export
            ai_data = {}
            if use_ai:
                ai_data = {
                    'AI_Timing_Classification': timing_classification or 'N/A',
                    'AI_Timing_Reason': timing_reason,
                    'AI_Target_15d_Mid': target_15d['mid'] if target_15d else 'N/A',
                    'AI_Target_15d_Change': f"{target_15d['mid_change_pct']:+.1f}%" if target_15d else 'N/A',
                    'AI_Target_30d_Mid': target_30d['mid'] if target_30d else 'N/A', 
                    'AI_Target_30d_Change': f"{target_30d['mid_change_pct']:+.1f}%" if target_30d else 'N/A',
                    'AI_Insight': ai_insight if 'ai_insight' in locals() else 'N/A'
                }

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

            export_data = {
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
            }
            
            # Add AI data to export if AI is enabled (without API key)
            if use_ai and 'ai_data' in locals():
                export_data.update(ai_data)
            
            export_list.append(export_data)

    # --- Export to Excel ---
    if len(export_list) > 0:
        df_export = pd.DataFrame(export_list)
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            # Main sheet
            df_export.to_excel(writer, index=False, sheet_name='WarrenDCA')
            
            # AI Analysis sheet (if AI is enabled)
            if use_ai:
                ai_columns = [col for col in df_export.columns if col.startswith('AI_')]
                if ai_columns:
                    ai_df = df_export[['หุ้น', 'ชื่อบริษัท'] + ai_columns].copy()
                    ai_df.to_excel(writer, index=False, sheet_name='AI_Analysis')
        
        download_label = "📥 ดาวน์โหลดผลลัพธ์เป็น Excel"
        if use_ai:
            download_label += " (รวม AI Analysis)"
            
        st.download_button(
            label=download_label,
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

st.caption("Powered by Yahoo Finance | วิเคราะห์หุ้นด้วย Buffett Checklist (ขยาย 18 เงื่อนไข) + DCA + ปันผลย้อนหลัง 1 ปี")