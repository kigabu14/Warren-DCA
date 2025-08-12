import streamlit as st
import yfinance as yf
import pandas as pd
import io
import datetime
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

# ----------------- Bollinger Bands Strategy Functions -----------------
def calculate_bollinger_bands(prices: pd.Series, period: int = 20, multiplier: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Bollinger Bands for given price series.
    
    Args:
        prices: Price series (typically Close prices)
        period: Moving average period (default: 20)
        multiplier: Standard deviation multiplier (default: 2)
    
    Returns:
        Tuple of (upper_band, middle_band, lower_band)
    """
    if len(prices) < period:
        raise ValueError(f"Insufficient data: need at least {period} data points, got {len(prices)}")
    
    # Calculate moving average (middle band)
    middle_band = prices.rolling(window=period).mean()
    
    # Calculate standard deviation
    std = prices.rolling(window=period).std()
    
    # Calculate upper and lower bands
    upper_band = middle_band + (multiplier * std)
    lower_band = middle_band - (multiplier * std)
    
    return upper_band, middle_band, lower_band

def bollinger_bands_strategy(
    hist_data: pd.DataFrame, 
    period: int = 20, 
    multiplier: float = 2,
    initial_capital: float = 10000,
    stop_loss_pct: float = 5.0,
    take_profit_pct: float = 10.0
) -> Dict:
    """
    Implement Bollinger Bands trading strategy.
    
    Args:
        hist_data: Historical price data with OHLC columns
        period: BB period (default: 20)
        multiplier: BB multiplier (default: 2)
        initial_capital: Starting capital
        stop_loss_pct: Stop loss percentage
        take_profit_pct: Take profit percentage
    
    Returns:
        Dictionary with trading results and metrics
    """
    if hist_data.empty or len(hist_data) < period:
        return {"error": "Insufficient historical data"}
    
    try:
        prices = hist_data['Close'].copy()
        
        # Calculate Bollinger Bands
        upper_band, middle_band, lower_band = calculate_bollinger_bands(prices, period, multiplier)
        
        # Initialize trading variables
        position = 0  # 0: no position, 1: long position
        entry_price = 0
        capital = initial_capital
        shares = 0
        trades = []
        equity_curve = []
        
        # Trading signals and execution
        for i in range(period, len(hist_data)):
            current_price = prices.iloc[i]
            current_date = hist_data.index[i]
            
            # Buy signal: price crosses below lower band
            if position == 0 and current_price < lower_band.iloc[i] and not pd.isna(lower_band.iloc[i]):
                shares = capital / current_price
                entry_price = current_price
                position = 1
                capital = 0
                
                trades.append({
                    'date': current_date,
                    'action': 'BUY',
                    'price': current_price,
                    'shares': shares,
                    'value': shares * current_price
                })
            
            # Sell signal: price crosses above upper band OR stop loss/take profit
            elif position == 1:
                sell_signal = False
                sell_reason = ""
                
                # Check sell conditions
                if current_price > upper_band.iloc[i] and not pd.isna(upper_band.iloc[i]):
                    sell_signal = True
                    sell_reason = "Upper band crossed"
                elif (current_price - entry_price) / entry_price * 100 >= take_profit_pct:
                    sell_signal = True
                    sell_reason = f"Take profit ({take_profit_pct}%)"
                elif (entry_price - current_price) / entry_price * 100 >= stop_loss_pct:
                    sell_signal = True
                    sell_reason = f"Stop loss ({stop_loss_pct}%)"
                
                if sell_signal:
                    capital = shares * current_price
                    profit = capital - last_buy_value if last_buy_value is not None else 0
                    
                    trades.append({
                        'date': current_date,
                        'action': 'SELL',
                        'price': current_price,
                        'shares': shares,
                        'value': capital,
                        'profit': profit,
                        'reason': sell_reason
                    })
                    
                    shares = 0
                    position = 0
            
            # Calculate current portfolio value
            current_value = capital + (shares * current_price if shares > 0 else 0)
            equity_curve.append(current_value)
        
        # Calculate performance metrics
        if not trades:
            return {
                "error": "No trades generated",
                "bb_data": {
                    "upper_band": upper_band,
                    "middle_band": middle_band,
                    "lower_band": lower_band,
                    "prices": prices
                }
            }
        
        final_value = capital + (shares * prices.iloc[-1] if shares > 0 else 0)
        total_return = (final_value - initial_capital) / initial_capital * 100
        
        # Calculate additional metrics
        winning_trades = [t for t in trades if t.get('profit', 0) > 0]
        losing_trades = [t for t in trades if t.get('profit', 0) < 0]
        
        buy_trades = [t for t in trades if t['action'] == 'BUY']
        total_trades = len(buy_trades)
        
        win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
        
        avg_win = np.mean([t['profit'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['profit'] for t in losing_trades]) if losing_trades else 0
        
        max_drawdown = 0
        peak = initial_capital
        for value in equity_curve:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak * 100
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        return {
            "success": True,
            "initial_capital": initial_capital,
            "final_value": round(final_value, 2),
            "total_return": round(total_return, 2),
            "total_trades": total_trades,
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": round(win_rate, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "max_drawdown": round(max_drawdown, 2),
            "trades": trades,
            "equity_curve": equity_curve,
            "bb_data": {
                "upper_band": upper_band,
                "middle_band": middle_band,
                "lower_band": lower_band,
                "prices": prices
            },
            "parameters": {
                "period": period,
                "multiplier": multiplier,
                "stop_loss_pct": stop_loss_pct,
                "take_profit_pct": take_profit_pct
            }
        }
        
    except Exception as e:
        return {"error": f"Strategy calculation error: {str(e)}"}

def plot_bollinger_bands_chart(result: Dict, ticker: str) -> plt.Figure:
    """
    Create a chart showing price data with Bollinger Bands and trade signals.
    
    Args:
        result: Result dictionary from bollinger_bands_strategy
        ticker: Stock ticker symbol
    
    Returns:
        Matplotlib figure
    """
    if "bb_data" not in result:
        return None
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[3, 1])
    
    bb_data = result["bb_data"]
    prices = bb_data["prices"]
    upper_band = bb_data["upper_band"]
    middle_band = bb_data["middle_band"]
    lower_band = bb_data["lower_band"]
    
    # Plot price and Bollinger Bands
    ax1.plot(prices.index, prices, label='Price', color='black', linewidth=1)
    ax1.plot(upper_band.index, upper_band, label='Upper Band', color='red', linestyle='--', alpha=0.7)
    ax1.plot(middle_band.index, middle_band, label='Middle Band (SMA)', color='blue', linestyle='-', alpha=0.7)
    ax1.plot(lower_band.index, lower_band, label='Lower Band', color='green', linestyle='--', alpha=0.7)
    
    # Fill between bands
    ax1.fill_between(upper_band.index, upper_band, lower_band, alpha=0.1, color='gray')
    
    # Plot trade signals
    if "trades" in result:
        buy_signals = [t for t in result["trades"] if t["action"] == "BUY"]
        sell_signals = [t for t in result["trades"] if t["action"] == "SELL"]
        
        if buy_signals:
            buy_dates = [t["date"] for t in buy_signals]
            buy_prices = [t["price"] for t in buy_signals]
            ax1.scatter(buy_dates, buy_prices, color='green', marker='^', s=100, label='Buy Signal', zorder=5)
        
        if sell_signals:
            sell_dates = [t["date"] for t in sell_signals]
            sell_prices = [t["price"] for t in sell_signals]
            ax1.scatter(sell_dates, sell_prices, color='red', marker='v', s=100, label='Sell Signal', zorder=5)
    
    ax1.set_title(f'{ticker} - Bollinger Bands Strategy', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot equity curve
    if "equity_curve" in result:
        equity_dates = prices.index[result["parameters"]["period"]:]
        ax2.plot(equity_dates, result["equity_curve"], color='purple', linewidth=2)
        ax2.set_title('Portfolio Value', fontsize=12)
        ax2.set_ylabel('Value ($)', fontsize=10)
        ax2.set_xlabel('Date', fontsize=10)
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig
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

### Backtesting Strategies
**Bollinger Bands Strategy:**
- ใช้ Bollinger Bands เพื่อหาจุดซื้อขาย
- ซื้อเมื่อราคาข้ามลงใต้ Lower Band
- ขายเมื่อราคาข้ามขึ้นเหนือ Upper Band
- รองรับการตั้ง Stop Loss และ Take Profit
- แสดงผลการ Backtest พร้อมกราฟและสถิติ

### หมายเหตุ
- ข้อมูลหุ้น US มักครบถ้วนกว่าหุ้นไทย
- ถ้าข้อมูลสำคัญไม่ครบ บางข้อจะขึ้น N/A
- ใช้งบการเงินย้อนหลัง (Annual) ตามที่ Yahoo ให้ (ปกติ 4 ปี)
- รองรับหุ้นจากตลาดทั่วโลก: US, SET100, Europe, Asia, Australia
""")
    st.stop()

elif menu == "Backtesting":
    st.header("📈 Backtesting Strategies")
    st.markdown("**Test trading strategies on historical data to evaluate performance.**")
    
    # Strategy selection
    strategy = st.selectbox(
        "เลือกกลยุทธ์การเทรด", 
        ["Bollinger Bands"],
        index=0,
        help="เลือกกลยุทธ์ที่ต้องการทดสอบ"
    )
    
    if strategy == "Bollinger Bands":
        st.subheader("🎯 Bollinger Bands Strategy")
        st.markdown("""
        **กลยุทธ์:** ซื้อเมื่อราคาทะลุลงใต้ Lower Band และขายเมื่อราคาทะลุขึ้นเหนือ Upper Band
        
        **หลักการ:** Bollinger Bands ช่วยระบุสภาวะ Oversold (ราคาต่ำเกินไป) และ Overbought (ราคาสูงเกินไป)
        """)
        
        # Stock selection
        col1, col2 = st.columns(2)
        
        with col1:
            selected_market = st.selectbox(
                "เลือกตลาดหุ้น",
                options=list(markets.keys()),
                index=0,
                help="เลือกตลาดหุ้นที่ต้องการทดสอบ"
            )
            
            available_tickers = markets[selected_market]
            ticker = st.selectbox(
                "เลือกหุ้น",
                available_tickers,
                index=0 if available_tickers else None,
                help="เลือกหุ้นที่ต้องการทดสอบกลยุทธ์"
            )
            
            period = st.selectbox(
                "ช่วงเวลาข้อมูล", 
                ["1y", "2y", "5y", "max"], 
                index=2,
                help="ช่วงเวลาของข้อมูลราคาที่ใช้ในการทดสอบ"
            )
        
        with col2:
            st.subheader("⚙️ พารามิเตอร์กลยุทธ์")
            
            # Bollinger Bands parameters
            bb_period = st.number_input(
                "Period (จำนวนวันสำหรับค่าเฉลี่ย)",
                min_value=5,
                max_value=100,
                value=20,
                step=1,
                help="จำนวนวันที่ใช้คำนวณ Moving Average และ Standard Deviation"
            )
            
            bb_multiplier = st.number_input(
                "Multiplier (ตัวคูณ Standard Deviation)",
                min_value=0.5,
                max_value=4.0,
                value=2.0,
                step=0.1,
                help="ตัวคูณสำหรับ Standard Deviation ในการกำหนดความกว้างของ Bands"
            )
            
            initial_capital = st.number_input(
                "เงินทุนเริ่มต้น ($)",
                min_value=1000.0,
                max_value=1000000.0,
                value=10000.0,
                step=1000.0,
                help="จำนวนเงินทุนเริ่มต้นสำหรับการทดสอบ"
            )
            
            stop_loss_pct = st.number_input(
                "Stop Loss (%)",
                min_value=0.0,
                max_value=50.0,
                value=5.0,
                step=0.5,
                help="เปอร์เซ็นต์ขาดทุนที่จะขายหุ้นออก"
            )
            
            take_profit_pct = st.number_input(
                "Take Profit (%)",
                min_value=0.0,
                max_value=100.0,
                value=10.0,
                step=0.5,
                help="เปอร์เซ็นต์กำไรที่จะขายหุ้นออก"
            )
        
        # Validation
        if bb_period <= 0 or bb_multiplier <= 0:
            st.error("❌ พารามิเตอร์ต้องมีค่าเป็นบวก")
        elif not ticker:
            st.error("❌ กรุณาเลือกหุ้นที่ต้องการทดสอบ")
        else:
            if st.button("🚀 เริ่มการทดสอบ", type="primary"):
                with st.spinner(f"กำลังทดสอบกลยุทธ์ Bollinger Bands สำหรับ {ticker}..."):
                    try:
                        # Fetch stock data
                        stock = yf.Ticker(ticker)
                        hist_data = stock.history(period=period)
                        
                        if hist_data.empty:
                            st.error(f"❌ ไม่พบข้อมูลราคาสำหรับ {ticker}")
                        else:
                            # Run strategy
                            result = bollinger_bands_strategy(
                                hist_data=hist_data,
                                period=bb_period,
                                multiplier=bb_multiplier,
                                initial_capital=initial_capital,
                                stop_loss_pct=stop_loss_pct,
                                take_profit_pct=take_profit_pct
                            )
                            
                            if "error" in result:
                                st.error(f"❌ {result['error']}")
                            else:
                                # Display results
                                st.success("✅ การทดสอบสำเร็จ!")
                                
                                # Performance summary
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    st.metric(
                                        "Total Return",
                                        f"{result['total_return']}%",
                                        delta=f"{result['total_return']:.2f}%"
                                    )
                                
                                with col2:
                                    st.metric(
                                        "Final Value",
                                        f"${result['final_value']:,.2f}",
                                        delta=f"${result['final_value'] - result['initial_capital']:,.2f}"
                                    )
                                
                                with col3:
                                    st.metric(
                                        "Total Trades",
                                        f"{result['total_trades']}",
                                        delta=f"Win Rate: {result['win_rate']}%"
                                    )
                                
                                with col4:
                                    st.metric(
                                        "Max Drawdown",
                                        f"{result['max_drawdown']}%",
                                        delta=f"-{result['max_drawdown']:.2f}%",
                                        delta_color="inverse"
                                    )
                                
                                # Charts
                                st.subheader("📊 กราฟและสัญญาณการเทรด")
                                
                                try:
                                    fig = plot_bollinger_bands_chart(result, ticker)
                                    if fig:
                                        st.pyplot(fig)
                                    else:
                                        st.warning("ไม่สามารถสร้างกราฟได้")
                                except Exception as e:
                                    st.error(f"เกิดข้อผิดพลาดในการสร้างกราฟ: {str(e)}")
                                
                                # Detailed statistics
                                st.subheader("📈 สถิติรายละเอียด")
                                
                                stats_col1, stats_col2 = st.columns(2)
                                
                                with stats_col1:
                                    st.markdown("**การเทรด:**")
                                    st.write(f"• จำนวนการเทรดทั้งหมด: {result['total_trades']}")
                                    st.write(f"• การเทรดที่ได้กำไร: {result['winning_trades']}")
                                    st.write(f"• การเทรดที่ขาดทุน: {result['losing_trades']}")
                                    st.write(f"• อัตราการชนะ: {result['win_rate']:.2f}%")
                                
                                with stats_col2:
                                    st.markdown("**ผลตอบแทน:**")
                                    st.write(f"• กำไรเฉลี่ยต่อเทรด: ${result['avg_win']:.2f}")
                                    st.write(f"• ขาดทุนเฉลี่ยต่อเทรด: ${result['avg_loss']:.2f}")
                                    st.write(f"• การลดลงสูงสุด: {result['max_drawdown']:.2f}%")
                                
                                # Trade history
                                if result['trades']:
                                    st.subheader("📋 ประวัติการเทรด")
                                    
                                    trades_df = pd.DataFrame(result['trades'])
                                    trades_df['date'] = pd.to_datetime(trades_df['date']).dt.strftime('%Y-%m-%d')
                                    trades_df['price'] = trades_df['price'].round(2)
                                    trades_df['value'] = trades_df['value'].round(2)
                                    
                                    st.dataframe(
                                        trades_df,
                                        use_container_width=True,
                                        hide_index=True
                                    )
                                
                                # Export option
                                if result['trades']:
                                    st.subheader("💾 ส่งออกผลลัพธ์")
                                    
                                    # Prepare export data
                                    export_data = {
                                        "Strategy": "Bollinger Bands",
                                        "Ticker": ticker,
                                        "Period": period,
                                        "BB_Period": bb_period,
                                        "BB_Multiplier": bb_multiplier,
                                        "Initial_Capital": initial_capital,
                                        "Stop_Loss_%": stop_loss_pct,
                                        "Take_Profit_%": take_profit_pct,
                                        "Total_Return_%": result['total_return'],
                                        "Final_Value": result['final_value'],
                                        "Total_Trades": result['total_trades'],
                                        "Win_Rate_%": result['win_rate'],
                                        "Max_Drawdown_%": result['max_drawdown'],
                                        "Avg_Win": result['avg_win'],
                                        "Avg_Loss": result['avg_loss']
                                    }
                                    
                                    # Create Excel file
                                    output = io.BytesIO()
                                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                                        # Summary sheet
                                        summary_df = pd.DataFrame([export_data])
                                        summary_df.to_excel(writer, sheet_name='Summary', index=False)
                                        
                                        # Trades sheet
                                        trades_df = pd.DataFrame(result['trades'])
                                        trades_df.to_excel(writer, sheet_name='Trades', index=False)
                                    
                                    st.download_button(
                                        label="📥 ดาวน์โหลดผลลัพธ์ (Excel)",
                                        data=output.getvalue(),
                                        file_name=f'BB_Backtest_{ticker}_{datetime.datetime.now().strftime("%Y%m%d_%H%M")}.xlsx',
                                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                                    )
                    
                    except Exception as e:
                        st.error(f"❌ เกิดข้อผิดพลาด: {str(e)}")
    
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

st.caption("Powered by Yahoo Finance | วิเคราะห์หุ้นด้วย Buffett Checklist (ขยาย 18 เงื่อนไข) + DCA + ปันผลย้อนหลัง 1 ปี")