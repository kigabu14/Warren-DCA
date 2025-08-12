import streamlit as st
import yfinance as yf
import pandas as pd
import io
import datetime
import numpy as np
import matplotlib.pyplot as plt

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

# ----------------- BACKTESTING STRATEGIES -----------------

def calculate_bollinger_bands(prices, period=20, multiplier=2):
    """Calculate Bollinger Bands"""
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper_band = sma + (std * multiplier)
    lower_band = sma - (std * multiplier)
    return sma, upper_band, lower_band

def calculate_rsi(prices, period=14):
    """Calculate RSI"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_sma(prices, period):
    """Calculate Simple Moving Average"""
    return prices.rolling(window=period).mean()

def apply_transaction_cost(value, transaction_cost_rate):
    """Apply transaction cost and return fee amount"""
    fee = abs(value) * transaction_cost_rate / 100
    return fee

def calc_max_drawdown(equity_curve):
    """Calculate maximum drawdown"""
    peak = equity_curve.expanding().max()
    drawdown = (equity_curve - peak) / peak
    return drawdown.min() * 100

def calc_cagr(equity_curve, start_date, end_date):
    """Calculate Compound Annual Growth Rate"""
    if isinstance(equity_curve, list):
        equity_curve = pd.Series(equity_curve)
    
    if len(equity_curve) < 2:
        return None
        
    years = (end_date - start_date).days / 365.25
    if years <= 0:
        return None
        
    start_value = equity_curve.iloc[0]
    end_value = equity_curve.iloc[-1]
    
    if start_value <= 0:
        return None
        
    cagr = ((end_value / start_value) ** (1/years) - 1) * 100
    return cagr

def calc_profit_factor(trades):
    """Calculate profit factor"""
    if not trades:
        return None
        
    gross_profit = sum(trade['profit'] for trade in trades if trade.get('profit', 0) > 0)
    gross_loss = abs(sum(trade['profit'] for trade in trades if trade.get('profit', 0) < 0))
    
    if gross_loss == 0:
        return None if gross_profit == 0 else float('inf')
    
    return gross_profit / gross_loss

def calc_win_rate(trades):
    """Calculate win rate percentage"""
    if not trades:
        return 0
        
    winning_trades = sum(1 for trade in trades if trade.get('profit', 0) > 0)
    return (winning_trades / len(trades)) * 100

def calc_average_profit_per_trade(trades):
    """Calculate average profit per trade percentage"""
    if not trades:
        return 0
        
    total_profit = sum(trade.get('profit', 0) for trade in trades)
    return total_profit / len(trades)

def bollinger_bands_strategy(data, period=20, multiplier=2, stop_loss_pct=5, take_profit_pct=10, 
                           transaction_cost_rate=0.1, initial_capital=10000):
    """Bollinger Bands Strategy"""
    if len(data) < period:
        return {
            'equity_curve': [initial_capital],
            'trades': [],
            'metrics': {},
            'parameters': {
                'period': period,
                'multiplier': multiplier,
                'stop_loss_pct': stop_loss_pct,
                'take_profit_pct': take_profit_pct,
                'transaction_cost_rate': transaction_cost_rate
            }
        }
    
    prices = data['Close']
    sma, upper_band, lower_band = calculate_bollinger_bands(prices, period, multiplier)
    
    capital = initial_capital
    position = 0
    entry_price = 0
    trades = []
    equity_curve = [capital]
    total_fees = 0
    
    for i in range(1, len(data)):
        current_price = prices.iloc[i]
        current_date = data.index[i]
        
        # Skip if we don't have enough data for Bollinger Bands
        if pd.isna(lower_band.iloc[i]) or pd.isna(upper_band.iloc[i]):
            equity_curve.append(capital + (position * current_price))
            continue
            
        # Check for stop loss or take profit
        if position > 0:
            profit_pct = (current_price - entry_price) / entry_price * 100
            
            if profit_pct <= -stop_loss_pct or profit_pct >= take_profit_pct:
                # Sell signal
                sell_value = position * current_price
                fee = apply_transaction_cost(sell_value, transaction_cost_rate)
                net_proceeds = sell_value - fee
                profit = net_proceeds - (position * entry_price)
                
                capital = net_proceeds
                total_fees += fee
                
                trades.append({
                    'date': current_date,
                    'action': 'SELL',
                    'price': current_price,
                    'shares': position,
                    'value': -sell_value,
                    'fee': fee,
                    'profit': profit,
                    'reason': f'Stop Loss' if profit_pct <= -stop_loss_pct else 'Take Profit',
                    'strategy': 'Bollinger Bands'
                })
                
                position = 0
                entry_price = 0
        
        # Entry signals
        if position == 0:
            if current_price <= lower_band.iloc[i]:
                # Buy signal
                shares_to_buy = calculate_shares_to_buy(capital, current_price)
                if shares_to_buy > 0:
                    purchase_value = shares_to_buy * current_price
                    fee = apply_transaction_cost(purchase_value, transaction_cost_rate)
                    total_cost = purchase_value + fee
                    
                    if total_cost <= capital:
                        capital -= total_cost
                        position = shares_to_buy
                        entry_price = current_price
                        total_fees += fee
                        
                        trades.append({
                            'date': current_date,
                            'action': 'BUY',
                            'price': current_price,
                            'shares': shares_to_buy,
                            'value': purchase_value,
                            'fee': fee,
                            'profit': 0,
                            'reason': 'Price below Lower Band',
                            'strategy': 'Bollinger Bands'
                        })
            
        
        elif position > 0:
            if current_price >= upper_band.iloc[i]:
                # Sell signal
                sell_value = position * current_price
                fee = apply_transaction_cost(sell_value, transaction_cost_rate)
                net_proceeds = sell_value - fee
                profit = net_proceeds - (position * entry_price)
                
                capital = net_proceeds
                total_fees += fee
                
                trades.append({
                    'date': current_date,
                    'action': 'SELL',
                    'price': current_price,
                    'shares': position,
                    'value': -sell_value,
                    'fee': fee,
                    'profit': profit,
                    'reason': 'Price above Upper Band',
                    'strategy': 'Bollinger Bands'
                })
                
                position = 0
        
        if position > 0 and current_price >= upper_band.iloc[i]:
            # Sell signal
            sell_value = position * current_price
            fee = apply_transaction_cost(sell_value, transaction_cost_rate)
            net_proceeds = sell_value - fee
            profit = net_proceeds - (position * entry_price)
            
            capital = net_proceeds
            total_fees += fee
            
            trades.append({
                'date': current_date,
                'action': 'SELL',
                'price': current_price,
                'shares': position,
                'value': -sell_value,
                'fee': fee,
                'profit': profit,
                'reason': 'Price above Upper Band',
                'strategy': 'Bollinger Bands'
            })
            
            position = 0
            entry_price = 0
        # Update equity curve
        current_equity = capital + (position * current_price)
        equity_curve.append(current_equity)
    
    # Close any open position at the end
    if position > 0:
        final_price = prices.iloc[-1]
        sell_value = position * final_price
        fee = apply_transaction_cost(sell_value, transaction_cost_rate)
        net_proceeds = sell_value - fee
        profit = net_proceeds - (position * entry_price)
        
        capital = net_proceeds
        total_fees += fee
        
        trades.append({
            'date': data.index[-1],
            'action': 'SELL',
            'price': final_price,
            'shares': position,
            'value': -sell_value,
            'fee': fee,
            'profit': profit,
            'reason': 'End of Period',
            'strategy': 'Bollinger Bands'
        })
        
        equity_curve[-1] = capital
    
    # Calculate metrics
    final_value = equity_curve[-1]
    net_return_pct = (final_value - initial_capital) / initial_capital * 100
    max_drawdown_pct = calc_max_drawdown(pd.Series(equity_curve))
    
    start_date = data.index[0]
    end_date = data.index[-1]
    cagr_pct = calc_cagr(equity_curve, start_date, end_date)
    
    num_trades = len([t for t in trades if t['action'] == 'SELL'])
    win_rate_pct = calc_win_rate([t for t in trades if t['action'] == 'SELL'])
    avg_profit_per_trade_pct = calc_average_profit_per_trade([t for t in trades if t['action'] == 'SELL'])
    profit_factor = calc_profit_factor([t for t in trades if t['action'] == 'SELL'])
    
    metrics = {
        'initial_capital': initial_capital,
        'final_value': final_value,
        'net_return_pct': net_return_pct,
        'cagr_pct': cagr_pct,
        'max_drawdown_pct': max_drawdown_pct,
        'num_trades': num_trades,
        'win_rate_pct': win_rate_pct,
        'avg_profit_per_trade_pct': avg_profit_per_trade_pct,
        'profit_factor': profit_factor,
        'total_fees': total_fees
    }
    
    return {
        'equity_curve': equity_curve,
        'trades': trades,
        'metrics': metrics,
        'parameters': {
            'period': period,
            'multiplier': multiplier,
            'stop_loss_pct': stop_loss_pct,
            'take_profit_pct': take_profit_pct,
            'transaction_cost_rate': transaction_cost_rate
        }
    }

def ma_crossover_strategy(data, short_period=10, long_period=50, stop_loss_pct=5, 
                         take_profit_pct=10, transaction_cost_rate=0.1, initial_capital=10000):
    """Moving Average Crossover Strategy"""
    if len(data) < long_period:
        return {
            'equity_curve': [initial_capital],
            'trades': [],
            'metrics': {},
            'parameters': {
                'short_period': short_period,
                'long_period': long_period,
                'stop_loss_pct': stop_loss_pct,
                'take_profit_pct': take_profit_pct,
                'transaction_cost_rate': transaction_cost_rate
            }
        }
    
    prices = data['Close']
    short_ma = calculate_sma(prices, short_period)
    long_ma = calculate_sma(prices, long_period)
    
    capital = initial_capital
    position = 0
    entry_price = 0
    trades = []
    equity_curve = [capital]
    total_fees = 0
    
    for i in range(1, len(data)):
        current_price = prices.iloc[i]
        current_date = data.index[i]
        
        # Skip if we don't have enough data for moving averages
        if pd.isna(short_ma.iloc[i]) or pd.isna(long_ma.iloc[i]):
            equity_curve.append(capital + (position * current_price))
            continue
            
        # Check for stop loss or take profit
        if position > 0:
            profit_pct = (current_price - entry_price) / entry_price * 100
            
            if profit_pct <= -stop_loss_pct or profit_pct >= take_profit_pct:
                # Sell signal
                sell_value = position * current_price
                fee = apply_transaction_cost(sell_value, transaction_cost_rate)
                net_proceeds = sell_value - fee
                profit = net_proceeds - (position * entry_price)
                
                capital = net_proceeds
                total_fees += fee
                
                trades.append({
                    'date': current_date,
                    'action': 'SELL',
                    'price': current_price,
                    'shares': position,
                    'value': -sell_value,
                    'fee': fee,
                    'profit': profit,
                    'reason': f'Stop Loss' if profit_pct <= -stop_loss_pct else 'Take Profit',
                    'strategy': 'MA Crossover'
                })
                
                position = 0
                entry_price = 0
        
        # Entry and exit signals
        if i > 0:  # Need previous values for crossover detection
            prev_short_ma = short_ma.iloc[i-1]
            prev_long_ma = long_ma.iloc[i-1]
            curr_short_ma = short_ma.iloc[i]
            curr_long_ma = long_ma.iloc[i]
            
            # Buy signal: short MA crosses above long MA
            if position == 0 and prev_short_ma <= prev_long_ma and curr_short_ma > curr_long_ma:
                shares_to_buy = calculate_shares_to_buy(capital, current_price)
                if shares_to_buy > 0:
                    purchase_value = shares_to_buy * current_price
                    fee = apply_transaction_cost(purchase_value, transaction_cost_rate)
                    total_cost = purchase_value + fee
                    
                    if total_cost <= capital:
                        capital -= total_cost
                        position = shares_to_buy
                        entry_price = current_price
                        total_fees += fee
                        
                        trades.append({
                            'date': current_date,
                            'action': 'BUY',
                            'price': current_price,
                            'shares': shares_to_buy,
                            'value': purchase_value,
                            'fee': fee,
                            'profit': 0,
                            'reason': 'Short MA crosses above Long MA',
                            'strategy': 'MA Crossover'
                        })
            
            # Sell signal: short MA crosses below long MA
            elif position > 0 and prev_short_ma >= prev_long_ma and curr_short_ma < curr_long_ma:
                sell_value = position * current_price
                fee = apply_transaction_cost(sell_value, transaction_cost_rate)
                net_proceeds = sell_value - fee
                profit = net_proceeds - (position * entry_price)
                
                capital = net_proceeds
                total_fees += fee
                
                trades.append({
                    'date': current_date,
                    'action': 'SELL',
                    'price': current_price,
                    'shares': position,
                    'value': -sell_value,
                    'fee': fee,
                    'profit': profit,
                    'reason': 'Short MA crosses below Long MA',
                    'strategy': 'MA Crossover'
                })
                
                position = 0
                entry_price = 0
        
        # Update equity curve
        current_equity = capital + (position * current_price)
        equity_curve.append(current_equity)
    
    # Close any open position at the end
    if position > 0:
        final_price = prices.iloc[-1]
        sell_value = position * final_price
        fee = apply_transaction_cost(sell_value, transaction_cost_rate)
        net_proceeds = sell_value - fee
        profit = net_proceeds - (position * entry_price)
        
        capital = net_proceeds
        total_fees += fee
        
        trades.append({
            'date': data.index[-1],
            'action': 'SELL',
            'price': final_price,
            'shares': position,
            'value': -sell_value,
            'fee': fee,
            'profit': profit,
            'reason': 'End of Period',
            'strategy': 'MA Crossover'
        })
        
        equity_curve[-1] = capital
    
    # Calculate metrics
    final_value = equity_curve[-1]
    net_return_pct = (final_value - initial_capital) / initial_capital * 100
    max_drawdown_pct = calc_max_drawdown(pd.Series(equity_curve))
    
    start_date = data.index[0]
    end_date = data.index[-1]
    cagr_pct = calc_cagr(equity_curve, start_date, end_date)
    
    num_trades = len([t for t in trades if t['action'] == 'SELL'])
    win_rate_pct = calc_win_rate([t for t in trades if t['action'] == 'SELL'])
    avg_profit_per_trade_pct = calc_average_profit_per_trade([t for t in trades if t['action'] == 'SELL'])
    profit_factor = calc_profit_factor([t for t in trades if t['action'] == 'SELL'])
    
    metrics = {
        'initial_capital': initial_capital,
        'final_value': final_value,
        'net_return_pct': net_return_pct,
        'cagr_pct': cagr_pct,
        'max_drawdown_pct': max_drawdown_pct,
        'num_trades': num_trades,
        'win_rate_pct': win_rate_pct,
        'avg_profit_per_trade_pct': avg_profit_per_trade_pct,
        'profit_factor': profit_factor,
        'total_fees': total_fees
    }
    
    return {
        'equity_curve': equity_curve,
        'trades': trades,
        'metrics': metrics,
        'parameters': {
            'short_period': short_period,
            'long_period': long_period,
            'stop_loss_pct': stop_loss_pct,
            'take_profit_pct': take_profit_pct,
            'transaction_cost_rate': transaction_cost_rate
        }
    }

def rsi_strategy(data, rsi_period=14, oversold=30, overbought=70, stop_loss_pct=5, 
                take_profit_pct=10, transaction_cost_rate=0.1, initial_capital=10000):
    """RSI Strategy"""
    if len(data) < rsi_period:
        return {
            'equity_curve': [initial_capital],
            'trades': [],
            'metrics': {},
            'parameters': {
                'rsi_period': rsi_period,
                'oversold': oversold,
                'overbought': overbought,
                'stop_loss_pct': stop_loss_pct,
                'take_profit_pct': take_profit_pct,
                'transaction_cost_rate': transaction_cost_rate
            }
        }
    
    prices = data['Close']
    rsi = calculate_rsi(prices, rsi_period)
    
    capital = initial_capital
    position = 0
    entry_price = 0
    trades = []
    equity_curve = [capital]
    total_fees = 0
    
    for i in range(1, len(data)):
        current_price = prices.iloc[i]
        current_date = data.index[i]
        
        # Skip if we don't have enough data for RSI
        if pd.isna(rsi.iloc[i]):
            equity_curve.append(capital + (position * current_price))
            continue
            
        current_rsi = rsi.iloc[i]
        
        # Check for stop loss or take profit
        if position > 0:
            profit_pct = (current_price - entry_price) / entry_price * 100
            
            if profit_pct <= -stop_loss_pct or profit_pct >= take_profit_pct or current_rsi > overbought:
                # Sell signal
                sell_value = position * current_price
                fee = apply_transaction_cost(sell_value, transaction_cost_rate)
                net_proceeds = sell_value - fee
                profit = net_proceeds - (position * entry_price)
                
                capital = net_proceeds
                total_fees += fee
                
                reason = 'RSI Overbought' if current_rsi > overbought else ('Stop Loss' if profit_pct <= -stop_loss_pct else 'Take Profit')
            if profit_pct <= -stop_loss_pct:
                # Sell signal - Stop Loss
                # reason will be set below based on the trigger
            if profit_pct <= -stop_loss_pct:
                # Sell signal - Stop Loss
                sell_value = position * current_price
                fee = apply_transaction_cost(sell_value, transaction_cost_rate)
                net_proceeds = sell_value - fee
                profit = net_proceeds - (position * entry_price)
                capital = net_proceeds
                total_fees += fee
                trades.append({
                    'date': current_date,
                    'action': 'SELL',
                    'price': current_price,
                    'shares': position,
                    'value': -sell_value,
                    'fee': fee,
                    'profit': profit,
                    'reason': reason,
                    'strategy': 'RSI'
                })
                
                position = 0
                entry_price = 0
        
        # Entry signals
        elif position == 0 and current_rsi < oversold:
            # Buy signal
            shares_to_buy = int(np.floor(capital / current_price))
            if shares_to_buy > 0:
                purchase_value = shares_to_buy * current_price
                fee = apply_transaction_cost(purchase_value, transaction_cost_rate)
                total_cost = purchase_value + fee
                
                if total_cost <= capital:
                    capital -= total_cost
                    position = shares_to_buy
                    entry_price = current_price
                    total_fees += fee
                    
                    trades.append({
                        'date': current_date,
                        'action': 'BUY',
                        'price': current_price,
                        'shares': shares_to_buy,
                        'value': purchase_value,
                        'fee': fee,
                        'profit': 0,
                        'reason': 'RSI Oversold',
                        'strategy': 'RSI'
                    })
        
        # Update equity curve
        current_equity = capital + (position * current_price)
        equity_curve.append(current_equity)
    
    # Close any open position at the end
    if position > 0:
        final_price = prices.iloc[-1]
        sell_value = position * final_price
        fee = apply_transaction_cost(sell_value, transaction_cost_rate)
        net_proceeds = sell_value - fee
        profit = net_proceeds - (position * entry_price)
        
        capital = net_proceeds
        total_fees += fee
        
        trades.append({
            'date': data.index[-1],
            'action': 'SELL',
            'price': final_price,
            'shares': position,
            'value': -sell_value,
            'fee': fee,
            'profit': profit,
            'reason': 'End of Period',
            'strategy': 'RSI'
        })
        
        equity_curve[-1] = capital
    
    # Calculate metrics
    final_value = equity_curve[-1]
    net_return_pct = (final_value - initial_capital) / initial_capital * 100
    max_drawdown_pct = calc_max_drawdown(pd.Series(equity_curve))
    
    start_date = data.index[0]
    end_date = data.index[-1]
    cagr_pct = calc_cagr(equity_curve, start_date, end_date)
    
    num_trades = len([t for t in trades if t['action'] == 'SELL'])
    win_rate_pct = calc_win_rate([t for t in trades if t['action'] == 'SELL'])
    avg_profit_per_trade_pct = calc_average_profit_per_trade([t for t in trades if t['action'] == 'SELL'])
    profit_factor = calc_profit_factor([t for t in trades if t['action'] == 'SELL'])
    
    metrics = {
        'initial_capital': initial_capital,
        'final_value': final_value,
        'net_return_pct': net_return_pct,
        'cagr_pct': cagr_pct,
        'max_drawdown_pct': max_drawdown_pct,
        'num_trades': num_trades,
        'win_rate_pct': win_rate_pct,
        'avg_profit_per_trade_pct': avg_profit_per_trade_pct,
        'profit_factor': profit_factor,
        'total_fees': total_fees
    }
    
    return {
        'equity_curve': equity_curve,
        'trades': trades,
        'metrics': metrics,
        'parameters': {
            'rsi_period': rsi_period,
            'oversold': oversold,
            'overbought': overbought,
            'stop_loss_pct': stop_loss_pct,
            'take_profit_pct': take_profit_pct,
            'transaction_cost_rate': transaction_cost_rate
        }
    }

def buy_and_hold_strategy(data, transaction_cost_rate=0.1, initial_capital=10000,
                         stop_loss_pct=5, take_profit_pct=10):
    """Buy and Hold Strategy"""
    if len(data) < 2:
        return {
            'equity_curve': [initial_capital],
            'trades': [],
            'metrics': {},
            'parameters': {
                'transaction_cost_rate': transaction_cost_rate,
                'stop_loss_pct': stop_loss_pct,  # Ignored but kept for uniformity
                'take_profit_pct': take_profit_pct  # Ignored but kept for uniformity
            }
        }
    
    prices = data['Close']
    
    # Buy at first available bar
    first_price = prices.iloc[0]
    first_date = data.index[0]
    
    shares_to_buy = int(np.floor(initial_capital / first_price))
    purchase_value = shares_to_buy * first_price
    buy_fee = apply_transaction_cost(purchase_value, transaction_cost_rate)
    total_cost = purchase_value + buy_fee
    
    remaining_capital = initial_capital - total_cost
    
    trades = [{
        'date': first_date,
        'action': 'BUY',
        'price': first_price,
        'shares': shares_to_buy,
        'value': purchase_value,
        'fee': buy_fee,
        'profit': 0,
        'reason': 'Buy and Hold Start',
        'strategy': 'Buy & Hold'
    }]
    
    # Calculate equity curve
    equity_curve = []
    for price in prices:
        current_equity = remaining_capital + (shares_to_buy * price)
        equity_curve.append(current_equity)
    
    # Sell at last bar
    final_price = prices.iloc[-1]
    final_date = data.index[-1]
    
    sell_value = shares_to_buy * final_price
    sell_fee = apply_transaction_cost(sell_value, transaction_cost_rate)
    net_proceeds = sell_value - sell_fee
    profit = net_proceeds - purchase_value
    
    final_capital = remaining_capital + net_proceeds
    total_fees = buy_fee + sell_fee
    
    trades.append({
        'date': final_date,
        'action': 'SELL',
        'price': final_price,
        'shares': shares_to_buy,
        'value': -sell_value,
        'fee': sell_fee,
        'profit': profit,
        'reason': 'Buy and Hold End',
        'strategy': 'Buy & Hold'
    })
    
    # Update final equity value
    equity_curve[-1] = final_capital
    
    # Calculate metrics
    final_value = final_capital
    net_return_pct = (final_value - initial_capital) / initial_capital * 100
    max_drawdown_pct = calc_max_drawdown(pd.Series(equity_curve))
    
    start_date = data.index[0]
    end_date = data.index[-1]
    cagr_pct = calc_cagr(equity_curve, start_date, end_date)
    
    num_trades = 1  # Only one complete trade cycle
    win_rate_pct = 100.0 if profit > 0 else 0.0
    avg_profit_per_trade_pct = (profit / purchase_value) * 100
    profit_factor = None  # N/A for single trade
    
    metrics = {
        'initial_capital': initial_capital,
        'final_value': final_value,
        'net_return_pct': net_return_pct,
        'cagr_pct': cagr_pct,
        'max_drawdown_pct': max_drawdown_pct,
        'num_trades': num_trades,
        'win_rate_pct': win_rate_pct,
        'avg_profit_per_trade_pct': avg_profit_per_trade_pct,
        'profit_factor': profit_factor,
        'total_fees': total_fees
    }
    
    return {
        'equity_curve': equity_curve,
        'trades': trades,
        'metrics': metrics,
        'parameters': {
            'transaction_cost_rate': transaction_cost_rate,
            'stop_loss_pct': stop_loss_pct,
            'take_profit_pct': take_profit_pct
        }
    }

# Strategy Registry
STRATEGIES = {
    'bollinger_bands': {
        'display_name': 'Bollinger Bands',
        'function': bollinger_bands_strategy,
        'parameters': {
            'period': {'type': 'number', 'default': 20, 'min': 5, 'max': 50, 'label': 'ช่วงเวลา (Period)'},
            'multiplier': {'type': 'number', 'default': 2.0, 'min': 1.0, 'max': 3.0, 'step': 0.1, 'label': 'ตัวคูณ (Multiplier)'},
            'stop_loss_pct': {'type': 'number', 'default': 5.0, 'min': 1.0, 'max': 20.0, 'step': 0.5, 'label': 'Stop Loss (%)'},
            'take_profit_pct': {'type': 'number', 'default': 10.0, 'min': 1.0, 'max': 50.0, 'step': 0.5, 'label': 'Take Profit (%)'}
        }
    },
    'ma_crossover': {
        'display_name': 'MA Crossover',
        'function': ma_crossover_strategy,
        'parameters': {
            'short_period': {'type': 'number', 'default': 10, 'min': 5, 'max': 30, 'label': 'MA สั้น (Short Period)'},
            'long_period': {'type': 'number', 'default': 50, 'min': 20, 'max': 200, 'label': 'MA ยาว (Long Period)'},
            'stop_loss_pct': {'type': 'number', 'default': 5.0, 'min': 1.0, 'max': 20.0, 'step': 0.5, 'label': 'Stop Loss (%)'},
            'take_profit_pct': {'type': 'number', 'default': 10.0, 'min': 1.0, 'max': 50.0, 'step': 0.5, 'label': 'Take Profit (%)'}
        }
    },
    'rsi': {
        'display_name': 'RSI',
        'function': rsi_strategy,
        'parameters': {
            'rsi_period': {'type': 'number', 'default': 14, 'min': 5, 'max': 30, 'label': 'RSI Period'},
            'oversold': {'type': 'number', 'default': 30, 'min': 10, 'max': 40, 'label': 'Oversold Level'},
            'overbought': {'type': 'number', 'default': 70, 'min': 60, 'max': 90, 'label': 'Overbought Level'},
            'stop_loss_pct': {'type': 'number', 'default': 5.0, 'min': 1.0, 'max': 20.0, 'step': 0.5, 'label': 'Stop Loss (%)'},
            'take_profit_pct': {'type': 'number', 'default': 10.0, 'min': 1.0, 'max': 50.0, 'step': 0.5, 'label': 'Take Profit (%)'}
        }
    },
    'buy_and_hold': {
        'display_name': 'Buy & Hold',
        'function': buy_and_hold_strategy,
        'parameters': {
            'stop_loss_pct': {'type': 'number', 'default': 5.0, 'min': 1.0, 'max': 20.0, 'step': 0.5, 'label': 'Stop Loss (%) - ไม่ใช้', 'disabled': True},
            'stop_loss_pct': {'type': 'number', 'default': 5.0, 'min': 1.0, 'max': 20.0, 'step': 0.5, 'label': 'Stop Loss (%)', 'help': 'ไม่ใช้ (Not used in Buy & Hold strategy)', 'disabled': True},
            'take_profit_pct': {'type': 'number', 'default': 10.0, 'min': 1.0, 'max': 50.0, 'step': 0.5, 'label': 'Take Profit (%)', 'help': 'ไม่ใช้ (Not used in Buy & Hold strategy)', 'disabled': True}
        }
    }
}

def create_excel_export(all_results, strategy_key):
    """Create Excel file with summary and individual ticker sheets"""
    try:
        output = io.BytesIO()
        
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            # Summary sheet
            summary_data = []
            for ticker, result in all_results.items():
                metrics = result['metrics']
                summary_data.append({
                    'Ticker': ticker,
                    'Strategy': strategy_key,
                    'Initial Capital': metrics.get('initial_capital', 0),
                    'Final Value': metrics.get('final_value', 0),
                    'Net Return (%)': metrics.get('net_return_pct', 0),
                    'CAGR (%)': metrics.get('cagr_pct', 0) if metrics.get('cagr_pct') else 0,
                    'Max Drawdown (%)': metrics.get('max_drawdown_pct', 0),
                    'Number of Trades': metrics.get('num_trades', 0),
                    'Win Rate (%)': metrics.get('win_rate_pct', 0),
                    'Avg Profit per Trade': metrics.get('avg_profit_per_trade_pct', 0),
                    'Profit Factor': metrics.get('profit_factor', 0) if metrics.get('profit_factor') else 0,
                    'Total Fees': metrics.get('total_fees', 0)
                })
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Individual ticker sheets
            for ticker, result in all_results.items():
                # Clean sheet name - remove invalid characters and limit length
                sheet_name = f"{ticker}_{strategy_key}"
                # Replace invalid Excel sheet name characters with underscore
                sheet_name = re.sub(r'[:\\/?*\[\]]', '_', sheet_name)
                sheet_name = sheet_name[:31]  # Excel sheet name limit
                sheet_name = sanitize_sheet_name(f"{ticker}_{strategy_key}")
                
                
                
                if result['trades']:
                    trades_df = pd.DataFrame(result['trades'])
                    trades_df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        return output.getvalue()
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการสร้างไฟล์ Excel: {str(e)}")
        return None

# ----------------- UI & Main -----------------
st.set_page_config(page_title="Warren-DCA วิเคราะห์หุ้น", layout="wide")

# Add version display
st.sidebar.markdown("**เวอร์ชัน:** v0.2.0")

menu = st.sidebar.radio("เลือกหน้าที่ต้องการ", ["วิเคราะห์หุ้น", "Backtesting", "คู่มือการใช้งาน"])

if menu == "Backtesting":
    st.header("🚀 Backtesting ระบบ")
    st.markdown("ทดสอบกลยุทธ์การลงทุนต่างๆ กับข้อมูลในอดีต")
    
    # Strategy selection
    selected_strategy_key = st.selectbox(
        "เลือกกลยุทธ์",
        options=list(STRATEGIES.keys()),
        format_func=lambda x: STRATEGIES[x]['display_name'],
        help="เลือกกลยุทธ์ที่ต้องการทดสอบ"
    )
    
    selected_strategy = STRATEGIES[selected_strategy_key]
    
    # Market and ticker selection (reuse existing logic)
    selected_market = st.selectbox(
        "เลือกตลาดหุ้น",
        options=list(markets.keys()),
        index=0,
        help="เลือกตลาดหุ้นที่ต้องการวิเคราะห์"
    )
    
    available_tickers = markets[selected_market]
    
    default_tickers = []
    if selected_market == "US":
        default_tickers = ["AAPL"]
    elif selected_market == "SET100":
        default_tickers = ["PTT.BK"]
    elif selected_market == "Global":
        default_tickers = ["AAPL", "PTT.BK"]
    else:
        default_tickers = [available_tickers[0]] if available_tickers else []
    
    tickers = st.multiselect(
        f"เลือกหุ้น ({selected_market})",
        available_tickers,
        default=default_tickers,
        help=f"เลือกหุ้นจากตลาด {selected_market} ที่ต้องการทดสอบ (สามารถเลือกหลายตัว)"
    )
    
    # Common parameters
    col1, col2 = st.columns(2)
    with col1:
        initial_capital = st.number_input(
            "เงินทุนเริ่มต้น", 
            min_value=1000.0, 
            max_value=1000000.0, 
            value=10000.0, 
            step=1000.0
        )
        
    with col2:
        transaction_cost_rate = st.number_input(
            "ค่าธรรมเนียมการซื้อขาย (%)", 
            min_value=0.0, 
            max_value=5.0, 
            value=0.1, 
            step=0.01,
            help="ค่าธรรมเนียมที่จะถูกหักทั้งการซื้อและขาย"
        )
    
    # Strategy-specific parameters
    st.subheader(f"พารามิเตอร์สำหรับ {selected_strategy['display_name']}")
    strategy_params = {}
    
    if selected_strategy_key == 'buy_and_hold':
        st.info("📌 กลยุทธ์ Buy & Hold จะซื้อหุ้นในวันแรกและขายในวันสุดท้าย พารามิเตอร์ Stop Loss และ Take Profit จะไม่ถูกใช้งาน")
    
    # Create parameter inputs based on strategy configuration
    param_cols = st.columns(2)
    for i, (param_name, param_config) in enumerate(selected_strategy['parameters'].items()):
        with param_cols[i % 2]:
            if param_config['type'] == 'number':
                disabled = param_config.get('disabled', False)
                if disabled:
                    st.number_input(
                        param_config['label'],
                        value=param_config['default'],
                        disabled=True,
                        help="พารามิเตอร์นี้ไม่ถูกใช้งานในกลยุทธ์ Buy & Hold"
                    )
                    strategy_params[param_name] = param_config['default']
                else:
                    strategy_params[param_name] = st.number_input(
                        param_config['label'],
                        min_value=param_config.get('min', 0),
                        max_value=param_config.get('max', 100),
                        value=param_config['default'],
                        step=param_config.get('step', 1)
                    )
    
    # Date range selection
    period = st.selectbox("เลือกช่วงเวลาทดสอบ", ["1y", "2y", "5y", "max"], index=2)
    
    # Run backtest button
    if st.button("🚀 เริ่มทดสอบ", type="primary"):
        if not tickers:
            st.error("กรุณาเลือกหุ้นที่ต้องการทดสอบ")
        else:
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            results_summary = []
            all_results = {}
            
            for idx, ticker in enumerate(tickers):
                status_text.text(f"กำลังทดสอบ {ticker}...")
                progress_bar.progress((idx + 1) / len(tickers))
                
                try:
                    # Get stock data
                    stock = yf.Ticker(ticker)
                    hist = stock.history(period=period)
                    
                    if hist.empty:
                        st.warning(f"ไม่สามารถดึงข้อมูลสำหรับ {ticker}")
                        continue
                    
                    # Run strategy
                    strategy_params['transaction_cost_rate'] = transaction_cost_rate
                    strategy_params['initial_capital'] = initial_capital
                    
                    result = selected_strategy['function'](hist, **strategy_params)
                    all_results[ticker] = result
                    
                    # Add to summary
                    metrics = result['metrics']
                    results_summary.append({
                        'หุ้น': ticker,
                        'เงินทุนเริ่มต้น': f"{metrics.get('initial_capital', 0):,.2f}",
                        'มูลค่าสุดท้าย': f"{metrics.get('final_value', 0):,.2f}",
                        'ผลตอบแทน (%)': f"{metrics.get('net_return_pct', 0):.2f}%",
                        'CAGR (%)': f"{metrics.get('cagr_pct', 0):.2f}%" if metrics.get('cagr_pct') else "N/A",
                        'Max Drawdown (%)': f"{metrics.get('max_drawdown_pct', 0):.2f}%",
                        'จำนวนเทรด': metrics.get('num_trades', 0),
                        'อัตราชนะ (%)': f"{metrics.get('win_rate_pct', 0):.2f}%",
                        'กำไรเฉลี่ยต่อเทรด': f"{metrics.get('avg_profit_per_trade_pct', 0):.2f}",
                        'Profit Factor': f"{metrics.get('profit_factor', 0):.2f}" if metrics.get('profit_factor') else "N/A",
                        'ค่าธรรมเนียมรวม': f"{metrics.get('total_fees', 0):.2f}"
                    })
                    
                except Exception as e:
                    st.error(f"เกิดข้อผิดพลาดกับ {ticker}: {str(e)}")
            
            progress_bar.empty()
            status_text.empty()
            
            if results_summary:
                # Display summary table
                st.subheader("📊 สรุปผลการทดสอบ")
                summary_df = pd.DataFrame(results_summary)
                st.dataframe(summary_df, use_container_width=True)
                
                # Excel export
                if len(all_results) > 0:
                    excel_data = create_excel_export(all_results, selected_strategy_key)
                    if excel_data:
                        st.download_button(
                            label="📥 ดาวน์โหลดผลลัพธ์เป็น Excel",
                            data=excel_data,
                            file_name=f'Backtest_Results_{selected_strategy_key}.xlsx',
                            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                        )
                
                # Individual ticker results
                for ticker, result in all_results.items():
                    with st.expander(f"📈 รายละเอียด {ticker} - {selected_strategy['display_name']}", expanded=False):
                        
                        # Metrics
                        st.subheader("ตัวชี้วัดผลการดำเนินงาน")
                        metrics = result['metrics']
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("เงินทุนเริ่มต้น", f"{metrics.get('initial_capital', 0):,.2f}")
                            st.metric("มูลค่าสุดท้าย", f"{metrics.get('final_value', 0):,.2f}")
                        with col2:
                            st.metric("ผลตอบแทน (%)", f"{metrics.get('net_return_pct', 0):.2f}%")
                            st.metric("CAGR (%)", f"{metrics.get('cagr_pct', 0):.2f}%" if metrics.get('cagr_pct') else "N/A")
                        with col3:
                            st.metric("Max Drawdown (%)", f"{metrics.get('max_drawdown_pct', 0):.2f}%")
                            st.metric("จำนวนเทรด", metrics.get('num_trades', 0))
                        with col4:
                            st.metric("อัตราชนะ (%)", f"{metrics.get('win_rate_pct', 0):.2f}%")
                            st.metric("Profit Factor", f"{metrics.get('profit_factor', 0):.2f}" if metrics.get('profit_factor') else "N/A")
                        
                        # Equity curve chart
                        st.subheader("เส้นกราฟมูลค่าพอร์ต")
                        equity_df = pd.DataFrame({
                            'Equity': result['equity_curve']
                        })
                        st.line_chart(equity_df)
                        
                        # Drawdown chart (optional)
                        if len(result['equity_curve']) > 1:
                            equity_series = pd.Series(result['equity_curve'])
                            peak = equity_series.expanding().max()
                            drawdown = (equity_series - peak) / peak * 100
                            
                            st.subheader("กราฟ Drawdown")
                            drawdown_df = pd.DataFrame({
                                'Drawdown (%)': drawdown
                            })
                            st.line_chart(drawdown_df)
                        
                        # Trades table
                        if result['trades']:
                            st.subheader("ตารางธุรกรรม")
                            trades_df = pd.DataFrame(result['trades'])
                            trades_df['date'] = pd.to_datetime(trades_df['date'])
                            trades_df['date'] = trades_df['date'].dt.strftime('%Y-%m-%d')
                            st.dataframe(trades_df, use_container_width=True)
                        else:
                            st.info("ไม่มีธุรกรรมในช่วงเวลานี้")

elif menu == "คู่มือการใช้งาน":
    st.header("📚 คู่มือการใช้งาน (ภาษาไทย)")
    st.markdown("---")
    
    # Version display
    st.markdown("**เวอร์ชัน:** v0.2.0")
    st.markdown("---")
    
    # 1. ภาพรวมระบบ
    st.subheader("1. 📋 ภาพรวมระบบ")
    st.markdown("""
    **Warren-DCA** เป็นระบบวิเคราะห์หุ้นแบบครบวงจร ประกอบด้วย 3 ส่วนหลัก:
    
    - **วิเคราะห์หุ้น:** ตรวจสอบหุ้นตามหลัก Warren Buffett (18 เงื่อนไข) + จำลอง DCA
    - **Backtesting:** ทดสอบกลยุทธ์การลงทุนต่างๆ กับข้อมูลย้อนหลัง
    - **คู่มือการใช้งาน:** เอกสารประกอบการใช้งาน
    
    **แหล่งข้อมูล:** Yahoo Finance
    """)
    
    # 2. ขั้นตอนการใช้งาน
    st.subheader("2. 🛠️ ขั้นตอนการใช้งาน")
    
    tab1, tab2 = st.tabs(["📊 วิเคราะห์หุ้น", "🚀 Backtesting"])
    
    with tab1:
        st.markdown("""
        **ขั้นตอนการวิเคราะห์หุ้น:**
        1. เลือกตลาดหุ้น (US, SET100, Europe, Asia, Australia, Global)
        2. เลือกหุ้นที่ต้องการวิเคราะห์ (สามารถเลือกหลายตัว)
        3. ตั้งค่าช่วงเวลาราคาหุ้น (1y, 5y, max)
        4. ระบุจำนวนเงินลงทุน DCA ต่อเดือน
        5. คลิก "วิเคราะห์"
        6. ดูผลลัพธ์และดาวน์โหลด Excel
        """)
    
    with tab2:
        st.markdown("""
        **ขั้นตอนการ Backtesting:**
        1. เลือกกลยุทธ์ที่ต้องการทดสอบ
        2. เลือกตลาดหุ้นและหุ้นที่ต้องการ
        3. ตั้งค่าเงินทุนเริ่มต้นและค่าธรรมเนียม
        4. ปรับพารามิเตอร์ของกลยุทธ์
        5. เลือกช่วงเวลาทดสอบ
        6. คลิก "เริ่มทดสอบ"
        7. ดูผลลัพธ์และดาวน์โหลด Excel
        """)
    
    # 3. กลยุทธ์ที่มี
    st.subheader("3. 🎯 กลยุทธ์ที่มี")
    
    strategy_tabs = st.tabs(["📈 Bollinger Bands", "↗️ MA Crossover", "📊 RSI", "🏦 Buy & Hold"])
    
    with strategy_tabs[0]:
        st.markdown("""
        **Bollinger Bands Strategy**
        
        **หลักการ:**
        - ซื้อเมื่อราคาแตะแถบล่าง (Lower Band)
        - ขายเมื่อราคาแตะแถบบน (Upper Band)
        - มี Stop Loss และ Take Profit เพิ่มเติม
        
        **พารามิเตอร์:**
        - **ช่วงเวลา (Period):** 20 วัน (ค่าเริ่มต้น)
        - **ตัวคูณ (Multiplier):** 2.0 (ค่าเริ่มต้น)
        - **Stop Loss (%):** 5% (ค่าเริ่มต้น)
        - **Take Profit (%):** 10% (ค่าเริ่มต้น)
        """)
    
    with strategy_tabs[1]:
        st.markdown("""
        **MA Crossover Strategy**
        
        **หลักการ:**
        - ซื้อเมื่อ Moving Average สั้นตัดขึ้นเหนือ MA ยาว
        - ขายเมื่อ Moving Average สั้นตัดลงต่ำกว่า MA ยาว
        - มี Stop Loss และ Take Profit เพิ่มเติม
        
        **พารามิเตอร์:**
        - **MA สั้น (Short Period):** 10 วัน (ค่าเริ่มต้น)
        - **MA ยาว (Long Period):** 50 วัน (ค่าเริ่มต้น)
        - **Stop Loss (%):** 5% (ค่าเริ่มต้น)
        - **Take Profit (%):** 10% (ค่าเริ่มต้น)
        """)
    
    with strategy_tabs[2]:
        st.markdown("""
        **RSI Strategy**
        
        **หลักการ:**
        - ซื้อเมื่อ RSI ต่ำกว่าระดับ Oversold
        - ขายเมื่อ RSI สูงกว่าระดับ Overbought
        - มี Stop Loss และ Take Profit เพิ่มเติม
        
        **พารามิเตอร์:**
        - **RSI Period:** 14 วัน (ค่าเริ่มต้น)
        - **Oversold Level:** 30 (ค่าเริ่มต้น)
        - **Overbought Level:** 70 (ค่าเริ่มต้น)
        - **Stop Loss (%):** 5% (ค่าเริ่มต้น)
        - **Take Profit (%):** 10% (ค่าเริ่มต้น)
        """)
    
    with strategy_tabs[3]:
        st.markdown("""
        **Buy & Hold Strategy**
        
        **หลักการ:**
        - ซื้อหุ้นในวันแรกของช่วงทดสอบ
        - ถือหุ้นตลอดช่วงเวลา
        - ขายหุ้นในวันสุดท้าย
        
        **พารามิเตอร์:**
        - **Stop Loss/Take Profit:** ไม่ใช้งาน (เพื่อความสม่ำเสมอในระบบ)
        - เฉพาะค่าธรรมเนียมเท่านั้นที่มีผล
        """)
    
    # 4. คำอธิบายตัวชี้วัด
    st.subheader("4. 📊 คำอธิบายตัวชี้วัด")
    
    metric_tabs = st.tabs(["💰 ผลตอบแทน", "📈 ความเสี่ยง", "🔄 การซื้อขาย"])
    
    with metric_tabs[0]:
        st.markdown("""
        **ตัวชี้วัดผลตอบแทน:**
        
        - **Final Value:** มูลค่าสุดท้ายของพอร์ต
        - **Net Return (%):** ผลตอบแทนสุทธิ เปรียบเทียบกับเงินทุนเริ่มต้น
        - **CAGR (%):** อัตราผลตอบแทนแบบทบต้นต่อปี (ถ้าช่วงเวลา > 365 วัน)
        """)
    
    with metric_tabs[1]:
        st.markdown("""
        **ตัวชี้วัดความเสี่ยง:**
        
        - **Max Drawdown (%):** การลดลงสูงสุดจากจุดสูงสุด
        - **Sharpe Ratio:** อัตราส่วนผลตอบแทนต่อความเสี่ยง (ถ้ามี)
        """)
    
    with metric_tabs[2]:
        st.markdown("""
        **ตัวชี้วัดการซื้อขาย:**
        
        - **Number of Trades:** จำนวนรอบการซื้อขาย (นับเฉพาะการขาย)
        - **Win Rate (%):** อัตราการทำกำไรต่อจำนวนเทรด
        - **Profit Factor:** อัตราส่วนกำไรรวมต่อขาดทุนรวม
        - **Average Profit per Trade:** กำไรเฉลี่ยต่อเทรด
        - **Total Fees:** ค่าธรรมเนียมรวมทั้งหมด
        """)
    
    # 5. ข้อจำกัด & สมมติฐาน
    st.subheader("5. ⚠️ ข้อจำกัด & สมมติฐาน")
    st.markdown("""
    **ข้อจำกัดของระบบ:**
    - ไม่มีการคำนวณ Slippage (ส่วนต่างราคาจากการซื้อขายจริง)
    - ใช้ข้อมูลย้อนหลังเท่านั้น (Historical Data)
    - มี Survivorship Bias (หุ้นที่ยังคงซื้อขายได้)
    - ไม่รวมเงินปันผลใน Backtesting
    - ใช้ราคาปิดในการซื้อขาย (ไม่ใช่ราคาเปิด)
    
    **สมมติฐาน:**
    - สามารถซื้อขายได้ในปริมาณที่ต้องการ
    - ค่าธรรมเนียมคงที่ทุกครั้ง
    - ไม่มีต้นทุนการถือครอง (Carrying Cost)
    - ข้อมูลราคาถูกต้องและครบถ้วน
    """)
    
    # 6. คำเตือนความเสี่ยง
    st.subheader("6. 🚨 คำเตือนความเสี่ยง")
    st.error("""
    **คำเตือนสำคัญ:**
    
    - ผลการดำเนินงานในอดีตไม่ใช่การรับประกันผลตอบแทนในอนาคต
    - การลงทุนมีความเสี่ยง อาจได้รับกำไรหรือขาดทุนได้
    - ควรศึกษาข้อมูลและปรึกษาผู้เชี่ยวชาญก่อนตัดสินใจลงทุน
    - ระบบนี้เป็นเครื่องมือช่วยวิเคราะห์เท่านั้น ไม่ใช่คำแนะนำการลงทุน
    - ผู้ใช้ต้องรับผิดชอบการตัดสินใจลงทุนด้วยตนเอง
    """)
    
    # 7. Buffett 18 Checklist (from original)
    st.subheader("7. 📋 Buffett 18 Checklist (การวิเคราะห์หุ้น)")
    st.markdown("""
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
    """)
    
    st.markdown("---")
    st.markdown("**Warren-DCA v0.2.0** | Powered by Yahoo Finance")
    st.stop()

st.caption("Powered by Yahoo Finance | วิเคราะห์หุ้นด้วย Buffett Checklist (ขยาย 18 เงื่อนไข) + DCA + ปันผลย้อนหลัง 1 ปี | เวอร์ชัน v0.2.0")