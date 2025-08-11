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

def dca_simulation(hist_prices: pd.DataFrame, monthly_invest: float = 1000, div=None, commission_fee: float = 0.0):
    if hist_prices.empty:
        return {"error": "ไม่มีข้อมูลราคาหุ้น"}
    prices = hist_prices['Close'].resample('M').first().dropna()
    
    # Calculate commission fee per transaction
    commission_per_transaction = monthly_invest * (commission_fee / 100)
    effective_invest_amount = monthly_invest - commission_per_transaction
    
    units = effective_invest_amount / prices
    total_units = units.sum()
    total_invested = monthly_invest * len(prices)  # Total invested including fees
    total_fees = commission_per_transaction * len(prices)
    effective_total_invested = total_invested - total_fees  # Amount actually used to buy stocks
    avg_buy_price = effective_total_invested / total_units if total_units != 0 else 0
    latest_price = prices.iloc[-1]
    current_value = total_units * latest_price
    profit = current_value - total_invested  # Profit accounts for fees
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
        "ค่าคอมมิชชั่นรวม": round(total_fees, 2),
        "เงินซื้อหุ้นจริง": round(effective_total_invested, 2),
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

# ----------------- SET100/US/GLOBAL STOCKS -----------------
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
global_stocks = [
    # Europe
    "ASML", "NESN.SW", "RDSA.L", "SAP", "LVMH.PA", "MC.PA", "OR.PA", "SAN.PA", "INGA.AS", "ADS.DE",
    "SAP.DE", "SIE.DE", "ALV.DE", "DTE.DE", "BAS.DE", "BAYN.DE", "BMW.DE", "DAI.DE", "VOW3.DE", "FRE.DE",
    # UK
    "SHEL.L", "AZN.L", "ULVR.L", "DGE.L", "VOD.L", "GSK.L", "BP.L", "RIO.L", "BHP.L", "HSBA.L",
    # Japan
    "7203.T", "6758.T", "6861.T", "8306.T", "9984.T", "6954.T", "7267.T", "4502.T", "8316.T", "6098.T",
    # Australia
    "CBA.AX", "BHP.AX", "CSL.AX", "WBC.AX", "ANZ.AX", "NAB.AX", "WES.AX", "WOW.AX", "TLS.AX", "RIO.AX",
    # South Korea  
    "005930.KS", "000660.KS", "035420.KS", "051910.KS", "035720.KS", "028260.KS", "066570.KS", "105560.KS",
    # Hong Kong
    "0700.HK", "0941.HK", "1299.HK", "0388.HK", "2318.HK", "1398.HK", "3988.HK", "0005.HK", "0883.HK", "2628.HK"
]

markets = {
    "US": us_stocks,
    "SET100": set100,
    "Global": global_stocks,
    "All": us_stocks + set100 + global_stocks
}

all_tickers = us_stocks + set100 + global_stocks

# ----------------- UI & Main -----------------
st.set_page_config(page_title="Warren-DCA วิเคราะห์หุ้น", layout="wide")
menu = st.sidebar.radio("เลือกหน้าที่ต้องการ", ["วิเคราะห์หุ้น", "คู่มือการใช้งาน"])

if menu == "คู่มือการใช้งาน":
    st.header("คู่มือการใช้งาน (ภาษาไทย)")
    st.markdown("""
**Warren-DCA คืออะไร?**  
โปรแกรมนี้ช่วยวิเคราะห์หุ้นตามแนวทางของ Warren Buffett (Buffett 11 Checklist แบบขยาย 18 เงื่อนไข) พร้อมจำลองการลงทุนแบบ DCA และคำนวณผลตอบแทนเงินปันผลย้อนหลัง 1 ปี  
**แหล่งข้อมูล:** Yahoo Finance  
**ตลาดหุ้นที่รองรับ:** US, SET100 (Thailand), Global (Europe, Asia, Australia)

### ฟีเจอร์ใหม่
- **ค่าคอมมิชชั่น:** สามารถระบุค่าคอมมิชชั่นต่อการซื้อขาย (เช่น 0.25% สำหรับโบรกเกอร์ไทย)
- **ชื่อบริษัทเต็ม:** แสดงชื่อบริษัทเต็มควบคู่กับสัญลักษณ์หุ้น
- **หุ้นสากล:** รองรับหุ้นจากยุโรป เอเชีย และออสเตรเลีย

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
- ค่าคอมมิชชั่นจะถูกหักออกจากเงินลงทุนก่อนซื้อหุ้นในแต่ละครั้ง
""")
    st.stop()

market_selection = st.selectbox("เลือกตลาด", ["All", "US", "SET100", "Global"], index=0)
available_tickers = markets[market_selection]

tickers = st.multiselect(
    f"เลือกหุ้น ({market_selection})",
    available_tickers,
    default=["AAPL", "PTT.BK"] if market_selection == "All" else available_tickers[:2]
)
period = st.selectbox("เลือกช่วงเวลาราคาหุ้น", ["1y", "5y", "max"], index=1)
monthly_invest = st.number_input("จำนวนเงินลงทุน DCA ต่อเดือน (บาทหรือ USD)", min_value=100.0, max_value=10000.0, value=1000.0, step=100.0)
commission_fee = st.number_input("ค่าคอมมิชชั่น (%)", min_value=0.0, max_value=5.0, value=0.0, step=0.1, help="ค่าคอมมิชชั่นต่อการซื้อแต่ละครั้ง เช่น 0.25% สำหรับโบรกเกอร์ไทย")
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
        company_name = info.get('longName', 'N/A')

        with st.expander(f"ดูรายละเอียดหุ้น {ticker} - {company_name}", expanded=False):
            st.subheader(f"ข้อมูลราคาหุ้นและปันผลล่าสุด - {company_name}")
            st.caption(f"บริษัท: {company_name}")

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
            dca_result = dca_simulation(hist, monthly_invest, div, commission_fee)
            st.write(pd.DataFrame(dca_result, index=['สรุปผล']).T)

            # สะสมผลรวม
            total_invest += dca_result["เงินลงทุนรวม"]
            total_profit += dca_result["กำไร/ขาดทุน"]
            total_div += dca_result["เงินปันผลรวม"]

            results_table.append({
                "หุ้น": ticker,
                "ชื่อบริษัท": company_name,
                "เงินลงทุน": dca_result["เงินลงทุนรวม"],
                "ค่าคอมมิชชั่น": dca_result["ค่าคอมมิชชั่นรวม"],
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
                "ค่าคอมมิชชั่นรวม": dca_result["ค่าคอมมิชชั่นรวม"],
                "เงินซื้อหุ้นจริง": dca_result["เงินซื้อหุ้นจริง"],
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