import streamlit as st
import yfinance as yf
import pandas as pd
import io
import datetime
import numpy as np
import matplotlib.pyplot as plt

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    genai = None
    GEMINI_AVAILABLE = True


def setup_gemini(api_key: str) -> bool:
    if not (GEMINI_AVAILABLE and api_key):
        return False
    try:
        genai.configure(api_key=api_key)
        return True
    except Exception:
        return False


def gemini_analyze_company(ticker: str, company_name: str, buffett_detail: dict, dca_result: dict) -> str:
    if not GEMINI_AVAILABLE:
        return "❌ ยังไม่ได้ติดตั้ง google-generativeai (pip install google-generativeai)"
    try:
        details = buffett_detail.get('details', [])
        passed = sum(1 for d in details if d['result'] == 1)
        failed = sum(1 for d in details if d['result'] == 0)
        na = sum(1 for d in details if d['result'] == -1)
        failed_list = [d['title'] for d in details if d['result'] == 0]
        failed_text = ", ".join(failed_list) if failed_list else "ผ่านทุกข้อ หรือไม่มีข้อไม่ผ่านที่ระบุ"
        score_pct = buffett_detail.get('score_pct', 0)
        prompt = f"""
วิเคราะห์หุ้น {ticker} ({company_name}) ภาษาไทยอย่างมีโครงสร้าง:
สรุป Buffett Checklist:
- ผ่าน {passed} ไม่ผ่าน {failed} ไม่มีข้อมูล {na} คะแนนรวม {score_pct}%
- ข้อไม่ผ่าน: {failed_text}

สรุป DCA (จำลอง):
- เงินลงทุนรวม: {dca_result.get('เงินลงทุนรวม')}
- มูลค่าปัจจุบัน: {dca_result.get('มูลค่าปัจจุบัน')}
- กำไร/ขาดทุน: {dca_result.get('กำไร/ขาดทุน')} ({dca_result.get('กำไร(%)')}%)
- เงินปันผลรวม: {dca_result.get('เงินปันผลรวม')}
- ราคาเฉลี่ยที่ซื้อ: {dca_result.get('ราคาเฉลี่ยที่ซื้อ')}
- ราคาปิดล่าสุด: {dca_result.get('ราคาปิดล่าสุด')}

หัวข้อที่ต้องการ:
1) ภาพรวมธุรกิจ/คุณภาพ
2) จุดเด่น
3) ความเสี่ยง / ข้อควรระวัง (อ้างอิงข้อไม่ผ่าน)
4) มุมมองต่อกลยุทธ์ DCA (ไม่ใช่คำแนะนำ)
5) สรุป + Disclaimer
ห้ามเดาตัวเลขใหม่ที่ไม่มีในข้อมูล
"""
        model = genai.GenerativeModel("gemini-1.5-flash")
        resp = model.generate_content(prompt)
        return resp.text or "(ไม่มีข้อความตอบกลับจากโมเดล)"
    except Exception as e:
        return f"⚠️ Gemini error: {str(e)[:200]}"


# ----------------- Helper Functions -----------------
def strip_timezone(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    for col in df.select_dtypes(include=['datetimetz']).columns:
        df[col] = df[col].dt.tz_localize(None)
    for col in df.columns:
        if df[col].dtype == 'object':
            sample = next((v for v in df[col] if isinstance(v, pd.Timestamp)), None)
            if sample is not None and sample.tz is not None:
                df[col] = pd.to_datetime(df[col], errors='coerce').dt.tz_localize(None)
    return df
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
   #DCA_AI_AVAILABLE = True
# Import new DCA modules
try:
    from dca_data_loader import DCADataLoader
    from dca_strategies import DCAStrategy, DCAStrategyFactory
    from dca_optimizer import DCAOptimizer
    from dca_metrics import DCAMetrics
    from ai_dca_helper import DCAAnalysisHelper
 
except ImportError as e:
    print(f"DCA AI modules not available: {e}")
    DCA_AI_AVAILABLE = True

# ----------------- UI & Main -----------------
st.set_page_config(page_title="Warren-DCA วิเคราะห์หุ้น", layout="wide")

# Menu options
menu_options = ["วิเคราะห์หุ้น", "คู่มือการใช้งาน", "DCA AI Optimizer"]
menu = st.sidebar.radio("เลือกหน้าที่ต้องการ", menu_options)
if DCA_AI_AVAILABLE:
    menu_options.insert(1, "DCA AI Optimizer")

menu = st.sidebar.radio("เลือกหน้าที่ต้องการ", menu_options)

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
""")
    st.stop()

# DCA AI Optimizer Menu
if menu == "DCA AI Optimizer":
    st.header("🤖 DCA AI Optimizer - เพิ่มประสิทธิภาพ DCA ด้วย AI")
    
    if not DCA_AI_AVAILABLE:
        st.error("ระบบ DCA AI ไม่พร้อมใช้งาน กรุณาติดต่อผู้ดูแลระบบ")
        st.stop()
    
    st.markdown("""
    ### ฟีเจอร์ใหม่: การเพิ่มประสิทธิภาพ DCA แบบอัจฉริยะ
    
    **ความสามารถหลัก:**
    - 🎯 วิเคราะห์หลายหุ้นพร้อมกัน (Multi-ticker)
    - 📊 กลยุทธ์ DCA ขั้นสูง 5 แบบ
    - 🔍 เพิ่มประสิทธิภาพพารามิเตอร์อัตโนมัติ
    - 🤖 วิเคราะห์ผลลัพธ์ด้วย AI (ภาษาไทย)
    - 📈 คาดการณ์จุดคุ้มทุน (Break-even Forecast)
    - 📋 Export ไฟล์ Excel แบบละเอียด
    """)
    
    # Initialize components
    data_loader = DCADataLoader()
    optimizer = DCAOptimizer()
    ai_helper = DCAAnalysisHelper()
    
    # Multi-ticker selection
    st.subheader("📈 เลือกหุ้นสำหรับวิเคราะห์")
    
    col1, col2 = st.columns(2)
    with col1:
        # Market selection for DCA AI
        selected_market_ai = st.selectbox(
            "เลือกตลาดหุ้น",
            options=list(markets.keys()),
            index=0,
            key="ai_market",
            help="เลือกตลาดหุ้นสำหรับ DCA AI Optimizer"
        )
        
        available_tickers_ai = markets[selected_market_ai]
        
        # Default selection
        default_ai = []
        if selected_market_ai == "US":
            default_ai = ["AAPL", "MSFT"]
        elif selected_market_ai == "SET100":
            default_ai = ["PTT.BK", "CPALL.BK"]
        else:
            default_ai = available_tickers_ai[:2] if len(available_tickers_ai) >= 2 else available_tickers_ai
        
        selected_tickers = st.multiselect(
            f"เลือกหุ้น ({selected_market_ai})",
            available_tickers_ai,
            default=default_ai,
            help="เลือกหุ้นที่ต้องการเพิ่มประสิทธิภาพ DCA (แนะนำ 2-5 หุ้น)",
            key="ai_tickers"
        )
    
    with col2:
        # Period and budget settings
        period_ai = st.selectbox(
            "ช่วงเวลาข้อมูล",
            ["1y", "2y", "3y", "5y"],
            index=3,
            key="ai_period",
            help="ช่วงเวลาข้อมูลประวัติสำหรับการวิเคราะห์"
        )
        
        budget_mode = st.radio(
            "โหมดงบประมาณ",
            ["งบประมาณรวม", "งบประมาณต่อหุ้น"],
            key="budget_mode",
            help="เลือกวิธีการกำหนดงบประมาณการลงทุน"
        )
        
        if budget_mode == "งบประมาณรวม":
            total_budget = st.number_input(
                "งบประมาณรวมต่อเดือน",
                min_value=500.0,
                max_value=50000.0,
                value=5000.0,
                step=500.0,
                key="total_budget"
            )
        else:
            per_ticker_budget = st.number_input(
                "งบประมาณต่อหุ้นต่อเดือน",
                min_value=200.0,
                max_value=10000.0,
                value=1000.0,
                step=100.0,
                key="per_ticker_budget"
            )
    
    # Strategy Configuration
    st.subheader("⚙️ การตั้งค่ากลยุทธ์ DCA")
    
    # Get available strategies
    available_strategies = DCAStrategyFactory.get_available_strategies()
    
    strategy_configs = []
    
    # Create tabs for strategy configuration
    strategy_tabs = st.tabs([f"{s[1]}" for s in available_strategies])
    
    for i, (strategy_enum, strategy_name, strategy_desc) in enumerate(available_strategies):
        with strategy_tabs[i]:
            st.markdown(f"**{strategy_desc}**")
            
            enabled = st.checkbox(
                f"เปิดใช้งาน {strategy_name}",
                value=(i < 3),  # Enable first 3 strategies by default
                key=f"enable_{strategy_enum.value}"
            )
            
            if enabled:
                # Default parameters
                default_params = DCAStrategyFactory.get_default_parameters(strategy_enum)
                
                # Optimization settings
                optimization_type = st.radio(
                    "วิธีการเพิ่มประสิทธิภาพ",
                    ["Grid Search", "Random Search", "Bayesian-like"],
                    key=f"opt_type_{strategy_enum.value}",
                    help="Grid Search: ทดสอบทุกชุดพารามิเตอร์, Random Search: สุ่มตัวอย่าง, Bayesian-like: ปรับปรุงแบบวนซ้ำ"
                )
                
                if optimization_type == "Random Search":
                    num_samples = st.slider(
                        "จำนวนตัวอย่างสุ่ม",
                        min_value=20,
                        max_value=200,
                        value=50,
                        key=f"samples_{strategy_enum.value}"
                    )
                elif optimization_type == "Bayesian-like":
                    num_iterations = st.slider(
                        "จำนวนรอบการปรับปรุง",
                        min_value=5,
                        max_value=30,
                        value=15,
                        key=f"iterations_{strategy_enum.value}"
                    )
                
                # Add to configs
                config = {
                    'strategy': strategy_enum,
                    'enabled': True,
                    'optimization': {
                        'type': optimization_type.lower().replace('-', '_').replace(' ', '_'),
                    }
                }
                
                if optimization_type == "Random Search":
                    config['optimization']['num_samples'] = num_samples
                elif optimization_type == "Bayesian-like":
                    config['optimization']['num_iterations'] = num_iterations
                
                strategy_configs.append(config)
    
    # Ranking criteria
    st.subheader("🏆 เกณฑ์การจัดอันดับ")
    ranking_criteria = st.selectbox(
        "เกณฑ์หลักในการจัดอันดับกลยุทธ์",
        [
            "total_return",  # ผลตอบแทนรวมสูงสุด
            "cost_basis",    # ราคาเฉลี่ยต่ำสุด
            "sharpe_ratio",  # Risk-adjusted return สูงสุด
            "break_even_speed"  # คุ้มทุนเร็วที่สุด
        ],
        format_func=lambda x: {
            "total_return": "ผลตอบแทนรวมสูงสุด",
            "cost_basis": "ราคาเฉลี่ยต่ำสุด (ประสิทธิภาพการซื้อ)",
            "sharpe_ratio": "ผลตอบแทนปรับความเสี่ยงสูงสุด",
            "break_even_speed": "คุ้มทุนเร็วที่สุด"
        }[x],
        key="ranking_criteria"
    )
    
    # AI Configuration
    st.subheader("🤖 การตั้งค่า AI")
    
    ai_provider = st.selectbox(
        "เลือก AI Provider",
        ["ไม่ใช้ AI", "Google Gemini", "OpenAI GPT"],
        key="ai_provider"
    )
    
    ai_api_key = None
    if ai_provider != "ไม่ใช้ AI":
        ai_api_key = st.text_input(
            f"API Key สำหรับ {ai_provider}",
            type="password",
            key="ai_api_key",
            help="API Key จะไม่ถูกบันทึกในระบบ"
        )
        
        if ai_api_key:
            if ai_provider == "Google Gemini":
                if ai_helper.setup_gemini(ai_api_key):
                    st.success("✅ Gemini API พร้อมใช้งาน")
                else:
                    st.error("❌ ไม่สามารถเชื่อมต่อ Gemini API")
            elif ai_provider == "OpenAI GPT":
                if ai_helper.setup_openai(ai_api_key):
                    st.success("✅ OpenAI API พร้อมใช้งาน")
                else:
                    st.error("❌ ไม่สามารถเชื่อมต่อ OpenAI API")
    
    # Run Optimization
    st.subheader("🚀 เรียกใช้การเพิ่มประสิทธิภาพ")
    
    if st.button("🔥 เริ่มการเพิ่มประสิทธิภาพ DCA", key="run_optimization"):
        if not selected_tickers:
            st.error("กรุณาเลือกหุ้นอย่างน้อย 1 ตัว")
        elif not strategy_configs:
            st.error("กรุณาเปิดใช้งานกลยุทธ์อย่างน้อย 1 แบบ")
        else:
            # Show progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Step 1: Load data
                status_text.text("📥 กำลังโหลดข้อมูลหุ้น...")
                progress_bar.progress(10)
                
                # Validate tickers first
                valid_tickers, invalid_tickers = data_loader.validate_ticker_list(selected_tickers)
                
                if invalid_tickers:
                    st.warning(f"ไม่สามารถโหลดข้อมูลหุ้น: {', '.join(invalid_tickers)}")
                
                if not valid_tickers:
                    st.error("ไม่สามารถโหลดข้อมูลหุ้นใดๆ ได้")
                    st.stop()
                
                # Load ticker data
                ticker_data_dict = {}
                for i, ticker in enumerate(valid_tickers):
                    status_text.text(f"📥 กำลังโหลดข้อมูล {ticker}...")
                    progress_bar.progress(10 + (i + 1) * 20 / len(valid_tickers))
                    
                    try:
                        ticker_data = data_loader.fetch_ticker_data(ticker, period_ai)
                        price_data = data_loader.get_price_data_for_dca(ticker_data)
                        dividend_data = data_loader.get_dividend_data_for_dca(ticker_data)
                        
                        ticker_data_dict[ticker] = {
                            'price_data': price_data,
                            'dividend_data': dividend_data,
                            'raw_data': ticker_data
                        }
                    except Exception as e:
                        st.warning(f"ไม่สามารถโหลดข้อมูล {ticker}: {str(e)}")
                
                if not ticker_data_dict:
                    st.error("ไม่สามารถโหลดข้อมูลหุ้นใดๆ ได้")
                    st.stop()
                
                # Step 2: Run optimization
                status_text.text("🔍 กำลังเพิ่มประสิทธิภาพกลยุทธ์...")
                progress_bar.progress(40)
                
                optimization_results = optimizer.optimize_multiple_tickers(
                    ticker_data_dict,
                    strategy_configs,
                    ranking_criteria
                )
                
                progress_bar.progress(70)
                
                # Step 3: Generate AI analysis (if enabled)
                ai_analysis = None
                if ai_provider != "ไม่ใช้ AI" and ai_api_key:
                    status_text.text("🤖 กำลังวิเคราะห์ผลลัพธ์ด้วย AI...")
                    progress_bar.progress(85)
                    
                    try:
                        ai_analysis = ai_helper.analyze_multi_ticker(
                            optimization_results,
                            ai_provider.lower().replace(' ', '_')
                        )
                    except Exception as e:
                        st.warning(f"การวิเคราะห์ AI ล้มเหลว: {str(e)}")
                
                # Step 4: Display results
                status_text.text("✅ เสร็จสิ้น!")
                progress_bar.progress(100)
                
                # Display results
                st.success("🎉 การเพิ่มประสิทธิภาพเสร็จสิ้น!")
                
                # Summary table
                st.subheader("📊 สรุปผลลัพธ์")
                
                summary_data = []
                for ticker, best_config in optimization_results.get('summary_by_ticker', {}).items():
                    if best_config:
                        metrics = best_config['metrics']
                        summary_data.append({
                            'หุ้น': ticker,
                            'กลยุทธ์ที่ดีที่สุด': best_config['strategy'],
                            'ผลตอบแทน (%)': f"{metrics.get('total_return_pct', 0):.2f}%",
                            'ราคาเฉลี่ย': f"{metrics.get('cost_basis', 0):.2f}",
                            'Sharpe Ratio': f"{metrics.get('sharpe_ratio', 0):.3f}",
                            'คุ้มทุนแล้ว': '✅' if metrics.get('break_even_achieved', False) else '❌',
                            'Max Drawdown': f"{metrics.get('max_drawdown_pct', 0):.2f}%"
                        })
                
                if summary_data:
                    summary_df = pd.DataFrame(summary_data)
                    st.dataframe(summary_df, use_container_width=True)
                
                # Detailed results for each ticker
                st.subheader("📈 ผลลัพธ์รายละเอียด")
                
                for ticker in ticker_data_dict.keys():
                    if ticker in optimization_results.get('detailed_results', {}):
                        ticker_results = optimization_results['detailed_results'][ticker]
                        
                        with st.expander(f"📊 รายละเอียด {ticker}", expanded=False):
                            for strategy_name, strategy_result in ticker_results.items():
                                if 'error' not in strategy_result:
                                    st.markdown(f"**{strategy_name}**")
                                    
                                    best_result = strategy_result.get('best_metrics', {})
                                    if best_result:
                                        col1, col2, col3, col4 = st.columns(4)
                                        with col1:
                                            st.metric("ผลตอบแทน", f"{best_result.get('total_return_pct', 0):.2f}%")
                                        with col2:
                                            st.metric("ราคาเฉลี่ย", f"{best_result.get('cost_basis', 0):.2f}")
                                        with col3:
                                            st.metric("เงินลงทุนรวม", f"{best_result.get('total_invested', 0):,.0f}")
                                        with col4:
                                            st.metric("Max Drawdown", f"{best_result.get('max_drawdown_pct', 0):.2f}%")
                                        
                                        # Parameters
                                        params = strategy_result.get('best_parameters', {})
                                        if params:
                                            st.json(params)
                
                # AI Analysis
                if ai_analysis:
                    st.subheader("🤖 การวิเคราะห์ด้วย AI")
                    st.markdown(ai_analysis)
                
                # Export functionality
                st.subheader("📥 ดาวน์โหลดผลลัพธ์")
                
                try:
                    # Create Excel export
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        # Summary sheet
                        if summary_data:
                            summary_df.to_excel(writer, sheet_name='Summary', index=False)
                        
                        # Detailed results for each ticker
                        for ticker in ticker_data_dict.keys():
                            if ticker in optimization_results.get('detailed_results', {}):
                                ticker_results = optimization_results['detailed_results'][ticker]
                                
                                all_results = []
                                for strategy_name, strategy_result in ticker_results.items():
                                    if 'error' not in strategy_result:
                                        for result in strategy_result.get('all_results', []):
                                            result_copy = result.copy()
                                            result_copy['strategy'] = strategy_name
                                            all_results.append(result_copy)
                                
                                if all_results:
                                    ticker_df = pd.DataFrame(all_results)
                                    # Clean up the dataframe for export
                                    export_columns = [
                                        'strategy', 'rank', 'total_return_pct', 'cost_basis',
                                        'total_invested', 'max_drawdown_pct', 'sharpe_ratio',
                                        'break_even_achieved', 'time_in_profit_pct', 'parameters'
                                    ]
                                    export_df = ticker_df[[col for col in export_columns if col in ticker_df.columns]]
                                    
                                    sheet_name = ticker.replace('.', '_')[:31]  # Excel sheet name limit
                                    export_df.to_excel(writer, sheet_name=sheet_name, index=False)
                        
                        # AI Analysis sheet
                        if ai_analysis:
                            ai_df = pd.DataFrame([{'AI_Analysis': ai_analysis}])
                            ai_df.to_excel(writer, sheet_name='AI_Analysis', index=False)
                    
                    st.download_button(
                        label="📥 ดาวน์โหลดผลลัพธ์ Excel",
                        data=output.getvalue(),
                        file_name=f'DCA_Optimization_Results_{datetime.now().strftime("%Y%m%d_%H%M")}.xlsx',
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                    )
                
                except Exception as e:
                    st.warning(f"ไม่สามารถสร้างไฟล์ Excel: {str(e)}")
                
            except Exception as e:
                st.error(f"เกิดข้อผิดพลาด: {str(e)}")
                st.exception(e)
    
    # Quick compare single ticker
    st.subheader("⚡ เปรียบเทียบกลยุทธ์แบบด่วน")
    st.markdown("เปรียบเทียบกลยุทธ์ DCA ทั้งหมดสำหรับหุ้น 1 ตัว (ใช้พารามิเตอร์เริ่มต้น)")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        quick_ticker = st.selectbox(
            "เลือกหุ้น",
            available_tickers_ai,
            key="quick_ticker"
        )
    with col2:
        quick_period = st.selectbox(
            "ช่วงเวลา",
            ["1y", "2y", "3y", "5y"],
            index=2,
            key="quick_period"
        )
    with col3:
        if st.button("🚀 เปรียบเทียบด่วน", key="quick_compare"):
            try:
                with st.spinner("กำลังเปรียบเทียบ..."):
                    # Load data
                    ticker_data = data_loader.fetch_ticker_data(quick_ticker, quick_period)
                    price_data = data_loader.get_price_data_for_dca(ticker_data)
                    dividend_data = data_loader.get_dividend_data_for_dca(ticker_data)
                    
                    # Compare strategies
                    comparison_result = optimizer.compare_strategies_for_ticker(
                        quick_ticker, price_data, dividend_data
                    )
                    
                    if 'error' not in comparison_result:
                        st.success(f"✅ เปรียบเทียบเสร็จสิ้น - กลยุทธ์ที่ดีที่สุด: **{comparison_result['best_strategy']}**")
                        
                        # Show comparison table
                        comparison_data = []
                        for result in comparison_result['all_results']:
                            comparison_data.append({
                                'กลยุทธ์': result['strategy_type'],
                                'อันดับ': result['rank'],
                                'ผลตอบแทน (%)': f"{result.get('total_return_pct', 0):.2f}%",
                                'ราคาเฉลี่ย': f"{result.get('cost_basis', 0):.2f}",
                                'Sharpe Ratio': f"{result.get('sharpe_ratio', 0):.3f}",
                                'Max Drawdown': f"{result.get('max_drawdown_pct', 0):.2f}%"
                            })
                        
                        comparison_df = pd.DataFrame(comparison_data)
                        st.dataframe(comparison_df, use_container_width=True)
                    else:
                        st.error(f"การเปรียบเทียบล้มเหลว: {comparison_result['error']}")
                        
            except Exception as e:
                st.error(f"เกิดข้อผิดพลาด: {str(e)}")
    
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