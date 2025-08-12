import streamlit as st
import yfinance as yf
import pandas as pd
import io
import datetime
from datetime import datetime as dt
import numpy as np
import matplotlib.pyplot as plt

# ----------------- Gemini AI Integration -----------------
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    genai = None
    GEMINI_AVAILABLE = False


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
    hist_prices = strip_timezone(hist_prices)
    if div is not None and not div.empty:
        div = strip_timezone(pd.DataFrame(div))
        div = div.iloc[:, 0] if len(div.columns) > 0 else div
    prices = hist_prices['Close'].resample('M').first().dropna()
    units = monthly_invest / prices
    total_units = units.sum()
    total_invested = monthly_invest * len(prices)
    avg_buy_price = total_invested / total_units if total_units != 0 else 0
    latest_price = prices.iloc[-1]
    current_value = total_units * latest_price
    profit = current_value - total_invested
    total_div_val = 0
    if div is not None and not div.empty:
        div_period = div[div.index >= prices.index[0]]
        if not div_period.empty:
            cum_units = units.cumsum()
            for i, dtm in enumerate(prices.index):
                div_in_month = div_period[(div_period.index.month == dtm.month) &
                                          (div_period.index.year == dtm.year)].sum()
                if div_in_month > 0:
                    total_div_val += div_in_month * cum_units.iloc[i]
    return {
        "เงินลงทุนรวม": round(total_invested, 2),
        "จำนวนหุ้นสะสม": round(total_units, 4),
        "มูลค่าปัจจุบัน": round(current_value, 2),
        "กำไร/ขาดทุน": round(profit, 2),
        "กำไร(%)": round(profit / total_invested * 100, 2) if total_invested != 0 else 0,
        "ราคาเฉลี่ยที่ซื้อ": round(avg_buy_price, 2),
        "ราคาปิดล่าสุด": round(latest_price, 2),
        "เงินปันผลรวม": round(total_div_val, 2)
    }


# ----------------- Buffett 11 Checklist (Expanded 18) -----------------
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
        inv_growth = all([inv[i] < inv[i + 1] for i in range(len(inv) - 1)]) if len(inv) >= 2 else True

        ni = []
        for col in financials.columns:
            v = financials.loc[financials.index.str.contains("Net Income", case=False), col]
            if not v.empty and v.iloc[0] is not None:
                ni.append(v.iloc[0])
        ni_growth = all([ni[i] < ni[i + 1] for i in range(len(ni) - 1)]) if len(ni) >= 2 else True
        res = 1 if inv_growth and ni_growth else 0
    except:
        res = -1
    results.append({'title': '1.1 Inventory & Net Earnings เพิ่มขึ้นต่อเนื่อง', 'result': res,
                    'desc': 'Inventory และ Net Income ต้องโตต่อเนื่อง'})
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
            growth = all([ppe[i] <= ppe[i + 1] for i in range(len(ppe) - 1)])
            spike = max([abs(ppe[i + 1] - ppe[i]) / ppe[i] if ppe[i] != 0 else 0 for i in range(len(ppe) - 1)]) < 1.0
            res = 1 if growth and spike else 0
        else:
            res = -1
    except:
        res = -1
    results.append({'title': '3. PPE เพิ่มขึ้น (ไม่มี spike)', 'result': res, 'desc': 'Property, Plant & Equipment โตต่อเนื่อง'})
    if res != -1:
        score += res
        evaluated += 1

    # 4.1 RTA ≥ 11%
    try:
        ebitda_vals = []
        ta = []
        for col in financials.columns:
            v = financials.loc[financials.index.str.contains("EBITDA", case=False), col]
            if not v.empty and v.iloc[0] is not None:
                ebitda_vals.append(v.iloc[0])
        for col in balance_sheet.columns:
            v = balance_sheet.loc[balance_sheet.index.str.contains("Total Assets", case=False), col]
            if not v.empty and v.iloc[0] is not None:
                ta.append(v.iloc[0])
        rtas = [ebitda_vals[i] / ta[i] for i in range(min(len(ebitda_vals), len(ta))) if ta[i] != 0]
        avg_rta = sum(rtas) / len(rtas) if rtas else 0
        res = 1 if avg_rta >= 0.11 else 0
    except:
        res = -1
        avg_rta = 0
    results.append({'title': '4.1 RTA ≥ 11%', 'result': res, 'desc': 'Return on Total Assets เฉลี่ย ≥ 11%'})
    if res != -1:
        score += res
        evaluated += 1

    # 4.2 RTA ≥ 17%
    try:
        res = 1 if avg_rta >= 0.17 else 0
    except:
        res = -1
    results.append({'title': '4.2 RTA ≥ 17%', 'result': res, 'desc': 'Return on Total Assets เฉลี่ย ≥ 17%'})
    if res != -1:
        score += res
        evaluated += 1

    # 5.1 LTD/Total Assets ≤ 0.5
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

    # 5.2 LTD repayable by EBITDA ≤ 4 yrs
    try:
        last_ebitda = ebitda_vals[-1] if ebitda_vals else None
        last_ltd = ltd[-1] if ltd else None
        if last_ebitda and last_ltd and last_ebitda > 0:
            res = 1 if last_ltd / last_ebitda <= 4 else 0
        else:
            res = -1
    except:
        res = -1
    results.append({'title': '5.2 EBITDA จ่ายหนี้ LTD หมดใน ≤ 4 ปี', 'result': res,
                    'desc': 'EBITDA ล่าสุดชำระหนี้ระยะยาว ≤ 4 ปี'})
    if res != -1:
        score += res
        evaluated += 1

    # 6.1 Negative equity any year? (logic เดิม: ถ้าติดลบให้ result=1)
    try:
        se = []
        for col in balance_sheet.columns:
            v = balance_sheet.loc[
                balance_sheet.index.str.contains("Total Stock", case=False) &
                balance_sheet.index.str.contains("Equity", case=False),
                col
            ]
            if not v.empty and v.iloc[0] is not None:
                se.append(v.iloc[0])
        neg_se = any([x < 0 for x in se])
        res = 1 if neg_se else 0
    except:
        res = -1
        neg_se = False
        se = []
    results.append({'title': '6.1 Equity ติดลบในปีใดหรือไม่', 'result': res, 'desc': 'ถ้าติดลบ ข้าม 6.2-6.3'})
    if res != -1:
        evaluated += 1  # ไม่รวมในคะแนน (ตามโค้ดเดิม)

    # 6.2 DSER ≤ 1.0
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

    # 6.3 DSER ≤ 0.8
    try:
        res = 1 if (not neg_se and avg_dser <= 0.8) else (-1 if neg_se else 0)
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
        re_growths = [(re[i + 1] - re[i]) / re[i] if re[i] != 0 else 0 for i in range(len(re) - 1)]
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

    # 10 ROE >= 23% (ใช้ EBITDA/SE แทน เพราะ data structure เดิม)
    try:
        roe = [ebitda_vals[i] / se[i] for i in range(min(len(ebitda_vals), len(se))) if se[i] != 0]
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
        res = 1 if all([gw[i] <= gw[i + 1] for i in range(len(gw) - 1)]) and len(gw) >= 2 else 0
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


# ----------------- STOCK LISTS -----------------
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

european_stocks = [
    "ASML.AS", "SAP.DE", "NESN.SW", "INGA.AS", "MC.PA", "OR.PA", "SAN.PA", "RDSA.AS", "NOVN.SW", "ROG.SW",
    "LONN.SW", "UNA.AS", "ADYEN.AS", "DSM.AS", "PHIA.AS", "DBK.DE", "EOAN.DE", "VOW3.DE", "SIE.DE", "ALV.DE",
    "AZN.L", "ULVR.L", "SHEL.L", "BP.L", "HSBA.L", "GSK.L", "DGE.L", "VODL.L", "BARC.L", "LLOY.L"
]

asian_stocks = [
    "7203.T", "9984.T", "6098.T", "6758.T", "8035.T", "9434.T", "4063.T", "7974.T", "6501.T", "9432.T",
    "005930.KS", "000660.KS", "035420.KS", "207940.KS", "035720.KS", "068270.KS", "012330.KS", "051910.KS",
    "0700.HK", "9988.HK", "0941.HK", "1299.HK", "0175.HK", "1398.HK", "3690.HK", "0388.HK", "2318.HK", "1810.HK",
    "000001.SS", "000002.SS", "000858.SS", "600036.SS", "600519.SS", "000725.SS", "600276.SS", "002415.SS"
]

australian_stocks = [
    "CBA.AX", "WBC.AX", "ANZ.AX", "NAB.AX", "CSL.AX", "BHP.AX", "WOW.AX", "TLS.AX", "WES.AX", "MQG.AX",
    "COL.AX", "TCL.AX", "RIO.AX", "WDS.AX", "REA.AX", "QBE.AX", "IAG.AX", "SUN.AX", "QAN.AX", "ALL.AX"
]

markets = {
    "US": us_stocks,
    "SET100": set100,
    "Europe": european_stocks,
    "Asia": asian_stocks,
    "Australia": australian_stocks,
    "Global": us_stocks + set100 + european_stocks + asian_stocks + australian_stocks
}

# Optional DCA AI modules
try:
    from dca_data_loader import DCADataLoader
    from dca_strategies import DCAStrategy, DCAStrategyFactory
    from dca_optimizer import DCAOptimizer
    from dca_metrics import DCAMetrics
    from ai_dca_helper import DCAAnalysisHelper
    DCA_AI_AVAILABLE = True
except ImportError:
    DCA_AI_AVAILABLE = False

# ----------------- UI & Main -----------------
st.set_page_config(page_title="Warren-DCA วิเคราะห์หุ้น", layout="wide")

menu_options = ["วิเคราะห์หุ้น", "คู่มือการใช้งาน"]
if DCA_AI_AVAILABLE:
    menu_options.insert(1, "DCA AI Optimizer")
menu = st.sidebar.radio("เลือกหน้าที่ต้องการ", menu_options)

# Gemini sidebar (เฉพาะโหมดวิเคราะห์หุ้นธรรมดา)
ai_ready = False
if menu == "วิเคราะห์หุ้น":
    st.sidebar.markdown("### 🤖 Gemini วิเคราะห์หุ้น")
    use_ai = st.sidebar.checkbox("เปิดใช้ Gemini", value=False)
    if use_ai:
        gemini_api_key_sidebar = st.sidebar.text_input("Gemini API Key", type="password")
        if gemini_api_key_sidebar:
            if setup_gemini(gemini_api_key_sidebar):
                st.sidebar.success("✅ Gemini พร้อมใช้งาน")
                ai_ready = True
            else:
                st.sidebar.error("❌ ใช้งาน Gemini ไม่สำเร็จ")

if menu == "คู่มือการใช้งาน":
    st.header("คู่มือการใช้งาน (ภาษาไทย)")
    st.markdown("""
**Warren-DCA คืออะไร?**  
เครื่องมือนี้ช่วยคุณ:
- วิเคราะห์หุ้นตาม Buffett 11 Checklist (แตกย่อย 18 ข้อ)
- จำลองการลงทุนแบบ DCA พร้อมปันผล
- สรุปผลรวมหลายหุ้น
- ปุ่ม AI (Gemini) วิเคราะห์สั้น ๆ (ไม่ใช่คำแนะนำการลงทุน)
""")
    st.markdown("""
### 18 ข้อ (Checklist)
1) Inventory & Net Earnings เพิ่มขึ้น  
2) ไม่มี R&D  
3) EBITDA > Current Liabilities  
4) PPE เพิ่มขึ้น (ไม่กระโดดผิดปกติ)  
5) RTA ≥ 11%  
6) RTA ≥ 17%  
7) LTD/Total Assets ≤ 0.5  
8) EBITDA จ่าย LTD หมด ≤ 4 ปี  
9) Equity ติดลบ?  
10) DSER ≤ 1.0  
11) DSER ≤ 0.8  
12) ไม่มี Preferred Stock  
13) Retained Earnings โต ≥ 7%  
14) Retained Earnings โต ≥ 13.5%  
15) Retained Earnings โต ≥ 17%  
16) มี Treasury Stock  
17) ROE ≥ 23%  
18) Goodwill เพิ่มขึ้น
""")
    st.info("ข้อมูลจาก Yahoo Finance อาจไม่ครบทุกหุ้น/ตลาด")
    st.stop()

# ===== Sidebar Settings =====
with st.sidebar:
    st.subheader("⚙️ ตั้งค่า Optimizer")
    opt_market = st.selectbox("ตลาดสำหรับค้นหา", list(markets.keys()), index=0, key="opt_market")
    candidate_universe = markets[opt_market]
    max_tickers = st.number_input("จำนวนหุ้นสูงสุด", 1, 15, 5)
    base_period = st.selectbox("ช่วงราคาประวัติ", ["1y", "5y", "max"], index=0, key="opt_period")
    total_budget = st.number_input("งบ DCA รวม / เดือน", min_value=100.0, max_value=300000.0, value=5000.0, step=100.0)
    step_unit = st.number_input("Step ต่อหุ้น (บาท/$ ต่อเดือน)", min_value=50.0, max_value=10000.0, value=500.0, step=50.0)
    objective = st.selectbox("วัตถุประสงค์", ["maximize_final_value", "maximize_dividends", "risk_adjusted"], index=0)
    run_btn = st.button("🚀 เริ่ม Optimize", use_container_width=True)

st.markdown("### เลือกหุ้นที่ต้องการ")
picked = st.multiselect(
    f"เลือกหุ้นจาก {opt_market}",
    candidate_universe[:70],
    default=candidate_universe[:min(4, len(candidate_universe))]
)
if len(picked) > max_tickers:
    st.warning(f"เลือกเกิน {max_tickers} ตัว ใช้เฉพาะ {max_tickers} ตัวแรก")
    picked = picked[:max_tickers]

# ===== Helper Functions =====
def fetch_basic_data(ticker: str, period: str):
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period)
    div = stock.dividends
    return hist, div

def simulate_single(hist, div, monthly_invest: float):
    if hist.empty:
        return {"monthly": monthly_invest, "final_value": 0, "profit": 0, "dividends": 0,
                "gain_pct": 0, "units": 0}
    hist = strip_timezone(hist)
    d = dca_simulation(hist, monthly_invest, div)
    return {
        "monthly": monthly_invest,
        "final_value": d["มูลค่าปัจจุบัน"],
        "profit": d["กำไร/ขาดทุน"],
        "gain_pct": d["กำไร(%)"],
        "dividends": d["เงินปันผลรวม"],
        "units": d["จำนวนหุ้นสะสม"]
    }

def brute_force_allocations(tickers, period, budget, step, objective):
    """
    Simple integer partition search:
    - แบ่ง budget เป็น step units
    - แจกจ่ายให้แต่ละ ticker (non-negative integers ที่ sum = steps)
    - คำนวณ performance และเลือกตาม objective
    NOTE: steps จำกัดที่ 40 เพื่อลดเวลาคำนวณ
    """
    if not tickers:
        return None

    cache = {}
    for t in tickers:
        h, d = fetch_basic_data(t, period)
        cache[t] = (h, d)

    steps = int(budget // step)
    steps = min(steps, 40)  # safety cap

    best = None

    def generate_partitions(n, k):
        if n == 1:
            yield [k]
        else:
            for i in range(k + 1):
                for rest in generate_partitions(n - 1, k - i):
                    yield [i] + rest

    for alloc_steps in generate_partitions(len(tickers), steps):
        monthly_map = {tickers[i]: alloc_steps[i] * step for i in range(len(tickers))}
        if sum(monthly_map.values()) == 0:
            continue

        portfolio_final = 0
        portfolio_profit = 0
        portfolio_div = 0
        gains = []
        detail_rows = []

        for tk, amt in monthly_map.items():
            if amt == 0:
                continue
            h, d = cache[tk]
            sim = simulate_single(h, d, amt)
            portfolio_final += sim["final_value"]
            portfolio_profit += sim["profit"]
            portfolio_div += sim["dividends"]
            gains.append(sim["gain_pct"])
            detail_rows.append((tk, sim))

        if not gains:
            continue

        avg_gain = sum(gains) / len(gains)
        variance = sum((g - avg_gain) ** 2 for g in gains) / len(gains)
        risk_adj = (portfolio_profit / (1 + variance)) if variance >= 0 else portfolio_profit

        if objective == "maximize_final_value":
            score = portfolio_final
        elif objective == "maximize_dividends":
            score = portfolio_div
        else:
            score = risk_adj

        if (best is None) or (score > best["score"]):
            best = {
                "score": score,
                "objective": objective,
                "allocation": monthly_map,
                "final_value": portfolio_final,
                "profit": portfolio_profit,
                "dividends": portfolio_div,
                "variance": variance,
                "details": detail_rows
            }

    return best

# ===== Advanced External Optimizer (If Available) =====
if run_btn:
    used_fallback = False
    external_result = None
    if DCA_AI_AVAILABLE:
        try:
            st.info("ใช้โมดูล DCA AI ภายนอก...")
            loader = DCADataLoader()
            prices_map = {}
            dividends_map = {}
            for tk in picked:
                data = loader.fetch(tk, period=base_period)
                prices_map[tk] = data["history"]
                dividends_map[tk] = data.get("dividends")
            strategies = [
                DCAStrategyFactory.create("equal_weight"),
                DCAStrategyFactory.create("value_weighted"),
            ]
            optimizer = DCAOptimizer(prices_map, dividends_map, strategies=strategies)
            external_result = optimizer.optimize(
                total_budget=total_budget,
                objective=objective,
                step=step_unit,
                max_allocation_per_ticker=None
            )
        except Exception as e:
            st.warning(f"External DCA AI Error: {e}")
            used_fallback = True
    else:
        used_fallback = True

    if used_fallback:
        with st.spinner("กำลังค้นหาการจัดสรร (Fallback Brute Force)..."):
            best = brute_force_allocations(picked, base_period, total_budget, step_unit, objective)
        if not best:
            st.error("ไม่พบผลลัพธ์ที่เหมาะสม / ข้อมูลไม่พอ")
        else:
            st.subheader("ผลลัพธ์ Fallback Optimizer")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Objective Score", f"{best['score']:.2f}")
            col2.metric("Final Value", f"{best['final_value']:.2f}")
            col3.metric("Total Profit", f"{best['profit']:.2f}")
            col4.metric("Total Dividends", f"{best['dividends']:.2f}")
            st.caption(f"Variance (gain%): {best['variance']:.5f}")

            alloc_df = pd.DataFrame(
                [{"Ticker": tk, "Monthly Invest": amt} for tk, amt in best["allocation"].items() if amt > 0]
            ).sort_values("Monthly Invest", ascending=False)
            st.markdown("#### การจัดสรรต่อเดือน")
            st.dataframe(alloc_df, use_container_width=True)

            detail_rows = []
            for tk, sim in best["details"]:
                detail_rows.append({
                    "Ticker": tk,
                    "Monthly": sim["monthly"],
                    "FinalValue": sim["final_value"],
                    "Profit": sim["profit"],
                    "Gain(%)": sim["gain_pct"],
                    "Dividends": sim["dividends"],
                    "Units": sim["units"]
                })
            st.markdown("#### รายละเอียดต่อหุ้น")
            st.dataframe(pd.DataFrame(detail_rows), use_container_width=True)

            if not alloc_df.empty:
                fig_alloc, ax_alloc = plt.subplots()
                ax_alloc.pie(
                    alloc_df["Monthly Invest"],
                    labels=alloc_df["Ticker"],
                    autopct='%1.1f%%',
                    startangle=90
                )
                ax_alloc.set_title("สัดส่วน DCA / เดือน")
                st.pyplot(fig_alloc)

            st.success("เสร็จสิ้นการ Optimize (Fallback)")

    else:
        st.subheader("ผลลัพธ์ External Optimizer")
        if external_result:
            summ = external_result.get("summary")
            allocs = external_result.get("allocations", {})
            if summ:
                st.write(summ)
            if allocs:
                st.markdown("#### Allocation (External)")
                st.dataframe(pd.DataFrame([
                    {"Ticker": k, "Monthly": v} for k, v in allocs.items()
                ]))
        else:
            st.info("ไม่มีข้อมูลสรุปจาก External Optimizer")

st.stop()
# -------- Main Stock Analysis --------
selected_market = st.selectbox("เลือกตลาดหุ้น", options=list(markets.keys()), index=0)
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
    help=f"เลือกหุ้นจากตลาด {selected_market}"
)

period = st.selectbox("ช่วงเวลาราคาหุ้น", ["1y", "5y", "max"], index=1)
monthly_invest = st.number_input("จำนวนเงิน DCA ต่อเดือน", min_value=100.0, max_value=100000.0, value=1000.0, step=100.0)
show_financials = st.checkbox("แสดงงบการเงิน (Income Statement)", value=False)

if st.button("วิเคราะห์"):
    export_list = []
    results_table = []
    total_invest = 0
    total_profit = 0
    total_div_all = 0

    for ticker in tickers:
        stock = yf.Ticker(ticker)
        fin = stock.financials
        bs = stock.balance_sheet
        cf = stock.cashflow
        div = stock.dividends
        hist = stock.history(period=period)
        info = stock.info if hasattr(stock, "info") else {}

        company_name = info.get('longName', ticker)
        company_symbol = ticker

        manual_yield = np.nan
        total_div1y = np.nan

        with st.expander(f"{ticker} - {company_name}", expanded=False):
            st.subheader(f"{company_name} ({company_symbol})")

            div_yield = info.get('dividendYield', None)
            div_yield_pct = round(div_yield * 100, 2) if div_yield is not None else "N/A"

            ex_div = info.get('exDividendDate', None)
            if ex_div:
                try:
                    ex_div_date = datetime.datetime.fromtimestamp(ex_div).strftime('%Y-%m-%d')
                except Exception:
                    ex_div_date = str(ex_div)
            else:
                ex_div_date = "N/A"

            w52_high = info.get('fiftyTwoWeekHigh', "N/A")
            w52_low = info.get('fiftyTwoWeekLow', "N/A")
            last_close = info.get('previousClose', "N/A")
            last_open = info.get('open', "N/A")

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Dividend Yield (%)", div_yield_pct)
                st.metric("Ex-Dividend Date", ex_div_date)
            with c2:
                st.metric("52W High", w52_high)
                st.metric("52W Low", w52_low)
            with c3:
                st.metric("ราคาปิดล่าสุด", last_close)
                st.metric("ราคาเปิดล่าสุด", last_open)

            # Dividend 1Y
            st.subheader("ปันผลย้อนหลัง 1 ปี")
            if not div.empty and not hist.empty:
                last_year = hist.index[-1] - pd.DateOffset(years=1)
                recent_div = div[div.index >= last_year]
                total_div1y = recent_div.sum()
                avg_price1y = hist['Close'][hist.index >= last_year].mean()
                price_base = avg_price1y if (avg_price1y and avg_price1y > 0) else hist['Close'].iloc[-1]
                manual_yield = (total_div1y / price_base) * 100 if price_base > 0 else np.nan
                st.markdown(
                    f"<b>เงินปันผลรวม 1 ปี:</b> {total_div1y:.2f} | "
                    f"<b>ราคาเฉลี่ย 1 ปี:</b> {price_base:.2f} | "
                    f"<b>Dividend Yield (คำนวณ):</b> "
                    f"<span style='color:red;font-size:1.2em'>{manual_yield:.2f}%</span>",
                    unsafe_allow_html=True
                )
            else:
                st.info("ไม่มีข้อมูลปันผลย้อนหลัง")

            # Checklist
            st.subheader("Buffett 11 Checklist (ขยาย 18)")
            detail = buffett_11_checks_detail(fin, bs, cf, div, hist)
            badge = get_badge(detail['score_pct'])
            st.write(f"คะแนน: {detail['score']} / {detail['evaluated']} "
                     f"({detail['score_pct']}%) | ป้าย: {badge}")

            df_detail = pd.DataFrame([
                {
                    'ข้อ': i + 1,
                    'รายการ': d['title'],
                    'ผลลัพธ์': "✅ ผ่าน" if d['result'] == 1 else ("❌ ไม่ผ่าน" if d['result'] == 0 else "⚪ N/A"),
                    'คำอธิบาย': d['desc']
                }
                for i, d in enumerate(detail['details'])
            ])
            st.dataframe(df_detail, hide_index=True, use_container_width=True)

            # DCA Simulation
            st.subheader("DCA Simulation")
            dca_result = dca_simulation(hist, monthly_invest, div)
            st.write(pd.DataFrame(dca_result, index=['สรุปผล']).T)

            # Gemini AI Button
            if ai_ready:
                if st.button(f"🤖 Gemini วิเคราะห์ {ticker}", key=f"ai_{ticker}"):
                    with st.spinner("กำลังประมวลผล Gemini..."):
                        ai_text = gemini_analyze_company(ticker, company_name, detail, dca_result)
                    st.markdown("### 🤖 ผลวิเคราะห์ AI")
                    st.markdown(ai_text)

            # Accumulate totals
            total_invest += dca_result["เงินลงทุนรวม"]
            total_profit += dca_result["กำไร/ขาดทุน"]
            total_div_all += dca_result["เงินปันผลรวม"]

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
                st.warning("ไม่มีข้อมูลราคา")

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

    if export_list:
        df_export = pd.DataFrame(export_list)
        df_export = strip_timezone(df_export)
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df_export.to_excel(writer, index=False, sheet_name='WarrenDCA')
        st.download_button(
            label="📥 ดาวน์โหลดผลลัพธ์ Excel",
            data=output.getvalue(),
            file_name='WarrenDCA_Result.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )

    st.header("สรุปผลรวมหุ้นที่เลือก (DCA Simulator)")
    st.write(f"💰 เงินลงทุนรวม: {total_invest:,.2f}")
    st.write(f"📈 กำไรรวม: {total_profit:,.2f}")
    st.write(f"💵 เงินปันผลรวม: {total_div_all:,.2f}")

    if results_table:
        st.subheader("ตารางสรุป (ต่อหุ้น)")
        st.dataframe(pd.DataFrame(results_table), use_container_width=True)

    pie_labels = ["เงินลงทุน", "กำไร(>0 เท่านั้น)", "ปันผล"]
    pie_values = [total_invest, total_profit if total_profit > 0 else 0, total_div_all]
    fig, ax = plt.subplots()
    colors = ['#2196f3', '#4caf50', '#ffc107']
    ax.pie(pie_values, labels=pie_labels, autopct='%1.1f%%', startangle=90, colors=colors)
    ax.set_title("โครงสร้างผลรวม")
    st.pyplot(fig)

st.caption("Powered by Yahoo Finance | Buffett Checklist (18) + DCA + Gemini AI | ไม่ใช่คำแนะนำการลงทุน")