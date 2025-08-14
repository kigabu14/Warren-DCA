import streamlit as st
import yfinance as yf
import pandas as pd
import io
import datetime
import numpy as np
import matplotlib.pyplot as plt
import uuid
from ai_helper import AIHelper
from ai_database import AIDatabase
from dca_data_loader import DCADataLoader
from external_optimizer import PortfolioOptimizer



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

# ----------------- AI Chat Interface -----------------
def initialize_ai_session():
    """Initialize AI-related session state variables."""
    if 'ai_helper' not in st.session_state:
        st.session_state.ai_helper = AIHelper()
    
    if 'ai_database' not in st.session_state:
        st.session_state.ai_database = AIDatabase()
    
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

def render_ai_interface():
    """Render the AI chat interface in the sidebar."""
    with st.sidebar:
                # ---------------- Portfolio Optimizer (AI) ----------------
        st.markdown("---")
        st.subheader("📊 Portfolio Optimizer (AI)")

        # เตรียม state
        if 'last_optimize_result' not in st.session_state:
            st.session_state.last_optimize_result = None

        if not st.session_state.get('analysis_done', False):
            st.info("โปรดกดปุ่ม 'วิเคราะห์' ในหน้าหลักก่อน แล้วค่อย Optimize.")
        else:
            total_budget_sidebar = st.number_input(
                "งบลงทุนรวม (Optimize)",
                min_value=1000.0, max_value=5_000_000.0,
                value=st.session_state.get("monthly_invest", 20000.0),
                step=1000.0,
                key="opt_total_budget"
            )

            objective_sidebar = st.selectbox(
                "วัตถุประสงค์",
                ["balanced", "maximize_return", "minimize_risk", "income"],
                index=0,
                key="opt_objective"
            )

            use_dca_budget = st.toggle("ใช้ค่า monthly_invest เป็นงบ", value=False)
            if use_dca_budget:
                total_budget_sidebar = float(st.session_state.get("monthly_invest", total_budget_sidebar))

            run_opt_button = st.button("🚀 Optimize Portfolio", type="primary", use_container_width=True)

            # แสดงผลล่าสุดแบบย่อ
            if st.session_state.get('last_optimize_result'):
                if st.checkbox("แสดงสรุป Optimize ล่าสุด"):
                    opt = st.session_state.last_optimize_result
                    st.write(f"Objective: {opt.get('objective','-')}")
                    st.write(f"Exp Return: {opt['expected_return']*100:.2f}% | Exp Yield: {opt['expected_yield']*100:.2f}%")

            if run_opt_button:
                import time
                now = time.time()
                last_run = st.session_state.get("last_opt_run_time", 0)
                if now - last_run < 3:
                    st.warning("กรุณารอสักครู่ก่อนรันใหม่ (กันกดรัว)")
                else:
                    st.session_state.last_opt_run_time = now
                    from external_optimizer import PortfolioOptimizer
                    from dca_data_loader import DCADataLoader
                    import pandas as pd

                    @st.cache_data(show_spinner=False, ttl=3600)
                    def load_price_and_div(ticker, period):
                        loader = DCADataLoader()
                        data = loader.fetch(ticker, period=period)
                        hist_df = data.get('history') or data.get('historical_prices')
                        if hist_df is None:
                            return None, None
                        if "Close" not in hist_df.columns and "Adj Close" in hist_df.columns:
                            hist_df = hist_df.copy()
                            hist_df["Close"] = hist_df["Adj Close"]
                        div_data = data.get("dividends")
                        if isinstance(div_data, pd.DataFrame) and {"Date","Dividend"}.issubset(div_data.columns):
                            div_series = div_data.set_index("Date")["Dividend"]
                        elif isinstance(div_data, pd.Series):
                            div_series = div_data
                        else:
                            div_series = pd.Series(dtype=float)
                        return hist_df, div_series

                    with st.spinner("กำลัง Optimize พอร์ต..."):
                        try:
                            tickers = st.session_state.get('selected_tickers', [])
                            period = st.session_state.get('period', '1y')
                            if not tickers:
                                st.warning("ยังไม่ได้เลือกหุ้น")
                            else:
                                prices_map = {}
                                dividends_map = {}
                                for tk in tickers:
                                    hist_df, div_series = load_price_and_div(tk, period)
                                    if hist_df is None or hist_df.empty:
                                        st.warning(f"ข้าม {tk} (ไม่มีข้อมูลราคา)")
                                        continue

    # ให้แน่ใจว่า index เป็น DatetimeIndex
                                    if not isinstance(hist_df.index, pd.DatetimeIndex):
                                        hist_df = hist_df.copy()
                                        hist_df.index = pd.to_datetime(hist_df.index, errors='coerce')
                                        hist_df = hist_df.dropna(axis=0, subset=['Close'])  # กัน index เพี้ยน

    # ตรวจว่ามีคอลัมน์ Close
                                    if "Close" not in hist_df.columns:
                                        st.warning(f"{tk}: ไม่มีคอลัมน์ Close ข้าม")
                                        continue

    # เก็บข้อมูลราคาปิดเท่านั้น
                                    prices_map[tk] = hist_df[['Close']].copy()
                                    dividends_map[tk] = div_series if div_series is not None else pd.Series(dtype=float)

# ตรวจสอบผลลัพธ์ (debug)
                                    
                                if not prices_map:
                                    st.error("ไม่มีข้อมูลราคาที่ใช้ได้")
                                else:
                                    if len(prices_map) < len(tickers):
                                        st.info(f"ใช้ข้อมูลได้ {len(prices_map)}/{len(tickers)} ตัว")

                                    optimizer = PortfolioOptimizer(prices_map, dividends_map)
                                    opt_result = optimizer.optimize(
                                        total_budget=total_budget_sidebar,
                                        objective=objective_sidebar
                                    )
                                    st.session_state.last_optimize_result = opt_result

                                    context_data = get_current_context()
                                    context_data.update({
                                        'total_budget': total_budget_sidebar,
                                        'objective': objective_sidebar,
                                        'tickers_used': list(prices_map.keys())
                                    })

                                    st.session_state.ai_database.store_optimization(
                                        st.session_state.session_id,
                                        context_data,
                                        opt_result
                                    )
                                    st.success("✅ Optimize เสร็จสิ้น! ดูรายละเอียดในหน้าหลัก")
                        except Exception as e:
                            st.error(f"เกิดข้อผิดพลาด: {e}")
        st.header("🤖 AI Financial Assistant")
        
        # Check if AI is configured
        if not st.session_state.ai_helper.is_ready():
            st.warning("⚠️ AI not configured. Please set GOOGLE_AI_API_KEY environment variable or in Streamlit secrets.")
            with st.expander("Setup Instructions"):
                st.markdown("""
                **To enable AI features:**
                1. Get a Google AI API key from [Google AI Studio](https://aistudio.google.com/)
                2. Set it as environment variable: `GOOGLE_AI_API_KEY=your_key`
                3. Or add to Streamlit secrets: `GOOGLE_AI_API_KEY = "your_key"`
                4. Restart the application
                """)
            return
        
        # Display AI statistics
        stats = st.session_state.ai_database.get_query_statistics()
        st.caption(f"💬 Total queries: {stats['total_queries']} | Today: {stats['queries_today']}")
        
        # Sample questions
        with st.expander("📝 คำถาม ถาม AI"):
            sample_questions = st.session_state.ai_helper.get_sample_questions()
            for i, question in enumerate(sample_questions[:5]):
                if st.button(f"📌 {question[:50]}...", key=f"sample_{i}", help=question):
                    st.session_state.ai_query_input = question
                    st.rerun()
        
        # AI Query Input
        ai_query = st.text_area(
            "AI จะตอบ ให้ ถามมา:",
            height=100,
            placeholder="e.g., 'เราจะมาวิเคราะห์ กันสวมิญญาณความคิดของ ปู่ Warren Buffett ว่าถ้าเป็นปู่จะทำไง'",
            key="ai_query_input"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            ask_button = st.button("🚀ส่งคำถาม ", type="primary", use_container_width=True)
        with col2:
            clear_button = st.button("🗑️ Clear Chat", use_container_width=True)
        
        if clear_button:
            st.session_state.conversation_history = []
            st.rerun()
        
        # Process AI query
        if ask_button and ai_query.strip():
            with st.spinner("🤔 ขอคิดก่อนะ ..."):
                # Get current context
                context_data = get_current_context()
                
                # Get AI response
                response = st.session_state.ai_helper.query_ai(
                    ai_query, 
                    context_data, 
                    st.session_state.conversation_history
                )
                
                # Store in database
                st.session_state.ai_database.store_query(
                    ai_query, 
                    response, 
                    context_data, 
                    st.session_state.session_id
                )
                
                # Add to conversation history
                st.session_state.conversation_history.append({
                    'user': ai_query,
                    'assistant': response,
                    'timestamp': datetime.datetime.now().strftime("%H:%M:%S")
                })
                
                # Clear input and rerun
                st.session_state.ai_query_input = ""
                st.rerun()
        
        # Display conversation history
        if st.session_state.conversation_history:
            st.subheader("💬 ความคิดเห็น ปู่")
            
            # Reverse to show latest first
            for i, msg in enumerate(reversed(st.session_state.conversation_history[-5:])):
                with st.container():
                    st.markdown(f"**👤 คุณถาม ({msg['timestamp']}):**")
                    st.markdown(msg['user'])
                    
                    st.markdown("**🤖 ปู่ AI ตอบ :**")
                    st.markdown(msg['assistant'])
                    
                    st.divider()
def get_current_context():
    """
    รวบรวม context ปัจจุบันส่งให้ AI / ใช้บันทึก DB
    """
    context = {}
    # ผล Optimize ล่าสุด
    if st.session_state.get('last_optimize_result'):
        context['last_optimize_result'] = st.session_state['last_optimize_result']
    # ตลาด
    context['market'] = st.session_state.get('selected_market', 'Unknown')
    # หุ้นที่เลือก
    context['selected_stocks'] = st.session_state.get('selected_tickers', [])
    # การตั้งค่า DCA
    context['dca_settings'] = {
        'monthly_invest': st.session_state.get('monthly_invest', 10000),
        'period': st.session_state.get('period', '1y')
    }
    # วิเคราะห์แล้วหรือยัง
    context['analysis_done'] = st.session_state.get('analysis_done', False)
    return context

# ----------------- UI & Main -----------------
st.set_page_config(page_title="Warren-DCA วิเคราะห์หุ้น", layout="wide")

# Initialize AI session
initialize_ai_session()

# Render AI interface in sidebar
render_ai_interface()

# Main navigation
menu = st.sidebar.radio("เลือกหน้าที่ต้องการ", ["วิเคราะห์หุ้น", "คู่มือการใช้งาน", "Optimization History" ,"AI Chat History"])

if menu == "AI Chat History":
    st.header("🤖 AI Chat History")
    
    if not st.session_state.ai_helper.is_ready():
        st.warning("AI is not configured. Please configure your Google AI API key to use this feature.")
        st.stop()
    
    # Display statistics
    stats = st.session_state.ai_database.get_query_statistics()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Queries", stats['total_queries'])
    with col2:
        st.metric("Queries Today", stats['queries_today'])
    with col3:
        if stats['last_query_time']:
            st.metric("Last Query", stats['last_query_time'][:16])
        else:
            st.metric("Last Query", "None")
    
    # Search functionality
    st.subheader("🔍 Search Chat History")
    search_term = st.text_input("Search in queries and responses:", placeholder="Enter search term...")
    
    if search_term:
        search_results = st.session_state.ai_database.search_queries(search_term, limit=20)
        if search_results:
            st.write(f"Found {len(search_results)} results for '{search_term}':")
            for result in search_results:
                with st.expander(f"Query on {result['timestamp'][:16]} - {result['query'][:50]}..."):
                    st.markdown(f"**Query:** {result['query']}")
                    st.markdown(f"**Response:** {result['response']}")
                    if result['context_data']:
                        st.json(result['context_data'])
        
        else:
            st.info("No results found.")
    
    # Recent queries
    st.subheader("📝 Recent Queries")
    recent_queries = st.session_state.ai_database.get_recent_queries(limit=10)
    
    if recent_queries:
        for query in recent_queries:
            with st.expander(f"{query['timestamp'][:16]} - {query['query'][:60]}..."):
                st.markdown(f"**Query:** {query['query']}")
                st.markdown(f"**Response:** {query['response']}")
                if query['context_data']:
                    st.markdown("**Context Data:**")
                    st.json(query['context_data'])
    else:
        st.info("No chat history found. Start a conversation with the AI assistant!")
    
    # Cleanup options
    st.subheader("🧹 Data Management")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Old Data (30+ days)"):
            deleted = st.session_state.ai_database.clear_old_queries(30)
            st.success(f"Deleted {deleted} old queries.")
            st.rerun()
    with col2:
        if st.button("⚠️ Clear All Data"):
            if st.checkbox("I understand this will delete all chat history"):
                deleted = st.session_state.ai_database.clear_old_queries(0)
                st.success(f"Deleted all {deleted} queries.")
                st.rerun()
    
    st.stop()

if menu == "คู่มือการใช้งาน":
    st.header("คู่มือการใช้งาน (ภาษาไทย)")
    st.markdown("""
**Warren-DCA คืออะไร?**  
โปรแกรมนี้ช่วยวิเคราะห์หุ้นตามแนวทางของ Warren Buffett (Buffett 11 Checklist แบบขยาย 18 เงื่อนไข) พร้อมจำลองการลงทุนแบบ DCA และคำนวณผลตอบแทนเงินปันผลย้อนหลัง 1 ปี  
**แหล่งข้อมูล:** Yahoo Finance

### 🤖 AI Financial Assistant (ใหม่!)
โปรแกรมมี AI Assistant ที่ช่วยวิเคราะห์และให้คำแนะนำการลงทุน:
- ใช้ Google Gemini AI เพื่อตอบคำถามเกี่ยวกับหุ้น
- วิเคราะห์ข้อมูลตามหลักการของ Warren Buffett
- ให้คำแนะนำการลงทุนแบบ DCA
- เก็บประวัติการสนทนาไว้ในฐานข้อมูล
- ไม่ทำให้หน้าเว็บรีเฟรชใหม่เมื่อถามคำถาม

**วิธีใช้ AI:**
1. ตั้งค่า Google AI API Key (ดูคำแนะนำในแถบด้านข้าง)
2. เลือกหุ้นที่ต้องการวิเคราะห์
3. ถามคำถามใน AI Assistant ที่แถบด้านข้าง
4. ดูประวัติการสนทนาในหน้า "AI Chat History"

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
- AI Assistant ต้องการ Google AI API Key เพื่อใช้งาน
""")
if menu == "Optimization History":
    st.header("📊 Optimization History")
    rec = st.session_state.ai_database.get_recent_optimizations(limit=10)
    if not rec:
        st.info("ยังไม่มีการ Optimize ที่บันทึกไว้")
    else:
        for r in rec:
            with st.expander(f"{r['timestamp']} | Session {r['session_id'][:8]}..."):
                st.json(r['context_data'])
                st.json(r['result'])
    st.stop()

# Market selection
selected_market = st.selectbox(
    "เลือกตลาดหุ้น",
    options=list(markets.keys()),
    index=0,  # Default to US
    help="เลือกตลาดหุ้นที่ต้องการวิเคราะห์",
    key="selected_market"
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
    help=f"เลือกหุ้นจากตลาด {selected_market} ที่ต้องการวิเคราะห์",
    key="selected_tickers"
)
period = st.selectbox("เลือกช่วงเวลาราคาหุ้น", ["1y", "5y", "max"], index=1, key="period")
monthly_invest = st.number_input("จำนวนเงินลงทุน DCA ต่อเดือน (บาทหรือ USD)", min_value=100.0, max_value=10000.0, value=1000.0, step=100.0, key="monthly_invest")
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
    st.session_state.analysis_done = True
    # --- Pie Chart ---
    pie_labels = ["TOTAL INVEST", "TOTAL PROFIT", "div"]
    pie_values = [total_invest, total_profit if total_profit > 0 else 0, total_div]
    fig, ax = plt.subplots()
    colors = ['#2196f3', '#4caf50', '#ffc107']
    ax.pie(pie_values, labels=pie_labels, autopct='%1.1f%%', startangle=90, colors=colors)
    ax.set_title("INVEST/Profit/DivyYield")
    st.pyplot(fig)
    st.markdown("---")
    st.header("🔧 ผลการ Optimize (ถ้ามี)")

    if st.session_state.get('last_optimize_result'):
        opt = st.session_state.last_optimize_result
        st.subheader(f"Objective: {opt.get('objective','-')}")
        st.metric("Expected Return (CAGR Blend)", f"{opt['expected_return']*100:.2f}%")
        st.metric("Expected Dividend Yield", f"{opt['expected_yield']*100:.2f}%")

        alloc_rows = []
        for tk, row in opt['allocations'].items():
            alloc_rows.append({
                "Ticker": tk,
                "Weight (%)": round(row['weight']*100, 2),
                "Allocation": round(row['allocation'], 2),
                "CAGR (%)": round(row['cagr']*100, 2),
                "Vol (annual)": round(row['vol'], 4),
                "Div Yield (%)": round(row['div_yield']*100, 2)
            })
        df_alloc = pd.DataFrame(alloc_rows)
        st.dataframe(df_alloc, use_container_width=True)

        if st.button("💬 ให้ AI อธิบายผลลัพธ์ Optimize"):
            follow_question = (
                "ช่วยอธิบายเหตุผลของสัดส่วนพอร์ตนี้ ข้อดี ข้อควรระวัง "
                "และเสนอวิธีปรับปรุงอีก 2 แบบ"
            )
            context_for_ai = {
                "opt_result": opt,
                "selected_stocks": st.session_state.get('selected_tickers', []),
                "period": st.session_state.get('period'),
                "monthly_invest": st.session_state.get('monthly_invest')
            }
            ai_text = st.session_state.ai_helper.query_ai(
                follow_question,
                context_for_ai,
                st.session_state.conversation_history
            )
            st.session_state.ai_database.store_query(
                follow_question,
                ai_text,
                context_for_ai,
                st.session_state.session_id
            )
            st.session_state.conversation_history.append({
                "user": follow_question,
                "assistant": ai_text,
                "timestamp": datetime.datetime.now().strftime("%H:%M:%S")
            })
            st.success("บันทึกคำอธิบายจาก AI แล้ว!")
    else:
        st.info("ยังไม่มีผล Optimize - ไปกดปุ่ม Optimize ที่ Sidebar")







st.caption("Powered by Yahoo Finance | วิเคราะห์หุ้นด้วย Buffett Checklist (ขยาย 18 เงื่อนไข) + DCA + ปันผลย้อนหลัง 1 ปี")



