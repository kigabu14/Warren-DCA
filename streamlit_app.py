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

def generate_portfolio_weights(tickers):
    """Generate equal weights for portfolio allocation"""
    if not tickers:
        return {}
    
    weight = 1.0 / len(tickers)
    return {ticker: weight for ticker in tickers}

def create_portfolio_pie_chart(tickers, title="การกระจายการลงทุนในพอร์ตโฟลิโอ"):
    """Create a pie chart showing portfolio allocation"""
    if not tickers or len(tickers) == 0:
        return None
    
    weights = generate_portfolio_weights(tickers)
    
    # Create the pie chart
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Extract data for pie chart
    labels = list(weights.keys())
    sizes = [weights[ticker] * 100 for ticker in labels]  # Convert to percentages
    
    # Generate colors - use a colormap for consistent colors
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
    
    # Create pie chart
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%', 
                                      startangle=90, colors=colors,
                                      textprops={'fontsize': 10})
    
    # Enhance the appearance
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Ensure the pie chart is circular
    ax.axis('equal')
    
    return fig

# ----------------- Buffett 11 Checklist (ละเอียดแบบ parameters.py) -----------------
def buffett_11_checks_detail(financials, balance_sheet, cashflow, dividends, hist_prices):
    results = []
    score = 0
    evaluated = 0

    try:
        inv = []
        for col in balance_sheet.columns:
            v = balance_sheet.loc[balance_sheet.index.str.contains("Inventor",case=False), col]
            if not v.empty and v.iloc[0] is not None:
                inv.append(v.iloc[0])
        inv_growth = all([inv[i] < inv[i+1] for i in range(len(inv)-1)]) if len(inv)>=2 else True
        ni = []
        for col in financials.columns:
            v = financials.loc[financials.index.str.contains("Net Income",case=False), col]
            if not v.empty and v.iloc[0] is not None:
                ni.append(v.iloc[0])
        ni_growth = all([ni[i] < ni[i+1] for i in range(len(ni)-1)]) if len(ni)>=2 else True
        res = 1 if inv_growth and ni_growth else 0
    except:
        res = -1
    results.append({'title':'1.1 Inventory & Net Earnings เพิ่มขึ้นต่อเนื่อง','result':res,'desc':'Inventory และ Net Income ต้องโตต่อเนื่อง'})
    if res != -1: score += res; evaluated += 1

    try:
        r_and_d = any(financials.index.str.contains('Research',case=False))
        res = 0 if r_and_d else 1
    except:
        res = -1
    results.append({'title':'1.2 ไม่มี Research & Development','result':res,'desc':'ไม่มีค่าใช้จ่าย R&D'})
    if res != -1: score += res; evaluated += 1

    try:
        ebitda = []
        cl = []
        for col in financials.columns:
            v = financials.loc[financials.index.str.contains("EBITDA",case=False), col]
            if not v.empty and v.iloc[0] is not None:
                ebitda.append(v.iloc[0])
        for col in balance_sheet.columns:
            v = balance_sheet.loc[balance_sheet.index.str.contains("Current Liab",case=False), col]
            if not v.empty and v.iloc[0] is not None:
                cl.append(v.iloc[0])
        res = 1 if all([ebitda[i] > cl[i] for i in range(min(len(ebitda),len(cl)))]) and len(ebitda)>0 else 0
    except:
        res = -1
    results.append({'title':'2. EBITDA > Current Liabilities ทุกปี','result':res,'desc':'EBITDA มากกว่าหนี้สินหมุนเวียนทุกปี'})
    if res != -1: score += res; evaluated += 1

    try:
        ppe = []
        for col in balance_sheet.columns:
            v = balance_sheet.loc[balance_sheet.index.str.contains("Property, Plant",case=False), col]
            if not v.empty and v.iloc[0] is not None:
                ppe.append(v.iloc[0])
        if len(ppe) >= 2:
            growth = all([ppe[i] <= ppe[i+1] for i in range(len(ppe)-1)])
            spike = max([abs(ppe[i+1]-ppe[i])/ppe[i] if ppe[i]!=0 else 0 for i in range(len(ppe)-1)]) < 1.0
            res = 1 if growth and spike else 0
        else:
            res = -1
    except:
        res = -1
    results.append({'title':'3. PPE เพิ่มขึ้น (ไม่มี spike)','result':res,'desc':'Property, Plant & Equipment โตต่อเนื่อง'})
    if res != -1: score += res; evaluated += 1

    try:
        ebitda = []
        ta = []
        for col in financials.columns:
            v = financials.loc[financials.index.str.contains("EBITDA",case=False), col]
            if not v.empty and v.iloc[0] is not None:
                ebitda.append(v.iloc[0])
        for col in balance_sheet.columns:
            v = balance_sheet.loc[balance_sheet.index.str.contains("Total Assets",case=False), col]
            if not v.empty and v.iloc[0] is not None:
                ta.append(v.iloc[0])
        rtas = [ebitda[i]/ta[i] for i in range(min(len(ebitda),len(ta))) if ta[i]!=0]
        avg_rta = sum(rtas)/len(rtas) if rtas else 0
        res = 1 if avg_rta >= 0.11 else 0
    except:
        res = -1
    results.append({'title':'4.1 RTA ≥ 11%','result':res,'desc':'Return on Total Assets เฉลี่ย ≥ 11%'})
    if res != -1: score += res; evaluated += 1

    try:
        res = 1 if avg_rta >= 0.17 else 0
    except:
        res = -1
    results.append({'title':'4.2 RTA ≥ 17%','result':res,'desc':'Return on Total Assets เฉลี่ย ≥ 17%'})
    if res != -1: score += res; evaluated += 1

    try:
        ltd = []
        for col in balance_sheet.columns:
            v = balance_sheet.loc[balance_sheet.index.str.contains("Long Term Debt",case=False), col]
            if not v.empty and v.iloc[0] is not None:
                ltd.append(v.iloc[0])
        ratios = [ltd[i]/ta[i] for i in range(min(len(ltd),len(ta))) if ta[i]!=0]
        avg_ratio = sum(ratios)/len(ratios) if ratios else 1
        res = 1 if avg_ratio <= 0.5 else 0
    except:
        res = -1
    results.append({'title':'5.1 LTD/Total Assets ≤ 0.5','result':res,'desc':'อัตราส่วนหนี้สินระยะยาว ≤ 0.5'})
    if res != -1: score += res; evaluated += 1

    try:
        last_ebitda = ebitda[-1] if ebitda else None
        last_ltd = ltd[-1] if ltd else None
        if last_ebitda and last_ltd and last_ebitda>0:
            res = 1 if last_ltd/last_ebitda <= 4 else 0
        else:
            res = -1
    except:
        res = -1
    results.append({'title':'5.2 EBITDA จ่ายหนี้ LTD หมดใน ≤ 4 ปี','result':res,'desc':'EBITDA ล่าสุดชำระหนี้ LTD หมดใน ≤ 4 ปี'})
    if res != -1: score += res; evaluated += 1

    try:
        se = []
        for col in balance_sheet.columns:
            v = balance_sheet.loc[balance_sheet.index.str.contains("Total Stock",case=False) & balance_sheet.index.str.contains("Equity",case=False), col]
            if not v.empty and v.iloc[0] is not None:
                se.append(v.iloc[0])
        neg_se = any([x < 0 for x in se])
        res = 1 if neg_se else 0
    except:
        res = -1
    results.append({'title':'6.1 Equity ติดลบในปีใดหรือไม่','result':res,'desc':'ถ้าติดลบ ข้าม 6.2-6.3'})
    if res != -1: evaluated += 1  # ไม่บวกคะแนน

    try:
        if not neg_se:
            tl = []
            ts = []
            for col in balance_sheet.columns:
                v = balance_sheet.loc[balance_sheet.index.str.contains("Total Liab",case=False), col]
                if not v.empty and v.iloc[0] is not None:
                    tl.append(v.iloc[0])
                v = balance_sheet.loc[balance_sheet.index.str.contains("Treasury Stock",case=False), col]
                if not v.empty and v.iloc[0] is not None:
                    ts.append(abs(v.iloc[0]))
            adj_se = [se[i]+(ts[i] if i<len(ts) else 0) for i in range(min(len(se),len(ts)))] if ts else se
            dser = [tl[i]/adj_se[i] for i in range(min(len(tl),len(adj_se))) if adj_se[i]!=0]
            avg_dser = sum(dser)/len(dser) if dser else 0
            res = 1 if avg_dser <= 1.0 else 0
        else:
            res = -1
    except:
        res = -1
    results.append({'title':'6.2 DSER ≤ 1.0','result':res,'desc':'Debt to Shareholder Equity Ratio ≤ 1.0'})
    if res != -1: score += res; evaluated += 1

    try:
        res = 1 if not neg_se and avg_dser <= 0.8 else ( -1 if neg_se else 0)
    except:
        res = -1
    results.append({'title':'6.3 DSER ≤ 0.8','result':res,'desc':'Debt to Shareholder Equity Ratio ≤ 0.8'})
    if res != -1: score += res; evaluated += 1

    try:
        pref = any(balance_sheet.index.str.contains('Preferred',case=False))
        res = 0 if pref else 1
    except:
        res = -1
    results.append({'title':'7. ไม่มี Preferred Stock','result':res,'desc':'ไม่มีหุ้นบุริมสิทธิ'})
    if res != -1: score += res; evaluated += 1

    try:
        re = []
        for col in balance_sheet.columns:
            v = balance_sheet.loc[balance_sheet.index.str.contains("Retained Earnings",case=False), col]
            if not v.empty and v.iloc[0] is not None:
                re.append(v.iloc[0])
        re_growths = [(re[i+1]-re[i])/re[i] if re[i]!=0 else 0 for i in range(len(re)-1)]
        avg_re_growth = sum(re_growths)/len(re_growths) if re_growths else 0
        res = 1 if avg_re_growth >= 0.07 else 0
    except:
        res = -1
    results.append({'title':'8.1 Retained Earnings เติบโต ≥ 7%','result':res,'desc':'Retained Earnings เติบโตเฉลี่ย ≥ 7%'})
    if res != -1: score += res; evaluated += 1

    try:
        res = 1 if avg_re_growth >= 0.135 else 0
    except:
        res = -1
    results.append({'title':'8.2 Retained Earnings เติบโต ≥ 13.5%','result':res,'desc':'Retained Earnings เติบโตเฉลี่ย ≥ 13.5%'})
    if res != -1: score += res; evaluated += 1

    try:
        res = 1 if avg_re_growth >= 0.17 else 0
    except:
        res = -1
    results.append({'title':'8.3 Retained Earnings เติบโต ≥ 17%','result':res,'desc':'Retained Earnings เติบโตเฉลี่ย ≥ 17%'})
    if res != -1: score += res; evaluated += 1

    try:
        ts = []
        for col in balance_sheet.columns:
            v = balance_sheet.loc[balance_sheet.index.str.contains("Treasury Stock",case=False), col]
            if not v.empty and v.iloc[0] is not None:
                ts.append(v.iloc[0])
        res = 1 if any([x!=0 for x in ts]) else 0
    except:
        res = -1
    results.append({'title':'9. มี Treasury Stock','result':res,'desc':'มี Treasury Stock หรือไม่'})
    if res != -1: score += res; evaluated += 1

    try:
        roe = [ebitda[i]/se[i] for i in range(min(len(ebitda),len(se))) if se[i]!=0]
        avg_roe = sum(roe)/len(roe) if roe else 0
        res = 1 if avg_roe >= 0.23 else 0
    except:
        res = -1
    results.append({'title':'10. ROE ≥ 23%','result':res,'desc':'Return on Shareholders Equity เฉลี่ย ≥ 23%'})
    if res != -1: score += res; evaluated += 1

    try:
        gw = []
        for col in balance_sheet.columns:
            v = balance_sheet.loc[balance_sheet.index.str.contains("Goodwill",case=False), col]
            if not v.empty and v.iloc[0] is not None:
                gw.append(v.iloc[0])
        res = 1 if all([gw[i] <= gw[i+1] for i in range(len(gw)-1)]) and len(gw)>=2 else 0
    except:
        res = -1
    results.append({'title':'11. Goodwill เพิ่มขึ้น','result':res,'desc':'Goodwill โตต่อเนื่อง'})
    if res != -1: score += res; evaluated += 1

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

def dca_simulation(hist_prices: pd.DataFrame, monthly_invest: float = 1000):
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
    return {
        "เงินลงทุนรวม": round(total_invested, 2),
        "จำนวนหุ้นสะสม": round(total_units, 4),
        "มูลค่าปัจจุบัน": round(current_value, 2),
        "กำไร/ขาดทุน": round(profit, 2),
        "กำไร(%)": round(profit/total_invested*100, 2) if total_invested != 0 else 0,
        "ราคาเฉลี่ยที่ซื้อ": round(avg_buy_price, 2),
        "ราคาปิดล่าสุด": round(latest_price, 2)
    }

# ----------------- SET100/US STOCKS -----------------
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
all_tickers = us_stocks + set100

# ----------------- UI & Main -----------------
st.set_page_config(page_title="Warren-DCA วิเคราะห์หุ้น", layout="wide")
menu = st.sidebar.radio("เลือกหน้าที่ต้องการ", ["วิเคราะห์หุ้น", "คู่มือการใช้งาน"])

if menu == "คู่มือการใช้งาน":
    st.header("คู่มือการใช้งาน (ภาษาไทย)")
    st.markdown("""
**Warren-DCA คืออะไร?**  
โปรแกรมนี้ช่วยวิเคราะห์หุ้นตามแนวทางของ Warren Buffett (Buffett 11 Checklist) พร้อมจำลองการลงทุนแบบถัวเฉลี่ย (DCA)  
**แหล่งข้อมูล:** Yahoo Finance

### กฎ 11 ข้อ (DCA Checklist แบบละเอียด)
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
- ข้อมูลหุ้น US จะครบถ้วนมากกว่าหุ้นไทย
- ถ้าข้อมูลหลักๆ ไม่ครบ บางข้อจะขึ้นว่า N/A
- ข้อมูลย้อนหลังงบการเงิน 4 ปี (Annual)  
""")
    st.stop()

tickers = st.multiselect(
    "เลือกหุ้น (US & SET100)",
    all_tickers,
    default=["AAPL", "PTT.BK"]
)
period = st.selectbox("เลือกช่วงเวลาราคาหุ้น", ["1y", "5y", "max"], index=1)
monthly_invest = st.number_input("จำนวนเงินลงทุน DCA ต่อเดือน (บาทหรือ USD)", min_value=100.0, max_value=10000.0, value=1000.0, step=100.0)
show_financials = st.checkbox("แสดงงบการเงิน (Income Statement)", value=False)

if st.button("วิเคราะห์"):
    export_list = []
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        fin = stock.financials
        bs = stock.balance_sheet
        cf = stock.cashflow
        div = stock.dividends
        hist = stock.history(period=period)
        info = stock.info

        with st.expander(f"ดูรายละเอียดหุ้น {ticker}", expanded=False):
            # ส่วนข้อมูลราคาหุ้นและปันผลล่าสุด
            st.subheader("ข้อมูลราคาหุ้นและปันผลล่าสุด")

            # 1. Dividend Yield (% ต่อปี)
            div_yield = info.get('dividendYield', None)
            div_yield_pct = round(div_yield * 100, 2) if div_yield is not None else "N/A"

            # 2. วันที่ปันผลล่าสุด (Ex-Dividend Date)
            ex_div = info.get('exDividendDate', None)
            if ex_div:
                try:
                    ex_div_date = datetime.datetime.fromtimestamp(ex_div).strftime('%Y-%m-%d')
                except:
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

            # --------- สรุปปันผลย้อนหลัง 1 ปี (แยกชัดเจน) ---------
            st.subheader("ผลตอบแทนเงินปันผลย้อนหลัง 1 ปี (คำนวณจากราคาจริง)")
            if not div.empty and not hist.empty:
                last_year = hist.index[-1] - pd.DateOffset(years=1)
                recent_div = div[div.index >= last_year]
                total_div = recent_div.sum()
                avg_price = hist['Close'][hist.index >= last_year].mean()
                price_base = avg_price if avg_price and avg_price > 0 else hist['Close'].iloc[-1]
                manual_yield = (total_div / price_base) * 100 if price_base > 0 else np.nan

                st.markdown(f"""
                <div style='font-size:1.1em;'>
                <b>เงินปันผลรวม 1 ปี:</b> <span style='color:green'>{total_div:.2f}</span><br>
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
                    'ข้อ': i+1,
                    'รายการ': d['title'],
                    'ผลลัพธ์': "✅ ผ่าน" if d['result']==1 else ("❌ ไม่ผ่าน" if d['result']==0 else "⚪ N/A"),
                    'คำอธิบาย': d['desc']
                }
                for i,d in enumerate(detail['details'])
            ])
            st.dataframe(df_detail, hide_index=True)

            st.subheader("DCA Simulation (จำลองลงทุนรายเดือน)")
            dca_result = dca_simulation(hist, monthly_invest)
            st.write(pd.DataFrame(dca_result, index=['สรุปผล']).T)

            if not hist.empty:
                st.line_chart(hist['Close'])
            else:
                st.warning("ไม่มีข้อมูลราคาหุ้น")

            if show_financials and fin is not None and not fin.empty:
                st.subheader("งบกำไรขาดทุน (Income Statement)")
                st.dataframe(df_human_format(fin))

            # -- เตรียม export (ดึงค่า manual yield ด้วย) --
            export_list.append({
                "หุ้น": ticker,
                "คะแนนรวม": f"{detail['score']}/{detail['evaluated']}",
                "เปอร์เซ็นต์": detail['score_pct'],
                "ป้ายคะแนน": badge,
                **dca_result,
                "Dividend Yield (%)": div_yield_pct,
                "Ex-Dividend Date": ex_div_date,
                "52W High": w52_high,
                "52W Low": w52_low,
                "ราคาปิดล่าสุด": last_close,
                "ราคาเปิดล่าสุด": last_open,
                "เงินปันผลย้อนหลัง 1 ปี": round(total_div,2) if not div.empty and not hist.empty else "N/A",
                "Yield ย้อนหลัง 1 ปี (%)": round(manual_yield,2) if not div.empty and not hist.empty else "N/A"
            })

    # --- Portfolio Allocation Pie Chart ---
    if len(tickers) > 0:
        st.header("📊 การกระจายการลงทุนในพอร์ตโฟลิโอ (Portfolio Allocation)")
        
        if len(tickers) == 1:
            st.info(f"💡 คุณเลือกหุ้นเพียงตัวเดียว ({tickers[0]}) - พอร์ตโฟลิโอประกอบด้วยหุ้นนี้ 100%")
        else:
            # Create and display pie chart
            fig = create_portfolio_pie_chart(tickers)
            if fig:
                st.pyplot(fig)
                plt.close(fig)  # Clean up to prevent memory issues
                
                # Display portfolio weights as a table
                weights = generate_portfolio_weights(tickers)
                st.subheader("📋 สัดส่วนการลงทุน")
                portfolio_df = pd.DataFrame([
                    {"หุ้น": ticker, "สัดส่วน (%)": f"{weight * 100:.1f}%", "น้ำหนัก": weight}
                    for ticker, weight in weights.items()
                ])
                st.dataframe(portfolio_df, hide_index=True)
                
                st.info("ℹ️ สัดส่วนการลงทุนแบ่งเท่าๆ กันสำหรับทุกหุ้นที่เลือก (Equal Weight Portfolio)")

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

st.caption("Powered by Yahoo Finance | วิเคราะห์หุ้นด้วย Buffett 11 Checklist (ละเอียด) + DCA + ปันผลย้อนหลัง พร้อม Export Excel (เมนูไทย)")
