import streamlit as st
import yfinance as yf
import pandas as pd
import io

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

# ----------------- Buffett 11 Checklist (ละเอียดแบบ parameters.py) -----------------

def buffett_11_checks_detail(financials, balance_sheet, cashflow, dividends, hist_prices):
    """
    คืน dict: {
      'details': list ของแต่ละข้อ [{'title':..., 'result':1/0/-1, 'desc':...}],
      'score': int, 'evaluated': int, 'score_pct': int
    }
    """
    results = []
    score = 0
    evaluated = 0

    # 1.1 Inventory & Net Earnings เพิ่มขึ้นต่อเนื่อง
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

    # 1.2 ไม่มี R&D
    try:
        r_and_d = any(financials.index.str.contains('Research',case=False))
        res = 0 if r_and_d else 1
    except:
        res = -1
    results.append({'title':'1.2 ไม่มี Research & Development','result':res,'desc':'ไม่มีค่าใช้จ่าย R&D'})
    if res != -1: score += res; evaluated += 1

    # 2. EBITDA > Current Liabilities ทุกปี
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

    # 3. PPE เพิ่มขึ้น (ไม่มี spike)
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

    # 4.1 RTA ≥ 11%
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

    # 4.2 RTA ≥ 17%
    try:
        res = 1 if avg_rta >= 0.17 else 0
    except:
        res = -1
    results.append({'title':'4.2 RTA ≥ 17%','result':res,'desc':'Return on Total Assets เฉลี่ย ≥ 17%'})
    if res != -1: score += res; evaluated += 1

    # 5.1 LTD/Total Assets ≤ 0.5
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

    # 5.2 EBITDA ปีล่าสุดจ่ายหนี้ LTD หมดใน ≤ 4 ปี
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

    # 6.1 มีปีไหน Equity ติดลบหรือไม่
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

    # 6.2 DSER ≤ 1.0
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

    # 6.3 DSER ≤ 0.8
    try:
        res = 1 if not neg_se and avg_dser <= 0.8 else ( -1 if neg_se else 0)
    except:
        res = -1
    results.append({'title':'6.3 DSER ≤ 0.8','result':res,'desc':'Debt to Shareholder Equity Ratio ≤ 0.8'})
    if res != -1: score += res; evaluated += 1

    # 7. ไม่มี Preferred Stock
    try:
        pref = any(balance_sheet.index.str.contains('Preferred',case=False))
        res = 0 if pref else 1
    except:
        res = -1
    results.append({'title':'7. ไม่มี Preferred Stock','result':res,'desc':'ไม่มีหุ้นบุริมสิทธิ'})
    if res != -1: score += res; evaluated += 1

    # 8.1 Retained Earnings เติบโต ≥ 7%
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

    # 8.2 ≥ 13.5%
    try:
        res = 1 if avg_re_growth >= 0.135 else 0
    except:
        res = -1
    results.append({'title':'8.2 Retained Earnings เติบโต ≥ 13.5%','result':res,'desc':'Retained Earnings เติบโตเฉลี่ย ≥ 13.5%'})
    if res != -1: score += res; evaluated += 1

    # 8.3 ≥ 17%
    try:
        res = 1 if avg_re_growth >= 0.17 else 0
    except:
        res = -1
    results.append({'title':'8.3 Retained Earnings เติบโต ≥ 17%','result':res,'desc':'Retained Earnings เติบโตเฉลี่ย ≥ 17%'})
    if res != -1: score += res; evaluated += 1

    # 9. มี Treasury Stock
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

    # 10. ROE ≥ 23%
    try:
        roe = [ebitda[i]/se[i] for i in range(min(len(ebitda),len(se))) if se[i]!=0]
        avg_roe = sum(roe)/len(roe) if roe else 0
        res = 1 if avg_roe >= 0.23 else 0
    except:
        res = -1
    results.append({'title':'10. ROE ≥ 23%','result':res,'desc':'Return on Shareholders Equity เฉลี่ย ≥ 23%'})
    if res != -1: score += res; evaluated += 1

    # 11. Goodwill เพิ่มขึ้น
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

# Badge function
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

# หน้า "วิเคราะห์หุ้น"
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

        with st.expander(f"ดูรายละเอียดหุ้น {ticker}", expanded=False):
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

            # export
            export_list.append({
                "หุ้น": ticker,
                "คะแนนรวม": f"{detail['score']}/{detail['evaluated']}",
                "เปอร์เซ็นต์": detail['score_pct'],
                "ป้ายคะแนน": badge,
                **dca_result
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

st.caption("Powered by Yahoo Finance | วิเคราะห์หุ้นด้วย Buffett 11 Checklist (ละเอียด) + DCA พร้อม Export Excel (เมนูไทย)")
