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

def buffett_11_checks(financials, balance_sheet, cashflow, dividends, hist_prices):
    summary = {}
    # 1. มีกำไรสุทธิ 4 ปีติด
    try:
        if financials is not None and not financials.empty:
            ni_row = [i for i in financials.index if "Net Income" in i or "NetIncome" in i]
            if ni_row:
                net_income = financials.loc[ni_row[0]]
                summary["1. มีกำไรสุทธิ 4 ปีติด"] = bool((net_income > 0).all())
            else:
                summary["1. มีกำไรสุทธิ 4 ปีติด"] = "ข้อมูลไม่ครบ"
        else:
            summary["1. มีกำไรสุทธิ 4 ปีติด"] = "ข้อมูลไม่ครบ"
    except Exception as e:
        summary["1. มีกำไรสุทธิ 4 ปีติด"] = f"Error: {e}"

    # 2. D/E < 0.5
    try:
        if balance_sheet is not None and not balance_sheet.empty:
            liab_row = [i for i in balance_sheet.index if "totalLiabilities" in i]
            eq_row = [i for i in balance_sheet.index if "shareholderEquity" in i]
            if liab_row and eq_row:
                total_liab = balance_sheet.loc[liab_row[0]]
                equity = balance_sheet.loc[eq_row[0]]
                if (equity > 0).all():
                    debt_equity = (total_liab / equity).mean()
                    summary["2. D/E < 0.5"] = debt_equity < 0.5
                else:
                    summary["2. D/E < 0.5"] = "Equity เป็นศูนย์"
            else:
                summary["2. D/E < 0.5"] = "ข้อมูลไม่ครบ"
        else:
            summary["2. D/E < 0.5"] = "ข้อมูลไม่ครบ"
    except Exception as e:
        summary["2. D/E < 0.5"] = f"Error: {e}"

    # 3. ROE > 15%
    try:
        if balance_sheet is not None and not balance_sheet.empty:
            if (equity > 0).all():
                roe = (net_income / equity).mean()
                summary["3. ROE > 15%"] = roe > 0.15
            else:
                summary["3. ROE > 15%"] = "Equity เป็นศูนย์"
        else:
            summary["3. ROE > 15%"] = "ข้อมูลไม่ครบ"
    except Exception as e:
        summary["3. ROE > 15%"] = f"Error: {e}"

    # 4. Margin > 10%
    try:
        rev_row = [i for i in financials.index if "totalRevenue" in i]
        if rev_row:
            revenue = financials.loc[rev_row[0]]
            margin = (net_income / revenue).mean()
            summary["4. กำไรสุทธิ > 10%"] = margin > 0.10
        else:
            summary["4. กำไรสุทธิ > 10%"] = "ข้อมูลไม่ครบ"
    except Exception as e:
        summary["4. กำไรสุทธิ > 10%"] = f"Error: {e}"

    # 5. กระแสเงินสดดำเนินงานบวก
    try:
        if cashflow is not None and not cashflow.empty:
            ocf_row = [i for i in cashflow.index if "totalInvestingCashFlows" in i]
            if ocf_row:
                ocf = cashflow.loc[ocf_row[0]]
                summary["5. กระแสเงินสดดำเนินงานบวก"] = (ocf > 0).all()
            else:
                summary["5. กระแสเงินสดดำเนินงานบวก"] = "ข้อมูลไม่ครบ"
        else:
            summary["5. กระแสเงินสดดำเนินงานบวก"] = "ข้อมูลไม่ครบ"
    except Exception as e:
        summary["5. กระแสเงินสดดำเนินงานบวก"] = f"Error: {e}"

    # 6. ความได้เปรียบทางแข่งขัน
    summary["6. ความได้เปรียบทางแข่งขัน"] = "ประเมินเอง"

    # 7. รายได้เติบโต
    try:
        if "revenue" in locals() and len(revenue) > 1:
            summary["7. รายได้เติบโต"] = revenue.iloc[0] < revenue.iloc[-1]
        else:
            summary["7. รายได้เติบโต"] = "ข้อมูลไม่ครบ"
    except Exception as e:
        summary["7. รายได้เติบโต"] = f"Error: {e}"

    # 8. กำไรสุทธิเติบโต
    try:
        if "net_income" in locals() and len(net_income) > 1:
            summary["8. กำไรสุทธิเติบโต"] = net_income.iloc[0] < net_income.iloc[-1]
        else:
            summary["8. กำไรสุทธิเติบโต"] = "ข้อมูลไม่ครบ"
    except Exception as e:
        summary["8. กำไรสุทธิเติบโต"] = f"Error: {e}"

    # 9. Current Ratio > 1
    try:
        if balance_sheet is not None and not balance_sheet.empty:
            ca_row = [i for i in balance_sheet.index if "totalAssets" in i]
            cl_row = [i for i in balance_sheet.index if "totalLiabilities" in i]
            if ca_row and cl_row:
                current_assets = balance_sheet.loc[ca_row[0]]
                current_liab = balance_sheet.loc[cl_row[0]]
                if (current_liab > 0).all():
                    current_ratio = (current_assets / current_liab).mean()
                    summary["9. Current Ratio > 1"] = current_ratio > 1
                else:
                    summary["9. Current Ratio > 1"] = "Liabilities เป็นศูนย์"
            else:
                summary["9. Current Ratio > 1"] = "ข้อมูลไม่ครบ"
        else:
            summary["9. Current Ratio > 1"] = "ข้อมูลไม่ครบ"
    except Exception as e:
        summary["9. Current Ratio > 1"] = f"Error: {e}"

    # 10. Margin of Safety
    summary["10. Margin of Safety"] = "ประเมินเอง"

    # 11. ปันผลสม่ำเสมอ
    try:
        if dividends is not None and len(dividends) > 0:
            summary["11. ปันผลสม่ำเสมอ"] = (dividends > 0).any()
        else:
            summary["11. ปันผลสม่ำเสมอ"] = "ข้อมูลไม่ครบ"
    except Exception as e:
        summary["11. ปันผลสม่ำเสมอ"] = f"Error: {e}"

    return summary

def dca_simulation(hist_prices: pd.DataFrame, monthly_invest: float = 1000, dividend_yield: float = None):
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

    # คำนวณรายได้จากปันผล (ถ้ามี dividend_yield)
    dividend_income = 0
    if dividend_yield is not None:
        dividend_income = total_units * latest_price * dividend_yield

    return {
        "เงินลงทุนรวม": round(total_invested, 2),
        "จำนวนหุ้นสะสม": round(total_units, 4),
        "มูลค่าปัจจุบัน": round(current_value, 2),
        "กำไร/ขาดทุน": round(profit, 2),
        "กำไร(%)": round(profit/total_invested*100, 2) if total_invested != 0 else 0,
        "ราคาเฉลี่ยที่ซื้อ": round(avg_buy_price, 2),
        "ราคาปิดล่าสุด": round(latest_price, 2),
        "รายได้จากปันผล (ประมาณการ)": round(dividend_income, 2)
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
โปรแกรมนี้ช่วยวิเคราะห์หุ้นตามแนวทางของ Warren Buffett (Buffett 11 Checklist) พร้อมจำลองการลงทุนแบบ DCA  
**แหล่งข้อมูล:** Yahoo Finance

### คำอธิบายแต่ละส่วน
- **Buffett 11 Checklist**:  
    1. มีกำไรสุทธิ 4 ปีติด (Net Income)
    2. D/E < 0.5 (Total Liabilities / Total Equity)
    3. ROE > 15% (Net Income / Equity)
    4. กำไรสุทธิ > 10% (Net Income / Revenue)
    5. กระแสเงินสดดำเนินงานบวก (Operating Cash Flow)
    6. ความได้เปรียบทางแข่งขัน: ต้องประเมินเอง
    7. รายได้เติบโต (Revenue)
    8. กำไรสุทธิเติบโต (Net Income)
    9. Current Ratio > 1 (Current Assets / Current Liabilities)
    10. Margin of Safety: ประเมินเอง
    11. ปันผลสม่ำเสมอ (Dividends)

- **DCA Simulation**: จำลองซื้อหุ้นทุกเดือนด้วยเงินเท่ากัน
- **เลขตัวย่อ**: K = พัน, M = ล้าน, B = พันล้าน, T = ล้านล้าน

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

        # ข้อมูลเพิ่มเติม
        forward_dividend_yield = stock.info.get('dividendYield', None)  # Forward Dividend Yield
        fifty_two_week_high = stock.info.get('fiftyTwoWeekHigh', None)  # 52 Week High
        fifty_two_week_low = stock.info.get('fiftyTwoWeekLow', None)    # 52 Week Low
        ex_dividend_date = stock.info.get('exDividendDate', None)       # Ex-Dividend Date
        earnings_date = stock.info.get('earningsDate', None)            # Earnings Date

        with st.expander(f"ดูรายละเอียดหุ้น {ticker}", expanded=False):
            st.subheader("Buffett 11 Checklist")
            buffett_checks = buffett_11_checks(fin, bs, cf, div, hist)
            st.write(pd.DataFrame(buffett_checks, index=['ผลลัพธ์']).T)

            st.subheader("ข้อมูลสำคัญเพิ่มเติม")
            st.write({
                "Forward Dividend & Yield": forward_dividend_yield,
                "52 Week High": fifty_two_week_high,
                "52 Week Low": fifty_two_week_low,
                "Ex-Dividend Date": ex_dividend_date,
                "Earnings Date": earnings_date,
            })

            st.subheader("DCA Simulation (จำลองลงทุนรายเดือน)")
            dca_result = dca_simulation(hist, monthly_invest, dividend_yield=forward_dividend_yield)
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
                **buffett_checks,
                **dca_result,
                "Forward Dividend & Yield": forward_dividend_yield,
                "52 Week High": fifty_two_week_high,
                "52 Week Low": fifty_two_week_low,
                "Ex-Dividend Date": ex_dividend_date,
                "Earnings Date": earnings_date,
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

st.caption("Powered by Yahoo Finance | วิเคราะห์หุ้นด้วย Buffett 11 Checklist และ DCA พร้อม Export Excel (เมนูไทย)")