import streamlit as st
import yfinance as yf
import pandas as pd
import io

st.set_page_config(page_title="Warren-DCA: วิเคราะห์หุ้นและ DCA (ครบสูตร)", layout="wide")
st.title("Warren-DCA: วิเคราะห์หุ้นและ DCA (Buffett 11 ข้อ + DCA + Export Excel)")

# ----------------- SET100 Thailand -----------------
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

# ----------------- Buffett 11 Checklist -----------------
def buffett_11_checks(financials, balance_sheet, cashflow, dividends, hist_prices):
    summary = {}
    # 1. กำไรสุทธิต่อเนื่อง 3-4 ปี
    try:
        net_income = financials.loc[[i for i in financials.index if "Net Income" in i][0]]
        summary["1. มีกำไรสุทธิต่อเนื่อง"] = bool((net_income > 0).all())
    except Exception:
        summary["1. มีกำไรสุทธิต่อเนื่อง"] = "N/A"

    # 2. หนี้สินรวมน้อย (Debt/Equity < 0.5)
    try:
        total_liab = balance_sheet.loc[[i for i in balance_sheet.index if "Total Liab" in i][0]]
        equity = balance_sheet.loc[[i for i in balance_sheet.index if "Total Stockholder Equity" in i][0]]
        debt_equity = (total_liab / equity).mean()
        summary["2. หนี้สินรวมน้อย (D/E < 0.5)"] = debt_equity < 0.5
    except Exception:
        summary["2. หนี้สินรวมน้อย (D/E < 0.5)"] = "N/A"

    # 3. ROE > 15%
    try:
        net_income = financials.loc[[i for i in financials.index if "Net Income" in i][0]]
        equity = balance_sheet.loc[[i for i in balance_sheet.index if "Total Stockholder Equity" in i][0]]
        roe = (net_income / equity).mean()
        summary["3. ROE > 15%"] = roe > 0.15
    except Exception:
        summary["3. ROE > 15%"] = "N/A"

    # 4. Net Profit Margin > 10%
    try:
        revenue = financials.loc[[i for i in financials.index if "Total Revenue" in i][0]]
        net_income = financials.loc[[i for i in financials.index if "Net Income" in i][0]]
        margin = (net_income / revenue).mean()
        summary["4. อัตรากำไรสุทธิ > 10%"] = margin > 0.10
    except Exception:
        summary["4. อัตรากำไรสุทธิ > 10%"] = "N/A"

    # 5. กระแสเงินสดดำเนินงานเป็นบวก
    try:
        ocf = cashflow.loc[[i for i in cashflow.index if "Total Cash From Operating Activities" in i][0]]
        summary["5. กระแสเงินสดดำเนินงานเป็นบวก"] = (ocf > 0).all()
    except Exception:
        summary["5. กระแสเงินสดดำเนินงานเป็นบวก"] = "N/A"

    # 6. ความได้เปรียบทางแข่งขัน (ให้ประเมินเอง)
    summary["6. มีความได้เปรียบทางแข่งขัน"] = "ประเมินเอง"

    # 7. รายได้เติบโต
    try:
        revenue = financials.loc[[i for i in financials.index if "Total Revenue" in i][0]]
        summary["7. รายได้เติบโต"] = revenue.iloc[0] < revenue.iloc[-1]
    except Exception:
        summary["7. รายได้เติบโต"] = "N/A"
    # 8. กำไรสุทธิเติบโต
    try:
        net_income = financials.loc[[i for i in financials.index if "Net Income" in i][0]]
        summary["8. กำไรสุทธิเติบโต"] = net_income.iloc[0] < net_income.iloc[-1]
    except Exception:
        summary["8. กำไรสุทธิเติบโต"] = "N/A"

    # 9. Current ratio > 1
    try:
        current_assets = balance_sheet.loc[[i for i in balance_sheet.index if "Total Current Assets" in i][0]]
        current_liab = balance_sheet.loc[[i for i in balance_sheet.index if "Total Current Liabilities" in i][0]]
        current_ratio = (current_assets / current_liab).mean()
        summary["9. Current Ratio > 1"] = current_ratio > 1
    except Exception:
        summary["9. Current Ratio > 1"] = "N/A"

    # 10. Margin of Safety (ประเมินเอง)
    summary["10. มี Margin of Safety"] = "ประเมินเอง"

    # 11. จ่ายปันผลสม่ำเสมอ
    try:
        summary["11. ปันผลสม่ำเสมอ"] = len(dividends) > 0 and (dividends > 0).any()
    except Exception:
        summary["11. ปันผลสม่ำเสมอ"] = "N/A"

    return summary

# ----------------- DCA Simulation -----------------
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

# ----------------- UI -----------------
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
        st.header(f"หุ้น: {ticker}")
        stock = yf.Ticker(ticker)

        fin = stock.financials
        bs = stock.balance_sheet
        cf = stock.cashflow
        div = stock.dividends
        hist = stock.history(period=period)

        # --- Buffett Score 11 ข้อ ---
        st.subheader("Buffett 11 Checklist")
        buffett_checks = buffett_11_checks(fin, bs, cf, div, hist)
        st.write(pd.DataFrame(buffett_checks, index=['ผลลัพธ์']).T)

        # --- DCA Simulation ---
        st.subheader("DCA Simulation (จำลองลงทุนรายเดือน)")
        dca_result = dca_simulation(hist, monthly_invest)
        st.write(pd.DataFrame(dca_result, index=['สรุปผล']).T)

        # --- Chart ---
        if not hist.empty:
            st.line_chart(hist['Close'])
        else:
            st.warning("ไม่มีข้อมูลราคาหุ้น")

        # --- Financials ---
        if show_financials and fin is not None and not fin.empty:
            st.subheader("งบกำไรขาดทุน (Income Statement)")
            st.dataframe(fin)

        # สำหรับ export
        export_list.append({
            "หุ้น": ticker,
            **buffett_checks,
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

st.caption("Powered by Yahoo Finance | วิเคราะห์หุ้นด้วย Buffett Checklist และ DCA พร้อม Export Excel (เมนูไทย)")
