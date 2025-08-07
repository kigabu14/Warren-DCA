import streamlit as st
import yfinance as yf
import pandas as pd

st.title("Warren-DCA: Buffett Score & DCA Simulation (Yahoo Finance)")

# --- Buffett Score Calculation ---
def calc_buffett_score(financials: pd.DataFrame):
    try:
        # ยอมรับโครงสร้างงบการเงินของ yfinance
        roe = None
        debt_equity = None
        profit_margin = None

        # หา index ที่ตรงกัน (ชื่ออาจต่างกันได้ในแต่ละหุ้น/ช่วงเวลา)
        def find_row(label):
            for idx in financials.index:
                if label.lower() in idx.lower():
                    return idx
            return None

        net_income_row = find_row('Net Income')
        equity_row = find_row('Total Stockholder Equity')
        total_liab_row = find_row('Total Liab')
        revenue_row = find_row('Total Revenue')

        if net_income_row and equity_row:
            roe = financials.loc[net_income_row, :] / financials.loc[equity_row, :]
        if total_liab_row and equity_row:
            debt_equity = financials.loc[total_liab_row, :] / financials.loc[equity_row, :]
        if net_income_row and revenue_row:
            profit_margin = financials.loc[net_income_row, :] / financials.loc[revenue_row, :]

        # ใช้ค่าเฉลี่ย 3-4 ปีย้อนหลัง
        roe_score = min(roe.mean() * 10, 10) if roe is not None else 0
        debt_score = max(10 - (debt_equity.mean() * 10), 0) if debt_equity is not None else 0
        margin_score = min(profit_margin.mean() * 10, 10) if profit_margin is not None else 0
        total_score = roe_score + debt_score + margin_score

        return {
            "ROE(Score)": round(roe_score, 2),
            "Debt/Equity(Score)": round(debt_score, 2),
            "Profit Margin(Score)": round(margin_score, 2),
            "Total Buffett Score": round(total_score, 2)
        }
    except Exception as e:
        return {"Error": str(e)}

# --- DCA Simulation ---
def dca_simulation(hist_prices: pd.DataFrame, monthly_invest: float = 1000):
    if hist_prices.empty:
        return {"Error": "No price data"}
    # ใช้ราคาปิดแรกของแต่ละเดือน (ถ้ามีข้อมูลวันหยุดจะข้ามเดือนนั้นไป)
    prices = hist_prices['Close'].resample('M').first().dropna()
    units = monthly_invest / prices
    total_units = units.sum()
    total_invested = monthly_invest * len(prices)
    avg_buy_price = total_invested / total_units if total_units != 0 else 0
    latest_price = prices.iloc[-1]
    current_value = total_units * latest_price
    profit = current_value - total_invested
    return {
        "Total Invested": round(total_invested, 2),
        "Units Accumulated": round(total_units, 4),
        "Current Value": round(current_value, 2),
        "Profit": round(profit, 2),
        "Profit(%)": round(profit/total_invested*100, 2) if total_invested != 0 else 0,
        "Average Buy Price": round(avg_buy_price, 2),
        "Latest Price": round(latest_price, 2)
    }

# --- UI ---
tickers = st.multiselect(
    "Select Tickers",
    [
        "AAPL", "TSLA", "NVDA", "GOOG", "MSFT", "SBUX", "AMD", "BABA", "T", "WMT",
        "SONY", "KO", "MCD", "MCO", "SNAP", "DIS", "NFLX", "GPRO", "CCL", "PLTR", "CBOE", "HD", "F", "COIN"
    ],
    default=["AAPL", "TSLA"]
)
period = st.selectbox("Select Period for price chart", ["1y", "5y", "max"], index=1)
monthly_invest = st.number_input("Monthly DCA Amount", min_value=100.0, max_value=10000.0, value=1000.0, step=100.0)
show_financials = st.checkbox("Show Financials (Income Statement)", value=False)

if st.button("Analyze"):
    for ticker in tickers:
        st.header(ticker)
        stock = yf.Ticker(ticker)

        # --- Financials and Buffett Score ---
        fin = stock.financials
        if fin is not None and not fin.empty:
            st.write("**Buffett Score**")
            buffett_score = calc_buffett_score(fin)
            st.json(buffett_score)
            if show_financials:
                st.write("**Income Statement:**")
                st.dataframe(fin)
        else:
            st.warning("No financials found for Buffett Score")

        # --- DCA Simulation ---
        hist = stock.history(period=period)
        if not hist.empty:
            st.write("**DCA Simulation**")
            dca_result = dca_simulation(hist, monthly_invest)
            st.json(dca_result)
            st.line_chart(hist['Close'])
        else:
            st.warning("No price data found")

st.caption("Powered by Yahoo Finance via yfinance | Buffett Score และ DCA Simulation พร้อมใช้")
