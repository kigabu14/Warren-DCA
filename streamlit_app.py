import streamlit as st
import yfinance as yf
from buffett_and_dca import calc_buffett_score, dca_simulation

st.title("Warren-DCA: Buffett Score & DCA Simulation")

tickers = st.multiselect("Select Tickers", ["AAPL", "TSLA", "NVDA", "GOOG", "MSFT"], default=["AAPL"])
monthly_invest = st.number_input("Monthly DCA Amount", min_value=100.0, max_value=10000.0, value=1000.0, step=100.0)
period = st.selectbox("Select Period for price chart", ["1y", "5y", "max"], index=1)

if st.button("Analyze"):
    for ticker in tickers:
        st.header(ticker)
        stock = yf.Ticker(ticker)
        fin = stock.financials
        if fin is not None and not fin.empty:
            st.write("**Buffett Score**")
            buffett_score = calc_buffett_score(fin)
            st.json(buffett_score)
        else:
            st.warning("No financials found for Buffett Score")

        hist = stock.history(period=period)
        if not hist.empty:
            st.write("**DCA Simulation**")
            dca_result = dca_simulation(hist, monthly_invest)
            st.json(dca_result)
            st.line_chart(hist['Close'])
        else:
            st.warning("No price data found")
