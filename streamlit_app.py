import streamlit as st
import yfinance as yf
from pprint import pprint

st.title("Yahoo Finance Stock Data")

tickers = st.multiselect("Select Tickers", ["AAPL", "TSLA", "NVDA", "GOOG", "MSFT", "SBUX", "AMD"], default=["AAPL", "TSLA"])
period = st.selectbox("Select Period", ["1y", "5y", "max"], index=0)
show_financials = st.checkbox("Show Financials (Income Statement)")

if st.button("Fetch Data"):
    for ticker in tickers:
        st.subheader(ticker)
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        st.line_chart(hist['Close'])

        if show_financials:
            st.write("**Income Statement:**")
            fin = stock.financials
            st.dataframe(fin)
