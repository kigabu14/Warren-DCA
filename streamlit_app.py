import streamlit as st
import yfinance as yf

st.title("Warren-DCA Yahoo Finance Results")

tickers = st.multiselect(
    "Select Tickers",
    [
        "AAPL", "TSLA", "NVDA", "GOOG", "MSFT", "SBUX", "AMD", "BABA", "T", "WMT",
        "SONY", "KO", "MCD", "MCO", "SNAP", "DIS", "NFLX", "GPRO", "CCL", "PLTR", "CBOE", "HD", "F", "COIN"
    ],
    default=["AAPL", "TSLA"]
)
period = st.selectbox("Select Period for price chart", ["1y", "5y", "max"], index=0)
show_financials = st.checkbox("Show Financials (Income Statement)", value=True)

if st.button("Fetch Data"):
    for ticker in tickers:
        st.subheader(ticker)
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        if not hist.empty:
            st.line_chart(hist['Close'])
        else:
            st.warning(f"No historical price data found for {ticker}")

        if show_financials:
            st.write("**Income Statement:**")
            fin = stock.financials
            if fin is not None and not fin.empty:
                st.dataframe(fin)
            else:
                st.warning(f"No financial data found for {ticker}")

st.caption("Powered by Yahoo Finance via yfinance")
