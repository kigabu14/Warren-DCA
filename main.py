import streamlit as st
from pprint import pprint
from edgar.client import EdgarClient
from dca.client import DCAClient

# Streamlit UI
st.title("Warren-DCA: Financials & DCA Analysis")

# Select tickers & period
all_tickers = [
    'CBOE', 'AAPL', 'WMT', 'F', 'HD', 'SONY', 'KO', 'GOOG', 'MCD', 'MCO', 'TSLA',
    'MSFT', 'ZM', 'SNAP', 'DIS', 'NFLX', 'GPRO', 'BABA', 'CCL', 'PLTR', 'AMD',
    'NVDA', 'SBUX', 'COIN', 'T'
]
selected_tickers = st.multiselect('Select Tickers to Analyze', all_tickers, default=['AAPL', 'TSLA', 'NVDA'])
period = st.selectbox('Select Period', ['annual', 'quarter'])

run_analysis = st.button("Run DCA Analysis")

if run_analysis:
    st.write("Running analysis...")
    edgarClient = EdgarClient()
    dcaClient = DCAClient()
    results = {}
    for ticker in selected_tickers:
        try:
            financials = edgarClient.financials(ticker, period)
            dcaClient.runParameters(financials)
            results[ticker] = financials
            st.success(f"Fetched & calculated for {ticker}")
        except Exception as e:
            st.error(f"Error for {ticker}: {e}")
    # Write to Excel
    dcaClient.writeToExcel()
    st.info("Parameters written to Excel file.")

    # Display one example result (for preview)
    if results:
        st.subheader("Sample Financial Data")
        sample_ticker = list(results.keys())[0]
        st.write(f"**{sample_ticker}**")
        st.write(results[sample_ticker])
        st.caption("See console or Excel output for full details.")

st.caption("Developed for Streamlit by adapting main.py")
