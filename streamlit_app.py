import streamlit as st
from edgar.client import EdgarClient
from dca.client import DCAClient

# Initialize the Edgar Client & DCA Client
edgarClient = EdgarClient()
dcaClient = DCAClient()

# Setup tickers and period to fetch
tickers = [
    'CBOE', 'AAPL', 'WMT', 'F', 'HD', 'SONY', 'KO', 'GOOG', 'MCD',
    'MCO', 'TSLA', 'MSFT', 'ZM', 'SNAP', 'DIS', 'NFLX', 'GPRO', 'BABA',
    'CCL', 'PLTR', 'AMD', 'NVDA', 'SBUX', 'COIN', 'T',
]
period = 'annual'

st.title("Warren-DCA Results")

results = []
for ticker in tickers:
    financials = edgarClient.financials(ticker, period)
    dca_results = dcaClient.runParameters(financials)
    results.append({"ticker": ticker, "result": dca_results})
    st.write(f"✅ {ticker} processed.")

# ตัวอย่างการแสดงผลรวมเป็นตาราง (ถ้าผลลัพธ์เหมาะสมกับตาราง)
st.write("## สรุปผลลัพธ์ทั้งหมด")
st.write(results)

# สุดท้ายยังคงบันทึกลง Excel
dcaClient.writeToExcel()
