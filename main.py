from pprint import pprint
from edgar.client import EdgarClient
from dca.client import DCAClient

# Initialize the Edgar Client & DCA Client
edgarClient = EdgarClient()
dcaClient = DCAClient()

# Setup tickers and period to fetch
tickers = [
    'CBOE',
    'AAPL',
    'WMT',
    'F',
    'HD',
    'SONY',
    'KO',
    'GOOG',
    'MCD',
    'MCO',
    'TSLA',
    'MSFT',
    'ZM',
    'SNAP',
    'DIS',
    'NFLX',
    'GPRO',
    'BABA',
    'CCL',
    'PLTR',
    'AMD',
    'NVDA',
    'SBUX',
    'COIN', # no financial statements found..?
    'T',
]
period = 'annual'

# Loop through each ticker
for ticker in tickers:
    # Fetch financial information
    financials = edgarClient.financials(ticker, period)
    
    # Run parameter using fetched financial information
    dcaClient.runParameters(financials)

# Finally write calculated parameters to an excel document
dcaClient.writeToExcel()