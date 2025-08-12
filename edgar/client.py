from edgar.financials import Financials

class EdgarClient():

    def financials(self, ticker: str, period='annual') -> Financials:
        return Financials(ticker, period)