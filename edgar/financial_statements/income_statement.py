from edgar.financial_statements.financial_statement import FinancialStatement

class IncomeStatement(FinancialStatement):

    def __init__(self, ticker: str, companyFacts: dict, period='annual') -> None:
        super().__init__(ticker=ticker, companyFacts=companyFacts, period=period)
    
    def construct(self, period='annual') -> dict:
        self.incomeStatement = {
            'ticker': self.ticker,
            'incomeStatements': []
        }

        return self.incomeStatement