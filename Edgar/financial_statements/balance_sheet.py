from edgar.financial_statements.financial_statement import FinancialStatement

class BalanceSheet(FinancialStatement):

    def __init__(self, ticker: str, companyFacts: dict, period='annual') -> None:
        super().__init__(ticker=ticker, companyFacts=companyFacts, period=period)
    
    def construct(self, period='annual') -> dict:
        self.balanceSheet = {
            'ticker': self.ticker,
            'name': self.companyFacts['entityName'],
            'balanceSheets': [
                {
                    # "reportDate": "2020-10-17",
                    # "filingType": "10-K",
                    # "fiscalDate": "2020-09-13",
                    # "fiscalQuarter": 4,
                    # "fiscalYear": 2010,
                    # "currency": "USD",
                    # "currentCash": 25913000000,
                    # "shortTermInvestments": null,
                    # "receivables": 23186000000,
                    # "inventory": 3956000000,
                    # "otherCurrentAssets": 12087000000,
                    # "currentAssets": 131339000000,
                    # "longTermInvestments": 170799000000,
                    # "propertyPlantEquipment": 41304000000,
                    # "goodwill": null,
                    # "intangibleAssets": null,
                    # "otherAssets": 22283000000,
                    # "totalAssets": 365725000000,
                    # "accountsPayable": 55888000000,
                    # "currentLongTermDebt": null,
                    # "otherCurrentLiabilities": null,
                    # "totalCurrentLiabilities": 116866000000,
                    # "longTermDebt": 93735000000,
                    # "otherLiabilities": null,
                    # "minorityInterest": 0,
                    # "totalLiabilities": 258578000000,
                    # "commonStock": 40201000000,
                    # "retainedEarnings": 70400000000,
                    # "treasuryStock": null,
                    # "capitalSurplus": null,
                    # "shareholderEquity": 107147000000,
                    # "netTangibleAssets": 107147000000,
                    # "id": "BALANCE_SHEET",
                    # "key": "AAPL",
                    # "subkey": "quarterly",
                    # "date": 1635273127391,
                    # "updated": 1635273127391
                }
            ]
        }

        # use the entity public float to get a list of balance sheets
        # entityPublicFloat = self.companyFacts['facts']['dei']['EntityPublicFloat']['units']['USD']
        # print(entityPublicFloat)
        # self.balanceSheet.currentCash = self.companyFacts['facts']['us-gaap']['Cash']

        # TODO finish constructing income statement
        # 'goodwill': self.companyfacts['facts']['us-gaap']['Goodwill']

        return self.balanceSheet