from edgar.financials import Financials
from dca.excel import Excel
from dca.parameters import Parameters

class DCAClient():

    def __init__(self) -> None:
        self.parametersResults = {}

    def runParameters(self, financials: Financials) -> None:
        self.financials = financials
        self.parameters = Parameters(financials)
        self.parametersResults[financials.ticker] = self.parameters.getResults()
    
    def getParametersResult(self, ticker=None) -> dict:
        return self.parametersResults[ticker or self.financials.ticker]

    def writeToExcel(self) -> None:
        Excel(self.parametersResults).write()