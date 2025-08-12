import time

class FinancialStatement:

    def __init__(self, ticker: str, companyFacts: dict, period='annual', companyName='') -> None:
        self.ticker = ticker
        self.companyFacts = companyFacts
        self.period = period
        self.companyName = companyName
        self.currency = "USD" # USD by default

        self.aggregateFinancials = None
        self.incomeStatement = None
        self.balanceSheet = None
        self.cashFlow = None

        self.construct()

    def construct(self) -> dict:
        # setup return variable
        self.aggregateFinancials = {
            'ticker': self.ticker,
            'name': self.companyName,
            'financials': []
        }

        # begin building financials
        for form in self.companyFacts['forms']:
            # construct template
            curFinancials = self.constructTemplate()

            # setup concepts
            for concept in curFinancials.keys():
                curFinancials[concept] = self.getConcept(concept, form)

            # append current financials
            self.aggregateFinancials['financials'].append(curFinancials)

        return self.aggregateFinancials

    def getConceptVal(self, rawConcept: str, form: dict, type='int'):
        return None if not self.hasConcept(rawConcept, form) else int(round(float(form[rawConcept]))) if type == 'int' else form[rawConcept]

    def hasConcept(self, rawConcept: str, form: dict) -> bool:
        return rawConcept in form.keys()

    def getConcept(self, concept: str, form: dict):
        if concept == 'EBITDA': # TODO: verify if below is correctly calculated...
            # EBITDA = Net Income + Taxes + Interest Expense + Depreciation & Amortization
            if not self.getConceptVal('OperatingIncomeLoss', form):
                return (self.getConceptVal('NetIncomeLoss', form) or self.getConceptVal('ProfitLoss', form) or 0) + (self.getConceptVal('InterestExpense', form) or 0) + (self.getConceptVal('IncomeTaxExpenseBenefit', form) or 0) + (self.getConceptVal('DepreciationDepletionAndAmortization', form) or self.getConceptVal('DepreciationAmortizationAndAccretionNet', form) or self.getConceptVal('DepreciationAndAmortization', form) or self.getConceptVal('DepreciationAmortizationAndOther', form) or 0)
            else: # EBITDA = Operating Income + Depreciation & Amortization
                return (self.getConceptVal('OperatingIncomeLoss', form) or 0) + (self.getConceptVal('DepreciationDepletionAndAmortization', form) or self.getConceptVal('DepreciationAmortizationAndAccretionNet', form) or self.getConceptVal('DepreciationAndAmortization', form) or self.getConceptVal('DepreciationAmortizationAndOther', form) or 0)
        if concept == 'accountsPayable': return self.getConceptVal('IncreaseDecreaseInAccountsPayable', form) # AccountsPayableCurrent?
        if concept == 'capitalSurplus': return None
        if concept == 'cashChange': return None
        if concept == 'cashFlow': return self.getConceptVal('NetCashProvidedByUsedInOperatingActivities', form)
        if concept == 'cashFlowFinancing': return self.getConceptVal('NetCashProvidedByUsedInFinancingActivities', form)
        if concept == 'changesInInventories': return self.getConceptVal('IncreaseDecreaseInInventories', form)
        if concept == 'changesInReceivables':  return self.getConceptVal('IncreaseDecreaseInAccountsReceivable', form)
        # TODO: commonStock gets 'Common Equity (Total)' from WSJ cash-flow financials sheet... ASK JONNY IF THIS IS THE RIGHT VALUE WE WANT
        # TODO: ask jonny if this is the right value... (below)
        if concept == 'commonStock': return self.getConceptVal('CommonStocksIncludingAdditionalPaidInCapital', form)
        # TODO: verify if this is the right value w/ jonny
        if concept == 'costOfRevenue': return self.getConceptVal('CostOfGoodsAndServicesSold', form)
        if concept == 'currency': return self.currency
        if concept == 'currentAssets': return self.getConceptVal('AssetsCurrent', form)
        # TODO: ask jonny if we want 'Cash' or 'Cash & Short Term Investments' from WSJ balance-sheet... etc..
        if concept == 'currentCash': return self.getConceptVal('CashAndCashEquivalentsAtCarryingValue', form)
        # TODO: ask jonny what this value is... short term or long term debt? bc next value asks for long term debt...
        if concept == 'currentDebt': return self.getConceptVal('DebtCurrent', form)
        if concept == 'currentLongTermDebt': return self.getConceptVal('LongTermDebtCurrent', form)
        if concept == 'currentLiabilities': return self.getConceptVal('LiabilitiesCurrent', form)
        if concept == 'depreciation': return self.getConceptVal('Depreciation', form)
        # TODO: check w/ jonny to see if this is the right value
        if concept == 'dividendsPaid': return self.getConceptVal('PaymentsOfDividendsCommonStock', form)
        # TODO: implement EBIT
        if concept == 'ebit': return None
        if concept == 'exchangeRateEffect': return None
        if concept == 'filingType': return self.getConceptVal('DocumentType', form, type='str')
        if concept == 'fiscalDate': return self.getConceptVal('DocumentPeriodEndDate', form, type='str')
        if concept == 'fiscalQuarter': return self.getConceptVal('DocumentFiscalPeriodFocus', form, type='str')
        if concept == 'fiscalYear': return self.getConceptVal('DocumentFiscalYearFocus', form)
        if concept == 'goodwill': return self.getConceptVal('Goodwill', form)
        if concept == 'grossProfit': return self.getConceptVal('GrossProfit', form)
        if concept == 'incomeTax': return self.getConceptVal('IncomeTaxExpenseBenefit', form)
        if concept == 'intangibleAssets': return None
        if concept == 'inventory': return self.getConceptVal('InventoryNet', form)
        # TODO: check interestIncome variable w/ jonny...
        if concept == 'interestExpense': return self.getConceptVal('InterestExpense', form)
        if concept == 'interestIncome': return self.getConceptVal('InterestPaidNet', form)
        # TODO: check if this variable uses PaymentsForProceedsFromOtherInvestingActivities or NetCashProvidedByUsedInInvestingActivities
        if concept == 'investingActivityOther': return self.getConceptVal('PaymentsForProceedsFromOtherInvestingActivities', form)
        # TODO: verify if PaymentsToAcquireOtherInvestments is the right raw concept
        if concept == 'investments': return self.getConceptVal('PaymentsToAcquireOtherInvestments', form)
        # TODO: figure out how to get CORRECT long term debt from MSFT... LongTermDebtAndCapitalLeaseObligations / LongTermDebt???
        if concept == 'longTermDebt': return self.getConceptVal('LongTermDebt', form) or self.getConceptVal('LongTermDebtFairValue', form) or self.getConceptVal('LongTermDebtNoncurrent', form)
        # TODO: this is probs wrong so check this variable...
        if concept == 'longTermInvestments': return self.getConceptVal('MarketableSecuritiesNoncurrent', form)
        if concept == 'minorityInterest': return None
        if concept == 'netBorrowings': return None
        if concept == 'netIncome': return self.getConceptVal('NetIncomeLoss', form) or self.getConceptVal('ProfitLoss', form)
        if concept == 'netIncomeBasic': return None
        if concept == 'netTangibleAssets': return None
        if concept == 'operatingExpense': return self.getConceptVal('OperatingExpenses', form)
        if concept == 'operatingIncome': return self.getConceptVal('OperatingIncomeLoss', form)
        if concept == 'operatingRevenue': return None
        # TODO: ask jonny if OtherAssetsNoncurrent is the right raw concept name?
        if concept == 'otherAssets': return self.getConceptVal('OtherAssetsNoncurrent', form)
        if concept == 'otherCurrentAssets': return self.getConceptVal('OtherAssetsCurrent', form)
        if concept == 'otherCurrentLiabilities': return None
        if concept == 'otherIncomeExpenseNet': return None
        # TODO: check if this variable is correct
        if concept == 'otherLiabilities': return self.getConceptVal('OtherLiabilitiesNoncurrent', form)
        if concept == 'pretaxIncome': return self.getConceptVal('IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest', form)
        # NET = 'Property, Plant, & Equipment Gross' - 'Property, Plant, & Equipment Accumulated Depreciation'
        if concept == 'propertyPlantEquipment': return self.getConceptVal('PropertyPlantAndEquipmentNet', form) or self.getConceptVal('PropertyPlantandEquipmentandFinanceLeaseRightofUseAssetafterAccumulatedDepreciationandAmortization', form) or self.getConceptVal('PropertyPlantAndEquipmentAndFinanceLeaseRightOfUseAssetAfterAccumulatedDepreciationAndAmortization', form)
        if concept == 'receivables': return self.getConceptVal('AccountsReceivableNetCurrent', form)
        if concept == 'reportDate': return self.getConceptVal('DocumentPeriodEndDate', form, type='str')
        if concept == 'researchAndDevelopment': return self.getConceptVal('ResearchAndDevelopmentExpense', form, type='str')
        if concept == 'retainedEarnings': return self.getConceptVal('RetainedEarningsAccumulatedDeficit', form)
        # TODO: find the difference between 'revenue' and 'totalRevenue'
        if concept == 'revenue': return self.getConceptVal('SalesRevenueNet', form)
        if concept == 'sellingGeneralAndAdmin': return self.getConceptVal('SellingGeneralAndAdministrativeExpense', form)
        if concept == 'shareholderEquity': return self.getConceptVal('StockholdersEquity', form)
        if concept == 'shortTermDebt': return None
        if concept == 'shortTermInvestments': return None
        if concept == 'symbol': return self.ticker
        if concept == 'totalAssets': return self.getConceptVal('Assets', form)
        # TODO: make sure this variable is correct
        if concept == 'totalCash': return self.getConceptVal('Cash', form)
        if concept == 'totalDebt': return None
        if concept == 'totalInvestingCashFlows': return self.getConceptVal('NetCashProvidedByUsedInInvestingActivities', form)
        if concept == 'totalLiabilities': return self.getConceptVal('Liabilities', form)
        if concept == 'totalRevenue': return self.getConceptVal('SalesRevenueNet', form)
        if concept == 'treasuryStock': return self.getConceptVal('TreasuryStockCommonValue', form)
        if concept == 'id': return 'FINANCIALS'
        if concept == 'key': return self.ticker
        if concept == 'subkey': return self.period
        if concept == 'updated': return int(round(time.time()))
    
        # base case
        return None

    def constructTemplate(self) -> dict:
        return {
            "EBITDA": None,
            "accountsPayable": None,
            "capitalSurplus": None,
            "cashChange": None,
            "cashFlow": None,
            "cashFlowFinancing": None,
            "changesInInventories": None,
            "changesInReceivables": None,
            "commonStock": None,
            "costOfRevenue": None,
            "currency": None,
            "currentAssets": None,
            "currentCash": None,
            "currentDebt": None,
            "currentLongTermDebt": None,
            "currentLiabilities": None,
            "depreciation": None,
            "dividendsPaid": None,
            "ebit": None,
            "exchangeRateEffect": None,
            "filingType": None,
            "fiscalDate": None,
            "fiscalQuarter": None,
            "fiscalYear": None,
            "goodwill": None,
            "grossProfit": None,
            "incomeTax": None,
            "intangibleAssets": None,
            "inventory": None,
            "interestExpense": None,
            "interestIncome": None,
            "investingActivityOther": None,
            "investments": None,
            "longTermDebt": None,
            "longTermInvestments": None,
            "minorityInterest": None,
            "netBorrowings": None,
            "netIncome": None,
            "netIncomeBasic": None,
            "netTangibleAssets": None,
            "operatingExpense": None,
            "operatingIncome": None,
            "operatingRevenue": None,
            "otherAssets": None,
            "otherCurrentAssets": None,
            "otherCurrentLiabilities": None,
            "otherIncomeExpenseNet": None,
            "otherLiabilities": None,
            "pretaxIncome": None,
            "propertyPlantEquipment": None,
            "receivables": None,
            "reportDate": None,
            "researchAndDevelopment": None,
            "retainedEarnings": None,
            "revenue": None,
            "sellingGeneralAndAdmin": None,
            "shareholderEquity": None,
            "shortTermDebt": None,
            "shortTermInvestments": None,
            "symbol": None,
            "totalAssets": None,
            "totalCash": None,
            "totalDebt": None,
            "totalInvestingCashFlows": None,
            "totalLiabilities": None,
            "totalRevenue": None,
            "treasuryStock": None,
            "id": None,
            "key": None,
            "subkey": None,
            "updated": None
        }

    def get(self, asReported=False) -> dict:
        # TODO: add functionality for asReported
        return self if asReported else self.aggregateFinancials if self.aggregateFinancials != None else self.incomeStatement if self.incomeStatement != None else self.balanceSheet if self.balanceSheet != None else self.cashFlow if self.cashFlow != None else Non