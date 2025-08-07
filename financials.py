from pydoc import doc
import re
import xml.etree.ElementTree as ET
import requests
from edgar.financial_statements import financial_statement as fS
from edgar.financial_statements import income_statement as iS
from edgar.financial_statements import balance_sheet as bS
from edgar.financial_statements import cash_flow as cF

HEADERS = { 'User-Agent': 'Sample Company Name AdminContact@<sample company domain>.com' }

class Financials():

    def __init__(self, ticker: str, period='annual') -> None:
        self.period = period

        # convert ticker to cik
        self.ticker = ticker.upper()
        self.setupCIK()

        # validate cik (adds leading zeros)
        self.cik = self.validateCIK(self.cik)

        # setup base api url & endpoint
        self.api_resource = 'https://data.sec.gov'
        self.endpoint = f'/submissions/CIK{self.cik}.json'

        # print url of data were processing
        print(f'~ Now Fetching Filings for {self.ticker} ({self.companyName}): {self.api_resource + self.endpoint}')

        # request the company facts and decode it
        self.companyFacts = self.fetchCompanyFacts()

        # construct financial statements from company facts
        self.constructFinancials()

    def fetchCompanyFacts(self) -> dict:
        # fetch submissions (list of filings) from CIK
        submissions = requests.get(self.api_resource + self.endpoint, headers=HEADERS).json()
        
        # loop through forms and get all forms based on period
        formIndexes = []
        for i, form in enumerate(submissions['filings']['recent']['form']):
            # 10-K: domestic companies, 20-F: foreign companies
            if form == ('10-K' if self.period == 'annual' else '10-Q') or form == ('20-F' if self.period == 'annual' else '10-Q'):
                formIndexes.append(i)
        
        # setup empty company facts obj
        companyFacts = {
            'ticker': self.ticker,
            'forms': []
        }

        # TODO: remove this in post... THIS IS ONLY SO WE GET THE MOST RECENT PERIODS'S FILING for quick testing...
        # limit to only get last 5 filings
        formIndexes = formIndexes[:5]
        # formIndexes = formIndexes[:2]

        # get company facts from each form
        for formIndex in formIndexes:
            accession_number = submissions['filings']['recent']['accessionNumber'][formIndex].replace('-', '')
            document_name = submissions['filings']['recent']['primaryDocument'][formIndex].replace('.htm', '_htm.xml')
            form_data_url = f"https://www.sec.gov/Archives/edgar/data/{self.cik}/{accession_number}/{document_name}"

            # xml read using element tree
            root = ET.fromstring(requests.get(form_data_url, headers=HEADERS).text)

            # if error, then use older filing format
            if root.tag == 'Error':
                document_name = submissions['filings']['recent']['primaryDocument'][formIndex].replace('.htm', '.xml').replace('10k_' if self.period == 'annual' else '10q_', '')
                form_data_url = f"https://www.sec.gov/Archives/edgar/data/{self.cik}/{accession_number}/{document_name}"
                root = ET.fromstring(requests.get(form_data_url, headers=HEADERS).text)
                # if error, then use alternate older filing format
                if root.tag == 'Error':
                    # ex: convert wmtform10-kx1312017.xml to wmt-20170131.xml
                    document_name = f"{self.ticker.lower()}-{submissions['filings']['recent']['reportDate'][formIndex].replace('-', '')}.xml"
                    form_data_url = f"https://www.sec.gov/Archives/edgar/data/{self.cik}/{accession_number}/{document_name}"
                    root = ET.fromstring(requests.get(form_data_url, headers=HEADERS).text)
                    if root.tag == 'Error':
                        # TODO: figure out how to get foreign 20-F for SONY when cannot fetch filing
                        print(f"> Error Fetching Filing for {self.ticker} ({self.companyName}): reported {submissions['filings']['recent']['reportDate'][formIndex]}")
                        continue

            # display form data url
            print(f'Fetching Data from Filing: {form_data_url}')

            # get report year to validate fetched data
            reportYear = submissions['filings']['recent']['reportDate'][formIndex].split('-')[0]

            # setup company facts
            formCompanyFacts = {}
            for child in root.findall('./'):
                match = re.search('{.*}', child.tag)
                xmlns = match.group(0) if match else ''
    
                # replace xml ns with nothing to get tag (company fact) name
                companyFact = child.tag.replace(xmlns, '')
                if companyFact not in formCompanyFacts.keys():
                    # validate data is from current fiscal year
                    # TODO: replace report year with DocumentFiscalYearFocus (fiscal year)
                    if 'contextRef' in child.attrib.keys() and reportYear in child.attrib['contextRef']:
                        formCompanyFacts[companyFact] = child.text

            companyFacts['forms'].append(formCompanyFacts)

        return companyFacts

    def setupCIK(self) -> None:
        # sec provided list of tickers : cik
        self.ticker_resource = 'https://www.sec.gov/files/company_tickers.json'
        
        # request tickers if not already fetched
        if not hasattr(self, 'companyTickers'): # TODO: save this ticker_resource to local file in case SEC api is ever unavailable (as what happened today 1/31/2022 @ 2:58PM)
            self.companyTickers = requests.get(self.ticker_resource, headers=HEADERS).json()

        # get cik and company name
        for index in self.companyTickers.keys():
            if self.companyTickers[index]['ticker'] == self.ticker:
                self.cik = str(self.companyTickers[index]['cik_str'])
                self.companyName = self.companyTickers[index]['title']
                return
    
    def validateCIK(self, cik: str) -> str:
        # add leading zeros to cik if missing
        if len(cik) < 10:
            num_of_zeros = 10 - len(cik)
            cik = num_of_zeros * "0" + cik
        return cik
    
    def constructFinancials(self) -> None:
        # setup generic aggregate financials dict for easy access to all financial variables
        self.aggregateFinancials = fS.FinancialStatement(self.ticker, self.companyFacts, self.period, self.companyName).aggregateFinancials

        # TODO: construct statements
        # self.constructIncomeStatement()
        # self.constructBalanceSheet()
        # self.constructCashFlow()

    def constructIncomeStatement(self) -> None:
        self.incomeStatement = iS.IncomeStatement(self.ticker, self.companyFacts, self.period)

    def constructBalanceSheet(self) -> None:
        self.balanceSheet =  bS.BalanceSheet(self.ticker, self.companyFacts, self.period)

    def constructCashFlow(self) -> None:
        self.cashFlow =  cF.CashFlow(self.ticker, self.companyFacts, self.period)

    def getFinancials(self, asReported=False) -> dict:
        # TODO: add asReported support to display all raw concept names/values
        return self.aggregateFinancials

    def getIncomeStatement(self, asReported=False) -> dict | iS.IncomeStatement:
        return self.incomeStatement.get(asReported)

    def getBalanceSheet(self, asReported=False) -> dict | bS.BalanceSheet:
        return self.balanceSheet.get(asReported)

    def getCashFlow(self, asReported=False) -> dict | cF.CashFlow:
        return self.cashFlow.get(asReported)