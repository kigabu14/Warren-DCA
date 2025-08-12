def __init__(self, cache_dir: str = "data/cache"):
    """Initialize data loader with cache directory"""
    self.cache_dir = Path(cache_dir)
    try:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Error: Failed to create cache directory '{self.cache_dir}': {e}")
        raise
    
def _get_cache_path(self, ticker: str, period: str) -> Path:
    """Get cache file path for ticker and period"""
    return self.cache_dir / f"{ticker}_{period}_data.pkl"

def _is_cache_valid(self, cache_path: Path, max_age_hours: int = 1) -> bool:
    """Check if cache file is valid (exists and not too old)"""
    if not cache_path.exists():
        return False
    file_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
    return file_age < timedelta(hours=max_age_hours)

def fetch_ticker_data(self, ticker: str, period: str = "5y") -> Dict:
    """
    Fetch comprehensive data for a single ticker
    Returns dict with: historical_prices, dividends, info, metadata
    """
    cache_path = self._get_cache_path(ticker, period)
    
    # Try cache
    if self._is_cache_valid(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except:
            pass  # Ignore cache read failure
    
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        if hist.empty:
            raise ValueError(f"No historical data found for {ticker}")
        
        dividends = stock.dividends
        if not dividends.empty:
            start_date = hist.index[0]
            dividends = dividends[dividends.index >= start_date]
        
        try:
            info = stock.info
            company_name = info.get('longName', ticker)
            sector = info.get('sector', 'Unknown')
            currency = info.get('currency', 'USD')
        except:
            company_name = ticker
            sector = 'Unknown'
            currency = 'USD'
            info = {}
        
        data = {
            'ticker': ticker,
            'company_name': company_name,
            'sector': sector,
            'currency': currency,
            'period': period,
            'historical_prices': hist,
            'dividends': dividends,
            'info': info,
            'fetched_at': datetime.now(),
            'data_start': hist.index[0] if not hist.empty else None,
            'data_end': hist.index[-1] if not hist.empty else None
        }
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
        except:
            pass  # Ignore cache write errors
        
        return data
    
    except Exception as e:
        raise ValueError(f"Failed to fetch data for {ticker}: {str(e)}")

def fetch_multiple_tickers(self, tickers: List[str], period: str = "5y") -> Dict[str, Dict]:
    """Fetch data for a list of tickers; returns mapping ticker -> data dict"""
    results = {}
    errors = {}
    for ticker in tickers:
        try:
            results[ticker] = self.fetch_ticker_data(ticker, period)
        except Exception as e:
            errors[ticker] = str(e)
            print(f"Warning: Failed to fetch {ticker}: {e}")
    if errors:
        print(f"Failed to fetch data for {len(errors)} tickers: {list(errors.keys())}")
    return results

def get_price_data_for_dca(self, ticker_data: Dict,
                           start_date: Optional[datetime] = None,
                           end_date: Optional[datetime] = None) -> pd.DataFrame:
    """Return cleaned OHLCV slice for DCA simulation"""
    hist = ticker_data['historical_prices'].copy()
    if hist.empty:
        raise ValueError("No historical price data available")
    if start_date:
        hist = hist[hist.index >= start_date]
    if end_date:
        hist = hist[hist.index <= end_date]
    if hist.empty:
        raise ValueError("No data available in specified date range")
    dca_data = pd.DataFrame({
        'Date': hist.index,
        'Close': hist['Close'],
        'Volume': hist['Volume'],
        'High': hist['High'],
        'Low': hist['Low'],
        'Open': hist['Open']
    }).dropna(subset=['Close'])
    return dca_data

def get_dividend_data_for_dca(self, ticker_data: Dict,
                              start_date: Optional[datetime] = None,
                              end_date: Optional[datetime] = None) -> pd.DataFrame:
    """Return dividend data frame with Date, Dividend"""
    dividends = ticker_data['dividends'].copy()
    if dividends.empty:
        return pd.DataFrame(columns=['Date', 'Dividend'])
    if start_date:
        dividends = dividends[dividends.index >= start_date]
    if end_date:
        dividends = dividends[dividends.index <= end_date]
    return pd.DataFrame({'Date': dividends.index, 'Dividend': dividends.values})

def validate_ticker_list(self, tickers: List[str]) -> Tuple[List[str], List[str]]:
    """Quick validation using a short history request"""
    valid, invalid = [], []
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="5d")
            if not hist.empty:
                valid.append(ticker)
            else:
                invalid.append(ticker)
        except:
            invalid.append(ticker)
    return valid, invalid

def clear_cache(self, ticker: Optional[str] = None):
    """Clear cache (all or specific ticker)"""
    if ticker:
        for cache_file in self.cache_dir.glob(f"{ticker}_*.pkl"):
            cache_file.unlink()
    else:
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()

def get_cache_info(self) -> Dict:
    """Return metadata about cache contents"""
    cache_files = list(self.cache_dir.glob("*.pkl"))
    info = {
        'total_files': len(cache_files),
        'total_size_mb': sum(f.stat().st_size for f in cache_files) / (1024 * 1024),
        'files': []
    }
    for cache_file in cache_files:
        info['files'].append({
            'name': cache_file.name,
            'size_kb': cache_file.stat().st_size / 1024,
            'modified': datetime.fromtimestamp(cache_file.stat().st_mtime),
            'age_hours': (datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)).total_seconds() / 3600
        })
    return info

def fetch(self, ticker: str, period: str = "5y"):
    """
    Backward-compatible alias for external optimizer code that expects a .fetch() method.
    Delegates to fetch_ticker_data.
    """
    return self.fetch_ticker_data(ticker, period)