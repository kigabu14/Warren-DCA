from future import annotations

import osimport picklefrom datetime 
import datetime, timedeltafrom pathlib 
import Pathfrom typing 
import Dict, List, Tuple, Optional

import numpy as npimport pandas as pdimport yfinance as yf

class DCADataLoader:
  """Data loader for DCA (Dollar-Cost Averaging) analysis with caching support.

Responsibilities:
  - ดึงข้อมูลราคาหุ้น / กองทุน จาก yfinance
  - ดึงข้อมูลเงินปันผล (dividends)
  - เก็บ cache ลงไฟล์ .pkl เพื่อลดจำนวนการเรียก network ซ้ำ
  - คืนข้อมูลที่พร้อมใช้ในงานจำลองกลยุทธ์ DCA
"""

def __init__(self, cache_dir: str = "data/cache") -> None:
    """Initialize the loader and ensure cache directory exists."""
    self.cache_dir = Path(cache_dir)
    try:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:  # pragma: no cover (rare filesystem issue)
        print(f"Error: Failed to create cache directory '{self.cache_dir}': {e}")
        raise

# -------------------------------
# Internal cache helpers
# -------------------------------
def _get_cache_path(self, ticker: str, period: str) -> Path:
    """Return the cache file path for a given ticker & period."""
    safe_ticker = ticker.replace("/", "-")
    return self.cache_dir / f"{safe_ticker}_{period}_data.pkl"

def _is_cache_valid(self, cache_path: Path, max_age_hours: int = 1) -> bool:
    """Check whether a cache file exists and is younger than max_age_hours."""
    if not cache_path.exists():
        return False
    file_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
    return file_age < timedelta(hours=max_age_hours)

# -------------------------------
# Fetching data
# -------------------------------
def fetch_ticker_data(self, ticker: str, period: str = "5y", max_cache_age_hrs: int = 1) -> Dict:
    """Fetch comprehensive data for a single ticker.

    Returns a dict containing:
        ticker, company_name, sector, currency, period,
        historical_prices (DataFrame), dividends (Series), info (dict),
        fetched_at (datetime), data_start (Timestamp), data_end (Timestamp)
    """
    cache_path = self._get_cache_path(ticker, period)

    # Try cache first
    if self._is_cache_valid(cache_path, max_age_hours=max_cache_age_hrs):
        try:
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass  # Ignore corrupted cache

    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        if hist.empty:
            raise ValueError(f"No historical data found for {ticker}")

        dividends = stock.dividends
        if not dividends.empty:
            start_date = hist.index[0]
            dividends = dividends[dividends.index >= start_date]

        try:  # some tickers might fail on .info
            info = stock.info or {}
            company_name = info.get("longName", ticker)
            sector = info.get("sector", "Unknown")
            currency = info.get("currency", "USD")
        except Exception:
            info = {}
            company_name = ticker
            sector = "Unknown"
            currency = "USD"

        data: Dict[str, object] = {
            "ticker": ticker,
            "company_name": company_name,
            "sector": sector,
            "currency": currency,
            "period": period,
            "historical_prices": hist,
            "dividends": dividends,
            "info": info,
            "fetched_at": datetime.now(),
            "data_start": hist.index[0] if not hist.empty else None,
            "data_end": hist.index[-1] if not hist.empty else None,
        }

        # Cache write (best effort)
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(data, f)
        except Exception:
            pass

        return data
    except Exception as e:
        raise ValueError(f"Failed to fetch data for {ticker}: {e}") from e

def fetch_multiple_tickers(self, tickers: List[str], period: str = "5y", max_cache_age_hrs: int = 1) -> Dict[str, Dict]:
    """Fetch data for multiple tickers; returns mapping ticker -> data dict."""
    results: Dict[str, Dict] = {}
    errors: Dict[str, str] = {}
    for t in tickers:
        try:
            results[t] = self.fetch_ticker_data(t, period, max_cache_age_hrs=max_cache_age_hrs)
        except Exception as e:
            errors[t] = str(e)
            print(f"Warning: Failed to fetch {t}: {e}")
    if errors:
        print(f"Failed to fetch data for {len(errors)} tickers: {list(errors.keys())}")
    return results

# -------------------------------
# Data extraction for DCA simulation
# -------------------------------
def get_price_data_for_dca(
    self,
    ticker_data: Dict,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
) -> pd.DataFrame:
    """Return cleaned OHLCV slice prepared for DCA simulation."""
    hist: pd.DataFrame = ticker_data["historical_prices"].copy()
    if hist.empty:
        raise ValueError("No historical price data available")
    if start_date:
        hist = hist[hist.index >= start_date]
    if end_date:
        hist = hist[hist.index <= end_date]
    if hist.empty:
        raise ValueError("No data available in specified date range")

    dca_df = (
        pd.DataFrame(
            {
                "Date": hist.index,
                "Open": hist["Open"],
                "High": hist["High"],
                "Low": hist["Low"],
                "Close": hist["Close"],
                "Volume": hist["Volume"],
            }
        )
        .dropna(subset=["Close"]).reset_index(drop=True)
    )
    return dca_df

def get_dividend_data_for_dca(
    self,
    ticker_data: Dict,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
) -> pd.DataFrame:
    """Return dividend data (Date, Dividend) filtered by optional range."""
    dividends: pd.Series = ticker_data["dividends"].copy()
    if dividends.empty:
        return pd.DataFrame(columns=["Date", "Dividend"])
    if start_date:
        dividends = dividends[dividends.index >= start_date]
    if end_date:
        dividends = dividends[dividends.index <= end_date]
    return pd.DataFrame({"Date": dividends.index, "Dividend": dividends.values})

# -------------------------------
# Validation utilities
# -------------------------------
def validate_ticker_list(self, tickers: List[str]) -> Tuple[List[str], List[str]]:
    """Validate tickers by requesting a short 5d history."""
    valid: List[str] = []
    invalid: List[str] = []
    for t in tickers:
        try:
            hist = yf.Ticker(t).history(period="5d")
            if not hist.empty:
                valid.append(t)
            else:
                invalid.append(t)
        except Exception:
            invalid.append(t)
    return valid, invalid

# -------------------------------
# Cache management
# -------------------------------
def clear_cache(self, ticker: Optional[str] = None) -> None:
    """Clear all cache files or only those for a specific ticker."""
    if ticker:
        for cache_file in self.cache_dir.glob(f"{ticker}_*.pkl"):
            try:
                cache_file.unlink()
            except FileNotFoundError:
                pass
    else:
        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                cache_file.unlink()
            except FileNotFoundError:
                pass

def get_cache_info(self) -> Dict:
    """Return metadata about current cache contents."""
    cache_files = list(self.cache_dir.glob("*.pkl"))
    info: Dict[str, object] = {
        "total_files": len(cache_files),
        "total_size_mb": sum(f.stat().st_size for f in cache_files) / (1024 * 1024) if cache_files else 0.0,
        "files": [],
    }
    now = datetime.now()
    for f in cache_files:
        try:
            modified = datetime.fromtimestamp(f.stat().st_mtime)
            info["files"].append(
                {
                    "name": f.name,
                    "size_kb": f.stat().st_size / 1024,
                    "modified": modified,
                    "age_hours": (now - modified).total_seconds() / 3600,
                }
            )
        except Exception:
            continue
    return info

# -------------------------------
# Backwards compatibility
# -------------------------------
def fetch(self, ticker: str, period: str = "5y"):
    """Alias kept for backward compatibility with older code paths."""
    return self.fetch_ticker_data(ticker, period)