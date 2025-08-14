# external_optimizer.py
import pandas as pd
import numpy as np

class PortfolioOptimizer:
    """
    Optimizer แบบง่าย: คำนวณ CAGR + ความผันผวน (StdDev รายเดือน) แล้วจัดสรรงบตาม objective
    objective:
      - maximize_return : น้ำหนักตามสัดส่วน CAGR+
      - minimize_risk   : น้ำหนักผกผันความผันผวน
      - balanced        : ถัวเฉลี่ย Normalize(CAGR) และ Inverse Vol
    """
    def __init__(self, prices_map: dict, dividends_map: dict):
        self.prices_map = prices_map
        self.dividends_map = dividends_map

    def _monthly_returns(self, close_series: pd.Series):
        m = close_series.resample('ME').last().dropna()
        return m.pct_change().dropna()

    def _cagr(self, close_series: pd.Series):
        if len(close_series) < 2:
            return 0.0
        start = close_series.iloc[0]
        end = close_series.iloc[-1]
        days = (close_series.index[-1] - close_series.index[0]).days
        if days <= 0 or start <= 0:
            return 0.0
        years = days / 365.25
        return (end / start) ** (1 / years) - 1 if years > 0 else 0.0

    def _dividend_yield_trailing(self, ticker, close_series: pd.Series):
        div_series = self.dividends_map.get(ticker)
        if div_series is None or len(div_series) == 0:
            return 0.0
        last_year = close_series.index[-1] - pd.DateOffset(years=1)
        recent_div = div_series[div_series.index >= last_year]
        if recent_div.empty:
            return 0.0
        avg_price = close_series[close_series.index >= last_year].mean()
        return (recent_div.sum() / avg_price) if avg_price and avg_price > 0 else 0.0

    def optimize(self, total_budget: float, objective: str = "balanced"):
        metrics = []
        for tk, df in self.prices_map.items():
            if "Close" not in df.columns:
                continue
            close = df["Close"].dropna()
            if close.empty:
                continue

            cagr = self._cagr(close)
            mret = self._monthly_returns(close)
            vol = mret.std() * np.sqrt(12) if len(mret) > 0 else 0.0
            dy = self._dividend_yield_trailing(tk, close)

            metrics.append({
                "ticker": tk,
                "cagr": cagr,
                "vol": vol if vol > 0 else 1e-9,
                "div_yield": dy
            })

        if not metrics:
            return {
                "allocations": {},
                "expected_return": 0.0,
                "expected_yield": 0.0,
                "note": "ไม่มีข้อมูลเพียงพอ"
            }

        dfm = pd.DataFrame(metrics)

        # Normalize factors
        def nz_norm(col):
            x = dfm[col].values
            if np.allclose(x.max(), x.min()):
                return np.ones_like(x) / len(x)
            return (x - x.min()) / (x.max() - x.min())

        ret_norm = nz_norm("cagr")
        inv_vol_norm = nz_norm(1 / dfm["vol"])
        dy_norm = nz_norm("div_yield")

        if objective == "maximize_return":
            weight_raw = ret_norm
        elif objective == "minimize_risk":
            weight_raw = inv_vol_norm
        elif objective == "income":
            weight_raw = dy_norm
        else:  # balanced
            weight_raw = (ret_norm + inv_vol_norm + dy_norm) / 3.0

        if weight_raw.sum() == 0:
            weights = np.ones_like(weight_raw) / len(weight_raw)
        else:
            weights = weight_raw / weight_raw.sum()

        dfm["weight"] = weights
        dfm["allocation"] = dfm["weight"] * total_budget

        expected_return = float((dfm["cagr"] * dfm["weight"]).sum())
        expected_yield = float((dfm["div_yield"] * dfm["weight"]).sum())

        return {
            "allocations": {
                r.ticker: {
                    "allocation": float(r.allocation),
                    "weight": float(r.weight),
                    "cagr": float(r.cagr),
                    "vol": float(r.vol),
                    "div_yield": float(r.div_yield)
                } for _, r in dfm.iterrows()
            },
            "expected_return": expected_return,
            "expected_yield": expected_yield,
            "objective": objective,
        }