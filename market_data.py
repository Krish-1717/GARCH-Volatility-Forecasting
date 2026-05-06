"""
market_data.py — yfinance wrapper for GARCH volatility forecasting.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def get_returns(ticker: str, period: str = "2y") -> pd.Series:
    """Fetch log-returns for a ticker via yfinance."""
    try:
        import yfinance as yf
        df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        if df.empty:
            return pd.Series(dtype=float)
        close = df["Close"].squeeze()
        returns = np.log(close / close.shift(1)).dropna()
        return returns
    except Exception:
        return pd.Series(dtype=float)


def get_stock_info(ticker: str) -> dict:
    """Return basic stock metadata."""
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info
        return {
            "name": info.get("longName", ticker),
            "sector": info.get("sector", "N/A"),
            "price": info.get("regularMarketPrice") or info.get("currentPrice", None),
            "market_cap": info.get("marketCap", None),
        }
    except Exception:
        return {"name": ticker, "sector": "N/A", "price": None, "market_cap": None}


def realized_volatility(returns: pd.Series, window: int = 21) -> pd.Series:
    """Rolling annualised realised volatility."""
    return returns.rolling(window).std() * np.sqrt(252)
