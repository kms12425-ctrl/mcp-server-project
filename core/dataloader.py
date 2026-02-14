"""Tools: StockDataLoader with MinMax scaling, sequence serialization, caching and retry/backoff.

This module is intended for processing data (not resource registration).
"""
import logging
import os
import pickle
import time
import random
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler


from curl_cffi import requests


logger = logging.getLogger(__name__)


class StockDataLoader:
    """Download, normalize (MinMax) and serialize time-series into (X, y).

    Parameters
    ----------
    ticker: stock ticker symbol
    start, end: optional date strings
    period, interval: passed to yfinance
    sequence_length: number of past steps used to predict next step
    feature: column to use (default 'Close')
    cache_dir: directory to store cached pickles
    cache_ttl_seconds: cache lifetime in seconds
    max_retries, backoff_base: retry/backoff parameters
    """

    def __init__(
        self,
        ticker: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        period: str = "1y",
        interval: str = "1d",
        sequence_length: int = 30,
        feature: str = "Close",
        cache_dir: Optional[str] = None,
        cache_ttl_seconds: int = 24 * 3600,
        max_retries: int = 4,
        backoff_base: float = 1.0,
    ):
        self.ticker = ticker
        self.start = start
        self.end = end
        self.period = period
        self.interval = interval
        self.sequence_length = int(sequence_length)
        self.feature = feature

        self.cache_dir = cache_dir or ".cache"
        self.cache_ttl_seconds = int(cache_ttl_seconds)
        self.max_retries = int(max_retries)
        self.backoff_base = float(backoff_base)

        self.df: Optional[pd.DataFrame] = None
        self.scaler: Optional[MinMaxScaler] = None

        os.makedirs(self.cache_dir, exist_ok=True)

        # 【新增】初始化一个具备浏览器特征的持久化 Session
        self.session = requests.Session(impersonate="chrome")
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': '*/*',
            'Accept-Language': 'en-US,en;q=0.9',
        })

    def _cache_path(self) -> str:
        safe_ticker = self.ticker.replace('/', '_')
        name = f"yf_{safe_ticker}_{self.period}_{self.interval}.pkl"
        return os.path.join(self.cache_dir, name)

    def _load_cache(self, allow_expired: bool = False) -> Optional[pd.DataFrame]:
        path = self._cache_path()
        if not os.path.exists(path):
            return None
        try:
            mtime = int(os.path.getmtime(path))
            if not allow_expired:
                if time.time() - mtime > self.cache_ttl_seconds:
                    return None
            with open(path, 'rb') as fh:
                df = pickle.load(fh)
            if isinstance(df, pd.DataFrame):
                logger.debug('loaded cache %s', path)
                return df
        except Exception:
            logger.exception('failed loading cache')
        return None

    def _save_cache(self, df: pd.DataFrame) -> None:
        path = self._cache_path()
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'wb') as fh:
                pickle.dump(df, fh)
            logger.debug('saved cache %s', path)
        except Exception:
            logger.exception('failed saving cache')

    def fetch(self, use_cache: bool = True) -> pd.DataFrame:
        """Fetch data from yfinance with retry/backoff and cache support."""
        if use_cache:
            cached = self._load_cache()
            if cached is not None:
                self.df = cached
                return cached

        # Try to load GITHUB_TOKEN from environment if not already set
        # Using a .env file is recommended for local development
        if 'GITHUB_TOKEN' not in os.environ:
            # automatic loading from .env via python-dotenv if available could go here
            pass
        last_exc = None
        for attempt in range(1, self.max_retries + 1):
            try:
                # 【修改】直接使用 self.session 传入 yf.Ticker
                # 这样 yfinance 所有的内部请求（包括获取 Cookie 和 Crumb）都会通过 curl_cffi 模拟浏览器
                ticker_obj = yf.Ticker(self.ticker, session=self.session)

                df = ticker_obj.history(
                    period=self.period,
                    interval=self.interval,
                    start=self.start,
                    end=self.end,
                    timeout=15  # 增加超时保护
                )

                if df is None or df.empty:
                    # 如果返回 429，yfinance 0.2.59+ 可能会抛出特定异常，这里捕获通用异常
                    raise ValueError('no data downloaded or rate limited')

                self.df = df
                self._save_cache(df)
                return df

            except Exception as e:
                last_exc = e
                # 指数退避策略优化：429 错误建议等待更久
                backoff = self.backoff_base * (2 ** (attempt))  # 增加起始等待时长
                jitter = random.uniform(0, backoff * 0.2)
                wait = backoff + jitter
                logger.warning(
                    'fetch attempt %d failed: %s. retrying in %.1fs', attempt, e, wait)
                time.sleep(wait)

        # all retries failed, try stale cache
        cached = self._load_cache(allow_expired=True)
        if cached is not None:
            logger.warning('returning stale cache due to fetch failures')
            self.df = cached
            return cached

        raise last_exc

    def get_data(self, use_cache: bool = True) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
        """Return (X, y, scaler). X shape: (samples, seq_len, 1), y shape: (samples,).

        This method will call `fetch()` if data is not already present.
        """
        if self.df is None:
            self.fetch(use_cache=use_cache)

        df = self.df
        if self.feature not in df.columns:
            raise ValueError(
                f"feature column '{self.feature}' not found in data")

        values = df[self.feature].values.reshape(-1, 1).astype(np.float32)

        self.scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = self.scaler.fit_transform(values)

        seq = self.sequence_length
        if len(scaled) <= seq:
            raise ValueError(
                f"not enough data points ({len(scaled)}) for sequence_length={seq}")

        X_list = []
        y_list = []
        for i in range(seq, len(scaled)):
            X_list.append(scaled[i - seq: i, 0])
            y_list.append(scaled[i, 0])

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.float32)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        return X, y, self.scaler


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    loader = StockDataLoader('AAPL', period='6mo', sequence_length=30)
    try:
        X, y, scaler = loader.get_data()
        print('X.shape =', X.shape)
        print('y.shape =', y.shape)
    except Exception as e:
        print('fetch failed:', e)
