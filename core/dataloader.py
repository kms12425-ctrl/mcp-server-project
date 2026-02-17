"""Tools: ParquetDataLoader with QFQ (Forward Adjustment), MinMax scaling.

Uses 'data/full_data.parquet' and applies adjustment factor based on 'close' vs 'raw_close'.
"""
import logging
import os
import pickle
import time
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)


class ParquetDataLoader:
    """Download (from Parquet), normalize (MinMax) and serialize time-series into (X, y).

    Applies Forward Adjustment (QFQ) using:
      Factor = close / raw_close
      Adj_Open = open * Factor
      Adj_High = high * Factor
      Adj_Low  = low  * Factor
      (close is already adjusted in the dataset)

    Parameters
    ----------
    ticker: stock ticker symbol (e.g. '005930 KS')
    start, end: optional date strings
    period, interval: Ignored for local parquet, kept for compatibility/caching
    sequence_length: number of past steps used to predict next step
    feature: column to use (default 'Close')
    cache_dir: directory to store cached pickles
    cache_ttl_seconds: cache lifetime in seconds
    max_retries, backoff_base: ignored (no network calls)
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

        # Use a distinct cache directory for parquet-derived data
        self.cache_dir = cache_dir or ".cache_parquet"
        self.cache_ttl_seconds = int(cache_ttl_seconds)

        self.df: Optional[pd.DataFrame] = None
        self.scaler: Optional[MinMaxScaler] = None

        os.makedirs(self.cache_dir, exist_ok=True)

        # Local Parquet path
        # Assuming this file is in core/, and data is in ../data/
        self.parquet_path = os.path.join(os.path.dirname(
            os.path.dirname(__file__)), 'data', 'full_data.parquet')

    def _cache_path(self) -> str:
        safe_ticker = self.ticker.replace('/', '_').replace(' ', '_')
        # name includes qfq to differentiate
        name = f"qfq_{safe_ticker}_{self.period}_{self.interval}.pkl"
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
        """Fetch data from local Parquet with predicate pushdown and apply QFQ."""
        if use_cache:
            cached = self._load_cache()
            if cached is not None:
                self.df = cached
                return cached

        if not os.path.exists(self.parquet_path):
            raise FileNotFoundError(
                f"Parquet file not found at {self.parquet_path}")

        try:
            logger.info(
                f"Loading data from Parquet: {self.parquet_path} for {self.ticker}")

            # Efficiently load only specific ticker using pushdown filters
            # Parquet columns: ['ticker', 'date', 'close', 'raw_close', 'high', 'low', 'open', 'volume']
            df = pd.read_parquet(
                self.parquet_path,
                filters=[('ticker', '==', self.ticker)]
            )

            if df.empty:
                logger.warning(
                    f"Ticker {self.ticker} not found in Parquet dataset.")
                # Fallback check? No, strictly use Parquet here.
                raise ValueError(
                    f"No data found for ticker {self.ticker} in parquet dataset")

            # --- Data Processing & QFQ Logic ---

            # 1. Convert Date (int YYYYMMDD -> datetime)
            df['Date'] = pd.to_datetime(
                df['date'].astype(str), format='%Y%m%d')
            df.set_index('Date', inplace=True)
            df.sort_index(inplace=True)

            # 2. QFQ (Forward Adjustment)
            # Factor = close / raw_close
            # close in this dataset is typically the adjusted close.
            # raw_close is the unadjusted close.

            # Filter valid data
            df = df[df['raw_close'] != 0].copy()

            # Calculate Adjustment Factor
            adj_factor = df['close'] / df['raw_close']

            # Apply Factor to OHLC (Close is already adjusted)
            df['Open'] = df['open'] * adj_factor
            df['High'] = df['high'] * adj_factor
            df['Low'] = df['low'] * adj_factor
            df['Close'] = df['close']
            df['Volume'] = df['volume']  # Keep volume raw/as-is

            # Keep only standard columns
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()

            # 3. Filter by start/end date and period
            end_date_ref = df.index.max()
            if self.end:
                end_date_ref = pd.Timestamp(self.end)
                df = df[df.index <= end_date_ref]

            start_date_ref = None
            if self.start:
                start_date_ref = pd.Timestamp(self.start)
            elif self.period and self.period != 'max':
                # Calculate start date based on period relative to end_date_ref
                # Example: '1y' -> 1 year, '5y' -> 5 years, '6mo' -> 6 months
                try:
                    p = self.period.lower()
                    if p.endswith('y'):
                        years = int(p[:-1])
                        start_date_ref = end_date_ref - \
                            pd.DateOffset(years=years)
                    elif p.endswith('mo'):
                        months = int(p[:-2])
                        start_date_ref = end_date_ref - \
                            pd.DateOffset(months=months)
                    elif p.endswith('d'):
                        days = int(p[:-1])
                        start_date_ref = end_date_ref - \
                            pd.DateOffset(days=days)
                    else:
                        logger.warning(
                            f"Unknown period format '{self.period}', defaulting to full history.")
                except Exception as e:
                    logger.warning(
                        f"Failed to parse period '{self.period}': {e}")

            if start_date_ref:
                df = df[df.index >= start_date_ref]

            if df.empty:
                raise ValueError(
                    f"Ticker {self.ticker} found but no data in specified date range.")

            # Identify columns to float (ensure float32/64)
            cols_to_float = ["Open", "High", "Low", "Close", "Volume"]
            for col in cols_to_float:
                features_available = [
                    c for c in cols_to_float if c in df.columns]
                df[features_available] = df[features_available].apply(
                    pd.to_numeric, errors='coerce')

            # Drop any rows with NaN values (crucial for training)
            df.dropna(inplace=True)

            self.df = df
            self._save_cache(df)
            return df

        except Exception as e:
            logger.error(f"Failed to load from Parquet: {e}")
            raise e

    def get_data(self, use_cache: bool = True) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
        """Return (X, y, scaler). X shape: (samples, seq_len, 1), y shape: (samples,).

        This method will call `fetch()` if data is not already present.
        """
        if self.df is None:
            self.fetch(use_cache=use_cache)

        df = self.df
        if self.feature not in df.columns:
            raise ValueError(
                f"feature column '{self.feature}' not found in data. Options: {df.columns.tolist()}")

        # Extract values
        values = df[self.feature].values.reshape(-1, 1).astype(np.float32)

        # Normalize
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = self.scaler.fit_transform(values)

        seq = self.sequence_length
        if len(scaled) <= seq:
            raise ValueError(
                f"not enough data points ({len(scaled)}) for sequence_length={seq}")

        # Create Sequences
        X_list = []
        y_list = []
        for i in range(seq, len(scaled)):
            X_list.append(scaled[i - seq: i, 0])
            y_list.append(scaled[i, 0])

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.float32)

        # Reshape X to (samples, seq_len, 1)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        return X, y, self.scaler

    def get_latest_sequence(self, use_cache: bool = True) -> Tuple[np.ndarray, MinMaxScaler, pd.Timestamp]:
        """
        获取用于预测下一个时间步的最新序列数据。
        为了保持一致性，使用全量数据拟合Scaler。

        Returns:
            (last_sequence_reshaped, filled_scaler, last_date)
        """
        if self.df is None:
            self.fetch(use_cache=use_cache)

        df = self.df
        if self.feature not in df.columns:
            raise ValueError(f"Feature '{self.feature}' not found.")

        values = df[self.feature].values.reshape(-1, 1).astype(np.float32)

        # 必须使用与训练时相同的方式拟合Scaler
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = self.scaler.fit_transform(values)

        seq = self.sequence_length
        if len(scaled) < seq:
            raise ValueError(f"数据不足 ({len(scaled)}), 需要至少 {seq} 条")

        # 获取最后一段长度为 sequence_length 的数据
        last_seq = scaled[-seq:]
        last_seq = last_seq.reshape(1, seq, 1)

        last_date = df.index[-1]

        return last_seq, self.scaler, last_date


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # Test with a ticker known to be in the parquet (e.g., from user context '000060 KS')
    # Or just use AAPL if it exists.
    loader = ParquetDataLoader('000060 KS', sequence_length=30)
    try:
        print("Fetching data...")
        df = loader.fetch()
        print(f"Data Loaded. Shape: {df.shape}")
        print(df.head())
        print(df.tail())

        X, y, scaler = loader.get_data()
        print('X.shape =', X.shape)
        print('y.shape =', y.shape)
    except Exception as e:
        print('fetch failed:', e)
