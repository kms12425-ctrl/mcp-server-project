import logging
from typing import Any, Optional

import pandas as pd

from resources import YA_MCPServer_Resource
from tools.data_loader import StockDataLoader

logger = logging.getLogger(__name__)


@YA_MCPServer_Resource(
    "data://stocks/{ticker}",
    name="get_stock_data",
    title="Get Stock Data",
    description="从本地 parquet 获取指定股票的历史数据并以 JSON 返回",
)
def get_stock_data(ticker: str) -> Any:
    """
    下载并返回指定股票的历史数据（JSON 格式）。

    Args:
        ticker (str): 股票代码，例如 "AAPL"。

    Returns:
        Any: 成功时返回包含 OHLCV 和 volume 的 JSON（dict），失败时返回包含错误信息的 dict。
    """
    try:
        if not ticker:
            return {"error": "ticker 参数不能为空"}

        period = "1y"
        interval = "1d"

        try:
            loader = StockDataLoader(ticker, period=period, interval=interval)
            df = loader.fetch()
        except Exception as e:
            logger.warning(f"Error fetching data for {ticker}: {e}")
            return {"error": str(e), "ticker": ticker}

        if df is None or df.empty:
            return {"error": "no data", "ticker": ticker, "period": period, "interval": interval}

        df_reset = df.reset_index()
        if 'Date' in df_reset.columns:
            df_reset['Date'] = df_reset['Date'].dt.strftime(
                '%Y-%m-%d %H:%M:%S')
        else:
            try:
                # Fallback if index name is missing but it is the first column
                df_reset.iloc[:, 0] = pd.to_datetime(
                    df_reset.iloc[:, 0]).dt.strftime('%Y-%m-%d %H:%M:%S')
            except Exception:
                pass

        return df_reset.to_dict(orient='records')
    except Exception as e:
        logger.exception("failed to fetch stock data")
        return {"error": str(e)}


# Expose StockDataLoader for convenience via this module
__all__ = ["get_stock_data", "StockDataLoader"]
