from typing import Any
from resources import YA_MCPServer_Resource
from core.dataloader import ParquetDataLoader
import pandas as pd
import json


@YA_MCPServer_Resource(
    uri="stock://{ticker}/latest",
    name="get_latest_stock_data",
    title="Latest Stock Data",
    description="Get the latest date's OHLCV data for a specific stock (e.g., AAPL).",
    mime_type="application/json"
)
def get_latest_stock_data(ticker: str) -> str:
    """
    返回指定股票的最新一条交易数据。
    """
    try:
        # 使用 core 中的加载器加载数据
        # sequence_length在这里不重要，只取最后一行
        loader = ParquetDataLoader(ticker, sequence_length=1)
        # 获取数据（如果本地有缓存则直接用，没有则从parquet加载）
        loader.fetch(use_cache=True)

        if loader.df is None or loader.df.empty:
            return json.dumps({"error": f"No data found for {ticker}"})

        # 获取最后一行数据
        last_row = loader.df.iloc[-1]

        # 转换为字典
        data = {
            "ticker": ticker,
            "date": str(last_row.name.date()),
            "open": float(last_row['Open']),
            "high": float(last_row['High']),
            "low": float(last_row['Low']),
            "close": float(last_row['Close']),
            "volume": float(last_row['Volume'])
        }

        return json.dumps(data, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})
