"""Tools: StockDataLoader wrapper for ParquetDataLoader.

Redirects requests to the core ParquetDataLoader to ensure consistent data usage.
"""
from tools import YA_MCPServer_Tool
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from sklearn.preprocessing import MinMaxScaler
from core.dataloader import ParquetDataLoader
import mcp.types as types

logger = logging.getLogger(__name__)


class StockDataLoader:
    """Compatibility wrapper for ParquetDataLoader."""

    def __init__(
        self,
        ticker: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        period: str = "1y",
        interval: str = "1d",
        sequence_length: int = 30,
        feature: str = "Close",
        **kwargs
    ):
        # 初始化核心的 Parquet 加载器
        self.loader = ParquetDataLoader(
            ticker=ticker,
            start=start,
            end=end,
            period=period,
            interval=interval,
            sequence_length=sequence_length,
            feature=feature
        )
        self.ticker = ticker

    def fetch(self, use_cache: bool = True) -> pd.DataFrame:
        """Fetch data using ParquetDataLoader."""
        # 调用核心加载器的 fetch
        # core/dataloader.py 的 get_data 会自动调用内部加载逻辑
        # 这里为了兼容，我们需要手动触发加载
        if self.loader.df is None:
            self.loader.get_data(use_cache=use_cache)

        if self.loader.df is None or self.loader.df.empty:
            raise ValueError(
                f"No data found for ticker {self.ticker} in parquet dataset")

        return self.loader.df

    def get_data(self, use_cache: bool = True) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
        """Return (X, y, scaler)."""
        return self.loader.get_data(use_cache=use_cache)


@YA_MCPServer_Tool(
    name="fetch_stock_history",
    title="Fetch Stock History Chart",
    description="从本地数据库获取股票历史数据并生成走势图 (返回图像内容)",
)
def fetch_stock_history(ticker: str, period: str = "1y") -> list:
    """
    获取股票历史数据并绘制走势图。
    返回包含图像内容的列表。
    """
    try:
        # 1. 加载数据 (StockDataLoader 内部兼容了 ParquetDataLoader)
        loader = StockDataLoader(ticker, period=period)
        df = loader.fetch()

        # 2. 绘图
        plt.figure(figsize=(10, 5))
        plt.plot(df.index, df['Close'], label='Close Price')
        if 'Open' in df.columns:
            plt.plot(df.index, df['Open'],
                     label='Open Price', alpha=0.5, linestyle='--')

        plt.title(f"{ticker} Stock Price History ({period})")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # 3. 保存图像到内存缓冲区 (用于返回 ImageContent)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

        # 4. 同时保存到文件 (作为备份/记录)
        import os
        image_dir = os.path.join("logs", "images")
        os.makedirs(image_dir, exist_ok=True)
        filename = f"{ticker}_{period}_chart.png"
        filepath = os.path.join(image_dir, filename)
        with open(filepath, "wb") as f:
            f.write(buf.getvalue())

        plt.close()  # 关闭图表释放内存

        # 5. 返回 ImageContent
        return [
            types.TextContent(
                type="text", text=f"Here is the stock history chart for {ticker} ({period}):"),
            types.ImageContent(
                type="image",
                data=img_base64,
                mimeType="image/png"
            )
        ]

    except Exception as e:
        return [types.TextContent(type="text", text=f"Error generating chart: {str(e)}")]


# 辅助函数 (不再需要 python_uri_fix，但可以留着或删除，暂时保留以免破坏其他可能的引用)
def python_uri_fix(path):
    from urllib.request import pathname2url
    return "file:" + pathname2url(path)


@YA_MCPServer_Tool(
    name="prepare_stock_sequences",
    title="Prepare Stock Sequences",
    description="返回用于训练的 (X,y) 形状和示例",
)
def prepare_stock_sequences(ticker: str, sequence_length: int = 30) -> Dict[str, Any]:
    """
    准备并返回训练序列数据的元数据。
    """
    try:
        loader = StockDataLoader(ticker, sequence_length=sequence_length)
        X, y, scaler = loader.get_data()

        return {
            "ticker": ticker,
            "X_shape": str(X.shape),
            "y_shape": str(y.shape),
            "sample_X_last": X[-1].flatten().tolist(),  # 最后一个输入序列
            "sample_y_last": float(y[-1]),  # 对应的目标值
            "data_range": f"{loader.loader.df.index.min()} to {loader.loader.df.index.max()}"
        }
    except Exception as e:
        return {"error": str(e)}
