# 股票数据资源 (Resources)
本目录包含 YA MCP Server 中用于获取股票原始数据的资源函数，所有资源均通过 `@YA_MCPServer_Resource` 装饰器注册，可通过指定 URI 调用。

## 资源列表
| 资源名称 | URI 格式 | 功能描述 | 输入参数 | 输出格式 | 核心依赖 |
| :------: | :------: | :------: | :-------: | :------: | :------: |
| get_latest_stock_data | stock://{ticker}/latest | 获取股票最新一个交易日的 OHLCV 数据 | ticker (股票代码) | JSON 字符串（含日期、开盘/最高/最低/收盘/成交量） | core.dataloader.ParquetDataLoader |
| get_stock_data | data://stocks/{ticker} | 获取股票近1年的日度历史数据 | ticker (股票代码，不能为空) | 字典列表（JSON 格式，含日期、OHLCV、成交量） | tools.data_loader.StockDataLoader |


##  资源调用示例
```python
from resources import get_latest_stock_data, get_stock_data

# 获取最新单条数据
latest_data = get_latest_stock_data(ticker="AAPL")  # 返回 JSON 字符串

# 获取历史数据列表
history_data = get_stock_data(ticker="AAPL")  # 返回字典列表
```


##  关键特性
### 缓存机制
get_latest_stock_data 优先使用本地缓存数据，无缓存时从 Parquet 文件加载；
### 数据格式化
get_stock_data 自动将日期字段格式化为 %Y-%m-%d %H:%M:%S；
### 参数校验
get_stock_data 会校验 ticker 参数非空，避免无效调用；
### 异常返回
所有资源失败时返回含 error 键的 JSON / 字典，包含上下文信息（如 ticker、period 等）。