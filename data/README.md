# 数据目录 (data/)
本目录是股票分析系统的**核心原始数据源存储目录**，所有股票时序数据均从该目录下的 Parquet 文件加载。

## 核心文件说明
| 文件名 | 格式 | 作用 | 必选 |
|--------|------|------|------|
| `full_data.parquet` | Parquet | 全量股票时序数据文件，系统唯一的原始数据源 |  是 |

## 数据加载逻辑
1. `core/dataloader.ParquetDataLoader` 会从本目录加载 `full_data.parquet`；
2. 加载时会按 `ticker` 过滤数据，仅加载指定股票的记录；
3. 自动执行 QFQ 前复权计算（基于 `close/raw_close` 复权因子）；
4. 处理后的数据会缓存到 `.cache_parquet/` 目录，而非本目录。

## 目录特性
1. **只读目录**：本目录仅存储原始数据，程序不会修改/新增本目录下的文件；
2. **非缓存目录**：缓存数据存储在 `.cache_parquet/`，与原始数据分离；
3. **核心依赖**：删除/缺失 `full_data.parquet` 会导致整个系统无法加载数据。

## 数据更新说明
1. 如需更新股票数据，直接替换 `full_data.parquet` 文件即可；
2. 更新后建议清空 `.cache_parquet/` 目录，避免缓存数据与新数据不一致；
3. 从https://www.kaggle.com/datasets/code1110/yfinance-stock-price-data-for-numerai-signals/data可以获取最新的股价信息数据集，这个数据通过yfinance api爬取获得，每天更新，保证价格最新。

## 异常处理
- 文件缺失：程序会抛出 `FileNotFoundError`，提示 `Parquet file not found at ../data/full_data.parquet`；
- 字段缺失：加载数据时会因字段不存在导致 KeyError，需补充对应字段；
- 格式错误：日期字段无法转换、数值字段非浮点型等，会导致数据处理失败。

