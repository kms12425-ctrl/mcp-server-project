[中文](#中文) | [English](#english)

<a id="中文"></a>
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


---

<a id="english"></a>
# Data Directory (data/)

This directory is the **core raw data source storage directory** of the stock analysis system. All stock time series data are loaded from Parquet files in this directory.

## Core File Description
| File Name | Format | Function | Required |
|---|---|---|---|
| `full_data.parquet` | Parquet | Full stock time series data file, the unique raw data source of the system | Yes |

## Data Loading Logic
1. `core/dataloader.ParquetDataLoader` will load `full_data.parquet` from this directory;
2. When loading, it filters data by `ticker`, loading only the records of the specified stock;
3. Automatically performs QFQ adjustment calculation (based on `close/raw_close` adjustment factor);
4. Processed data will be cached to the `.cache_parquet/` directory, not this directory.

## Directory Features
1. **Read-only Directory**: This directory only stores raw data; the program will not modify/add files in this directory;
2. **Non-cache Directory**: Cached data is stored in `.cache_parquet/`, separated from raw data;
3. **Core Dependency**: Deletion/missing of `full_data.parquet` will cause the entire system to fail to load data.

## Data Update Instructions
1. To update stock data, simply replace the `full_data.parquet` file;
2. After updating, it is recommended to clear the `.cache_parquet/` directory to avoid inconsistency between cached data and new data;
3. You can get the latest stock price information dataset from [Kaggle Dataset](https://www.kaggle.com/datasets/code1110/yfinance-stock-price-data-for-numerai-signals/data). This data is crawled via yfinance api and updated daily to ensure the latest prices.

## Exception Handling
- **File Missing**: The program will raise `FileNotFoundError`, prompting Parquet file not found.
- **Field Missing**: Missing fields when loading data will cause `KeyError`, requiring the corresponding fields to be supplemented;
- **Format Error**: Failure to convert date fields, or non-float numeric fields, will lead to data processing failure.
