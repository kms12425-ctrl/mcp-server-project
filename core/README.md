[中文](#中文) | [English](#english)

<a id="中文"></a>
# 核心模块 (core/)
本目录是股票分析系统的底层核心层，包含数据加载、深度学习模型构建、模型训练与预测的全流程实现，为上层 `tools`/`resources` 模块提供核心支撑。

## 目录结构
core/ 

├── dataloader.py -- ParquetDataLoader 数据加载核心类

├── models.py -- LSTM/GRU 模型构建函数

└── train2.py -- 模型训练、对比、预测全流程函数



## 1. 数据加载 (dataloader.py)

从本地 Parquet 文件加载股票数据，完成 QFQ 前复权、数据标准化、时序序列生成，是整个系统的数据源核心。

#### 关键特性
| 特性 | 说明 |
|------|------|
| 数据来源 | 本地 Parquet 文件（默认路径：`../data/full_data.parquet`） |
| 复权处理 | 基于 `close/raw_close` 计算复权因子，对 Open/High/Low 执行 QFQ 前复权 |
| 缓存机制 | 自动缓存处理后的数据到 `.cache_parquet/`，缓存有效期 24 小时 |
| 日期过滤 | 支持 `start/end` 日期或 `period`（1y/5y/6mo 等）过滤数据 |
| 序列生成 | 生成适配 LSTM/GRU 的 (X,y) 时序训练数据 |



## 2.模型构建 (models.py)

提供标准化的 LSTM/GRU 模型构建函数，适配股票价格时序预测场景。


## 3. 训练与预测 (train2.py)

提供标准化的 LSTM/GRU 模型构建函数，适配股票价格时序预测场景。


#### 关键特性
| 特性 | 说明 |
|------|------|
| 时序数据拆分 | 7:2:1 划分训练 / 验证 / 测试集（不打乱，保证时序性） |
| 早停机制 | EarlyStopping 监控 val_loss，patience=5，防止过拟合 |
| 预测逻辑 | 基于最新序列预测下一个交易日价格 |
| 日期过滤 | 支持 `start/end` 日期或 `period`（1y/5y/6mo 等）过滤数据 |
| 序列生成 | 生成适配 LSTM/GRU 的 (X,y) 时序训练数据 |

## 异常处理

#### 数据缺失
Parquet 文件不存在 / 股票无数据时，抛出 ValueError 并提示具体原因

#### 序列不足
数据量小于 sequence_length 时，抛出 ValueError
#### 训练失败
自动捕获异常，保证上层调用鲁棒性

---

<a id="english"></a>
# Core Modules (core/)

This directory is the underlying core layer of the stock analysis system, containing the full process implementation of data loading, deep learning model construction, model training, and prediction, providing core support for the upper-level `tools`/`resources` modules.

## Directory Structure
core/ 
├── dataloader.py -- ParquetDataLoader: Core class for data loading
├── models.py -- LSTM/GRU model construction functions
└── train2.py -- Full process functions for model training, comparison, and prediction

## 1. Data Loading (dataloader.py)

Loads stock data from local Parquet files, completes QFQ (Quote Forward Qualification) adjustment, data standardization, and time series generation. It is the data source core of the entire system.

#### Key Features
| Feature | Description |
|---|---|
| Data Source | Local Parquet file (default path: `../data/full_data.parquet`) |
| Adjustment Processing | Calculates adjustment factor based on `close/raw_close`, performs QFQ adjustment on Open/High/Low |
| Caching Mechanism | Automatically caches processed data to `.cache_parquet/`, cache validity is 24 hours |
| Date Filtering | Supports filtering data by start/end date or period (1y/5y/6mo, etc.) |
| Sequence Generation | Generates (X,y) time series training data adapted for LSTM/GRU |

## 2. Model Construction (models.py)

Provides standardized LSTM/GRU model construction functions, adapted for stock price time series prediction scenarios.

## 3. Training and Prediction (train2.py)

Provides full process functions for model training, comparison, and prediction.

#### Key Features
| Feature | Description |
|---|---|
| Time Series Split | 7:2:1 split for training/validation/test sets (no shuffle, ensuring temporal order) |
| Early Stopping | EarlyStopping monitors `val_loss`, patience=5, preventing overfitting |
| Prediction Logic | Predicts the next trading day's price based on the latest sequence |

## Exception Handling
