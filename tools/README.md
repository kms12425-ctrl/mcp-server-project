[中文](#中文) | [English](#english)

<a id="中文"></a>
# 股票分析工具集 (Tools)
本目录包含 YA MCP Server 中用于股票数据可视化、训练序列准备、价格预测的核心工具函数，所有工具均通过 `@YA_MCPServer_Tool` 装饰器注册为可调用的 MCP 工具。

## 工具列表
| 工具名称 | 功能描述 | 输入参数 | 输出格式 | 核心依赖 |
| :------: | :------: | :-------: | :------: | :------: |
| prepare_stock_sequences | 生成股票训练序列数据元信息 | ticker (股票代码)、sequence_length (序列长度，默认30) | 字典（含序列形状、示例值、数据时间范围） | StockDataLoader (ParquetDataLoader 封装类) |
| fetch_stock_history | 生成股票历史价格走势图 | ticker (股票代码)、period (时间范围，默认1y) | 列表（文本说明 + Base64 编码PNG图片） | matplotlib、pandas |
| prediction_stock_price | 股票价格预测（LSTM/GRU） | ticker (股票代码)、lookback (回溯天数，默认30)、epochs (训练轮数，默认50) | 列表（预测报告文本 + 可视化图表） | core.train2.run_analysis、StockDataLoader |

## 核心类说明
### StockDataLoader
ParquetDataLoader 的兼容封装类，确保数据使用一致性：
- **初始化参数**：ticker、start/end、period、interval、sequence_length、feature
- **核心方法**：
  - `fetch()`：获取原始 DataFrame 格式股票数据
  - `get_data()`：获取归一化后的 (X, y, scaler) 训练数据


##  资源调用示例
```python
from tools import fetch_stock_history, prepare_stock_sequences, predict_stock_price

# 获取股票历史走势图
result = fetch_stock_history(ticker="AAPL", period="1y")

# 准备训练序列数据
seq_info = prepare_stock_sequences(ticker="AAPL", sequence_length=30)

# 预测股票价格
pred_result = predict_stock_price(ticker="AAPL", lookback=30, epochs=50)

```
## 使用说明
fetch_stock_history 生成的走势图会自动保存到 logs/images/ 目录，命名规则：{ticker}_{period}_chart.png

异常处理:

所有工具函数均包含完整的异常捕获

数据获取失败：返回含错误信息的文本内容

绘图 / 训练失败：返回具体错误描述

文件保存失败：不影响核心功能，仅日志提示
---

<a id="english"></a>
# Stock Analysis Toolkit (Tools)
This directory contains core tool functions for stock data visualization, training sequence preparation, and price prediction in YA MCP Server. All tools are registered as callable MCP tools via the `@YA_MCPServer_Tool` decorator.

## Tool List
| Tool Name | Description | Input Parameters | Output Format | Core Dependencies |
| :---: | :---: | :---: | :---: | :---: |
| prepare_stock_sequences | Generates metadata for stock training sequence data | ticker (stock code), sequence_length (sequence length, default 30) | Dictionary (contains sequence shape, sample values, data time range) | StockDataLoader (ParquetDataLoader wrapper class) |
| fetch_stock_history | Generates stock historical price trend chart | ticker (stock code), period (time range, default 1y) | List (text description + Base64 encoded PNG image) | matplotlib, pandas |
| prediction_stock_price | Stock price prediction (LSTM/GRU) | ticker (stock code), lookback (lookback days, default 30), epochs (training epochs, default 50) | List (prediction report text + visualization chart) | core.train2.run_analysis, StockDataLoader |

## Core Class Description
### StockDataLoader
Compatible wrapper class for ParquetDataLoader, ensuring consistency in data usage:
- **Initialization Parameters**: ticker, start/end, period, interval, sequence_length, feature
- **Core Methods**:
  - `fetch()`: Get raw DataFrame format stock data
  - `get_data()`: Get normalized (X, y, scaler) training data

## Resource Call Example
```python
from tools import fetch_stock_history, prepare_stock_sequences, predict_stock_price

# Get stock historical trend chart
result = fetch_stock_history(ticker="AAPL", period="1y")

# Prepare training sequence data
seq_info = prepare_stock_sequences(ticker="AAPL", sequence_length=30)

# Predict stock price
pred_result = predict_stock_price(ticker="AAPL", lookback=30)
```

## Usage Instructions
`fetch_stock_history` generated trend charts are automatically saved to `logs/images/` directory, naming rule: `{ticker}_{period}_chart.png`

**Exception Handling:**
All tool functions include complete exception capturing.
- **Data acquisition failure**: Returns text content containing error information.
- **Plotting / Training failure**: Returns specific error description.
- **File saving failure**: Does not affect core functions, only log prompts.