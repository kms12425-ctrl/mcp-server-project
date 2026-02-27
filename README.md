<a id="chinese"></a>
[中文](#chinese) | [English](#english)

## 交互式时间序列预测智能体 (LSTM/GRU)

一款支持即时训练模式的股票时序预测智能助手，用户输入股票代码后，系统自动提取对应股票数据、现场训练 LSTM/GRU 双模型并择优预测股价，直观展示预测结果与模型对比分析。

### Tool 列表

| 工具名称 | 功能描述 | 输入 | 输出 | 备注 |
| :------: | :------: | :--: | :--: | :--: |
| prepare_stock_sequences | 准备并返回股票训练序列数据的元数据，包含输入输出序列形状、示例值和数据时间范围 | ticker (股票代码，字符串)、sequence_length (序列长度，整数，默认 30) | 字典：包含 ticker、X_shape、y_shape、sample_X_last、sample_y_last、data_range 的字典 | - |
| Fetch Stock History Chart | 从本地 Parquet 数据库获取股票历史价格数据，生成并返回可视化走势图 | ticker (股票代码，字符串)、period (时间范围，字符串，默认 "1y") |列表：包含文本说明和Base64编码的PNG走势图 | - |
| Stock Price Prediction (LSTM/GRU) | 训练 LSTM/GRU 深度学习模型，预测指定股票下一日价格并生成预测报告 | ticker (股票代码，字符串)、lookback (回溯天数，整数，默认 30)、epochs (训练轮数，整数，默认 50) |列表：包含预测报告文本，若有图表则附加Base64编码的PNG图片 | - |

### Resource 列表

| 资源名称 | 功能描述 | 输入 | 输出 | 备注 |
| :------: | :------: | :--: | :--: | :--: |
|   get_latest_stock_data       |   获取指定股票最新一个交易日的 OHLCV（开盘、最高、最低、收盘、成交量）数据       |  ticker（股票代码，字符串）    |   JSON 格式字符串：包含 ticker（股票代码）、date（日期）、open（开盘价）、high（最高价）、low（最低价）、close（收盘价）、volume（成交量）   |      |
|   get_stock_data       |    从本地 Parquet 数据库获取指定股票近 1 年的日度历史数据，并转换为 JSON 格式返回      |   ticker（股票代码，字符串）   |  包含日期、OHLCV、volume 的字典列表（JSON 格式）    |      |
|          |          |      |      |      |

### Prompts 列表

| 指令名称 | 功能描述 | 输入 | 输出 | 备注 |
| :------: | :------: | :--: | :--: | :--: |
|  greet_user        |   生成用户问候消息       |  	name（用户名，字符串）    |   	字符串：包含用户名的问候语（如 “你好，Alice！欢迎使用 YA MCP Server。”）   |      |
|    analyze_stock_prediction      |    	生成标准化的股票分析请求指令文本，指导系统验证股票数据并执行价格预测      |  ticker（股票代码，字符串）    |  字符串：英文指令文本，包含检查指定股票最新数据的资源调用、执行深度学习价格预测工具的要求    |      |
|          |          |      |      |      |

### 项目结构

- `core`: init.py;hello_secrets.py;models.py;train2.py;dataloader.py
- `prompts`:init.py;hello_prompt.py;stock_prompt.py
- `resources`:init.py;hello_resource.py;data_loader.py;stock_resource.py
- `tools`: init.py;hello_tool.py;data_loader.py;stock_prediction.py
- `data`:存放项目依赖的Parquet格式股票数据集文件

### 其他需要说明的情况

- 项目未使用 sops 模块及密钥变量，无需配置加密相关内容。
- 使用 TensorFlow/Keras 深度学习框架构建 LSTM/GRU 模型
- 核心使用深度学习模型（LSTM/GRU 循环神经网络）实现时间序列预测，通过即时训练模式为单只股票定制化模型，对比双模型的训练速度与预测精度。

### 快速开始 (Quick Start)

#### 1. 下载数据集

本项目所有股票时序数据均从 `data` 目录下的 Parquet 文件加载。
请从 [Kaggle 数据集页面](https://www.kaggle.com/datasets/code1110/yfinance-stock-price-data-for-numerai-signals/data) 获取最新的股价信息数据集。
下载对应数据集后，将其放入/替换 `data` 文件夹下即可。详情可见 `data` 目录下的 `README.md` 文件。

#### 2. 环境准备
确保您的系统已安装 Python 3.10 或更高版本。

#### 3. 创建并激活虚拟环境
建议在虚拟环境中运行此项目，以避免依赖冲突。

**Windows:**
```powershell
python -m venv .venv
.venv\Scripts\activate
```

**Linux/macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### 4. 安装依赖
在虚拟环境激活状态下，安装项目所需的依赖包。
```bash
pip install -e .
```
> 注意：本项目依赖 `tensorflow` 或其他深度学习库，请确保网络连接正常以便下载。依赖列表请参考 `pyproject.toml`。

#### 5. 启动 MCP Server

**方式一：STDIO 模式 (默认，用于接入 MCP 客户端)**  
这是 MCP Server 的标准运行模式，通常配合 MCP 客户端（如 Claude Desktop）使用。  
请在客户端配置文件中添加以下配置（路径需根据实际情况修改）：

```json
{
  "mcpServers": {
    "StockPrediction": {
      "command": "python",
      "args": ["path/to/YA_MCPServer_Stockprediction/server.py"]
    }
  }
}
```
*注意：`path/to/...` 需要替换为您的实际路径。请确保 `command` 使用的是虚拟环境中的 python 解释器路径，或者直接使用绝对路径引用虚拟环境解释器。*

**方式二：SSE 模式 (用于调试或远程访问)**  
1. 修改 `config.yaml` 文件，将 `transport` 下的 `type` 改为 `sse`。  
2. 运行服务器：  
```bash
python server.py
```
服务器将启动在 `config.yaml` 中配置的地址和端口（默认 http://127.0.0.1:12345）。

**方式三：使用 MCP Inspector (通过 npx 调试)**  
此方法用于开发调试，它会启动一个可视化界面来测试工具和资源。需确保系统已安装 Node.js。  
在虚拟环境激活状态下运行：
```bash
npx @modelcontextprotocol/inspector python server.py
# 如果使用 uv:
# npx @modelcontextprotocol/inspector uv run server.py
```
运行后，访问终端显示的 URL（通常为 http://localhost:5173）即可开始调试。





---

<a id="english"></a>
## Interactive Time Series Prediction Agent (LSTM/GRU)

An intelligent stock time series prediction assistant supporting instant training mode. After the user inputs the stock ticker, the system automatically extracts corresponding stock data, trains LSTM/GRU dual models on-site, selects the best model for price prediction, and intuitively displays prediction results and model comparison analysis.

### Tool List

| Tool Name | Description | Input | Output | Remarks |
| :---: | :---: | :--: | :--: | :--: |
| prepare_stock_sequences | Prepares and returns metadata for stock training sequence data, including input/output sequence shapes, sample values, and data time range. | ticker (string), sequence_length (integer, default 30) | Dictionary: Contains ticker, X_shape, y_shape, sample_X_last, sample_y_last, data_range | - |
| Fetch Stock History Chart | Retrieves stock historical price data from local Parquet database, generates and returns a visual trend chart. | ticker (string), period (string, default "1y") | List: Contains text description and Base64 encoded PNG trend chart | - |
| Stock Price Prediction (LSTM/GRU) | Trains LSTM/GRU deep learning models, predicts the next day's price for the specified stock, and generates a prediction report. | ticker (string), lookback (integer, default 30), epochs (integer, default 50) | List: Contains prediction report text, and Base64 encoded PNG chart if available | - |

### Resource List

| Resource Name | Description | Input | Output | Remarks |
| :---: | :---: | :--: | :--: | :--: |
| get_latest_stock_data | Gets the latest trading day's OHLCV (Open, High, Low, Close, Volume) data for the specified stock. | ticker (string) | JSON string: Contains ticker, date, open, high, low, close, volume | - |
| get_stock_data | Retrieves daily historical data for the specified stock for the last 1 year from local Parquet database, returned in JSON format. | ticker (string) | List of dictionaries containing date, OHLCV, volume (JSON format) | - |

### Prompts List

| Prompt Name | Description | Input | Output | Remarks |
| :---: | :---: | :--: | :--: | :--: |
| greet_user | Generates a user greeting message. | name (string) | String: Greeting message including the username (e.g., "Hello, Alice! Welcome to YA MCP Server.") | - |
| analyze_stock_prediction | Generates a standardized stock analysis request instruction text, guiding the system to verify stock data and execute price prediction. | ticker (string) | String: English instruction text including resource calls to check latest stock data and requirements to execute deep learning price prediction tool | - |

### Project Structure

- core: init.py; hello_secrets.py; models.py; train2.py; dataloader.py
- prompts: init.py; hello_prompt.py; stock_prompt.py
- resources: init.py; hello_resource.py; data_loader.py; stock_resource.py
- tools: init.py; hello_tool.py; data_loader.py; stock_prediction.py
- data: Stores Parquet format stock dataset files required by the project

### Other Information

- The project does not use the sops module or secret variables, so no encryption configuration is needed.
- Uses TensorFlow/Keras deep learning framework to build LSTM/GRU models.
- The core uses deep learning models (LSTM/GRU Recurrent Neural Networks) for time series prediction, customizing models for single stocks through instant training mode, comparing the training speed and prediction accuracy of dual models.

### Quick Start

#### 1. Download Dataset

All stock time series data in this project is loaded from Parquet files in the data directory.
Please get the latest stock price information dataset from the [Kaggle Dataset Page](https://www.kaggle.com/datasets/code1110/yfinance-stock-price-data-for-numerai-signals/data).
After downloading the corresponding dataset, place/replace it in the data folder. See the README.md file in the data directory for details.

#### 2. Environment Preparation
Ensure your system has Python 3.10 or higher installed.

#### 3. Create and Activate Virtual Environment
It is recommended to run this project in a virtual environment to avoid dependency conflicts.

**Windows:**
```powershell
python -m venv .venv
.venv\Scripts\activate
```

**Linux/macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### 4. Install Dependencies
With the virtual environment activated, install the required dependencies for the project.
```bash
pip install -e .
```
> Note: This project depends on tensorflow or other deep learning libraries, please ensure a normal network connection for downloading. Refer to pyproject.toml for the dependency list.

#### 5. Start MCP Server

**Method 1: STDIO Mode (Default, for connecting to MCP Client)**
This is the standard running mode of MCP Server, usually used with MCP Clients (such as Claude Desktop).
Please add the following configuration to the client configuration file (modify the path according to the actual situation):

```json
{
  "mcpServers": {
    "StockPrediction": {
      "command": "python",
      "args": ["path/to/YA_MCPServer_Stockprediction/server.py"]
    }
  }
}
```
*Note: path/to/... needs to be replaced with your actual path. Please ensure command uses the python interpreter path in the virtual environment, or directly use the absolute path to reference the virtual environment interpreter.*

**Method 2: SSE Mode (For debugging or remote access)**
1. Modify the `config.yaml` file and change `type` under `transport` to `sse`.
2. Run the server:
```bash
python server.py
```
The server will start on the address and port configured in `config.yaml` (default http://127.0.0.1:12345).

**Method 3: Use MCP Inspector (Debug via npx)**
This method is used for development debugging; it starts a visual interface to test tools and resources. Ensure Node.js is installed on the system.
Run in the activated virtual environment:
```bash
npx @modelcontextprotocol/inspector python server.py
# If using uv:
# npx @modelcontextprotocol/inspector uv run server.py
```
After running, visit the URL displayed in the terminal (usually http://localhost:5173) to start debugging.
