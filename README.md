## 交互式时间序列预测智能体 (LSTM/GRU)

一款支持即时训练模式的股票时序预测智能助手，用户输入股票代码后，系统自动提取对应股票数据、现场训练 LSTM/GRU 双模型并择优预测股价，直观展示预测结果与模型对比分析。

### 组员信息

| 姓名 | 学号 | 分工 | 备注 |
| :------: | :------: | :--: | :--: |
|   马俊豪   | U202490042     |  智能体逻辑、交互与可视化的核心开发、项目调试   |    -  |
|   胡志锋   | U202414748     |    数据获取、演示视频制作           |  -    |
|   陈润泽   | U202414773     |      模型定义、训练引擎、项目整理               |  -    | 

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

#### 1. 环境准备
确保您的系统已安装 Python 3.10 或更高版本。

#### 2. 创建并激活虚拟环境
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

#### 3. 安装依赖
在虚拟环境激活状态下，安装项目所需的依赖包。
```bash
pip install -e .
```
> 注意：本项目依赖 `tensorflow` 或其他深度学习库，请确保网络连接正常以便下载。依赖列表请参考 `pyproject.toml`。

#### 4. 启动 MCP Server

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




