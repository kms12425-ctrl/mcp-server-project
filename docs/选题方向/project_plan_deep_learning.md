# 项目推进流程：交互式时间序列预测智能体 (LSTM/GRU)

## 1. 项目概述
**选题：** 基于 LSTM/GRU 的交互式股票/时序预测助手
**模式：** **On-Demand Training (即时训练模式)**
**核心理念：** 不做通用的预训练模型，而是根据用户输入的股票代码，**现场下载**该股票数据，**现场训练**专属模型，**现场预测**。
**目标：** 构建一个智能体，用户告知股票代码（如 AAPL, TSLA），系统自动完成“数据准备 -> 模型训练(LSTM vs GRU) -> 择优 -> 预测”的全流程。

### 1.1 为什么采用“即时训练”模式？
1.  **针对性强：** 科技股、能源股、医药股走势规律截然不同。为每只股票单独训练模型，比训练一个“大概率也不准”的通用模型精度更高。
2.  **无限覆盖：** 不需要提前下载几千只股票的数据存在硬盘里。用户问什么，就现学什么。
3.  **轻量级优势：** 我们使用的 LSTM/GRU 模型结构简单（2-3层），在普通 PC 上训练一只股票仅需 **30-60秒**，用户等待体验完全可接受。

### 1.2 为什么必须是 LSTM 和 GRU？
*   **普通神经网络痛点：** 无法记忆长期的历史信息（健忘）。
*   **LSTM (长短期记忆)：** 拥有“遗忘门”和“输入门”，能精准记住关键的历史趋势。
*   **GRU (门控循环单元)：** LSTM 的轻量化变体，计算更快。**本项目正是要对比这两者在实时训练中的速度和精度差异。**

---

## 2. 阶段一：基础组件构建 (第1-2天)

### 2.1 环境搭建
```bash
python -m venv venv
pip install tensorflow pandas matplotlib yfinance scikit-learn
```

### 2.2 动态数据加载器 (`data_loader.py`)
编写 `StockDataLoader` 类，不再硬编码股票代码，而是将其作为参数。
*   **输入：** 股票代码 (Ticker)，开始时间，结束时间。
*   **动作：** 
    1.  调用 `yfinance` 实时下载数据。
    2.  **归一化：** 使用 `MinMaxScaler` 将价格压缩到 0-1 (神经网络无论大小，只认 0-1)。
    3.  **序列化：** 将时间序列切分为 `(X_train, y_train)`。例如：用过去 30 天预测下一天。

---

## 3. 阶段二：模型定义与训练引擎 (第3-4天)

### 3.1 定义模型 (`models.py`)
定义两个函数来“生产”模型，确保每次调用都返回一个新的、未训练的初始网络：
*   `build_lstm_model()`: 包含 LSTM 层 + Dense 层。
*   `build_gru_model()`: 包含 GRU 层 + Dense 层。
*   **对比点：** 结构完全一致，只换核心算子，公平对比性能。

### 3.2 训练引擎 (`train_agent.py`)
编写一个核心函数 `train_and_evaluate(model, X, y)`。
*   **交互式优化：** 
    *   使用 **Early Stopping (早停机制)**。
    *   设定 `patience=3`，如果训练 3 轮 Loss 还没降，就提前结束，**节省用户等待时间**。
*   **返回：** 训练好的模型对象 + 最终 Loss 值。

---

## 4. 阶段三：智能体逻辑与交互 (第5-6天)

### 4.1 智能体主控 (`agent.py`)
这是智能体的“大脑”，负责调度整个流程：
```python
def run_prediction_task(ticker):
    print(f"收到指令：分析 {ticker}...")
    
    # 1. 现场获取数据
    loader = StockDataLoader(ticker)
    X, y = loader.get_data()
    
    # 2. 现场训练双模型
    print("正在训练 LSTM 大脑...")
    lstm_model = train(lstm, X, y)
    
    print("正在训练 GRU 大脑...")
    gru_model = train(gru, X, y)
    
    # 3. 择优录取
    if lstm_loss < gru_loss:
        return "LSTM胜出", lstm_predict
    else:
        return "GRU胜出", gru_predict
```

### 4.2 反归一化
*   模型输出的是 `0.55` 这种数字，必须用 `scaler.inverse_transform` 还原成 `$185.5` 这种真实股价展示给用户。

---

## 5. 阶段四：可视化与最终演示 (第7天)

### 5.1 动态绘图
*   当训练完成后，弹出一张对比图。
*   **黑线：** 真实历史走势。
*   **蓝线：** LSTM 拟合曲线。
*   **红线：** GRU 拟合曲线。
*   **亮点：** 在图的末尾画出“明日预测点”，直观展示。

### 5.2 项目产出标准
*   **运行脚本：** `python main.py`
*   **交互过程：**
    ```text
    > 请输入股票代码: NVDA
    > [系统] 正在下载英伟达(NVDA)数据...
    > [系统] 正在训练模型 (预计耗时 40s)...
    > [结果] 训练完成！
    > [结论] GRU 模型表现更好 (Error: 0.002)。
    > [预测] 英伟达明日股价预计为: $880.50
    ```

---

## 6. 代码结构示例

```text
project/
│
├── data_loader.py     # 负责动态下载和预处理 (MinMaxScaling)
├── models.py          # 存储 LSTM 和 GRU 的网络结构定义
├── agent.py           # 核心逻辑：接收Ticker -> 调度训练 -> 对比结果
├── main.py            # CLI 入口，处理用户输入 loop
└── requirements.txt   # 依赖列表
```