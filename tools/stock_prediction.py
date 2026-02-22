from typing import Any, Dict, List
from tools import YA_MCPServer_Tool
from core.train2 import run_analysis
import traceback
import mcp.types as types
import base64
import os


@YA_MCPServer_Tool(
    name="prediction_stock_price",
    title="Stock Price Prediction (LSTM/GRU)",
    description="训练深度学习模型(LSTM/GRU)并预测下一日价格。参数: ticker(股票代码), lookback(回溯天数,默认30), epochs(训练轮数,默认50)。"
)
def predict_stock_price(ticker: str, lookback: int = 30, epochs: int = 50) -> list:
    """
    对指定股票进行深度学习建模和预测。
    :param ticker: 股票代码 (如 'AAPL', '005930 KS')
    :param lookback: 历史回溯窗口大小 (即用过去多少天的数据预测)，默认为 30。
    :param epochs: 模型训练迭代轮数，默认为 50。
    """
    try:
        result = run_analysis(ticker=ticker, lookback=lookback, epochs=epochs)

        # 构建文本报告
        report_text = (
            f"### 📈 预测报告: {result['ticker']}\n"
            f"- **优胜模型**: {result['winner_model']} (MSE: {result['accuracy_mse']:.5f})\n"
            f"- **数据截止**: {result['last_actual_date']}\n"
            f"- **预测日期**: {result['next_prediction_date']}\n"
            f"- **预测价格**: ${result['predicted_price']:.2f}\n"
        )

        content: List[types.TextContent | types.ImageContent] = [
            types.TextContent(type="text", text=report_text)
        ]

        # 读取生成的图片并转换为 ImageContent
        plot_path = result.get('plot_path')
        if plot_path and os.path.exists(plot_path):
            with open(plot_path, "rb") as img_file:
                img_data = base64.b64encode(img_file.read()).decode('utf-8')

            content.append(types.ImageContent(
                type="image",
                data=img_data,
                mimeType="image/png"
            ))
        else:
            content[0].text += "\n*(Chart image not found)*"

        return content

    except ValueError as ve:
        return [types.TextContent(type="text", text=f"❌ 数据错误: {str(ve)}")]
    except Exception as e:
        traceback.print_exc()
        return [types.TextContent(type="text", text=f"❌ 分析失败: {str(e)}")]
