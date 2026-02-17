from typing import Any, Dict
from tools import YA_MCPServer_Tool
from core.train2 import run_analysis
import traceback


@YA_MCPServer_Tool(
    name="prediction_stock_price",
    title="Stock Price Prediction (LSTM/GRU)",
    description="训练深度学习模型(LSTM/GRU)并预测指定股票的下一日价格。输入股票代码(如 'AAPL' 或 '005930 KS')。"
)
def predict_stock_price(ticker: str) -> str:
    """
    对指定股票进行深度学习建模和预测。
    流程：加载数据 -> 训练双模型 -> 择优 -> 预测。
    """
    try:
        result = run_analysis(ticker=ticker)

        return (
            f"### 📈 预测报告: {result['ticker']}\n"
            f"- **优胜模型**: {result['winner_model']} (MSE: {result['accuracy_mse']:.5f})\n"
            f"- **数据截止**: {result['last_actual_date']}\n"
            f"- **预测日期**: {result['next_prediction_date']}\n"
            f"- **预测价格**: ${result['predicted_price']:.2f}\n\n"
            f"![Loss Curve]({result['plot_path']})"
        )

    except ValueError as ve:
        return f"❌ 数据错误: {str(ve)}"
    except Exception as e:
        traceback.print_exc()
        return f"❌ 分析失败: {str(e)}"
