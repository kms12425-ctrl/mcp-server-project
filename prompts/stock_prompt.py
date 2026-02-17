from prompts import YA_MCPServer_Prompt


@YA_MCPServer_Prompt(
    name="analyze_stock_prediction",
    title="Stock Prediction Analysis",
    description="Template for analyzing a stock using the deep learning prediction tool."
)
def analyze_stock_prompt(ticker: str) -> str:
    """
    生成一个标准化的股票分析请求。
    """
    # 直接返回 Prompt 文本内容，FastMCP 会自动封装为 message
    return (
        f"Please verify the data for {ticker} by checking resource 'stock://{ticker}/latest', "
        f"and then run the 'prediction_stock_price' tool to train the model and predict the price for the next trading day."
    )
