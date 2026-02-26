# 指令模板集 (Prompts)
本目录包含 YA MCP Server 中用于生成标准化指令的 Prompt 函数，所有 Prompt 均通过 `@YA_MCPServer_Prompt` 装饰器注册，用于指导 MCP 服务器执行指定逻辑。

## Prompt 列表
| 指令名称 | 功能描述 | 输入参数 | 输出格式 | 函数类型 |
| :------: | :------: | :-------: | :------: | :------: |
| greet_user | 生成个性化用户问候语 | name (用户名) | 字符串（中文问候语，如“你好，Alice！欢迎使用 YA MCP Server。”） | 异步函数 (async) |
| analyze_stock_prediction | 生成股票分析标准化指令 | ticker (股票代码) | 字符串（英文指令文本，含资源调用和工具执行要求） | 同步函数 |


## Prompt 调用示例
```python
from prompts import hello_prompt, analyze_stock_prompt

# 生成问候语
greeting = await hello_prompt(name="Alice")

# 生成股票分析指令
analysis_prompt = analyze_stock_prompt(ticker="AAPL")
```


