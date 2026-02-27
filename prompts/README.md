[中文](#中文) | [English](#english)

<a id="中文"></a>
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

---

<a id="english"></a>
# Prompt Template Set (Prompts)
This directory contains Prompt functions in YA MCP Server used to generate standardized instructions. All Prompts are registered via the `@YA_MCPServer_Prompt` decorator and are used to guide the MCP server to execute specific logic.

## Prompt List
| Prompt Name | Description | Input Parameters | Output Format | Function Type |
| :---: | :---: | :---: | :---: | :---: |
| greet_user | Generates personalized user greeting | name (username) | String (Chinese greeting, e.g., "你好, Alice! 欢迎使用 YA MCP Server.") | Asynchronous function (async) |
| analyze_stock_prediction | Generates standardized stock analysis instructions | ticker (stock code) | String (English instruction text, containing resource calls and tool execution requirements) | Synchronous function |

## Prompt Call Example
```python
from prompts import hello_prompt, analyze_stock_prompt

# Generate greeting
greeting = await hello_prompt(name="Alice")

# Generate stock analysis instruction
analysis_prompt = analyze_stock_prompt(ticker="AAPL")
```


