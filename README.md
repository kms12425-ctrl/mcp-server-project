# My First MCP Agent

这是一个基于 Python 的 MCP (Model Context Protocol) Server 基础项目。

## 1. 环境准备

项目已包含虚拟环境 `.venv`。如果需要重新安装依赖：

```powershell
# 激活虚拟环境
.venv\Scripts\activate

# 安装依赖
pip install mcp
```

## 2. 配置 Claude Desktop

要让 Claude Desktop 使用这个 Server，请编辑您的配置文件：

- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
- **Mac**: `~/Library/Application Support/Claude/claude_desktop_config.json`

将以下内容添加到配置文件中（请确保路径正确）：

```json
{
  "mcpServers": {
    "my-agent": {
      "command": "c:/Users/Owner/code/project/MCP server/.venv/Scripts/python.exe",
      "args": ["c:/Users/Owner/code/project/MCP server/server.py"]
    }
  }
}
```

> **注意**: 请检查上述路径是否与您实际的文件路径完全匹配。

## 3. 测试

1. 重启 Claude Desktop。
2. 在输入框右侧若看到插头图标 🔌，说明连接正常。
3. 尝试发送消息："请帮我计算 100 + 200"。
4. Claude 应该会调用 `add` 工具并返回 300。
