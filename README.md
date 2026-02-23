## YA_MCPServer

[一句话功能简介]

### 组员信息

| 姓名 | 学号 | 分工 | 备注 |
| :------: | :------: | :--: | :--: |
|   马俊豪   | U202490042     |      |      |
|   胡志锋   | U202414748     |      |      |
|   陈润泽   | U202414773     |      |      | 

### Tool 列表

| 工具名称 | 功能描述 | 输入 | 输出 | 备注 |
| :------: | :------: | :--: | :--: | :--: |
|          |          |      |      |      |
|          |          |      |      |      |
|          |          |      |      |      |

### Resource 列表

| 资源名称 | 功能描述 | 输入 | 输出 | 备注 |
| :------: | :------: | :--: | :--: | :--: |
|          |          |      |      |      |
|          |          |      |      |      |
|          |          |      |      |      |

### Prompts 列表

| 指令名称 | 功能描述 | 输入 | 输出 | 备注 |
| :------: | :------: | :--: | :--: | :--: |
|          |          |      |      |      |
|          |          |      |      |      |
|          |          |      |      |      |

### 项目结构

- `core`: [XXXX]
- `tools`: [XXXX]
- `config.yaml`: [XXXX(添加 XX 额外配置)]
- [XXXX(其他新添加的文件与目录介绍)]

### 其他需要说明的情况

- 在 `sops` 模块中添加的密钥变量分别用于什么功能
- 是否使用了 PyTorch、Tensorflow 等深度学习框架
- 是否使用了机器学习、深度学习模型

### 快速开始

#### 1. 环境准备

确保系统已安装 Python 3.10 或更高版本。

#### 2. 创建并激活虚拟环境

Windows:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Linux/macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### 3. 安装依赖

安装项目及其依赖（使用 editable 模式，方便开发调试）：
```bash
pip install -e .
```

#### 4. 配置 Sops 密钥（可选）

如果项目使用了加密配置，请确保已配置好 `sops` 和相关密钥。

#### 5. 启动 MCP Server

运行以下命令启动服务：

```bash
npx @modelcontextprotocol/inspector python server.py
```

或者使用 `uvicorn` (如果配置为 SSE 模式)：

```bash
uvicorn server:app --reload
```

#### 6. 配置 MCP 客户端 (Claude Desktop)

在 `claude_desktop_config.json` 中添加如下配置：

```json
{
  "mcpServers": {
    "ya-mcp-server": {
      "command": "python",
      "args": [
        "C:\\Users\\Owner\\code\\project\\mcp-server-project\\server.py"
      ]
    }
  }
}
```
**注意**：请根据你的实际项目路径修改 `args` 中的路径。

