from resources import YA_MCPServer_Resource
from typing import Any


@YA_MCPServer_Resource(
    "file:///config.yaml",  # 资源 URI
    name="server_files",  # 资源 ID
    title="Server Files",  # 资源标题
    description="返回配置文件内容",  # 资源描述
)
def get_config() -> Any:
    """
    返回配置文件的内容。

    Args:
        path (str): 文件路径。

    Returns:
        Any: 文件内容。
    """
    try:
        with open("config.yaml", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return {"error": "File not found"}


@YA_MCPServer_Resource(
    "file:///{path}",  # 资源模板 URI
    name="server_files",  # 资源 ID
    title="Server Files",  # 资源标题
    description="返回文件内容",  # 资源描述
)
def server_files(path: str) -> Any:
    """
    返回服务器文件的内容。
    这个资源可以被 MCP 客户端访问以获取文件内容。

    Args:
        path (str): 文件路径。

    Returns:
        Any: 文件内容。
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return {"error": "File not found"}
