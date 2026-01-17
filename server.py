from mcp.server.fastmcp import FastMCP

# 初始化 Server，名字可以自定义
mcp = FastMCP("MyFirstAgent")

# 定义一个简单的工具 (Tool)
# 智能体可以通过调用这个函数来执行操作


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

# 定义一个动态资源 (Resource)
# 智能体可以读取这里返回的数据


@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """Get a personalized greeting"""
    return f"Hello, {name}!"


if __name__ == "__main__":
    # 运行 Server
    mcp.run()
