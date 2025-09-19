import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

SERVER_CONFIG = {
    "jarvis-appium": {
        "command": "/opt/homebrew/opt/node@20/bin/npx",
        "args": ["-y", "jarvis-appium"],
        "transport": "stdio",
        "env": {
            "CAPABILITIES_CONFIG": "/Users/raiko.funakami/GitHub/test_robot/capabilities.json",
            "ANDROID_HOME_SDK_ROOT": "/Users/raiko.funakami/Library/Android/sdk",
            "ANDROID_SDK_ROOT": "/Users/raiko.funakami/Library/Android/sdk",
        }
    }
}

async def main():
    client = MultiServerMCPClient(SERVER_CONFIG)
    async with client.session("jarvis-appium") as session:
        tools = await load_mcp_tools(session)
        print("jarvis-appium ツール一覧:")
        for t in tools:
            print(f"- {t.name}")

if __name__ == "__main__":
    asyncio.run(main())
