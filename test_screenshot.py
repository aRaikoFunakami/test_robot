import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
import base64
from PIL import Image
import io

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
        select_platform = next(t for t in tools if t.name == "select_platform")
        create_session = next(t for t in tools if t.name == "create_session")
        screenshot_tool = next(t for t in tools if t.name == "appium_screenshot")

        print("select_platform 実行...")
        platform = await select_platform.ainvoke({"platform": "android"})
        print("select_platform結果:", platform)

        print("create_session 実行...")
        session = await create_session.ainvoke({"platform": "android"})
        print("create_session結果:", session)

        print("screenshot_tool 実行...")
        screenshot = await screenshot_tool.ainvoke({})
        print("screenshot_tool 結果:", screenshot[:100])
        
        if isinstance(screenshot, str) and screenshot:
            img_bytes = base64.b64decode(screenshot)
            with open("test_screenshot.png", "wb") as f:
                f.write(img_bytes)
            img = Image.open(io.BytesIO(img_bytes))
            if img.mode == "RGBA":
                img = img.convert("RGB")
            img.save("test_screenshot.jpg", "JPEG")
        
        print("test_screenshot.png と test_screenshot.jpg 保存完了")

if __name__ == "__main__":
    asyncio.run(main())
