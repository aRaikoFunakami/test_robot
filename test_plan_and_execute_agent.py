"""
LangGraph: Plan-and-Execute Agent Example (Refactored for Testing)
参考: https://langchain-ai.github.io/langgraph/tutorials/plan-and-execute/plan-and-execute/#conclusion

このテスト用のリファクタリング版では、MCPセッション内ですべての処理を実行します。
"""
import operator
from typing import Annotated, List, Tuple, Union
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
import asyncio
from colorama import Fore, init

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, START, END
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_core.messages import HumanMessage, SystemMessage
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
    },
    "jarvis-appium-sse": {
        "url": "http://localhost:7777/sse",
        "transport": "sse",
    },
}

init(autoreset=True)


# --- 状態定義 ---
class PlanExecute(TypedDict):
    input: str
    plan: List[str]
    past_steps: Annotated[List[Tuple], operator.add]
    response: str

# --- プランモデル ---
class Plan(BaseModel):
    steps: List[str] = Field(description="different steps to follow, should be in sorted order")

# --- 応答モデル ---
class Response(BaseModel):
    response: str

class Act(BaseModel):
    action: Union[Response, Plan] = Field(description="Action to perform. If you want to respond to user, use Response. If you need to further use tools to get the answer, use Plan.")

# --- シンプルなプランナークラス ---
class SimplePlanner:
    """テスト用のシンプルなプランナー"""
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    async def create_plan(self, user_input: str, locator: str = "", image_url: str = "") -> Plan:
        content = f"""For the given objective, come up with a simple step by step plan.
This plan should involve individual tasks, that if executed correctly will yield the correct answer.
Do not add any superfluous steps. The result of the final step should be the final answer.
Make sure that each step has all the information needed - do not skip steps.

Objective: {user_input}"""
        
        if locator:
            content += f"\n\nScreen locator information: {locator}"
        
        messages = [SystemMessage(content=content)]
        
        if image_url:
            messages.append(HumanMessage(content=[
                {"type": "text", "text": "Based on this screen, create the plan."},
                {"type": "image_url", "image_url": {"url": image_url}}
            ]))
        else:
            messages.append(HumanMessage(content="Create the plan for this objective."))
        
        structured_llm = self.llm.with_structured_output(Plan)
        plan = await structured_llm.ainvoke(messages)
        return plan
    
    async def replan(self, state: PlanExecute, locator: str = "", image_url: str = "") -> Act:
        content = f"""Your objective was: {state["input"]}
Your original plan was: {str(state["plan"])}
You have currently done the follow steps: {str(state["past_steps"])}

Update your plan accordingly. If an error occurred in the previous step, take the error into account and come up with an alternative approach.
If there are any steps remaining, always return them as a Plan.
Only use Response if absolutely no steps remain and the user can be informed of completion."""
        
        if locator:
            content += f"\n\nCurrent screen locator information: {locator}"
        
        messages = [SystemMessage(content=content)]
        
        if image_url:
            messages.append(HumanMessage(content=[
                {"type": "text", "text": "Based on the current screen state, should I continue with the plan or provide a response?"},
                {"type": "image_url", "image_url": {"url": image_url}}
            ]))
        else:
            messages.append(HumanMessage(content="Should I continue with the plan or provide a response?"))
        
        structured_llm = self.llm.with_structured_output(Act)
        act = await structured_llm.ainvoke(messages)
        return act

# --- ヘルパー関数 ---
async def generate_screen_info(screenshot_tool, generate_locators):
    """スクリーンショットとロケーター情報を取得する"""
    print("screenshot_tool 実行...")
    screenshot = await screenshot_tool.ainvoke({})
    print("screenshot_tool 結果:", screenshot[:100] if screenshot else "No screenshot")

    print("generate_locators 実行...")
    locator = await generate_locators.ainvoke({})
    print("generate_locators 結果:", locator[:100] if locator else "No locator")

    if not screenshot:
        return str(locator), ""

    try:
        img_bytes = base64.b64decode(screenshot)
        img = Image.open(io.BytesIO(img_bytes))
        if img.mode == "RGBA":
            img = img.convert("RGB")
        
        # 横幅1280px以上ならリサイズ
        if img.width > 1280:
            ratio = 1280 / img.width
            new_size = (1280, int(img.height * ratio))
            img = img.resize(new_size, Image.LANCZOS)

        # Vision API用にJPEG形式でbase64化
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        img_bytes_jpeg = buf.getvalue()
        image_url = "data:image/jpeg;base64," + base64.b64encode(img_bytes_jpeg).decode()
        
        return str(locator), image_url
    except Exception as e:
        print(f"画像処理エラー: {e}")
        return str(locator), ""

# --- ワークフロー関数の定義 ---
def create_workflow_functions(planner: SimplePlanner, agent_executor, screenshot_tool, generate_locators):
    """ワークフロー関数を作成する（セッション内のツールを使用）"""
    
    async def execute_step(state: PlanExecute):
        plan = state["plan"]
        if not plan:
            return {"past_steps": [("error", "Plan is empty")]}
        
        plan_str = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(plan))
        task = plan[0]
        task_formatted = f"""For the following plan: {plan_str}\n\nYou are tasked with executing step 1: {task}. if calling tools, return the output of the tool calls directly. Do not add any extra commentary."""
        
        try:
            agent_response = await agent_executor.ainvoke(
                {"messages": [("user", task_formatted)]}
            )
            print(Fore.RED + f"Agent Response for step '{task}': {agent_response['messages'][-1].content}")
            return {
                "past_steps": [(task, agent_response["messages"][-1].content)],
            }
        except Exception as e:
            print(Fore.RED + f"Error in execute_step: {e}")
            return {"past_steps": [(task, f"Error: {str(e)}")]}

    async def plan_step(state: PlanExecute):
        try:
            locator, image_url = await generate_screen_info(screenshot_tool, generate_locators)
            plan = await planner.create_plan(state["input"], locator, image_url)
            print(Fore.GREEN + f"Generated Plan: {plan}")
            return {"plan": plan.steps}
        except Exception as e:
            print(Fore.RED + f"Error in plan_step: {e}")
            # フォールバック: 基本的なプランを作成
            basic_plan = await planner.create_plan(state["input"])
            return {"plan": basic_plan.steps}

    async def replan_step(state: PlanExecute):
        try:
            locator, image_url = await generate_screen_info(screenshot_tool, generate_locators)
            output = await planner.replan(state, locator, image_url)
            print(Fore.YELLOW + f"Replanner Output: {output}")
            
            if isinstance(output.action, Response):
                return {"response": output.action.response}
            else:
                return {"plan": output.action.steps}
        except Exception as e:
            print(Fore.RED + f"Error in replan_step: {e}")
            # エラーの場合は終了
            return {"response": f"エラーが発生しました: {str(e)}"}

    def should_end(state: PlanExecute):
        if "response" in state and state["response"]:
            return END
        else:
            return "agent"
    
    return execute_step, plan_step, replan_step, should_end

# --- メイン実行関数 ---
async def main():
    """MCPセッション内ですべての処理を実行するメイン関数"""
    config = {"recursion_limit": 50}
    past_steps = []

    client = MultiServerMCPClient(SERVER_CONFIG)
    async with client.session("jarvis-appium-sse") as session:
        # ツールを取得
        tools = await load_mcp_tools(session)

        # 必要なツールを取得
        select_platform = next(t for t in tools if t.name == "select_platform")
        create_session = next(t for t in tools if t.name == "create_session")
        screenshot_tool = next(t for t in tools if t.name == "appium_screenshot")
        generate_locators = next(t for t in tools if t.name == "generate_locators")

        # プラットフォーム選択とセッション作成
        print("select_platform 実行...")
        platform = await select_platform.ainvoke({"platform": "android"})
        print("select_platform結果:", platform)
        past_steps.append(("select_platform", str(platform)))

        print("create_session 実行...")
        session_result = await create_session.ainvoke({"platform": "android"})
        print("create_session結果:", session_result)
        past_steps.append(("create_session", str(session_result)))

        # エージェントエグゼキューターを作成
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        prompt = "You are a helpful assistant."
        agent_executor = create_react_agent(llm, tools, prompt=prompt)

        # プランナーを作成
        planner = SimplePlanner()

        # ワークフロー関数を作成（セッション内のツールを使用）
        execute_step, plan_step, replan_step, should_end = create_workflow_functions(
            planner, agent_executor, screenshot_tool, generate_locators
        )

        # ワークフローを構築
        workflow = StateGraph(PlanExecute)
        workflow.add_node("planner", plan_step)
        workflow.add_node("agent", execute_step)
        workflow.add_node("replan", replan_step)
        workflow.add_edge(START, "planner")
        workflow.add_edge("planner", "agent")
        workflow.add_edge("agent", "replan")
        workflow.add_conditional_edges("replan", should_end, ["agent", END])
        app = workflow.compile()

        # 実行
        knowhow = "ユーザーや計画が「Enter（エンター）を押す」と指示した場合、必ずソフトウェアキーボードを呼び出して、Enterキークリックしなさい。矢印アイコン（→）や (↵) がエンターキーである場合が多いです。"
        inputs = {
            "input": knowhow + "Androidで動作するChromeを起動して、URLバーにyahoo.co.jpを入力する。エンターキーで確定し、yahooのトップページを表示してください。すべて日本語で回答してください。",
            "past_steps": past_steps,
        }
        
        print(Fore.CYAN + "=== Plan-and-Execute Agent 開始 ===")
        try:
            async for event in app.astream(inputs, config=config):
                for k, v in event.items():
                    if k != "__end__":
                        print(Fore.BLUE + str(v))
        except Exception as e:
            print(Fore.RED + f"実行中にエラーが発生しました: {e}")
        finally:
            print(Fore.CYAN + "=== Plan-and-Execute Agent 終了 ===")

if __name__ == "__main__":
    asyncio.run(main())
