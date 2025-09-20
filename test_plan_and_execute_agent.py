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
    replan_count: int  # リプラン回数の追跡

# --- プランモデル ---
class Plan(BaseModel):
    steps: List[str] = Field(description="実行すべき手順の一覧（順序通りに並べる）")

# --- 応答モデル ---
class Response(BaseModel):
    response: str

class Act(BaseModel):
    action: Union[Response, Plan] = Field(description="実行するアクション。ユーザーに応答する場合はResponse、さらにツールを使用してタスクを実行する場合はPlanを使用してください。")

# --- シンプルなプランナークラス ---
class SimplePlanner:
    """テスト用のシンプルなプランナー"""
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4.1", temperature=0)
    
    async def create_plan(self, user_input: str, locator: str = "", image_url: str = "") -> Plan:
        content = f"""与えられた目標に対して、シンプルなステップバイステップの計画を作成してください。
この計画は、正しく実行されれば正解を得られる個別のタスクで構成される必要があります。
不要なステップは追加しないでください。最終ステップの結果が最終的な答えとなります。
各ステップに必要な情報がすべて含まれていることを確認し、ステップを飛ばさないでください。

目標: {user_input}"""
        
        if locator:
            content += f"\n\n画面ロケーター情報: {locator}"
        
        messages = [SystemMessage(content=content)]
        
        if image_url:
            messages.append(HumanMessage(content=[
                {"type": "text", "text": "この画面に基づいて計画を作成してください。"},
                {"type": "image_url", "image_url": {"url": image_url}}
            ]))
        else:
            messages.append(HumanMessage(content="この目標のための計画を作成してください。"))
        
        structured_llm = self.llm.with_structured_output(Plan)
        plan = await structured_llm.ainvoke(messages)
        return plan
    
    async def replan(self, state: PlanExecute, locator: str = "", image_url: str = "") -> Act:
        content = f"""あなたの目標: {state["input"]}
元の計画: {str(state["plan"])}
現在完了したステップ: {str(state["past_steps"])}

重要な指示:
1. メインの目標が完全に達成されているかを必ず分析してください
2. メインの目標を完了するために残りのステップがある場合は、必ず残りのステップを含むPlanを返してください
3. 全体の目標が100%完了し、これ以上のアクションが不要な場合のみResponseを返してください
4. 次に必要なアクションが見えているだけでResponseを返さないでください - 実際にそれを行うためのPlanを提供してください
5. 次に取るべきアクションが見える場合は、それをPlanに含めてください

前のステップでエラーが発生した場合は、それを考慮して代替アプローチを考えてください。

覚えておいてください: あなたの仕事は、現在の状態を観察するだけでなく、実行可能なステップを提供することです。"""
        
        if locator:
            content += f"\n\n現在の画面ロケーター情報: {locator}"
        
        messages = [SystemMessage(content=content)]
        
        if image_url:
            messages.append(HumanMessage(content=[
                {"type": "text", "text": "現在の画面状態に基づいて、目標を完了するための残りのステップは何ですか？残りのステップがある場合はPlanとして返してください。目標が完全に達成された場合のみResponseを使用してください。"},
                {"type": "image_url", "image_url": {"url": image_url}}
            ]))
        else:
            messages.append(HumanMessage(content="目標を完了するための残りのステップは何ですか？残りのステップがある場合はPlanとして返してください。"))
        
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
def create_workflow_functions(planner: SimplePlanner, agent_executor, screenshot_tool, generate_locators, max_replan_count: int = 5):
    """ワークフロー関数を作成する（セッション内のツールを使用）
    
    Args:
        max_replan_count: 最大リプラン回数（デフォルト5回）
    """
    
    async def execute_step(state: PlanExecute):
        plan = state["plan"]
        if not plan:
            return {"past_steps": [("error", "計画が空です")]}
        
        plan_str = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(plan))
        task = plan[0]
        task_formatted = f"""以下の計画について: {plan_str}\n\nあなたはステップ1の実行を担当します: {task}。ツールを呼び出す場合は、ツール呼び出しの出力を直接返してください。余計なコメントは追加しないでください。"""
        
        try:
            agent_response = await agent_executor.ainvoke(
                {"messages": [("user", task_formatted)]}
            )
            print(Fore.RED + f"ステップ '{task}' のエージェント応答: {agent_response['messages'][-1].content}")
            return {
                "past_steps": [(task, agent_response["messages"][-1].content)],
            }
        except Exception as e:
            print(Fore.RED + f"execute_stepでエラー: {e}")
            return {"past_steps": [(task, f"エラー: {str(e)}")]}

    async def plan_step(state: PlanExecute):
        try:
            locator, image_url = await generate_screen_info(screenshot_tool, generate_locators)
            plan = await planner.create_plan(state["input"], locator, image_url)
            print(Fore.GREEN + f"生成された計画: {plan}")
            return {"plan": plan.steps, "replan_count": 0}  # 初期化時はreplan_countを0に設定
        except Exception as e:
            print(Fore.RED + f"plan_stepでエラー: {e}")
            # フォールバック: 基本的なプランを作成
            basic_plan = await planner.create_plan(state["input"])
            return {"plan": basic_plan.steps, "replan_count": 0}

    async def replan_step(state: PlanExecute):
        current_replan_count = state.get("replan_count", 0)
        
        # リプラン回数制限チェック
        if current_replan_count >= max_replan_count:
            print(Fore.YELLOW + f"リプラン回数が制限に達しました（{max_replan_count}回）。処理を終了します。")
            return {
                "response": f"リプラン回数が制限（{max_replan_count}回）に達したため、処理を終了しました。現在の進捗: {len(state['past_steps'])}ステップ完了。",
                "replan_count": current_replan_count + 1
            }
        
        try:
            locator, image_url = await generate_screen_info(screenshot_tool, generate_locators)
            output = await planner.replan(state, locator, image_url)
            print(Fore.YELLOW + f"Replanner Output (replan #{current_replan_count + 1}): {output}")
            
            if isinstance(output.action, Response):
                return {
                    "response": output.action.response,
                    "replan_count": current_replan_count + 1
                }
            else:
                return {
                    "plan": output.action.steps,
                    "replan_count": current_replan_count + 1
                }
        except Exception as e:
            print(Fore.RED + f"Error in replan_step: {e}")
            # エラーの場合は終了
            return {
                "response": f"エラーが発生しました: {str(e)}",
                "replan_count": current_replan_count + 1
            }

    def should_end(state: PlanExecute):
        # レスポンスがある場合は終了
        if "response" in state and state["response"]:
            return END
            
        # それ以外は継続（replan制限チェックはreplan_step内で行う）
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
        llm = ChatOpenAI(model="gpt-4.1", temperature=0)
        prompt = "あなたは親切なアシスタントです。与えられたタスクを正確に実行してください。"
        agent_executor = create_react_agent(llm, tools, prompt=prompt)

        # プランナーを作成
        planner = SimplePlanner()

        # ワークフロー関数を作成（セッション内のツールを使用）
        max_replan_count = 10
        execute_step, plan_step, replan_step, should_end = create_workflow_functions(
            planner, agent_executor, screenshot_tool, generate_locators, max_replan_count
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
        knowhow = """
        1. アプリを実行するときは `appium_activate_app` ツールを使用します。
        例えば:
            await appium_activate_app.ainvoke({"id": "com.android.chrome"})
        2. エンターキーを最後に入力して確定させる場合には、`appium_set_value()` を使う時に最後に '\n' を追加しますを使用します。
        例えば:
            await appium_set_value.ainvoke({"args.elementUUID": "xxxx", "args.text": 'www.google.com\n'})
        """
        #query = "Androidで動作するChromeを起動して、メニューを開いて、新しいタブを開く。すべて日本語で回答してください。"
        query = "Androidで動作するChromeを起動して、yahoo.co.jp を開いてください"
        inputs = {
            "input": knowhow + query,
            "past_steps": past_steps,
            "replan_count": 0  # 初期化
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
