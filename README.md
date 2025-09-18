# Appium Mobile Test Agent

LangGraphのReActエージェントとmcp-appium-visualを使用したAndroidデバイスのテスト自動化アプリケーション

## 概要

このアプリケーションは以下の技術を組み合わせて、自然言語でAndroidデバイスのテスト自動化を実現します：

- **LangGraph**: ReActエージェントパターンによる推論と行動の組み合わせ
- **jarvis-appium**: Model Context Protocolを通じたAppiumツールの提供
- **OpenAI GPT-4o**: 自然言語理解と推論エンジン
- **EventLogger**: エージェントの動作ログとAllure統合

## 機能

- 自然言語による Androidデバイス操作指示
- Appiumを通じた実際のデバイス制御
- ReActパターンによる段階的推論と実行
- 詳細なログ出力とAllure統合
- インタラクティブモードと単発実行モード

## 必要な環境

### システム要件
- Python 3.13+
- Node.js
- Appiumサーバー
- Androidデバイスまたはエミュレーター

### 依存関係
- langchain
- langgraph
- langchain-openai
- langchain-mcp-adapters
- allure-pytest
- colorama
- appium-python-client

## セットアップ

### 1. リポジトリのクローン
```bash
git clone <repository-url>
cd test_robot
```

### 2. 依存関係のインストール
```bash
uv sync
```

### 3. 環境変数の設定
- エラー詳細

## 開発者向け情報

### カスタムツールの追加

`mcp_client.py`でMCPクライアント設定をカスタマイズできます。

### ログのカスタマイズ

# simple_chat.py を使った jarvis-appium の使い方

このリポジトリは、`simple_chat.py` を使って Model Context Protocol (MCP) サーバー「jarvis-appium」をインタラクティブに操作するためのサンプルです。

## 概要

- ユーザーが自然言語で Android デバイス操作を指示
- LangGraph の ReAct エージェントが指示を分解し、Appium ツールを自動選択
- EventLogger による詳細なイベントログ出力
- GPT-4o（画像・マルチモーダル対応）で高度な推論

## セットアップ手順

1. リポジトリのクローン
      ```bash
      git clone <repository-url>
      cd test_robot
      ```
2. 依存関係のインストール
      ```bash
      uv sync
      ```
3. OpenAI APIキーの設定
      ```bash
      export OPENAI_API_KEY="your-openai-api-key"
      ```
4. Node.jsとjarvis-appiumのインストール
      - https://github.com/AppiumTestDistribution/jarvis-appium
5. Androidデバイス/エミュレーターの準備
      - USBデバッグ有効化
      - `adb devices` で接続確認

## simple_chat.py の使い方

インタラクティブモードで起動：
```bash
uv run python simple_chat.py
```

プロンプト例：
```
>>> YouTubeアプリを起動して
>>> スクリーンショットを撮影して
>>> ChromeでGoogleを開いて
>>> 設定アプリでWi-FiをONにして
```

終了するには `exit` または `quit` を入力してください。

## 仕組み

- `simple_chat.py` は MultiServerMCPClient で jarvis-appium を起動
- ユーザー入力を ReAct エージェントに渡し、Appiumツール（initialize-appium, tap-element, appium-screenshot など）を自動選択
- EventLogger が各イベント（ツール呼び出し、LLM応答など）をリアルタイムで出力
- GPT-4oモデルを使うことで画像解析や複雑な推論も可能

## ファイル構成

```
test_robot/
├── simple_chat.py             # jarvis-appium用インタラクティブクライアント（推奨）
├── event_logger.py            # ログ機能とAllure統合
├── capabilities.json          # Appiumセッション設定
├── ...
```

## よくある質問

- **Q: 画像（スクリーンショット）をLLMに渡せますか？**
   - A: GPT-4oモデルなら画像入力に対応しています。gpt-4.1は画像非対応です。
- **Q: 任意のAndroidアプリを起動できますか？**
   - A: capabilities.json で設定したアプリのみ
- **Q: MCP/Appium/LLMのエラーはどこで確認できますか？**
   - A: EventLoggerの出力や標準出力に詳細なエラーが表示されます。

## ライセンス

Apache License 2.0

