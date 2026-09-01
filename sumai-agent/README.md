# スマイエージェント — セットアップ & 起動手順

マルチAIエージェント型 住宅意思決定支援サービス（モック）。
LLMはローカルの Ollama を使用（外部APIキーは不要）。

## 必要なもの
- Python 3.10 以上
- [Ollama](https://ollama.com/)（ローカルで起動しておく）

## 初回セットアップ

```bash
# 1. sumai-agent ディレクトリに移動
cd sumai-agent

# 2. venv 作成
python -m venv venv

# 3. 依存ライブラリのインストール
# Windows
./venv/Scripts/python.exe -m pip install -r requirements.txt
# macOS / Linux
./venv/bin/python -m pip install -r requirements.txt

# 4. .env ファイルを作成（.env.example をコピー）
# Windows
copy .env.example .env
# macOS / Linux
cp .env.example .env
```

`.env` の `SUMAI_MODEL` が実際に使用するモデル名です（既定値は `qwen2.5:7b`）。変更する場合はここを書き換えてください。

## ⚡ 応答速度について

チャット1回の送信で LLM を **最大2〜3回**呼び出すため、モデルの大きさが応答時間に直結します。

| モデル | 目安応答時間（CPU） | 備考 |
|---|---|---|
| `qwen2.5:3b` | 15〜40秒 | 速度優先。GPU なし環境に最適 |
| `qwen2.5:7b` | 30〜90秒 | バランス型（**デフォルト**） |
| `qwen2.5:14b` | 2〜5分 | 高品質。GPU (VRAM 8GB+) 推奨 |

> **GPU非搭載の場合は `qwen2.5:3b` の使用を強く推奨します。**
> `.env` の `SUMAI_MODEL=qwen2.5:3b` に変更してください。

## Ollama モデルの取得

```bash
# 速度重視（GPU非搭載PC向け）
ollama pull qwen2.5:3b

# バランス重視（デフォルト）
ollama pull qwen2.5:7b

# Ollama が起動していてモデルが取得済みか確認
ollama list
```

## 起動

```bash
# Windows
./venv/Scripts/python.exe run.py
# macOS / Linux
./venv/bin/python run.py
```

サーバー起動後、ブラウザで以下にアクセス：

```
http://localhost:8000
```

## API エンドポイント

| エンドポイント | 説明 |
|---|---|
| `GET /` | チャット UI（ブラウザ） |
| `GET /api/health` | ヘルスチェック |
| `POST /api/chat` | チャット API |
| `GET /api/demo-presets` | デモ用プリセット一覧 |
| `GET /api/makers` | ハウスメーカー一覧（デモ用・13社） |
| `GET /docs` | Swagger UI（API ドキュメント） |

## テスト（LLM モック）

```bash
# Windows
./venv/Scripts/python.exe -m pytest tests/test_acceptance.py -v
```

## 使い方

1. ブラウザで `http://localhost:8000` を開く
2. 左パネルの「デモシナリオ」から試したいペルソナを選ぶ（または自由にチャット）
3. テキストエリアにメッセージを入力して「送信」
4. ヒアリングAI が要件を深掘りし、要件が揃ったら自動で間取り3案を生成
5. 間取り生成後、**メーカー推薦AI** が要件・間取りをもとに最適なハウスメーカー・ポータルを最大3件推薦

## ディレクトリ構成

```
sumai-agent/
├── app/
│   ├── agents/
│   │   ├── orchestrator.py    # オーケストレーターAI（LangGraph）
│   │   ├── hearing_agent.py   # ヒアリングAI
│   │   ├── planning_agent.py  # 間取り生成AI
│   │   └── maker_agent.py     # メーカー推薦AI ★NEW
│   ├── api/
│   │   └── chat.py            # FastAPI ルーター
│   ├── data/
│   │   └── demo_data.py       # デモ用サンプルデータ（メーカー13社収録）
│   ├── schemas/
│   │   ├── chat.py            # チャット関連スキーマ
│   │   ├── floorplan.py       # 間取り関連スキーマ
│   │   ├── requirements.py    # 要件関連スキーマ
│   │   └── maker.py           # メーカー推薦スキーマ ★NEW
│   ├── tools/                 # エージェント用ツール（今後拡張）
│   └── main.py                # FastAPI アプリ本体
├── frontend/
│   └── index.html             # チャット UI
├── tests/
│   └── test_acceptance.py     # MVP 受入基準テスト
├── run.py                     # 起動スクリプト
├── requirements.txt
├── .env.example
└── README.md
```

## マルチエージェント構成

```
ユーザー（ブラウザ）
    ↕ チャット
オーケストレーターAI  ─→  ヒアリングAI（要件整理）
                     ─→  間取り生成AI（3案生成）
                     ─→  メーカー推薦AI（最大3社推薦）★NEW
```

### 収録メーカー・ポータル（13社）

| カテゴリ | 名称 |
|---|---|
| プレミアム | 積水ハウス・大和ハウス工業・住友林業・ヘーベルハウス・ミサワホーム・パナソニック ホームズ |
| ミドル | 住友不動産・トヨタホーム |
| ローコスト | タマホーム・アイダ設計 |
| 情報ポータル | SUUMO・カナリー・LIFULL HOME'S |

## Phase 2 予定機能

- ④ 概算見積AI（坪単価ベース）
- ⑤ 法規チェックAI（建ぺい率・容積率等）
- ⑥ 来場予約連携（外部サイト）
- 不動産情報ライブラリ API 連携
