# スマイエージェント — セットアップ & 起動手順

## 必要なもの
- Python 3.10 以上
- Anthropic API キー（https://console.anthropic.com/）

## 初回セットアップ

```bash
# 1. sumai-agent ディレクトリに移動
cd sumai-agent

# 2. 依存ライブラリのインストール
pip install -r requirements.txt

# 3. .env ファイルを作成（.env.example をコピーして編集）
copy .env.example .env
# .env を開き、ANTHROPIC_API_KEY=sk-ant-... を設定する
```

## 起動

```bash
# sumai-agent ディレクトリで実行
python run.py
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
| `GET /api/makers` | ハウスメーカー一覧（デモ用） |
| `GET /docs` | Swagger UI（API ドキュメント） |

## テスト（LLM モック）

```bash
python -m pytest tests/test_acceptance.py -v
```

## 使い方

1. ブラウザで `http://localhost:8000` を開く
2. 左パネルの「デモシナリオ」から試したいペルソナを選ぶ（または自由にチャット）
3. テキストエリアにメッセージを入力して「送信」
4. ヒアリングAI が要件を深掘りし、要件が揃ったら自動で間取り3案を生成

## ディレクトリ構成

```
sumai-agent/
├── app/
│   ├── agents/
│   │   ├── orchestrator.py    # オーケストレーターAI（LangGraph）
│   │   ├── hearing_agent.py   # ヒアリングAI
│   │   └── planning_agent.py  # 間取り生成AI
│   ├── api/
│   │   └── chat.py            # FastAPI ルーター
│   ├── data/
│   │   └── demo_data.py       # デモ用サンプルデータ
│   ├── schemas/
│   │   └── models.py          # Pydantic スキーマ
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

## マルチエージェント構成（MVP）

```
ユーザー（ブラウザ）
    ↕ チャット
オーケストレーターAI  ─→  ヒアリングAI（要件整理）
                     ─→  間取り生成AI（3案生成）
```

## Phase 2 予定機能

- ④ 概算見積AI（坪単価ベース）
- ⑤ 法規チェックAI（建ぺい率・容積率等）
- ⑥ メーカー推薦AI（来場予約 CTA）
- 不動産情報ライブラリ API 連携
