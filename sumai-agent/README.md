# スマイエージェント — セットアップ & 起動手順

マルチAIエージェント型 住宅意思決定支援サービス（モック）。
LLMはローカルの Ollama を使用（外部APIキーは不要）。

> 📘 **初めての方は [`docs/法規チェックAI_ガイド.html`](docs/法規チェックAI_ガイド.html) をブラウザで開いてください。**
> 環境構築からデモ実行、法規チェックの判定内容、エージェントの追加方法まで通しで解説しています。

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

`.env` の `SUMAI_MODEL` が実際に使用するモデル名です（既定値は `qwen2.5:3b`。初期フェーズはチーム全員この値で揃えます）。変更する場合はここを書き換えてください。
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
# .env の SUMAI_MODEL に指定したモデルを取得（既定なら qwen2.5:3b）
ollama pull qwen2.5:3b
# 速度重視（GPU非搭載PC向け）
ollama pull qwen2.5:3b

# バランス重視（デフォルト）
ollama pull qwen2.5:7b

# Ollama が起動していてモデルが取得済みか確認
ollama list
```

## 起動

### 1. Ollama を起動する

```bash
# Windows 版 Ollama … インストール済みなら常駐しているので通常は不要
# WSL / Linux に手動で入れた場合は毎回起動する（自動起動しません）
ollama serve &

# 起動確認（{"version":"..."} が返る）
curl http://localhost:11434/api/version
```

### 2. アプリを起動する

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

## モデル選択と応答時間

LLM は `.env` の `SUMAI_MODEL` で切り替える。**初期フェーズはチーム全員 `qwen2.5:3b` に統一する**
（14b / 7b は CPU 推論では重すぎて、開発中の試行サイクルが回らない）。

| モデル | サイズ | CPU 推論での間取り生成（3案） | 用途 |
|---|---|---|---|
| `qwen2.5:3b` | 1.9 GB | 未実測（7b より速い見込み） | **既定・チーム統一。開発／動作確認用。品質は要検証** |
| `qwen2.5:7b` | 4.7 GB | 6 分（実測・14コアCPU／メモリ15GB） | 実測で間取りが 3 案 → 1 案に減った |
| `qwen2.5:14b` | 9.0 GB | 13〜39 分（実測・同一条件） | JSON 出力が安定。GPU がある環境の本番デモ用 |

- **3b は品質が要検証。** 7b の時点で「3案 → 1案」の指示追従の劣化が実測で出ているため、
  3b でも同様かそれ以上の劣化が起きうる。デモで 3 案提示が必要な場合は、
  下記の応答キャッシュを 14b で作成しておくか、その場面だけモデルを上げる
- Ollama が GPU を使うのは NVIDIA（CUDA）／AMD（ROCm）のみ。Intel 内蔵GPU では CPU 推論になる
- メモリが不足すると同じモデルでも所要時間が数倍にばらつく（14b は 16GB 以上を推奨）
- 応答時間はヒアリング1ターンで数分、間取り生成で数十分になり得るため、
  デモ収録時は事前に1度通して**モデルをメモリに載せた状態**にしておく（初回はモデルのロード時間が加算される）
- モデルは 5 分間呼ばれないと Ollama がメモリから降ろす。デモ中の解説で間が空くと再ロードが走るため、
  事前に `keep_alive` を延ばしておく：

  ```bash
  curl -s http://localhost:11434/api/generate \
    -d '{"model":"qwen2.5:3b","prompt":"1+1=","stream":false,"keep_alive":"2h"}' > /dev/null
  ```

**根本対策は次節の応答キャッシュ（`SUMAI_LLM_CACHE`）。** モデル選択やメモリ調整は待ち時間を
短くするだけで、CPU 推論である限り待ちは残る。デモの待ち時間を 0 にできるのはキャッシュのみ。
軽量モデルへの切り替えは開発の回転を上げるためのもので、品質要件（3 案提示）の担保にはならない。
**開発は 3b、デモは品質を確認したモデル ＋ キャッシュ**という使い分けにする。

## LLM応答のキャッシュ（NFR-06 オフライン再生 / NFR-01 再現性）

LLM の応答を一度実機で取得してリポジトリに保存し、2回目以降はそのキャッシュを返す。
**Ollama 停止中・ネットワーク断でもデモが完走し、同じ入力に対して常に同じ応答になる。**

`.env` の `SUMAI_LLM_CACHE` で切り替える：

| 値 | 動作 | 用途 |
|---|---|---|
| `off` | 常に実機のLLMを呼ぶ | 既定。開発時 |
| `auto` | キャッシュがあれば返し、無ければ実機を呼んで保存 | **デモ準備・デモ本番** |
| `replay` | キャッシュのみを使う。無ければエラー | デモ前の最終確認 |
| `record` | 常に実機を呼んでキャッシュを上書き | プロンプト変更後の作り直し |

### 手順

```bash
# 1. デモシナリオのキャッシュを作る（初回のみ・時間がかかるので裏で流す）
./venv/Scripts/python.exe scripts/record_demo.py --persona C

# 2. キャッシュだけで完走するか検証する（Ollama を止めてから実行して確実に確認する）
./venv/Scripts/python.exe scripts/record_demo.py --persona C --verify

# 3. .env を SUMAI_LLM_CACHE=auto にしてアプリを起動 → 即座に応答が返る
```

`scripts/record_demo.py` はターンごとの所要時間も出すため、**モデル比較のベンチにも使える**：

```bash
./venv/Scripts/python.exe scripts/record_demo.py --persona C --model qwen2.5:14b
```

### 注意点

- キャッシュキーは **モデル名 + JSONモード + 全会話履歴** のハッシュ。
  モデルを変えるとキャッシュは命中しない（14b の応答を 3b のデモで黙って返す事故を防ぐため）
- 同様に、**デモ当日の発話はキャッシュ作成時と一字一句同じである必要がある**。
  `scripts/record_demo.py` はヒアリング未完時に `提案に移ってください。` を送るため、
  ブラウザからも同じ文言を使う（`auto` なら外れても実機にフォールバックするだけで止まらない）
- プロンプトを変更するとキャッシュは無効になる。`record` で作り直す
- キャッシュには応答本文だけでなくプロンプトも残るため、何を保存したのかレビューできる

## API エンドポイント

| エンドポイント | 説明 |
|---|---|
| `GET /` | チャット UI（ブラウザ） |
| `GET /api/health` | ヘルスチェック |
| `POST /api/chat` | チャット API |
| `GET /api/artifact/{session_id}` | 機械可読アーティファクト（要件・間取り・敷地情報・法規チェック結果の統合 JSON） |
| `GET /api/demo-presets` | デモ用プリセット一覧 |
| `GET /api/makers` | ハウスメーカー一覧（デモ用・13社） |
| `GET /docs` | Swagger UI（API ドキュメント） |

## テスト（LLM モック・Ollama 不要）

```bash
# Windows
./venv/Scripts/python.exe -m pytest tests/ -v
```

| ファイル | 内容 |
|---|---|
| `tests/test_acceptance.py` | MVP 受入基準（AC-1〜3） |
| `tests/test_legal.py` | 法規チェックAI（LAW-1〜4） |
| `tests/test_pipeline.py` | エージェントパイプライン・応答合成・修正ループの停止性 |
| `tests/test_area_utils.py` | 面積表記のパースと坪/㎡ 取り違えの自動補正 |
| `tests/test_llm_cache.py` | LLM応答キャッシュ（オフライン再生の成立確認） |

## 使い方

1. ブラウザで `http://localhost:8000` を開く
2. 左パネルの「デモシナリオ」から試したいペルソナを選ぶ（または自由にチャット）
3. テキストエリアにメッセージを入力して「送信」
4. ヒアリングAI が要件を深掘りし、要件が揃ったら自動で間取り3案を生成 → 続けて法規チェックを実行

デモシナリオの見え方の違い：

| ペルソナ | 法規チェックの挙動 |
|---|---|
| **C：土地ありこだわり層** | 敷地面積・用途地域・道路幅員が発話に含まれるため、実データで数値判定が走る（**法規チェックの見せ場**） |
| A：初めて検討層（土地なし） | 敷地未確定のため標準的な敷地条件を仮定した参考判定＋「敷地確定後の再チェック」フラグ |
4. ヒアリングAI が要件を深掘りし、要件が揃ったら自動で間取り3案を生成
5. 間取り生成後、**メーカー推薦AI** が要件・間取りをもとに最適なハウスメーカー・ポータルを最大3件推薦

## ディレクトリ構成

```
sumai-agent/
├── app/
│   ├── agents/
│   │   ├── orchestrator.py    # オーケストレーターAI（LangGraph・ノード定義・ステップ宣言）
│   │   ├── state.py           # グラフ状態とエージェント間の受け渡し契約
│   │   ├── pipeline.py        # 専門エージェントの直列実行と拡張点（AgentStep）
│   │   ├── hearing_agent.py   # ヒアリングAI
│   │   ├── planning_agent.py  # 間取り生成AI
│   │   └── legal_agent.py     # 法規チェックAI
│   │   └── maker_agent.py     # メーカー推薦AI ★NEW
│   ├── api/
│   │   └── chat.py            # FastAPI ルーター
│   ├── data/
│   │   └── demo_data.py       # デモ用サンプルデータ（メーカー13社収録）
│   ├── schemas/
│   │   ├── chat.py            # チャット関連スキーマ
│   │   ├── floorplan.py       # 間取り関連スキーマ
│   │   ├── requirements.py    # 要件関連スキーマ
│   │   ├── legal.py           # 法規チェック関連スキーマ
│   │   └── artifact.py        # 機械可読アーティファクト
│   ├── tools/
│   │   ├── zoning.py          # 用途地域マスタ（建築基準法の法定値）
│   │   ├── legal_rules.py     # 法規計算ツール（決定論的な判定エンジン）
│   │   ├── area_utils.py      # 面積表記のパースと単位取り違えの補正
│   │   ├── llm_cache.py       # LLM応答のキャッシュ（NFR-06）
│   │   └── site_lookup.py     # 敷地照会ツール（外部API／プリセット／仮定値）
│   │   └── maker.py           # メーカー推薦スキーマ ★NEW
│   ├── tools/                 # エージェント用ツール（今後拡張）
│   └── main.py                # FastAPI アプリ本体
├── frontend/
│   └── index.html             # チャット UI
├── docs/
│   └── 法規チェックAI_ガイド.html  # 環境構築〜デモ実行の通し手順（ブラウザで開く）
├── fixtures/
│   └── llm/                   # LLM応答のキャッシュ（SUMAI_LLM_CACHE で利用）
├── scripts/
│   └── record_demo.py         # キャッシュ作成＋モデル別の所要時間計測
├── tests/
│   ├── test_acceptance.py     # MVP 受入基準テスト
│   ├── test_legal.py          # 法規チェックAI テスト
│   ├── test_pipeline.py       # パイプライン・応答合成テスト
│   ├── test_area_utils.py     # 面積正規化テスト
│   └── test_llm_cache.py      # 応答キャッシュのテスト
├── run.py                     # 起動スクリプト
├── requirements.txt
├── .env.example
└── README.md
```

## マルチエージェント構成

```
ユーザー（ブラウザ）
    ↕ チャット
オーケストレーターAI ─→ ヒアリングAI（要件整理）
                    ─→ 間取り生成AI（3案生成）
                    ─→ 法規チェックAI（建ぺい率・容積率・高さ判定＋要確認フラグ）
```

グラフ形状（`app/agents/orchestrator.py`）：

```
orchestrator ─┬─(間取り生成済み)→ follow_up ──────→ compose → END
              └─→ hearing ─┬─(要件不足)──────────→ compose → END
                           └─→ planning → legal ─→ compose → END
                                  ▲          │
                                  └──────────┘ 法規NG時の自律修正（既定OFF・最大1回）
```

## 法規チェックAI

要件定義書 §7.6（LAW-1〜4）に対応。**判定は決定論的計算**で行い、LLM は敷地情報の抽出にのみ使う。
同じ入力に対して常に同じ判定を返すため、デモの再現性（NFR-01）を損なわない。

| 区分 | 項目 |
|---|---|
| **自動判定**（LAW-2） | 建ぺい率（第53条）／容積率（第52条・**基準容積率 = min(指定容積率, 前面道路幅員×法定係数) を適用**）／絶対高さ制限（第55条）／用途制限（別表第二） |
| **要確認フラグ**（LAW-3） | 道路斜線・北側斜線・日影規制・外壁後退・防火/準防火地域の仕様規制・セットバック・接道義務・自治体条例 |
| **判定材料不足** | 数値が読み取れない項目は NG ではなく `unknown` として区別する |

- 建ぺい率・容積率の緩和（角地・防火地域・車庫の不算入等）は**見込まない＝安全側**の判定
- 建築面積は延床面積の 55%（2階建ての1階床面積相当）、高さは階高3m＋屋根2mで概算する
- 判定に用いた敷地情報の出典（ご入力内容／外部API／プリセット／仮定値）と仮定内容を必ず表示する
- **適法性は保証しない**（LAW-4）。全出力に免責を明記

### 敷地情報の取得（LAW-1）

`REINFOLIB_API_KEY` を設定すると不動産情報ライブラリ（国土交通省）API を照会し、
未設定・障害時は対象自治体のプリセット → 標準的な敷地条件の仮定値へ静かにフォールバックする
（NFR-06 オフライン再生：ネットワーク断でもデモが完走する）。

> 注意：同 API の用途地域データはタイル配信のため、住所からの点照会にはジオコーディングと
> point-in-polygon が必要。エンドポイントとレスポンス項目名は技術検証（スパイク）で確定させる前提で、
> 環境変数（`REINFOLIB_API_BASE` / `REINFOLIB_ZONING_ENDPOINT`）で差し替えられるようにしている。

### 自律修正ループ（既定OFF）

`SUMAI_LEGAL_AUTOFIX=1` にすると、法規NG時に上限値を制約として間取りを再生成し、再チェックする。
無限ループ防止のため最大1回で打ち切る（NFR-07 停止性）。間取り生成が2回走るため応答時間は約2倍。

## エージェントの追加方法（見積AI・メーカー推薦AI）

専門エージェントの実行順は `app/agents/orchestrator.py` の `POST_HEARING_STEPS` が唯一の宣言箇所。
グラフの分岐を書き換える必要はない。

```python
POST_HEARING_STEPS = [
    AgentStep("planning", "planning", planning_node),
    AgentStep("legal",    "legal",    legal_node, route_override=_legal_autofix_route),
    AgentStep("estimate", "estimate", estimate_node),   # ← 追加はこの1行
]
```

追加の手順：

1. `app/schemas/` に出力スキーマを追加
2. `app/agents/<name>_agent.py` に `run_xxx()`（構造化結果）と `build_xxx_reply()`（説明文）を実装
   — 既存エージェントと同様、LLM は引数で受け取る純関数にするとテストでモックできる
3. `app/agents/state.py` の `SumaiState` に結果の格納スロットを1行追加
4. `orchestrator.py` にノード関数を書き、`POST_HEARING_STEPS` に `AgentStep` を1行追加
   — ノードは結果をスロットに入れ、説明文を `section("<name>", "見出し", markdown)` で返す
   （応答全体の組み立てと免責の付与は `compose_node` が一括で行う）
5. `app/schemas/artifact.py` と `app/api/chat.py` の `ChatResponse` に項目を追加

`AgentStep` のオプション：

| 引数 | 用途 |
|---|---|
| `is_enabled` | `False` を返すとそのステップを飛ばす（条件付き実行） |
| `route_override` | ノード名を返すとそこへ遷移する（生成→検査→修正の逆流。法規の自動修正で使用） |

## Phase 2 予定機能

- ⑤ 概算見積AI（坪単価ベース）
- ⑥ メーカー推薦AI（来場予約 CTA）
- 間取りの視覚化（SVG）
- e-Gov 法令API による条文 RAG（`LegalCheckOutput.references` に受け口あり）
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
