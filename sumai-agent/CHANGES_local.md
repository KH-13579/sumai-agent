# ローカル変更メモ

既存コードのバグ修正・調整。

## 2026-08-03

ヒアリングAIの対話が同じ質問を繰り返すループになる不具合の対応（原因: LLMのJSON出力パース失敗時に汎用フォールバック文が返っていた）。

| # | 変更 | ファイル | 内容 |
|---|---|---|---|
| 1 | Ollamaの JSON 強制モード | `app/agents/orchestrator.py` | `_get_llm()` に `json_mode` 引数を追加。ヒアリング/間取り生成ノード（`hearing_node`, `planning_node`）では `format="json"` を付けて呼び出し、Ollama側にJSON整形を強制。フォローアップ会話ノード（`orchestrator_node`）は自然文回答のため `json_mode` を付けず現状維持 |
| 2 | モデル変更 | `app/agents/orchestrator.py` | デフォルトモデルを `qwen2.5:7b` → `qwen2.5:14b` に変更（`_get_llm()` 内、`SUMAI_MODEL` 環境変数で上書き可） |
| 3 | パース失敗時のリトライ | `app/agents/hearing_agent.py`, `app/agents/planning_agent.py` | `run_hearing` / `run_planning` 内のJSON解析を最大2回試行するループに変更。1回目が `json.JSONDecodeError` になった場合のみ再度LLMを呼び直す。2回とも失敗した場合のみ既存のフォールバック応答を返す |

### 前提条件
- `qwen2.5:14b` をローカルのOllamaに `ollama pull qwen2.5:14b` で取得しておく必要あり（未取得だと起動時にモデルダウンロードが走る/エラーになる）

## 2026-08-05

構造整理（動作は変更なし）。

| # | 変更 | ファイル | 内容 |
|---|---|---|---|
| 1 | スキーマ分割 | `app/schemas/models.py` → `requirements.py`（ヒアリング系）/ `floorplan.py`（間取り系）/ `chat.py`（API系） | 1ファイルに全モデルが混在していたのを関心ごとに分割。`models.py` は削除。呼び出し側（`orchestrator.py`, `hearing_agent.py`, `planning_agent.py`, `api/chat.py`, `tests/test_acceptance.py`）のimportを新パスに追従 |
| 2 | `app/tools/` 新設 | `app/tools/__init__.py` | 今後の間取りジオメトリ計算・SVG描画等のロジック用に空パッケージを用意（現時点では未実装、配置先の受け皿のみ） |
| 3 | グラフの遅延初期化 | `app/api/chat.py` | `build_graph()` をモジュールロード時の即時実行から、初回 `/chat` リクエスト時の遅延初期化（`_get_graph()`）に変更。importだけでOllama接続が発生しないようにした |

### 動作確認
- `tests/test_acceptance.py` を直接実行し、既存4テスト（AC-1, AC-2, AC-3, AC-2補）が全てPASSすることを確認
- `app.main` / `app.api.chat` のimportがエラーなく完了すること、遅延初期化により `_graph` が `None` のまま保持されることを確認
