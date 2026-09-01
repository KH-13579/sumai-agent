"""FastAPI ルーター — チャット API エンドポイント"""
from __future__ import annotations

import os
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from langchain_core.messages import HumanMessage

from app.agents.orchestrator import GLOBAL_DISCLAIMER, build_graph
from app.data.demo_data import DEMO_MAKERS, DEMO_PRESETS
from app.schemas.artifact import SumaiArtifact
from app.schemas.chat import ChatRequest, ChatResponse
from app.tools.llm_cache import LLMCacheMiss

router = APIRouter()

# 接続拒否のときに OS が返す文言。日本語 Windows は「対象のコンピューターによって拒否された」
# となり原因が読み取れないため、ここで拾ってサーバー側の状況に翻訳する。
_CONNECT_ERROR_HINTS = (
    "10061",            # WinError 10061（Windows・接続拒否）
    "connection refused",
    "connectionrefused",
    "failed to establish a new connection",
    "max retries exceeded",
    "all connection attempts failed",
    "拒否された",
)


def _explain_agent_error(exc: Exception) -> str:
    """例外を、利用者が次に何をすればよいか分かる日本語に翻訳する"""
    text = str(exc)
    mode = os.getenv("SUMAI_LLM_CACHE", "off").strip().lower()
    model = os.getenv("SUMAI_MODEL", "qwen2.5:3b")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # ① キャッシュのみで動かす設定なのに、その入力のキャッシュが無い
    if isinstance(exc, LLMCacheMiss):
        return (
            f"この入力に対する応答キャッシュがありません（SUMAI_LLM_CACHE={mode} は"
            "キャッシュのみで動作するため実機を呼びません）。デモ用に用意した文言と"
            "一字一句同じ入力のみ応答できます。自由な入力を試す場合は Ollama を起動し、"
            "SUMAI_LLM_CACHE=auto に変更してください。"
        )

    # ② Ollama に繋がらない。キャッシュに無い入力＝実機呼び出しで初めて露出する
    if any(hint in text.lower() for hint in _CONNECT_ERROR_HINTS):
        cached_note = (
            "キャッシュ済みの入力なら Ollama なしでも応答できますが、"
            "それ以外の入力は実機の推論が必要です。"
            if mode in ("auto", "replay")
            else ""
        )
        return (
            f"LLM（Ollama）に接続できません: {base_url} が応答していません。"
            f"{cached_note}"
            f"『ollama serve』で起動し、モデル {model} が導入済みか"
            "（ollama list）確認してください。"
        )

    # ③ モデル未導入
    if "not found" in text.lower() and "model" in text.lower():
        return f"モデル {model} が Ollama に導入されていません。『ollama pull {model}』を実行してください。"

    return f"エージェント実行エラー: {text}"

# グラフは初回リクエスト時に遅延初期化する（インポート時にOllama接続を発生させないため）
_graph = None
_memory = None


def _get_graph():
    global _graph, _memory
    if _graph is None:
        _graph, _memory = build_graph()
    return _graph


@router.get("/health")
async def health():
    return {"status": "ok", "service": "スマイエージェント"}


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """メインチャットエンドポイント"""
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="メッセージが空です")

    config = {"configurable": {"thread_id": req.session_id}}
    graph = _get_graph()

    try:
        # requirements/floor_plans/hearing_turns 等は明示的に渡さない。
        # チェックポイント（MemorySaver）に蓄積された前回までの状態を
        # 上書きしてしまうため、messages のみを差分として渡す。
        result = graph.invoke(
            {"messages": [HumanMessage(content=req.message)]},
            config=config,
        )
    except Exception as e:
        # 503 = 依存サービス（Ollama）側の都合。利用者の入力の誤りではない
        status = 503 if "接続できません" in (detail := _explain_agent_error(e)) else 500
        raise HTTPException(status_code=status, detail=detail)

    return ChatResponse(
        reply=result.get("reply", ""),
        requirements=result.get("requirements"),
        floor_plans=result.get("floor_plans"),
        site_info=result.get("site_info"),
        legal_checks=result.get("legal_checks"),
        maker_recommendations=result.get("maker_recommendations"),
        stage=result.get("stage", "hearing"),
        done=result.get("done", False),
    )


@router.get("/artifact/{session_id}", response_model=SumaiArtifact)
async def artifact(session_id: str):
    """機械可読アーティファクトを返す（要件定義書 §16）

    セッションの成果物（要件・間取り・敷地情報・法規チェック結果）を1つの JSON に
    まとめて出力する。ハウスメーカーへの持ち込み用データであり、下流エージェント
    （見積AI・メーカー推薦AI）の入力としても使える。
    """
    config = {"configurable": {"thread_id": session_id}}
    graph = _get_graph()

    snapshot = graph.get_state(config)
    values = snapshot.values if snapshot else None
    if not values:
        raise HTTPException(status_code=404, detail="該当するセッションが見つかりません")

    return SumaiArtifact(
        session_id=session_id,
        generated_at=datetime.now(timezone.utc),
        requirements=values.get("requirements"),
        floor_plans=values.get("floor_plans") or [],
        site_info=values.get("site_info"),
        legal_checks=values.get("legal_checks") or [],
        estimate=values.get("estimate"),
        maker_recommendation=values.get("maker_recommendation"),
        disclaimer=GLOBAL_DISCLAIMER,
    )


@router.get("/demo-presets")
async def demo_presets():
    """デモ用プリセットデータを返す"""
    return {"presets": DEMO_PRESETS}


@router.get("/makers")
async def makers():
    """デモ用ハウスメーカー一覧を返す"""
    return {"makers": DEMO_MAKERS}
