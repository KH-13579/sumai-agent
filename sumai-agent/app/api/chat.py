"""FastAPI ルーター — チャット API エンドポイント"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from langchain_core.messages import HumanMessage

from app.schemas.chat import ChatRequest, ChatResponse
from app.agents.orchestrator import build_graph
from app.data.demo_data import DEMO_PRESETS, DEMO_MAKERS

router = APIRouter()

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
        raise HTTPException(status_code=500, detail=f"エージェント実行エラー: {str(e)}")

    return ChatResponse(
        reply=result.get("reply", ""),
        requirements=result.get("requirements"),
        floor_plans=result.get("floor_plans"),
        maker_recommendations=result.get("maker_recommendations"),
        stage=result.get("stage", "hearing"),
        done=result.get("done", False),
    )


@router.get("/demo-presets")
async def demo_presets():
    """デモ用プリセットデータを返す"""
    return {"presets": DEMO_PRESETS}


@router.get("/makers")
async def makers():
    """デモ用ハウスメーカー一覧を返す"""
    return {"makers": DEMO_MAKERS}
