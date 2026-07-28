"""FastAPI ルーター — チャット API エンドポイント"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from langchain_core.messages import HumanMessage

from app.schemas.models import ChatRequest, ChatResponse
from app.agents.orchestrator import build_graph
from app.data.demo_data import DEMO_PRESETS, DEMO_MAKERS

router = APIRouter()

# グラフをモジュールロード時に一度だけ初期化
_graph, _memory = build_graph()


@router.get("/health")
async def health():
    return {"status": "ok", "service": "スマイエージェント"}


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """メインチャットエンドポイント"""
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="メッセージが空です")

    config = {"configurable": {"thread_id": req.session_id}}

    try:
        result = _graph.invoke(
            {
                "messages": [HumanMessage(content=req.message)],
                "requirements": None,
                "floor_plans": None,
                "stage": "hearing",
                "reply": "",
                "done": False,
            },
            config=config,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"エージェント実行エラー: {str(e)}")

    return ChatResponse(
        reply=result.get("reply", ""),
        requirements=result.get("requirements"),
        floor_plans=result.get("floor_plans"),
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
