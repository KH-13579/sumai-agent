"""オーケストレーターAI — LangGraph による会話フロー制御"""
from __future__ import annotations

import os
from typing import TypedDict, Annotated, List, Optional, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from app.schemas.models import RequirementBaseline, FloorPlan, ChatResponse
from app.agents.hearing_agent import run_hearing
from app.agents.planning_agent import run_planning


# ─────────────────────────────────────────
# グラフ状態
# ─────────────────────────────────────────

class SumaiState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    requirements: Optional[RequirementBaseline]
    floor_plans: Optional[List[FloorPlan]]
    stage: str            # "hearing" | "planning" | "follow_up"
    reply: str
    done: bool


# ─────────────────────────────────────────
# LLM 初期化
# ─────────────────────────────────────────

def _get_llm() -> ChatOllama:
    model = os.getenv("SUMAI_MODEL", "qwen2.5:7b")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    return ChatOllama(model=model, base_url=base_url, num_predict=4096)


# ─────────────────────────────────────────
# ノード実装
# ─────────────────────────────────────────

def orchestrator_node(state: SumaiState) -> dict:
    """ルーティングとフォローアップ応答を担当"""
    # 間取り生成済み → フォローアップ
    if state.get("floor_plans"):
        llm = _get_llm()
        system = SystemMessage(content=(
            "あなたは住宅AIコンシェルジュです。"
            "すでに間取り案を提案した後のフォローアップ対話を行います。"
            "ユーザーの質問や感想に丁寧に答え、必要に応じてハウスメーカーへの相談・来場予約を提案してください。"
            "「※本提案は概算・参考プランです。詳細は建築士・ハウスメーカーにご確認ください。」"
            "を文末に添えてください。"
        ))
        messages = [system] + state["messages"]
        resp = llm.invoke(messages)
        return {
            "reply": resp.content,
            "stage": "follow_up",
            "done": True,
        }

    # 未生成 → ヒアリングフェーズへ委譲
    return {
        "stage": "hearing",
        "reply": "",
        "done": False,
    }


def hearing_node(state: SumaiState) -> dict:
    """ヒアリングAIを実行して要件を構造化"""
    llm = _get_llm()
    result = run_hearing(state["messages"], llm)
    req = result.requirements

    if req.is_complete:
        return {
            "requirements": req,
            "stage": "planning",
            "reply": "",
            "done": False,
        }
    else:
        question = result.follow_up_question or "もう少し詳しく教えていただけますか？"
        return {
            "requirements": req,
            "stage": "hearing",
            "reply": question,
            "done": False,
        }


def planning_node(state: SumaiState) -> dict:
    """間取り生成AIを実行して3案を生成"""
    llm = _get_llm()
    result = run_planning(state["requirements"], llm)

    # 自然言語応答の生成
    plans_text = ""
    for i, plan in enumerate(result.plans, 1):
        rooms_text = "、".join([f"{r.name}（{r.area}）" for r in plan.rooms])
        plans_text += f"\n\n### 案{i}：{plan.concept}\n"
        plans_text += f"- 延床面積：{plan.total_floor_area}（{plan.floors}）\n"
        plans_text += f"- 主要な部屋：{rooms_text}\n"
        plans_text += f"- 概算費用：{plan.estimated_cost or '要確認'}\n"
        plans_text += f"- 間取りのポイント：{plan.layout_description}\n"
        plans_text += f"- この案をお勧めする理由：{plan.rationale}\n"

    reply = f"""ご要望をもとに、コンセプトの異なる**3つの間取り案**をご提案します！
{plans_text}

---
{result.summary}

> ⚠️ 本提案はAIによる概算・参考プランです。詳細な設計・法規確認・正確な見積は、建築士やハウスメーカーにご相談ください。

気になる案はありましたか？変更したい点や追加の要望があれば、お気軽にお申し付けください。ハウスメーカーへの相談・来場予約のご案内もできます。"""

    return {
        "floor_plans": result.plans,
        "stage": "follow_up",
        "reply": reply,
        "done": True,
    }


# ─────────────────────────────────────────
# ルーティング関数
# ─────────────────────────────────────────

def route_from_orchestrator(state: SumaiState) -> str:
    if state.get("floor_plans"):
        return END
    return "hearing"


def route_from_hearing(state: SumaiState) -> str:
    if state.get("stage") == "planning":
        return "planning"
    return END


# ─────────────────────────────────────────
# グラフ構築
# ─────────────────────────────────────────

def build_graph() -> tuple:
    """LangGraph グラフを構築してコンパイル済みグラフとメモリを返す"""
    memory = MemorySaver()

    builder = StateGraph(SumaiState)
    builder.add_node("orchestrator", orchestrator_node)
    builder.add_node("hearing", hearing_node)
    builder.add_node("planning", planning_node)

    builder.set_entry_point("orchestrator")
    builder.add_conditional_edges("orchestrator", route_from_orchestrator, {END: END, "hearing": "hearing"})
    builder.add_conditional_edges("hearing", route_from_hearing, {"planning": "planning", END: END})
    builder.add_edge("planning", END)

    graph = builder.compile(checkpointer=memory)
    return graph, memory
