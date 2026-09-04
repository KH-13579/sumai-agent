"""オーケストレーターAI — LangGraph による会話フロー制御"""
from __future__ import annotations

import os
from typing import TypedDict, Annotated, List, Optional, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from app.schemas.requirements import RequirementBaseline
from app.schemas.floorplan import FloorPlan
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
    hearing_turns: int


# 間取り生成に必須の4項目（この4つが揃うまでヒアリングを続ける）
REQUIRED_FIELDS = ["family_structure", "budget", "land_info", "desired_size"]

# ヒアリングを続ける最大ターン数。これを超えたら未確定項目に仮定を入れて先に進む
MAX_HEARING_TURNS = 3

# 「ヒアリングを打ち切って提案してほしい」という明示的な意図を示す語句
SKIP_KEYWORDS = [
    "提案に移って", "提案してください", "提案をお願い", "提案して",
    "スキップして", "そのまま提案", "進んでください", "間取りを見せて",
    "間取り提案", "先に進んで",
]

# ヒアリングを打ち切った際、未確定項目に入れる仮定値（プランニングAIへの前提として渡す）
FALLBACK_DEFAULTS = {
    "family_structure": "家族構成未確認（標準的な家族構成として想定）",
    "budget": "予算未確認（3,000万円台の標準プランとして想定）",
    "land_info": "土地未定（30坪前後の一般的な整形地を想定）",
    "desired_size": "広さ・部屋数未指定（3LDK・30坪前後を想定）",
}


def _merge_requirements(
    old: Optional[RequirementBaseline], new: RequirementBaseline
) -> RequirementBaseline:
    """前回までの要件と今回の抽出結果を統合する（新しい値がnullなら旧値を保持）"""
    if old is None:
        merged_data = new.model_dump()
    else:
        merged_data = old.model_dump()
        new_data = new.model_dump()
        for field in RequirementBaseline.model_fields:
            if field in ("is_complete", "missing_fields"):
                continue
            if new_data.get(field) is not None:
                merged_data[field] = new_data[field]
    merged = RequirementBaseline(**merged_data)
    missing = [f for f in REQUIRED_FIELDS if getattr(merged, f) is None]
    merged.missing_fields = missing
    merged.is_complete = not missing
    return merged


def _apply_fallback_defaults(req: RequirementBaseline, missing: List[str]) -> RequirementBaseline:
    """ヒアリング打ち切り時、未確定の必須項目に仮定値を補完する"""
    data = req.model_dump()
    for field in missing:
        data[field] = FALLBACK_DEFAULTS[field]
    data["missing_fields"] = []
    data["is_complete"] = True
    return RequirementBaseline(**data)


def _wants_to_skip_hearing(message: str) -> bool:
    return any(keyword in message for keyword in SKIP_KEYWORDS)


def _last_human_message(messages: Sequence[BaseMessage]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return str(msg.content)
    return ""


# ─────────────────────────────────────────
# LLM 初期化
# ─────────────────────────────────────────

def _get_llm(json_mode: bool = False) -> ChatOllama:
    model = os.getenv("SUMAI_MODEL", "qwen2.5:14b")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    kwargs = {"model": model, "base_url": base_url, "num_predict": 4096}
    if json_mode:
        # ヒアリング/間取り生成はJSON応答が前提のため、Ollama側にJSON整形を強制させる
        kwargs["format"] = "json"
    return ChatOllama(**kwargs)


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
    llm = _get_llm(json_mode=True)
    prev_req = state.get("requirements")
    turns = state.get("hearing_turns", 0) + 1

    result = run_hearing(state["messages"], llm, known_requirements=prev_req)
    merged = _merge_requirements(prev_req, result.requirements)
    missing = merged.missing_fields

    skip_requested = _wants_to_skip_hearing(_last_human_message(state["messages"]))
    should_proceed = not missing or skip_requested or turns >= MAX_HEARING_TURNS

    if should_proceed:
        if missing:
            merged = _apply_fallback_defaults(merged, missing)
        return {
            "requirements": merged,
            "stage": "planning",
            "reply": "",
            "hearing_turns": turns,
            "done": False,
        }
    else:
        question = result.follow_up_question or "もう少し詳しく教えていただけますか？"
        return {
            "requirements": merged,
            "stage": "hearing",
            "reply": question,
            "hearing_turns": turns,
            "done": False,
        }


def planning_node(state: SumaiState) -> dict:
    """間取り生成AIを実行して3案を生成"""
    llm = _get_llm(json_mode=True)
    result = run_planning(state["requirements"], llm)

    # 自然言語応答の生成
    plans_text = ""
    for i, plan in enumerate(result.plans, 1):
        rooms_text = "、".join([f"{r.name}（{r.area}）" for r in plan.rooms])
        plans_text += f"\n\n### 案{i}：{plan.concept}\n"
        plans_text += f"- 延床面積：{plan.total_floor_area}（{plan.floors}）\n"
        plans_text += f"- 主要な部屋：{rooms_text}\n"
        plans_text += f"- 概算費用：{plan.estimated_cost or '右側の見積AIで算定'}\n"
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
