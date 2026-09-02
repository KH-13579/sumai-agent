"""オーケストレーターAI — LangGraph による会話フロー制御（要件定義書 §7.1）

グラフ形状:

    orchestrator ─┬─(間取り生成済み)→ follow_up ────→ compose → END
                  └─→ hearing ─┬─(要件不足)─────────→ compose → END
                               └─→ [ POST_HEARING_STEPS ] → compose → END

POST_HEARING_STEPS が専門エージェントの実行順を宣言する唯一の場所であり、
見積AI（Phase 2）はここに1行追加するだけで組み込める。
"""
from __future__ import annotations

import os
from typing import List, Optional, Sequence

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from app.schemas.requirements import RequirementBaseline
from app.schemas.floorplan import FloorPlan
from app.schemas.maker import MakerRecommendation, MakerRecommendationOutput
from app.agents.hearing_agent import run_hearing
from app.agents.legal_agent import build_legal_reply, build_replan_constraints, run_legal_check
from app.agents.pipeline import AgentStep, add_pipeline_to_graph, first_enabled
from app.agents.planning_agent import run_planning
from app.agents.state import ReplySection, SumaiState, section
from app.tools.llm_cache import CachedChatModel, cache_mode
from app.agents.maker_agent import run_maker_recommendation


# 間取り生成に必須の4項目（この4つが揃うまでヒアリングを続ける）
REQUIRED_FIELDS = ["family_structure", "budget", "land_info", "desired_size"]

# ヒアリングを続ける最大ターン数。これを超えたら未確定項目に仮定を入れて先に進む
MAX_HEARING_TURNS = 3

# 法規NGによる間取り再生成の最大回数（NFR-07 停止性）
MAX_LEGAL_RETRY = 1

# 「ヒアリングを打ち切って提案してほしい」という明示的な意図を示す語句
SKIP_KEYWORDS = [
    "提案に移って", "提案してください", "提案をお願い", "提案して",
    "スキップして", "そのまま提案", "進んでください", "間取りを見せて",
    "間取り提案", "先に進んで",
]

# 「その項目に希望はない」という回答。追質問への答えとして正当なので、
# 同じことを聞き直さず FALLBACK_DEFAULTS の仮定値で先に進む。
# 「土地はありません」のように意味のある否定を拾わないよう、単独の「ありません」は含めない。
NO_PREFERENCE_KEYWORDS = [
    "特にない", "特になし", "特に無い", "とくにない", "とくになし",
    "特にありません", "特にございません", "とくにありません", "特にこだわり",
    "こだわりはない", "こだわりなし", "こだわりません", "こだわりありません",
    "希望はない", "希望なし", "希望はありません", "希望ありません",
    "お任せ", "おまかせ", "任せます", "どちらでも", "どれでも",
    "なんでもいい", "何でもいい", "なんでも構いません", "何でも構いません",
    "分からない", "わからない", "分かりません", "わかりません", "未定です",
]

# ヒアリングを打ち切った際、未確定項目に入れる仮定値（プランニングAIへの前提として渡す）
FALLBACK_DEFAULTS = {
    "family_structure": "家族構成未確認（標準的な家族構成として想定）",
    "budget": "予算未確認（3,000万円台の標準プランとして想定）",
    "land_info": "土地未定（30坪前後の一般的な整形地を想定）",
    "desired_size": "広さ・部屋数未指定（3LDK・30坪前後を想定）",
}

# 追質問を組み立てるための必須項目の表示名
REQUIRED_FIELD_LABELS = {
    "family_structure": "ご家族の構成",
    "budget": "ご予算",
    "land_info": "土地の有無や場所",
    "desired_size": "ご希望の広さ・部屋数",
}

# すべての成果物に添える免責表示（NFR-05／LAW-4）
GLOBAL_DISCLAIMER = (
    "⚠️ 本提案はAIによる概算・参考情報です。間取り・概算費用・法規判定はいずれも推定値であり、"
    "適法性や金額を保証するものではありません。"
    "詳細設計・法規確認・正確な見積は建築士・ハウスメーカーにご相談ください。"
)

# 間取り提示後にユーザーへ次の行動を促す文（ORC-3 送客提案）
FOLLOW_UP_PROMPT = (
    "気になる案はありましたか？変更したい点や追加のご要望があれば、お気軽にお申し付けください。"
    "ハウスメーカーへの相談・来場予約のご案内もできます。"
)


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


def _has_no_preference(message: str) -> bool:
    """「特にないです」のように、希望がないことを答えた発言か"""
    return any(keyword in message for keyword in NO_PREFERENCE_KEYWORDS)


def _missing_fields_question(missing: List[str]) -> str:
    """不足項目を名指しした追質問を組み立てる

    LLM が follow_up_question を省略したときに使う。何が足りないかは
    こちら側で把握しているため、内容のない聞き返しにしない。
    """
    labels = [REQUIRED_FIELD_LABELS[f] for f in missing if f in REQUIRED_FIELD_LABELS]
    if not labels:
        return "もう少し詳しく教えていただけますか？"
    return (
        f"あと{'・'.join(labels)}をうかがえますか？"
        "「特にありません」とお答えいただければ、一般的な想定で間取りをご提案します。"
    )


def _last_human_message(messages: Sequence[BaseMessage]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return str(msg.content)
    return ""


def _legal_autofix_enabled() -> bool:
    """法規NG時の自律修正ループの有効・無効（既定はOFF）

    デモの見せ場として使う場合は環境変数 SUMAI_LEGAL_AUTOFIX=1 を設定する。
    間取り生成が2回走るため応答時間が伸びる点に注意（NFR-02 の許容範囲内）。
    """
    return os.getenv("SUMAI_LEGAL_AUTOFIX", "0").strip().lower() in ("1", "true", "yes", "on")


# ─────────────────────────────────────────
# LLM 初期化
# ─────────────────────────────────────────

def _temperature() -> float:
    """生成のばらつきを抑える温度（NFR-01 デモ再現性）。不正値は既定値に戻す"""
    try:
        return float(os.getenv("SUMAI_TEMPERATURE", "0.3"))
    except ValueError:
        return 0.3


def _get_llm(json_mode: bool = False, num_predict: int = 1024):
    """全ノード共通のLLM生成口

    ここが LLM 呼び出しの唯一の入口であるため、応答キャッシュ（NFR-06）の差し込みも
    この1箇所で完結する。SUMAI_LLM_CACHE=off（既定）のときは素の ChatOllama を返す。
    """
    model = os.getenv("SUMAI_MODEL", "qwen2.5:3b")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    kwargs = {
        "model": model,
        "base_url": base_url,
        "num_predict": num_predict,
        "temperature": _temperature(),
    }
    if json_mode:
        # ヒアリング/間取り生成/敷地照会はJSON応答が前提のため、Ollama側にJSON整形を強制させる
        kwargs["format"] = "json"

    llm = ChatOllama(**kwargs)

    mode = cache_mode()
    if mode == "off":
        return llm
    return CachedChatModel(llm, model=model, json_mode=json_mode, mode=mode)


# ─────────────────────────────────────────
# ノード実装
# ─────────────────────────────────────────

def orchestrator_node(state: SumaiState) -> dict:
    """ルーティング担当（ORC-1, ORC-2）

    毎ターンの入口として reply_sections をリセットする。MemorySaver は状態をターンを
    越えて保持するため、ここでリセットしないと前ターンの提案文が再表示されてしまう。
    """
    if state.get("floor_plans"):
        return {"reply_sections": None, "stage": "follow_up"}
    return {"reply_sections": None, "stage": "hearing", "done": False}


def follow_up_node(state: SumaiState) -> dict:
    """間取り提示後のフォローアップ応答（ORC-3）"""
    llm = _get_llm(num_predict=512)
    system = SystemMessage(content=(
        "あなたは住宅AIコンシェルジュです。"
        "すでに間取り案と法規チェック結果を提示した後のフォローアップ対話を行います。"
        "ユーザーの質問や感想に丁寧に答え、必要に応じてハウスメーカーへの相談・来場予約を提案してください。"
        "法規に関する質問には「参考判定であり、建築士・指定確認検査機関の確認が必要」と必ず添えてください。"
    ))
    resp = llm.invoke([system] + state["messages"])
    return {
        "reply_sections": section("follow_up", "フォローアップ", str(resp.content)),
    }


def hearing_node(state: SumaiState) -> dict:
    """ヒアリングAIを実行して要件を構造化（HEAR-1〜4）"""
    llm = _get_llm(json_mode=True, num_predict=512)
    prev_req = state.get("requirements")
    turns = state.get("hearing_turns", 0) + 1

    result = run_hearing(state["messages"], llm, known_requirements=prev_req)
    merged = _merge_requirements(prev_req, result.requirements)
    missing = merged.missing_fields

    last_message = _last_human_message(state["messages"])
    skip_requested = _wants_to_skip_hearing(last_message)

    # 「特にないです」は追質問への正当な回答。ただし1項目も聞けていない段階で
    # これを受けると全項目が仮定値になってしまうため、既に何か判明している場合に限る。
    known_something = any(
        getattr(merged, f) is not None for f in REQUIRED_FIELDS if f not in missing
    )
    no_preference = _has_no_preference(last_message) and known_something

    should_proceed = (
        not missing or skip_requested or no_preference or turns >= MAX_HEARING_TURNS
    )

    if should_proceed:
        if missing:
            merged = _apply_fallback_defaults(merged, missing)
        return {
            "requirements": merged,
            "stage": "planning",
            "hearing_turns": turns,
            "done": False,
        }

    question = result.follow_up_question or _missing_fields_question(missing)
    return {
        "requirements": merged,
        "stage": "hearing",
        "hearing_turns": turns,
        "done": False,
        "reply_sections": section("hearing", "追質問", question),
    }


def planning_node(state: SumaiState) -> dict:
    """間取り生成AIを実行して3案を生成（PLAN-1〜3）

    法規チェックからの修正指示（legal_constraints）が入っている場合は、
    それを制約として渡して再生成する（自律修正ループの2周目）。
    """
    # with_structured_output(method="json_schema")がJSON強制を担うため、
    # ここでformat="json"を重ねて指定しない（衝突回避）
    llm = _get_llm(json_mode=False, num_predict=2048)
    constraints = state.get("legal_constraints")
    retry = state.get("legal_retry", 0)

    result = run_planning(state["requirements"], llm, legal_constraints=constraints)

    plans_text = ""
    for i, plan in enumerate(result.plans, 1):
        rooms_text = "、".join([f"{r.name}（{r.area}）" for r in plan.rooms])
        plans_text += f"\n\n#### 案{i}：{plan.concept}\n"
        plans_text += f"- 延床面積：{plan.total_floor_area}（{plan.floors}）\n"
        plans_text += f"- 主要な部屋：{rooms_text}\n"
        plans_text += f"- 概算費用：{plan.estimated_cost or '要確認'}\n"
        plans_text += f"- 間取りのポイント：{plan.layout_description}\n"
        plans_text += f"- この案をお勧めする理由：{plan.rationale}\n"

    if constraints:
        heading = (
            f"### 🔁 間取り提案（法規チェックを反映して修正／{len(result.plans)}案）\n\n"
            "先の案は法規上の制限を超える可能性があったため、制限内に収まるよう調整しました。"
        )
    else:
        heading = (
            f"### ✨ 間取り提案（{len(result.plans)}案）\n\n"
            "ご要望をもとに、コンセプトの異なる間取り案をご提案します！"
        )

    markdown = f"{heading}\n{plans_text}\n\n{result.summary}"

    return {
        "floor_plans": result.plans,
        "stage": "planning",
        # 制約を消費したらクリアする（再生成の連鎖を防ぐ）
        "legal_constraints": None,
        "legal_retry": retry + 1 if constraints else retry,
        "reply_sections": section("planning", "間取り提案", markdown),
    }


def legal_node(state: SumaiState) -> dict:
    """法規チェックAIを実行して適合判定と要確認フラグを生成（LAW-1〜4）"""
    plans = state.get("floor_plans") or []
    if not plans:
        return {"stage": "legal"}

    llm = _get_llm(json_mode=True)
    # 2周目は1周目に確定した敷地情報を再利用し、余分なLLM呼び出しを避ける
    output = run_legal_check(
        state["requirements"], plans, llm, site_info=state.get("site_info")
    )

    # 不適合があり、かつ再生成の余地がある場合のみ修正指示を残す
    constraints = build_replan_constraints(output)
    can_retry = _legal_autofix_enabled() and state.get("legal_retry", 0) < MAX_LEGAL_RETRY

    return {
        "site_info": output.site_info,
        "legal_checks": output.checks,
        "stage": "legal",
        "legal_constraints": constraints if can_retry else None,
        "reply_sections": section(
            "legal",
            "法規チェック結果",
            build_legal_reply(output, include_disclaimer=False),
        ),
    }


def maker_node(state: SumaiState) -> dict:
    """メーカー推薦AIを実行してハウスメーカーを推薦する（MKR-1〜6）"""
    llm = _get_llm(json_mode=True, num_predict=1024)
    result = run_maker_recommendation(
        state["requirements"],
        state.get("floor_plans") or [],
        llm,
    )

    rec_text = ""
    for rec in result.recommendations:
        type_label = "注文住宅メーカー" if rec.type == "builder" else "情報ポータル"
        rec_text += f"\n\n### 第{rec.rank}位：{rec.name}（{type_label}）\n"
        rec_text += f"**推薦理由：** {rec.reason}\n"
        strengths_text = "、".join(rec.strengths[:3])
        rec_text += f"- 強み：{strengths_text}\n"
        rec_text += f"- 価格帯：{rec.price_band}\n"
        if rec.caution:
            rec_text += f"- ⚠️ 注意点：{rec.caution}\n"
        rec_text += f"- 🔗 [{rec.name} 公式サイト]({rec.website})\n"

    markdown = f"""### 🏠 おすすめハウスメーカー・サービス
{rec_text}

---
{result.summary}

> ⚠️ 本推薦はAIによる参考情報です。実際の費用・仕様・対応エリアは各社にご確認ください。"""

    return {
        "maker_recommendation": result.recommendations,
        "stage": "maker",
        "reply_sections": section("maker", "メーカー推薦", markdown),
    }


def compose_node(state: SumaiState) -> dict:
    """各エージェントのセクションを1つの応答に合成する（ORC-4）

    同じエージェントのセクションが複数ある場合（修正ループで2回実行された場合）は
    最後のものだけを採用する。
    """
    sections: List[ReplySection] = list(state.get("reply_sections") or [])

    latest: List[ReplySection] = []
    seen: set[str] = set()
    for item in reversed(sections):
        if item["agent"] in seen:
            continue
        seen.add(item["agent"])
        latest.append(item)
    latest.reverse()

    body = "\n\n---\n\n".join(s["markdown"] for s in latest if s["markdown"])
    done = bool(state.get("floor_plans"))

    parts = [body] if body else []
    # 提案を出したターンだけ次の行動を促す（ヒアリング中の追質問には付けない）
    if any(s["agent"] == "planning" for s in latest):
        parts.append(FOLLOW_UP_PROMPT)
    # 免責は成果物を含む応答に付ける（NFR-05）。追質問のみのターンでは冗長になるため省く
    if done:
        parts.append(f"> {GLOBAL_DISCLAIMER}")

    return {
        "reply": "\n\n---\n\n".join(parts),
        "stage": "follow_up" if done else state.get("stage", "hearing"),
        "done": done,
    }


# ─────────────────────────────────────────
# 専門エージェントの実行順（拡張点）
# ─────────────────────────────────────────

def _legal_autofix_route(state: SumaiState) -> Optional[str]:
    """法規NGなら間取り生成へ差し戻す（自律修正ループ。最大 MAX_LEGAL_RETRY 回）"""
    if state.get("legal_constraints"):
        return "planning"
    return None


# 要件が揃った後に走る専門エージェント。エージェント追加はこのリストへの1行で完結する。
POST_HEARING_STEPS: List[AgentStep] = [
    AgentStep("planning", "planning", planning_node),
    AgentStep("legal", "legal", legal_node, route_override=_legal_autofix_route),
    AgentStep("maker", "maker", maker_node),
    # ── Phase 2 の追加予定（要件定義書 §7.4）──
    # AgentStep("estimate", "estimate", estimate_node),   # 見積AI（EST-1〜4）
]

COMPOSE_NODE = "compose"
FOLLOW_UP_NODE = "follow_up"


# ─────────────────────────────────────────
# ルーティング関数
# ─────────────────────────────────────────

def route_from_orchestrator(state: SumaiState) -> str:
    if state.get("floor_plans"):
        return FOLLOW_UP_NODE
    return "hearing"


def route_from_hearing(state: SumaiState) -> str:
    if state.get("stage") == "planning":
        return first_enabled(POST_HEARING_STEPS, state, COMPOSE_NODE)
    return COMPOSE_NODE


# ─────────────────────────────────────────
# グラフ構築
# ─────────────────────────────────────────

def build_graph() -> tuple:
    """LangGraph グラフを構築してコンパイル済みグラフとメモリを返す"""
    memory = MemorySaver()

    builder = StateGraph(SumaiState)
    builder.add_node("orchestrator", orchestrator_node)
    builder.add_node("hearing", hearing_node)
    builder.add_node(FOLLOW_UP_NODE, follow_up_node)
    builder.add_node(COMPOSE_NODE, compose_node)

    # 専門エージェント群のノードとエッジは宣言から自動生成する
    add_pipeline_to_graph(builder, POST_HEARING_STEPS, terminal=COMPOSE_NODE)

    builder.set_entry_point("orchestrator")
    builder.add_conditional_edges(
        "orchestrator",
        route_from_orchestrator,
        {FOLLOW_UP_NODE: FOLLOW_UP_NODE, "hearing": "hearing"},
    )
    builder.add_conditional_edges(
        "hearing",
        route_from_hearing,
        {step.name: step.name for step in POST_HEARING_STEPS} | {COMPOSE_NODE: COMPOSE_NODE},
    )
    builder.add_edge(FOLLOW_UP_NODE, COMPOSE_NODE)
    builder.add_edge(COMPOSE_NODE, END)

    graph = builder.compile(checkpointer=memory)
    return graph, memory
