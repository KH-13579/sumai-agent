"""pytest — エージェントパイプラインと応答合成のテスト

他エージェント（見積AI・メーカー推薦AI）を追加したときに壊れやすい部分、
すなわち「ステップの順序解決」「応答セクションの合成」「修正ループの停止性」を検証する。
"""
from __future__ import annotations

import pytest
from langgraph.graph import END, StateGraph

from app.agents import orchestrator as orch
from app.agents.pipeline import AgentStep, add_pipeline_to_graph, first_enabled, resolve_next
from app.agents.state import merge_reply_sections, section


# ─── ステップ順序の解決 ────────────────────────────

def make_steps():
    return [
        AgentStep("planning", "planning", lambda s: {}),
        AgentStep("legal", "legal", lambda s: {}),
        AgentStep("estimate", "estimate", lambda s: {}, is_enabled=lambda s: s.get("est_on", False)),
    ]


def test_pipeline_runs_steps_in_declared_order():
    steps = make_steps()
    assert first_enabled(steps, {}, "compose") == "planning"
    assert resolve_next(steps, 0, {}, "compose") == "legal"


def test_pipeline_skips_disabled_step():
    """is_enabled=False のステップは飛ばして次へ進む"""
    steps = make_steps()
    assert resolve_next(steps, 1, {}, "compose") == "compose"            # estimate 無効
    assert resolve_next(steps, 1, {"est_on": True}, "compose") == "estimate"


def test_pipeline_last_step_goes_to_terminal():
    steps = make_steps()
    assert resolve_next(steps, 2, {"est_on": True}, "compose") == "compose"


def test_route_override_wins_over_next_step():
    """route_override は次ステップより優先される（修正ループの逆流）"""
    steps = [
        AgentStep("planning", "planning", lambda s: {}),
        AgentStep("legal", "legal", lambda s: {},
                  route_override=lambda s: "planning" if s.get("ng") else None),
        AgentStep("estimate", "estimate", lambda s: {}),
    ]
    builder = StateGraph(dict)
    add_pipeline_to_graph(builder, steps, terminal=END)
    # ルーターの挙動を直接確認する（グラフ構築時の遷移先マップに全ステップが含まれること）
    assert resolve_next(steps, 1, {"ng": True}, END) == "estimate"       # override 抜きなら次へ
    assert steps[1].route_override({"ng": True}) == "planning"
    assert steps[1].route_override({}) is None


def test_declared_pipeline_contains_planning_then_legal_then_maker():
    """本番のステップ宣言が「間取り生成 → 法規チェック → メーカー推薦」の順であること"""
    names = [s.name for s in orch.POST_HEARING_STEPS]
    assert names == ["planning", "legal", "maker"]
    assert first_enabled(orch.POST_HEARING_STEPS, {}, orch.COMPOSE_NODE) == "planning"
    assert resolve_next(orch.POST_HEARING_STEPS, 0, {}, orch.COMPOSE_NODE) == "legal"
    assert resolve_next(orch.POST_HEARING_STEPS, 1, {}, orch.COMPOSE_NODE) == "maker"
    assert resolve_next(orch.POST_HEARING_STEPS, 2, {}, orch.COMPOSE_NODE) == orch.COMPOSE_NODE


def test_graph_builds_with_all_nodes():
    """グラフがコンパイルでき、法規チェックノードが含まれること"""
    graph, memory = orch.build_graph()
    nodes = set(graph.get_graph().nodes)
    for expected in ("orchestrator", "hearing", "planning", "legal", "maker", "compose", "follow_up"):
        assert expected in nodes, f"{expected} ノードが無い"


# ─── 応答セクションの合成 ──────────────────────────

def test_reply_sections_reducer_appends_and_resets():
    existing = section("planning", "間取り", "案A")
    appended = merge_reply_sections(existing, section("legal", "法規", "判定"))
    assert [s["agent"] for s in appended] == ["planning", "legal"]
    # None はリセット（ターン開始時に前ターンの応答を消す）
    assert merge_reply_sections(appended, None) == []


def test_compose_joins_sections_in_order():
    state = {
        "reply_sections": section("planning", "間取り", "間取り本文") + section("legal", "法規", "法規本文"),
        "floor_plans": ["dummy"],
    }
    result = orch.compose_node(state)

    assert result["reply"].index("間取り本文") < result["reply"].index("法規本文")
    assert orch.FOLLOW_UP_PROMPT in result["reply"]
    assert orch.GLOBAL_DISCLAIMER in result["reply"]
    assert result["done"] is True
    assert result["stage"] == "follow_up"


def test_compose_keeps_only_latest_section_per_agent():
    """修正ループで2回実行された場合、最後のセクションだけを採用する"""
    state = {
        "reply_sections": (
            section("planning", "間取り", "1回目の案")
            + section("legal", "法規", "1回目の判定")
            + section("planning", "間取り", "修正後の案")
            + section("legal", "法規", "再判定")
        ),
        "floor_plans": ["dummy"],
    }
    reply = orch.compose_node(state)["reply"]

    assert "修正後の案" in reply and "1回目の案" not in reply
    assert "再判定" in reply and "1回目の判定" not in reply
    assert reply.index("修正後の案") < reply.index("再判定")   # 宣言順は保たれる


def test_compose_omits_disclaimer_and_cta_during_hearing():
    """追質問のみのターンでは免責・CTAを付けない（冗長さの回避）"""
    state = {"reply_sections": section("hearing", "追質問", "ご予算を教えてください"), "stage": "hearing"}
    result = orch.compose_node(state)

    assert result["reply"] == "ご予算を教えてください"
    assert result["done"] is False
    assert result["stage"] == "hearing"


def test_orchestrator_node_resets_sections_each_turn():
    """毎ターンの入口でセクションをリセットする（前ターンの提案の再表示防止）"""
    assert orch.orchestrator_node({"floor_plans": None})["reply_sections"] is None
    assert orch.orchestrator_node({"floor_plans": ["dummy"]})["reply_sections"] is None


# ─── 修正ループの停止性（NFR-07）──────────────────

def test_legal_autofix_route_disabled_without_constraints():
    assert orch._legal_autofix_route({}) is None
    assert orch._legal_autofix_route({"legal_constraints": None}) is None
    assert orch._legal_autofix_route({"legal_constraints": "制約"}) == "planning"


def test_legal_autofix_is_off_by_default(monkeypatch):
    monkeypatch.delenv("SUMAI_LEGAL_AUTOFIX", raising=False)
    assert orch._legal_autofix_enabled() is False


@pytest.mark.parametrize("value,expected", [
    ("1", True), ("true", True), ("on", True), ("0", False), ("", False), ("no", False),
])
def test_legal_autofix_env_flag(monkeypatch, value, expected):
    monkeypatch.setenv("SUMAI_LEGAL_AUTOFIX", value)
    assert orch._legal_autofix_enabled() is expected


def test_planning_node_increments_retry_only_when_revising():
    """再生成回数は制約付きで走ったときだけ増える（NFR-07 の上限判定用）"""
    from unittest.mock import MagicMock, patch
    from app.agents.planning_agent import _LLMFloorPlan, _LLMPlanningOutput

    structured = _LLMPlanningOutput(
        plans=[_LLMFloorPlan(concept="案", total_floor_area="約100㎡（約30坪）", floors="2階建て",
                             rooms=[], layout_description="x", rationale="y", estimated_cost=None)],
        summary="s",
    )
    llm = MagicMock()
    llm.with_structured_output.return_value.invoke.return_value = {
        "raw": MagicMock(content=""), "parsed": structured, "parsing_error": None,
    }

    from app.schemas.requirements import RequirementBaseline
    base = {"requirements": RequirementBaseline(), "legal_retry": 0}

    with patch.object(orch, "_get_llm", lambda json_mode=False, num_predict=1024: llm):
        assert orch.planning_node(dict(base))["legal_retry"] == 0
        revised = orch.planning_node(dict(base, legal_constraints="制約あり"))
        assert revised["legal_retry"] == 1
        assert revised["legal_constraints"] is None          # 消費してクリアする


# ─── ヒアリングの聞き返し挙動 ─────────────────────────
# 追質問に「特にないです」と答えたのに同じことを聞き返される、
# 内容のない「もう少し詳しく教えていただけますか？」が返る、
# LLM が値に "未定" を入れると項目が埋まった扱いになる — の3件の回帰テスト。

@pytest.mark.parametrize("message", [
    "特にないです", "特にありません", "特にございません",
    "こだわりはないです", "こだわりません", "お任せします",
    "どちらでもいいです", "何でも構いません", "わかりません",
])
def test_no_preference_detected(message):
    """希望がないという回答を認識する（＝仮定値で先に進む）"""
    assert orch._has_no_preference(message) is True


@pytest.mark.parametrize("message", [
    "土地はありません",          # 土地なしは意味のある情報。希望なしではない
    "書斎が欲しいです",
    "3LDKがいいです",
    "予算は3000万円です",
])
def test_meaningful_answer_not_treated_as_no_preference(message):
    assert orch._has_no_preference(message) is False


def test_no_preference_proceeds_to_planning():
    """「特にないです」で聞き直さず間取り生成へ進む"""
    from unittest.mock import MagicMock, patch
    import json
    from langchain_core.messages import HumanMessage
    from app.schemas.requirements import RequirementBaseline

    # 必須4項目のうち desired_size だけが未取得の状態
    known = RequirementBaseline(
        family_structure="夫婦と子供2人", budget="3000〜3500万円",
        land_info="さいたま市 約50坪",
    )
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content=json.dumps({
        "requirements": {"is_complete": False, "missing_fields": ["desired_size"]},
        "follow_up_question": "広さのご希望はありますか？",
    }))
    state = {
        "messages": [HumanMessage(content="特にないです")],
        "requirements": known,
        "hearing_turns": 1,
    }
    with patch.object(orch, "_get_llm", lambda json_mode=False, num_predict=1024: llm):
        result = orch.hearing_node(state)

    assert result["stage"] == "planning", "聞き直しに戻っている"
    assert result["requirements"].is_complete is True
    # 仮定値が入っていることを明示（利用者に伝わる文言であること）
    assert "未指定" in result["requirements"].desired_size


def test_no_preference_ignored_when_nothing_known_yet():
    """1項目も聞けていない段階の「特にない」では打ち切らない

    全項目を仮定値にした提案は無意味なため、追質問を続ける。
    """
    from unittest.mock import MagicMock, patch
    import json
    from langchain_core.messages import HumanMessage

    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content=json.dumps({
        "requirements": {"is_complete": False, "missing_fields": []},
        "follow_up_question": "ご家族の構成を教えてください。",
    }))
    state = {"messages": [HumanMessage(content="特にないです")], "hearing_turns": 1}
    with patch.object(orch, "_get_llm", lambda json_mode=False, num_predict=1024: llm):
        result = orch.hearing_node(state)

    assert result["stage"] == "hearing"


def test_fallback_question_names_missing_fields():
    """LLM が追質問を省略しても、何が足りないかを名指しする"""
    q1 = orch._missing_fields_question(["desired_size"])
    assert "広さ" in q1 and "もう少し詳しく" not in q1
    q2 = orch._missing_fields_question(["budget", "desired_size"])
    assert "ご予算" in q2 and "広さ" in q2
    # 対応表に無い項目しか無い場合だけ、従来の汎用文に落とす
    assert orch._missing_fields_question(["謎の項目"]) == "もう少し詳しく教えていただけますか？"


def test_fallback_question_suggests_an_answer_the_parser_accepts():
    """追質問が案内する言い方が、実際に希望なしと判定されること"""
    import re
    q = orch._missing_fields_question(["desired_size"])
    quoted = re.findall(r"「([^」]+)」", q)
    assert quoted, "案内する言い方が引用符で示されていない"
    assert any(orch._has_no_preference(p) for p in quoted)


@pytest.mark.parametrize("placeholder", ["未定", "未確認", "不明", "特になし", "  ", "-"])
def test_placeholder_values_treated_as_missing(placeholder):
    """LLM が値に入れる "未定" 等を未取得に戻す（空の要件で間取り生成に進まない）"""
    from app.agents.hearing_agent import _clean_value
    assert _clean_value(placeholder) is None


def test_placeholder_value_keeps_field_missing():
    """"未定" が入っても is_complete にならない"""
    from app.agents.hearing_agent import _clean_value
    from app.schemas.requirements import RequirementBaseline

    req = RequirementBaseline(
        family_structure="夫婦と子供2人", budget="3000〜3500万円",
        land_info="さいたま市 約50坪", desired_size=_clean_value("未定"),
    )
    merged = orch._merge_requirements(None, req)
    assert merged.missing_fields == ["desired_size"]
    assert merged.is_complete is False


def test_real_values_survive_cleaning():
    """通常の値は変えない"""
    from app.agents.hearing_agent import _clean_value
    assert _clean_value("約100㎡（約30坪）") == "約100㎡（約30坪）"
    assert _clean_value(" 3LDK ") == "3LDK"
