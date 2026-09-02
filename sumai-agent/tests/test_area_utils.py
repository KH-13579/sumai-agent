"""pytest — 面積表記の正規化テスト

実機（qwen2.5:14b）で観測された「坪の数値を㎡欄に書く」誤りを
決定論的に補正できることを確認する。
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from app.tools.area_utils import normalize_total_floor_area, sum_room_areas_sqm

# 実機で観測された 4LDK の部屋構成（合計 約68㎡ が読み取れる）
ROOMS_4LDK = ["18畳", "8畳", "6畳×2", "6畳", "標準サイズ", "2箇所"]


def test_sum_room_areas_ignores_unparseable():
    total = sum_room_areas_sqm(ROOMS_4LDK)
    # 18 + 8 + 12 + 6 = 44畳 × 1.62 ≈ 71.3㎡（「標準サイズ」「2箇所」は除外）
    assert total == pytest.approx(71.3, abs=0.2)


@pytest.mark.parametrize("bad,expected_sqm,expected_tsubo", [
    ("約35㎡（約10.5坪）", 116, 35),           # 実機の誤り: 35坪 → 約35㎡ と書かれた
    ("約35㎡（約10.5坪）/ 2階建て", 116, 35),   # 階数が混入した場合も補正する
    ("約30㎡（約10坪）", 99, 30),
    ("約40㎡（約12坪）", 132, 40),
])
def test_normalize_corrects_tsubo_written_as_sqm(bad, expected_sqm, expected_tsubo):
    fixed, note = normalize_total_floor_area(bad, ROOMS_4LDK)
    assert fixed == f"約{expected_sqm}㎡（約{expected_tsubo}坪）"
    assert note is not None and "補正" in note
    assert "坪として解釈" in note


@pytest.mark.parametrize("good", [
    "約116㎡（約35坪）",
    "約110㎡（約33坪）",
    "約99㎡（約30坪）",
])
def test_normalize_leaves_correct_values_untouched(good):
    """正しい表記は書き換えない（誤検出しない）"""
    fixed, note = normalize_total_floor_area(good, ROOMS_4LDK)
    assert fixed == good
    assert note is None


def test_normalize_skips_when_correction_does_not_resolve():
    """坪として解釈しても矛盾が解消しない場合は補正しない（法規チェック側で保留）"""
    # 部屋合計が非常に大きく、5㎡→16.5㎡ にしても足りないケース
    fixed, note = normalize_total_floor_area("約5㎡", ROOMS_4LDK)
    assert fixed == "約5㎡"
    assert note is None


def test_normalize_skips_when_result_implausible():
    """補正後が戸建てとして妥当な範囲を外れる場合は補正しない"""
    fixed, note = normalize_total_floor_area("約500㎡", ["300㎡"])
    assert fixed == "約500㎡"          # 500㎡ は部屋合計300㎡と整合するので触らない
    assert note is None


def test_normalize_handles_unreadable_rooms():
    """部屋面積が読み取れない場合も、延床が極端に小さければ補正する"""
    rooms = ["標準サイズ", "2箇所", "1台", "適宜"]
    fixed, note = normalize_total_floor_area("約35㎡", rooms)
    assert fixed == "約116㎡（約35坪）"
    assert note is not None

    # 部屋数が少なければ判断材料が乏しいため触らない
    fixed2, note2 = normalize_total_floor_area("約35㎡", ["標準サイズ"])
    assert fixed2 == "約35㎡" and note2 is None


def test_normalize_handles_missing_input():
    assert normalize_total_floor_area(None, ROOMS_4LDK) == (None, None)
    assert normalize_total_floor_area("", ROOMS_4LDK) == ("", None)
    assert normalize_total_floor_area("未定", ROOMS_4LDK) == ("未定", None)


# ─── 間取り生成AI に組み込まれているか ──────────────

def _make_planning_mock(concept, total_floor_area, floors, rooms, summary):
    """with_structured_output(..., include_raw=True) を使う run_planning 向けのモックを作る"""
    from app.agents.planning_agent import _LLMFloorPlan, _LLMPlanningOutput, _LLMRoom
    llm_rooms = [_LLMRoom(name=r["name"], note=r.get("note"), room_type="LDK", area_m2=29.8, floor=1) for r in rooms]
    structured = _LLMPlanningOutput(
        plans=[_LLMFloorPlan(
            concept=concept, total_floor_area=total_floor_area, floors=floors,
            rooms=llm_rooms, layout_description="x", rationale="y", estimated_cost=None,
        )],
        summary=summary,
    )
    mock = MagicMock()
    mock.with_structured_output.return_value.invoke.return_value = {
        "raw": MagicMock(content=""),
        "parsed": structured,
        "parsing_error": None,
    }
    return mock


def test_run_planning_normalizes_and_discloses():
    """run_planning がリスケール補正を適用して間取りを生成する（決定論エンジン経路）"""
    from app.agents.planning_agent import run_planning
    from app.schemas.requirements import RequirementBaseline

    llm = _make_planning_mock(
        concept="コスパ重視案",
        total_floor_area="約35㎡（約10.5坪）",   # 実機で観測された誤り（坪を㎡に書いた）
        floors="2階建て",
        rooms=[{"name": "LDK"}, {"name": "主寝室"}, {"name": "子供部屋"}, {"name": "書斎"}],
        summary="比較サマリー",
    )

    result = run_planning(RequirementBaseline(), llm)

    # 決定論エンジンが geometry を生成し、plan が少なくとも1件返ること
    assert len(result.plans) == 1
    assert result.plans[0].concept == "コスパ重視案"
    # geometry が生成されること（座標付き）
    assert result.plans[0].geometry is not None


def test_run_planning_keeps_valid_area_and_summary():
    """正しい面積表記でも run_planning が間取りを生成できること"""
    from app.agents.planning_agent import run_planning
    from app.schemas.requirements import RequirementBaseline

    llm = _make_planning_mock(
        concept="コスパ重視案",
        total_floor_area="約116㎡（約35坪）",
        floors="2階建て",
        rooms=[{"name": "LDK"}],
        summary="比較サマリー",
    )

    result = run_planning(RequirementBaseline(), llm)
    assert len(result.plans) == 1
    assert result.plans[0].concept == "コスパ重視案"
    assert result.summary == "比較サマリー"


def test_normalized_plan_passes_legal_consistency_check():
    """補正後の間取りは法規チェックの整合性検査を通り、判定が出る"""
    from app.schemas.floorplan import FloorPlan, Room
    from app.schemas.legal import SiteInfo
    from app.tools.legal_rules import check_plan, check_plan_consistency

    fixed_text, _ = normalize_total_floor_area("約35㎡（約10.5坪）", ROOMS_4LDK)
    plan = FloorPlan(
        concept="コスパ重視案", total_floor_area=fixed_text, floors="2階建て",
        rooms=[Room(name=f"room{i}", area=a, note=None) for i, a in enumerate(ROOMS_4LDK)],
        layout_description="x", rationale="y",
    )
    assert check_plan_consistency(plan) == (True, None)

    site = SiteInfo(site_area_sqm=165.0, zoning="第一種低層住居専用地域",
                    building_coverage_ratio=50.0, floor_area_ratio=100.0,
                    road_width_m=4.0, has_land=True, source="user_input")
    check = check_plan(plan, 1, site)
    # 補正により unknown ではなく実際の判定が出る
    assert check.status == "ok"
    assert all(i.status != "unknown" for i in check.items)
