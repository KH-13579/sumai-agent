"""pytest — 法規チェックAI テスト（要件定義書 §7.6 LAW-1〜4）

判定エンジンは決定論的なため、LLM モックは敷地情報の抽出にのみ使用する。
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from app.schemas.floorplan import FloorPlan, Room
from app.schemas.legal import SiteInfo
from app.schemas.requirements import RequirementBaseline
from app.tools.legal_rules import (
    check_plan,
    check_plans,
    effective_far,
    estimate_footprint_sqm,
    estimate_height_m,
    parse_area_sqm,
    parse_floor_count,
)
from app.tools.zoning import get_rule, normalize_zoning


# ─── テスト用ヘルパー ─────────────────────────────

def make_plan(concept="コスパ重視案", area="約100㎡（約30坪）", floors="2階建て") -> FloorPlan:
    return FloorPlan(
        concept=concept,
        total_floor_area=area,
        floors=floors,
        rooms=[Room(name="LDK", area="20畳", note=None)],
        layout_description="1階にLDK、2階に寝室。",
        rationale="ご要望に沿った標準設計。",
        estimated_cost="3,000万円",
    )


def make_site(**overrides) -> SiteInfo:
    defaults = dict(
        address="さいたま市",
        site_area_sqm=125.0,
        zoning="第一種住居地域",
        building_coverage_ratio=60.0,
        floor_area_ratio=200.0,
        road_width_m=6.0,
        fire_zone="指定なし",
        has_land=True,
        source="user_input",
    )
    defaults.update(overrides)
    return SiteInfo(**defaults)


def find_item(check, name):
    return next(i for i in check.items if i.item == name)


def mock_site_llm(payload: dict) -> MagicMock:
    """敷地情報抽出をモックするLLM"""
    mock = MagicMock()
    mock.invoke.return_value = MagicMock(content=json.dumps(payload))
    return mock


# ─── 数値パース ───────────────────────────────────

@pytest.mark.parametrize("text,expected", [
    ("約100㎡（約30坪）", 100.0),      # ㎡表記を優先する
    ("約110㎡／約33坪", 110.0),
    ("30坪", 99.2),                    # 坪のみなら㎡へ換算
    ("165", 165.0),                    # 単位なしは㎡とみなす
    ("未定", None),                    # 数値がなければ判定材料なし
    (None, None),
])
def test_parse_area_sqm(text, expected):
    assert parse_area_sqm(text) == expected


@pytest.mark.parametrize("text,expected", [
    ("2階建て", 2), ("3階建て", 3), ("平屋", 1), ("１階建て", 1), ("不明", None), (None, None),
])
def test_parse_floor_count(text, expected):
    assert parse_floor_count(text) == expected


def test_estimate_footprint_uses_first_floor_ratio():
    """建築面積は延床÷階数ではなく、1階が大きい前提の安全側で見積もる"""
    plan = make_plan(area="約100㎡", floors="2階建て")
    assert estimate_footprint_sqm(plan) == 55.0      # 100 × 0.55（単純な50.0より安全側）
    assert estimate_footprint_sqm(make_plan(area="約80㎡", floors="平屋")) == 80.0


def test_estimate_height():
    assert estimate_height_m(1) == 5.0
    assert estimate_height_m(2) == 8.0
    assert estimate_height_m(3) == 11.0
    assert estimate_height_m(None) == 8.0            # 階数不明時は2階建てを仮定


# ─── LAW-2: 自動判定 ──────────────────────────────

def test_law2_bcr_over_limit_is_ng():
    """建ぺい率超過を ng として検出する（建築基準法 第53条）"""
    # 敷地125㎡・建ぺい60% → 建築面積の上限は75㎡。
    # 平屋150㎡なら建築面積150㎡ = 120% で明確に超過する。
    site = make_site(site_area_sqm=125.0, building_coverage_ratio=60.0)
    check = check_plan(make_plan(area="約150㎡", floors="平屋"), 1, site)

    bcr = find_item(check, "建ぺい率")
    assert bcr.status == "ng"
    assert "超過" in bcr.margin
    assert bcr.basis == "建築基準法 第53条"
    assert check.status == "ng"
    print(f"✅ LAW-2a PASS: {bcr.actual} / 上限 {bcr.limit}")


def test_law2_bcr_within_limit_is_ok():
    site = make_site(site_area_sqm=200.0, building_coverage_ratio=60.0)
    check = check_plan(make_plan(area="約100㎡", floors="2階建て"), 1, site)

    bcr = find_item(check, "建ぺい率")
    assert bcr.status == "ok"
    assert "余裕" in bcr.margin


def test_law2_base_far_from_road_width_applies():
    """基準容積率 min(指定容積率, 幅員×法定係数) が適用される（第52条）"""
    rule = get_rule("第一種住居地域")
    # 住居系の法定係数は 4/10。幅員4m → 4 × 0.4 × 100 = 160% が指定200%より厳しい
    limit, basis = effective_far(rule, 200.0, 4.0)
    assert limit == 160.0
    assert "基準容積率を適用" in basis

    # 幅員12m以上なら道路幅員による制限は掛からない
    assert effective_far(rule, 200.0, 12.0)[0] == 200.0
    # 幅員不明なら指定容積率をそのまま使い、その旨を明示する
    limit_unknown, basis_unknown = effective_far(rule, 200.0, None)
    assert limit_unknown == 200.0
    assert "不明" in basis_unknown
    print(f"✅ LAW-2b PASS: 幅員4m → 上限 {limit}%（{basis}）")


def test_law2_far_uses_base_far_for_judgement():
    """基準容積率で判定され、指定容積率だけを見れば適合する案が ng になる"""
    site = make_site(site_area_sqm=100.0, floor_area_ratio=200.0, road_width_m=4.0)
    # 延床180㎡ / 敷地100㎡ = 180%。指定200%なら適合だが基準容積率160%では超過
    check = check_plan(make_plan(area="約180㎡", floors="2階建て"), 1, site)

    far = find_item(check, "容積率")
    assert far.status == "ng"
    assert "160%" in far.limit
    print(f"✅ LAW-2b補 PASS: {far.actual} / 上限 {far.limit}")


def test_law2_absolute_height_limit_in_low_rise_zone():
    """第一種低層住居専用地域の3階建ては絶対高さ制限10mを超過する（第55条）"""
    site = make_site(zoning="第一種低層住居専用地域", building_coverage_ratio=50.0,
                     floor_area_ratio=100.0, site_area_sqm=200.0)
    ng_check = check_plan(make_plan(area="約100㎡", floors="3階建て"), 1, site)
    height = find_item(ng_check, "絶対高さ制限")
    assert height.status == "ng"           # 推定11m > 10m
    assert height.basis == "建築基準法 第55条"

    ok_check = check_plan(make_plan(area="約100㎡", floors="2階建て"), 1, site)
    assert find_item(ok_check, "絶対高さ制限").status == "ok"   # 推定8m ≦ 10m
    print(f"✅ LAW-2c PASS: {height.actual} / 上限 {height.limit}")


def test_law2_no_height_item_when_no_absolute_limit():
    """絶対高さ制限のない用途地域では高さ項目を出さない（自明合格を並べない）"""
    check = check_plan(make_plan(floors="3階建て"), 1, make_site(zoning="商業地域"))
    assert all(i.item != "絶対高さ制限" for i in check.items)


def test_room_area_parsing():
    from app.tools.area_utils import parse_room_area_sqm

    assert parse_room_area_sqm("18畳") == 29.2          # 18 × 1.62
    assert parse_room_area_sqm("6畳×2") == 19.4         # 6 × 1.62 × 2
    assert parse_room_area_sqm("20㎡") == 20.0
    assert parse_room_area_sqm("標準サイズ") is None     # 面積として読めない表記は除外
    assert parse_room_area_sqm("2箇所") is None
    assert parse_room_area_sqm(None) is None


def test_inconsistent_plan_area_is_detected():
    """部屋面積の合計が延床面積を超える間取りを矛盾として検出する

    実機で qwen2.5:14b が「35坪」を「約35㎡」と誤変換した事例に対応する。
    """
    from app.tools.legal_rules import check_plan_consistency

    broken = FloorPlan(
        concept="コスパ重視案",
        total_floor_area="約35㎡（約10.5坪）/ 2階建て",   # 本来は約116㎡
        floors="2階建て",
        rooms=[
            Room(name="LDK", area="14畳", note=None),
            Room(name="主寝室", area="8畳", note=None),
            Room(name="子供部屋", area="6畳×2", note=None),
            Room(name="書斎", area="6畳", note=None),
            Room(name="浴室・洗面", area="標準サイズ", note=None),
        ],
        layout_description="x", rationale="y",
    )
    ok, reason = check_plan_consistency(broken)
    assert ok is False
    assert "矛盾" in reason and "坪" in reason

    # 正しい延床面積なら矛盾とみなさない（誤検出しない）
    fixed = broken.model_copy(update={"total_floor_area": "約116㎡（約35坪）"})
    assert check_plan_consistency(fixed) == (True, None)


def test_inconsistent_plan_is_not_reported_as_compliant():
    """矛盾した間取りに対して「適合」と判定しない（誤った安心を与えない）"""
    broken = FloorPlan(
        concept="コスパ重視案", total_floor_area="約35㎡（約10.5坪）", floors="2階建て",
        rooms=[Room(name="LDK", area="14畳", note=None),
               Room(name="主寝室", area="8畳", note=None),
               Room(name="子供部屋", area="6畳×2", note=None)],
        layout_description="x", rationale="y",
    )
    # 敷地165㎡・建ぺい50% に対し 35㎡ なら数値上は余裕だが、矛盾のため判定しない
    check = check_plan(broken, 1, make_site(site_area_sqm=165.0, building_coverage_ratio=50.0))

    assert check.status == "unknown"
    assert find_item(check, "建ぺい率").status == "unknown"
    assert find_item(check, "容積率").status == "unknown"
    assert find_item(check, "建ぺい率").actual is None
    flags = {f.item: f for f in check.manual_flags}
    assert "間取りデータの整合性" in flags
    assert flags["間取りデータの整合性"].severity == "high"
    print("✅ 整合性チェック PASS: 矛盾した間取りは unknown（適合と言わない）")


def test_plan_specific_flag_shown_per_plan_in_reply():
    """案固有のフラグは案ごとに、敷地共通のフラグは1回だけ表示する"""
    from app.agents.legal_agent import build_legal_reply, run_legal_check

    good = make_plan(area="約110㎡（約33坪）")
    broken = FloorPlan(
        concept="広さ重視案", total_floor_area="約40㎡", floors="2階建て",
        rooms=[Room(name="LDK", area="20畳", note=None),
               Room(name="主寝室", area="10畳", note=None),
               Room(name="子供部屋", area="8畳×2", note=None)],
        layout_description="x", rationale="y",
    )
    output = run_legal_check(RequirementBaseline(), [good, broken], MagicMock(),
                            site_info=make_site(site_area_sqm=200.0))
    reply = build_legal_reply(output)

    assert output.checks[0].status == "ok"
    assert output.checks[1].status == "unknown"
    # 案固有のフラグは1回だけ（案2の中に）出る
    assert reply.count("間取りデータの整合性") == 1
    # 敷地共通のフラグは共通セクションに1回だけ出る
    assert reply.count("道路斜線制限") == 1


def test_law2_missing_numbers_are_unknown_not_ng():
    """判定材料が不足している場合は ng ではなく unknown にする"""
    site = make_site(site_area_sqm=None)
    check = check_plan(make_plan(area="未定"), 1, site)

    assert check.status == "unknown"
    assert find_item(check, "建ぺい率").status == "unknown"
    assert find_item(check, "容積率").status == "unknown"
    assert all(i.margin is None for i in check.items)
    print("✅ LAW-2d PASS: 判定材料不足は unknown として NG と区別される")


def test_law2_industrial_exclusive_zone_rejects_housing():
    """工業専用地域は住宅を建てられないため用途制限で ng（別表第二）"""
    check = check_plan(make_plan(), 1, make_site(zoning="工業専用地域"))
    use = find_item(check, "用途制限")
    assert use.status == "ng"
    assert check.status == "ng"


# ─── LAW-3: 要確認フラグ ──────────────────────────

def test_law3_manual_flags_cover_non_computable_items():
    """自動判定できない項目が要確認フラグとして出力される"""
    site = make_site(zoning="第一種低層住居専用地域")
    check = check_plan(make_plan(), 1, site)
    flags = {f.item for f in check.manual_flags}

    assert "道路斜線制限" in flags          # 全用途地域で該当
    assert "北側斜線制限" in flags          # 低層住居専用地域で該当
    assert "日影規制" in flags
    assert "外壁の後退距離" in flags        # 低層住居専用地域で該当
    assert "接道義務" in flags
    assert "自治体条例・地区計画・高度地区" in flags
    # 自動判定した項目が要確認フラグに重複していないこと（LAW-3 の切り分け）
    assert not ({"建ぺい率", "容積率", "絶対高さ制限"} & flags)
    print(f"✅ LAW-3 PASS: 要確認フラグ {len(check.manual_flags)}件 = {sorted(flags)}")


def test_law3_north_slope_not_flagged_outside_residential_zones():
    """北側斜線が適用されない用途地域ではフラグを立てない"""
    flags = {f.item for f in check_plan(make_plan(), 1, make_site(zoning="商業地域")).manual_flags}
    assert "北側斜線制限" not in flags
    assert "外壁の後退距離" not in flags
    assert "道路斜線制限" in flags


def test_law3_setback_flag_when_road_narrower_than_4m():
    """幅員4m未満はセットバックの要確認フラグを立てる（第42条2項）"""
    flags = {f.item: f for f in check_plan(make_plan(), 1, make_site(road_width_m=3.0)).manual_flags}
    assert "セットバック（2項道路）" in flags
    assert flags["セットバック（2項道路）"].severity == "high"


def test_law3_unknown_fire_zone_is_high_severity():
    """防火地域が未確認なら優先度 high の要確認フラグにする"""
    flags = {f.item: f for f in check_plan(make_plan(), 1, make_site(fire_zone=None)).manual_flags}
    assert flags["防火・準防火地域の指定"].severity == "high"

    # 指定済みなら仕様規制の確認として medium
    flags2 = {f.item: f for f in check_plan(make_plan(), 1, make_site(fire_zone="準防火地域")).manual_flags}
    assert "準防火地域の仕様規制" in flags2
    assert "防火・準防火地域の指定" not in flags2


def test_law3_no_land_adds_recheck_flag():
    """土地未確定時は仮定による参考判定であることを要確認フラグで明示する"""
    site = make_site(has_land=False, source="assumed",
                     assumptions=["土地が未確定のため、標準的な敷地条件による参考判定"])
    output = check_plans([make_plan()], site)

    flags = {f.item: f for f in output.checks[0].manual_flags}
    assert "敷地確定後の再チェック" in flags
    assert flags["敷地確定後の再チェック"].severity == "high"
    assert output.site_info.assumptions
    assert "参考判定" in output.summary
    print("✅ 敷地なし PASS: 仮定値による参考判定として要確認フラグが立つ")


# ─── LAW-4: 免責表示 ──────────────────────────────

def test_law4_disclaimer_states_no_guarantee():
    """法規判定は適法性を保証しない旨を明記する"""
    output = check_plans([make_plan()], make_site())
    assert "保証するものではありません" in output.disclaimer
    assert "建築士" in output.disclaimer
    assert "参考情報" in output.disclaimer
    print("✅ LAW-4 PASS: 免責表示あり")


def test_references_collect_cited_articles():
    """根拠条文が参照リストに集約される（e-Gov 法令API 連携の受け口）"""
    output = check_plans([make_plan()], make_site())
    assert "建築基準法 第53条" in output.references
    assert "建築基準法 第52条" in output.references
    assert len(output.references) == len(set(output.references))   # 重複しない


# ─── LAW-1: 敷地照会 ──────────────────────────────

def test_law1_site_lookup_from_user_input():
    """ユーザー発話から敷地情報を抽出し、用途地域から法定既定値を補完する"""
    from app.tools.site_lookup import lookup_site

    req = RequirementBaseline(
        land_info="さいたま市の相続した土地、約50坪、第一種低層住居専用地域",
        desired_size="4LDK 35坪",
    )
    llm = mock_site_llm({
        "address": "さいたま市", "site_area_sqm": 165.3,
        "zoning": "第一種低層住居専用地域", "road_width_m": None,
        "fire_zone": None, "has_land": True,
    })
    site = lookup_site(req, llm)

    assert site.has_land is True
    assert site.zoning == "第一種低層住居専用地域"
    assert site.site_area_sqm == 165.3
    assert site.height_limit_m == 10.0            # 用途地域から絶対高さ制限を補完
    assert site.building_coverage_ratio is not None
    assert site.floor_area_ratio is not None
    assert site.assumptions                       # 補完した内容が記録される
    print(f"✅ LAW-1 PASS: source={site.source} / {site.zoning} / 建ぺい{site.building_coverage_ratio}%")


def test_law1_site_lookup_without_land_uses_assumption():
    """土地なしの場合は仮定値で埋め、has_land=False と仮定内容を残す"""
    from app.tools.site_lookup import lookup_site

    req = RequirementBaseline(land_info="土地はまだ持っていないので、埼玉県内で探す予定です")
    llm = mock_site_llm({
        "address": None, "site_area_sqm": None, "zoning": None,
        "road_width_m": None, "fire_zone": None, "has_land": False,
    })
    site = lookup_site(req, llm)

    assert site.has_land is False
    assert site.source == "assumed"
    assert site.site_area_sqm is not None         # 仮定値で判定は可能にする
    assert any("未確定" in a for a in site.assumptions)


def test_law1_site_lookup_survives_broken_llm_output():
    """LLM が壊れた出力を返してもデモを止めない（NFR-06）"""
    from app.tools.site_lookup import lookup_site

    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content="JSONではない応答")
    site = lookup_site(RequirementBaseline(land_info="土地あり"), llm)

    assert site.site_area_sqm is not None         # 仮定値にフォールバックする
    assert site.source == "assumed"


def test_law1_reinfolib_disabled_without_api_key(monkeypatch):
    """APIキー未設定なら外部APIを呼ばない（オフラインでデモ完走可能）"""
    from app.tools.site_lookup import fetch_from_reinfolib

    monkeypatch.delenv("REINFOLIB_API_KEY", raising=False)
    assert fetch_from_reinfolib("さいたま市") is None


def test_zoning_normalization_handles_variants():
    assert normalize_zoning("第1種低層住居専用地域") == "第一種低層住居専用地域"
    assert normalize_zoning("一低層") == "第一種低層住居専用地域"
    assert normalize_zoning("よくわからない") is None
    assert get_rule(None).name == "用途地域不明"


# ─── 法規チェックAI（エージェント）─────────────────

def test_legal_agent_checks_all_plans():
    """全案ぶんの判定結果が返る"""
    from app.agents.legal_agent import build_legal_reply, run_legal_check

    plans = [
        make_plan("コスパ重視案", "約100㎡（約30坪）", "2階建て"),
        make_plan("広さ重視案", "約200㎡（約60坪）", "3階建て"),
        make_plan("収納重視案", "約115㎡（約35坪）", "2階建て"),
    ]
    site = make_site(zoning="第一種低層住居専用地域", building_coverage_ratio=50.0,
                     floor_area_ratio=100.0, site_area_sqm=165.3, road_width_m=4.0)
    output = run_legal_check(RequirementBaseline(), plans, MagicMock(), site_info=site)

    assert len(output.checks) == 3
    assert [c.plan_index for c in output.checks] == [1, 2, 3]
    # 広さ重視案（3階建て・延床200㎡）は容積率・高さで不適合になる
    assert output.checks[1].status == "ng"

    reply = build_legal_reply(output)
    assert "法規チェック結果" in reply
    assert "要確認項目" in reply
    assert "保証するものではありません" in reply

    # compose ノードが免責を一括付与する場合は個別の免責を出さない
    assert "保証するものではありません" not in build_legal_reply(output, include_disclaimer=False)
    print(f"✅ 法規チェックAI PASS: 3案判定 = {[c.status for c in output.checks]}")


def test_legal_agent_skips_llm_when_site_given():
    """site_info を渡した場合は敷地照会（LLM）を呼ばない"""
    from app.agents.legal_agent import run_legal_check

    llm = MagicMock()
    run_legal_check(RequirementBaseline(), [make_plan()], llm, site_info=make_site())
    llm.invoke.assert_not_called()


def test_replan_constraints_only_when_ng():
    """不適合がある場合のみ間取り生成AIへの修正指示を作る"""
    from app.agents.legal_agent import build_replan_constraints, run_legal_check

    site = make_site(zoning="第一種低層住居専用地域", building_coverage_ratio=50.0,
                     floor_area_ratio=100.0, site_area_sqm=165.3)

    ok_output = run_legal_check(RequirementBaseline(), [make_plan(area="約100㎡", floors="2階建て")],
                                MagicMock(), site_info=site)
    assert build_replan_constraints(ok_output) is None

    ng_output = run_legal_check(RequirementBaseline(), [make_plan(area="約250㎡", floors="3階建て")],
                                MagicMock(), site_info=site)
    constraints = build_replan_constraints(ng_output)
    assert constraints is not None
    assert "建築面積の上限" in constraints
    assert "延床面積の上限" in constraints
    assert "建物高さの上限" in constraints
    print("✅ 修正指示 PASS: NG時のみ制約テキストを生成")


def test_check_plans_with_no_plans():
    """間取り案が空でも例外を出さない"""
    output = check_plans([], make_site())
    assert output.checks == []
    assert "ありません" in output.summary
