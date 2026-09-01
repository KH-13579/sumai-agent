"""法規計算ツール — 建ぺい率・容積率・高さの決定論的判定（LAW-2/LAW-3）

要件定義書 v1.0 §7.6 のうち「自動判定できる部分」を算術で処理し、自動判定できない
項目（斜線・日影・条例・防火地域の仕様規制）を要確認フラグとして生成する。

LLM を介在させないため、同一入力に対して常に同一の出力を返す（NFR-01 デモ再現性）。
"""
from __future__ import annotations

from typing import Optional

from app.schemas.floorplan import FloorPlan
from app.schemas.legal import (
    CheckStatus,
    LegalCheckItem,
    LegalCheckOutput,
    ManualCheckFlag,
    PlanLegalCheck,
    SiteInfo,
)
from app.tools.area_utils import (
    ROOM_SUM_TOLERANCE,
    parse_area_sqm,
    parse_floor_count,
    sum_room_areas_sqm,
)
from app.tools.zoning import ZoningRule, get_rule

# 浮動小数の丸め誤差で不適合と判定しないための許容値（パーセントポイント／m）
_EPSILON = 0.05

# 延床面積に対する1階床面積（≒建築面積）の比率。
# 一般的な戸建てでは1階が2階より大きいため、単純な「延床÷階数」より安全側（大きめ）に見積もる。
_FOOTPRINT_RATIO = {1: 1.00, 2: 0.55, 3: 0.40}

# 階数から推定する建物高さ（m）。階高3m ＋ 屋根・基礎で 2m を加算。
_HEIGHT_BY_FLOORS = {1: 5.0, 2: 8.0, 3: 11.0}

# 階数が読み取れない場合に仮定する階数
_DEFAULT_FLOORS = 2

LEGAL_DISCLAIMER = (
    "※ 本判定はAIによる参考情報であり、適法性を保証するものではありません（LAW-4）。"
    "建ぺい率・容積率・高さは入力条件と概算値からの機械判定です。"
    "斜線制限・日影規制・自治体条例・防火地域の仕様規制は自動判定の対象外です。"
    "設計・着工前に必ず建築士または指定確認検査機関にご確認ください。"
)


# ─────────────────────────────────────────
# 間取りからの数値推定
# ─────────────────────────────────────────

def estimate_total_floor_area_sqm(plan: FloorPlan) -> Optional[float]:
    """間取り案の延床面積（㎡）を推定する"""
    return parse_area_sqm(plan.total_floor_area)


def estimate_footprint_sqm(plan: FloorPlan) -> Optional[float]:
    """間取り案の建築面積（≒1階床面積、㎡）を推定する"""
    total = estimate_total_floor_area_sqm(plan)
    if total is None:
        return None
    floors = parse_floor_count(plan.floors) or _DEFAULT_FLOORS
    ratio = _FOOTPRINT_RATIO.get(floors, min(1.0, 1.1 / floors))
    return round(total * ratio, 1)


def estimate_height_m(floors: Optional[int]) -> float:
    """階数から建物高さ（m）を推定する"""
    count = floors or _DEFAULT_FLOORS
    return _HEIGHT_BY_FLOORS.get(count, count * 3.0 + 2.0)


# ─────────────────────────────────────────
# 間取りデータの整合性確認
# ─────────────────────────────────────────

def sum_room_area_sqm(plan: FloorPlan) -> float:
    """面積として読み取れた部屋の合計（㎡）。延床面積の下限の目安になる"""
    return sum_room_areas_sqm([r.area for r in plan.rooms])


def check_plan_consistency(plan: FloorPlan) -> tuple[bool, Optional[str]]:
    """間取りデータの内部整合性を確認する

    部屋面積の合計が延床面積を超えている場合、延床面積の桁・単位が誤っている
    （坪と㎡の混同など）可能性が高い。この状態で法規判定を「適合」と出すと
    誤った安心を与えるため、矛盾を検出して呼び出し側に知らせる。

    戻り値は (整合しているか, 矛盾の説明文)。
    """
    total = estimate_total_floor_area_sqm(plan)
    room_sum = sum_room_area_sqm(plan)
    if total is None or room_sum <= 0:
        return True, None

    if room_sum > total * ROOM_SUM_TOLERANCE:
        return False, (
            f"延床面積 {total:.1f}㎡ に対して主要な部屋の面積合計が {room_sum:.1f}㎡ あり、"
            "数値が矛盾しています（延床面積の単位が坪と㎡で混同されている可能性があります）。"
            "面積の前提が確定できないため、建ぺい率・容積率は判定できません。"
        )
    return True, None


# ─────────────────────────────────────────
# 容積率の上限（第52条）
# ─────────────────────────────────────────

def effective_far(
    rule: ZoningRule, designated_far: float, road_width_m: Optional[float]
) -> tuple[float, str]:
    """容積率の上限を min(指定容積率, 前面道路幅員 × 法定係数) で求める

    幅員 12m 未満の場合、指定容積率より基準容積率（道路幅員による制限）が厳しくなるのが
    通例のため、両者の小さい方を採用する。角地・防火地域の緩和は見込まない（安全側）。
    戻り値は (上限%, 適用根拠の説明文)。
    """
    if road_width_m is None:
        return designated_far, "指定容積率（前面道路幅員が不明のため基準容積率は未考慮）"
    if road_width_m >= 12:
        return designated_far, "指定容積率（前面道路幅員12m以上のため基準容積率の制限なし）"

    road_far = road_width_m * rule.far_road_coefficient * 100
    if road_far < designated_far:
        return (
            round(road_far, 1),
            f"基準容積率を適用 — 前面道路幅員 {road_width_m}m × {rule.far_road_coefficient:.1f} "
            f"が指定容積率 {designated_far:.0f}% より厳しい",
        )
    return designated_far, f"指定容積率を適用 — 基準容積率 {road_far:.0f}% より厳しい"


# ─────────────────────────────────────────
# 自動判定（LAW-2）
# ─────────────────────────────────────────

def _ratio_item(
    *,
    item: str,
    actual_value: Optional[float],
    limit_value: float,
    numerator: Optional[float],
    numerator_label: str,
    site_area: Optional[float],
    basis: str,
    limit_note: str,
    assumption_note: str,
) -> LegalCheckItem:
    """建ぺい率・容積率のような「面積比 ≦ 上限%」型の判定項目を生成する"""
    if actual_value is None:
        return LegalCheckItem(
            item=item,
            status="unknown",
            actual=None,
            limit=f"{limit_value:.0f}%",
            basis=basis,
            message=(
                f"{numerator_label}または敷地面積が特定できないため自動判定できません。"
                "敷地面積と延床面積を確定のうえ再チェックが必要です。"
            ),
        )

    over = actual_value - limit_value
    status: CheckStatus = "ok" if over <= _EPSILON else "ng"
    margin = (
        f"余裕 {abs(over):.1f} ポイント" if status == "ok" else f"超過 {over:.1f} ポイント"
    )
    verdict = "上限内に収まっています" if status == "ok" else "上限を超過しています"
    return LegalCheckItem(
        item=item,
        status=status,
        actual=f"{actual_value:.1f}%（{numerator_label} {numerator:.1f}㎡ ÷ 敷地 {site_area:.1f}㎡）",
        limit=f"{limit_value:.0f}%（{limit_note}）",
        margin=margin,
        basis=basis,
        message=f"{item} {actual_value:.1f}% は上限 {limit_value:.0f}% に対し{verdict}。{assumption_note}",
    )


def _check_zoning_use(rule: ZoningRule) -> Optional[LegalCheckItem]:
    """用途制限（別表第二）— 住宅を建てられない用途地域かを判定する"""
    if rule.residential_allowed:
        return None
    return LegalCheckItem(
        item="用途制限",
        status="ng",
        actual=rule.name,
        limit="住宅の建築が可能な用途地域",
        basis="建築基準法 別表第二",
        message=f"{rule.name}では住宅を建築できません。敷地の用途地域をご確認ください。",
    )


def _check_bcr(plan: FloorPlan, site: SiteInfo, rule: ZoningRule) -> LegalCheckItem:
    """建ぺい率（第53条）"""
    limit = site.building_coverage_ratio if site.building_coverage_ratio is not None else rule.default_bcr
    footprint = estimate_footprint_sqm(plan)
    ratio = None
    if footprint is not None and site.site_area_sqm:
        ratio = footprint / site.site_area_sqm * 100

    floors = parse_floor_count(plan.floors) or _DEFAULT_FLOORS
    ratio_pct = int(_FOOTPRINT_RATIO.get(floors, min(1.0, 1.1 / floors)) * 100)
    limit_note = (
        "指定建ぺい率" if site.building_coverage_ratio is not None else f"{rule.name}の一般的な指定値を仮定"
    )
    return _ratio_item(
        item="建ぺい率",
        actual_value=ratio,
        limit_value=limit,
        numerator=footprint,
        numerator_label="建築面積",
        site_area=site.site_area_sqm,
        basis="建築基準法 第53条",
        limit_note=limit_note,
        assumption_note=(
            f"建築面積は延床面積の{ratio_pct}%（{floors}階建ての1階床面積相当）として概算しており、"
            "軒の出・バルコニー・角地緩和は未考慮です。"
        ),
    )


def _check_far(plan: FloorPlan, site: SiteInfo, rule: ZoningRule) -> LegalCheckItem:
    """容積率（第52条）"""
    designated = site.floor_area_ratio if site.floor_area_ratio is not None else rule.default_far
    limit, limit_note = effective_far(rule, designated, site.road_width_m)
    total = estimate_total_floor_area_sqm(plan)
    ratio = None
    if total is not None and site.site_area_sqm:
        ratio = total / site.site_area_sqm * 100

    return _ratio_item(
        item="容積率",
        actual_value=ratio,
        limit_value=limit,
        numerator=total,
        numerator_label="延床面積",
        site_area=site.site_area_sqm,
        basis="建築基準法 第52条",
        limit_note=limit_note,
        assumption_note=(
            "車庫・地下室の容積率不算入や共同住宅の共用部分の緩和は未考慮です（安全側の判定）。"
        ),
    )


def _check_height(plan: FloorPlan, site: SiteInfo, rule: ZoningRule) -> Optional[LegalCheckItem]:
    """絶対高さ制限（第55条）— 低層住居専用地域・田園住居地域のみ該当"""
    limit = site.height_limit_m if site.height_limit_m is not None else rule.absolute_height_limit_m
    if limit is None:
        return None

    floors = parse_floor_count(plan.floors)
    height = estimate_height_m(floors)
    over = height - limit
    status: CheckStatus = "ok" if over <= _EPSILON else "ng"
    floors_label = f"{floors}階建て" if floors else f"階数不明（{_DEFAULT_FLOORS}階建てと仮定）"
    verdict = "制限内です" if status == "ok" else "制限を超過する可能性があります"
    return LegalCheckItem(
        item="絶対高さ制限",
        status=status,
        actual=f"約{height:.1f}m（{floors_label}）",
        limit=f"{limit:.0f}m",
        margin=f"余裕 約{abs(over):.1f}m" if status == "ok" else f"超過 約{over:.1f}m",
        basis="建築基準法 第55条",
        message=(
            f"推定建物高さ 約{height:.1f}m は{rule.name}の絶対高さ制限 {limit:.0f}m に対し{verdict}。"
            "高さは階高3m＋屋根・基礎2mの概算であり、実際の屋根形状・地盤高さで変動します。"
        ),
    )


# ─────────────────────────────────────────
# 要確認フラグ（LAW-3）
# ─────────────────────────────────────────

def build_manual_flags(site: SiteInfo, rule: ZoningRule) -> list[ManualCheckFlag]:
    """自動判定できない法規項目を要確認フラグとして生成する"""
    flags: list[ManualCheckFlag] = [
        ManualCheckFlag(
            item="道路斜線制限",
            reason=(
                "前面道路の反対側境界線からの勾配制限。敷地形状・道路との高低差・"
                "セットバックによる緩和の判断が必要で、数値のみでは判定できません。"
            ),
            basis="建築基準法 第56条1項1号",
            severity="high",
        ),
    ]

    if rule.has_north_slope:
        flags.append(
            ManualCheckFlag(
                item="北側斜線制限",
                reason=(
                    f"{rule.name}では北側隣地の日照を守るための勾配制限があります。"
                    "真北方向・隣地の地盤高さの測量が必要です。"
                ),
                basis="建築基準法 第56条1項3号",
                severity="high",
            )
        )

    if rule.has_shadow_regulation:
        flags.append(
            ManualCheckFlag(
                item="日影規制",
                reason=(
                    "冬至日の日影時間の規制。建物形状・周辺敷地との関係を用いた"
                    "日影図の作成が必要です（自治体が定める測定水平面・規制時間により異なります）。"
                ),
                basis="建築基準法 第56条の2・別表第四",
                severity="medium",
            )
        )

    if rule.has_wall_setback:
        flags.append(
            ManualCheckFlag(
                item="外壁の後退距離",
                reason=(
                    f"{rule.name}では都市計画により外壁を隣地境界から1mまたは1.5m"
                    "後退させる定めがある場合があります。都市計画の指定内容の確認が必要です。"
                ),
                basis="建築基準法 第54条",
                severity="medium",
            )
        )

    if site.fire_zone in (None, "", "不明"):
        flags.append(
            ManualCheckFlag(
                item="防火・準防火地域の指定",
                reason=(
                    "防火地域・準防火地域の指定が未確認です。指定がある場合、"
                    "外壁・軒裏・開口部の仕様や構造に制限が加わり、工事費にも影響します。"
                ),
                basis="建築基準法 第61条・第62条",
                severity="high",
            )
        )
    elif site.fire_zone not in ("指定なし",):
        flags.append(
            ManualCheckFlag(
                item=f"{site.fire_zone}の仕様規制",
                reason=(
                    f"{site.fire_zone}に指定されています。外壁・軒裏・開口部の防火仕様および"
                    "構造制限への適合を設計段階で確認してください。"
                ),
                basis="建築基準法 第61条・第62条",
                severity="medium",
            )
        )

    if site.road_width_m is not None and site.road_width_m < 4:
        flags.append(
            ManualCheckFlag(
                item="セットバック（2項道路）",
                reason=(
                    f"前面道路の幅員が {site.road_width_m}m と4m未満です。道路中心線から2mまでの"
                    "後退（セットバック）が必要となり、有効な敷地面積が減少します。"
                ),
                basis="建築基準法 第42条2項",
                severity="high",
            )
        )

    flags.append(
        ManualCheckFlag(
            item="接道義務",
            reason=(
                "幅員4m以上の道路に2m以上接している必要があります。"
                "接道の幅・位置は敷地図と現況の確認が必要です。"
            ),
            basis="建築基準法 第43条",
            severity="medium",
        )
    )

    flags.append(
        ManualCheckFlag(
            item="自治体条例・地区計画・高度地区",
            reason=(
                "建築基準法に加えて自治体の条例・地区計画・高度地区・景観計画による"
                "独自の制限がある場合があります。所管の建築指導課へご確認ください。"
            ),
            basis="建築基準法 第68条の2・各自治体条例",
            severity="high",
        )
    )

    if not site.has_land:
        flags.append(
            ManualCheckFlag(
                item="敷地確定後の再チェック",
                reason=(
                    "土地が未確定のため、標準的な敷地条件を仮定した参考判定です。"
                    "土地の取得後に実際の敷地面積・用途地域・接道条件で再判定が必要です。"
                ),
                basis="—",
                severity="high",
            )
        )

    return flags


# ─────────────────────────────────────────
# 判定の実行
# ─────────────────────────────────────────

def _overall_status(items: list[LegalCheckItem]) -> CheckStatus:
    """総合判定。ng > unknown > ok の優先順で決める"""
    statuses = {item.status for item in items}
    if "ng" in statuses:
        return "ng"
    if "unknown" in statuses:
        return "unknown"
    return "ok"


def _inconsistent_area_item(item: str, basis: str, reason: str) -> LegalCheckItem:
    """間取りデータが矛盾している場合の面積比項目（判定不能として扱う）"""
    return LegalCheckItem(
        item=item,
        status="unknown",
        actual=None,
        limit=None,
        basis=basis,
        message=reason,
    )


def check_plan(plan: FloorPlan, plan_index: int, site: SiteInfo) -> PlanLegalCheck:
    """間取り1案に対して法規チェックを実行する"""
    rule = get_rule(site.zoning)
    is_consistent, inconsistency = check_plan_consistency(plan)

    items: list[LegalCheckItem] = []
    use_item = _check_zoning_use(rule)
    if use_item:
        items.append(use_item)

    if is_consistent:
        items.append(_check_bcr(plan, site, rule))
        items.append(_check_far(plan, site, rule))
    else:
        # 面積の前提が壊れている状態で「適合」と出すと誤った安心を与えるため判定を保留する
        items.append(_inconsistent_area_item("建ぺい率", "建築基準法 第53条", inconsistency or ""))
        items.append(_inconsistent_area_item("容積率", "建築基準法 第52条", inconsistency or ""))

    height_item = _check_height(plan, site, rule)
    if height_item:
        items.append(height_item)

    status = _overall_status(items)
    ng_items = [i.item for i in items if i.status == "ng"]
    unknown_items = [i.item for i in items if i.status == "unknown"]

    if status == "ng":
        summary = f"自動判定で不適合の可能性があります（{'・'.join(ng_items)}）。間取りの調整または敷地条件の再確認が必要です。"
    elif status == "unknown":
        summary = f"判定材料が不足しています（{'・'.join(unknown_items)}）。敷地面積・延床面積を確定のうえ再チェックしてください。"
    else:
        summary = "自動判定できる項目（建ぺい率・容積率・高さ）はいずれも制限内です。"

    flags = build_manual_flags(site, rule)
    if not is_consistent:
        flags.insert(
            0,
            ManualCheckFlag(
                item="間取りデータの整合性",
                reason=(inconsistency or "") + "延床面積を確定のうえ再チェックしてください。",
                basis="—",
                severity="high",
            ),
        )

    return PlanLegalCheck(
        plan_index=plan_index,
        plan_concept=plan.concept,
        status=status,
        items=items,
        manual_flags=flags,
        summary=summary,
    )


def check_plans(plans: list[FloorPlan], site: SiteInfo) -> LegalCheckOutput:
    """間取り全案に対して法規チェックを実行し、総括をまとめる"""
    checks = [check_plan(plan, i, site) for i, plan in enumerate(plans, 1)]

    ok_count = sum(1 for c in checks if c.status == "ok")
    ng_count = sum(1 for c in checks if c.status == "ng")
    unknown_count = sum(1 for c in checks if c.status == "unknown")

    parts = [f"全{len(checks)}案のうち"]
    if ok_count:
        parts.append(f"{ok_count}案が自動判定項目をクリア")
    if ng_count:
        ng_labels = "・".join(f"案{c.plan_index}" for c in checks if c.status == "ng")
        parts.append(f"{ng_count}案（{ng_labels}）に不適合の可能性")
    if unknown_count:
        parts.append(f"{unknown_count}案は判定材料が不足")
    summary = "、".join(parts) + "。" if checks else "対象となる間取り案がありません。"

    if not site.has_land:
        summary += "土地が未確定のため、標準的な敷地条件を仮定した参考判定です。"

    # 参照した条文（e-Gov 法令API による条文取得の受け口）
    references: list[str] = []
    for check in checks:
        for item in check.items:
            if item.basis not in references and item.basis != "—":
                references.append(item.basis)
        for flag in check.manual_flags:
            if flag.basis not in references and flag.basis != "—":
                references.append(flag.basis)

    return LegalCheckOutput(
        site_info=site,
        checks=checks,
        summary=summary,
        disclaimer=LEGAL_DISCLAIMER,
        references=references,
    )
