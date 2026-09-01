"""法規チェックAI — 間取り案の法規適合を判定するエージェント（要件定義書 §7.6）

| 要件 | 実装 |
|---|---|
| LAW-1 外部API連携 | `app/tools/site_lookup.py`（不動産情報ライブラリ API／プリセット／仮定値） |
| LAW-2 建ぺい率・容積率・高さの判定 | `app/tools/legal_rules.py`（決定論的な算術判定） |
| LAW-3 自動判定と要確認の切り分け | `LegalCheckItem`（自動）と `ManualCheckFlag`（要確認フラグ） |
| LAW-4 適法性を保証しない旨の表示 | `LEGAL_DISCLAIMER` を出力に必ず含める |

LLM は「自由文からの敷地情報の抽出」にのみ用い、判定と文面生成は決定論的に行う。
同じ敷地情報・間取りに対して常に同じ判定文を返すため、デモの再現性を損なわない（NFR-01）。
"""
from __future__ import annotations

from typing import Optional

from app.schemas.floorplan import FloorPlan
from app.schemas.legal import LegalCheckOutput, PlanLegalCheck, SiteInfo
from app.schemas.requirements import RequirementBaseline
from app.tools.legal_rules import check_plans
from app.tools.site_lookup import lookup_site
from app.tools.zoning import get_rule

# 判定ステータスの表示記号
_STATUS_MARK = {"ok": "✅ 適合", "ng": "⚠️ 不適合の可能性", "unknown": "❓ 判定材料不足"}

# 要確認フラグの優先度の表示記号
_SEVERITY_MARK = {"high": "🔴", "medium": "🟡", "low": "⚪"}

# 敷地情報の取得元の表示名（どこまでが実データかを利用者に開示する）
_SOURCE_LABEL = {
    "user_input": "ご入力内容",
    "reinfolib": "不動産情報ライブラリ（国土交通省）",
    "preset": "対象自治体のプリセットデータ",
    "assumed": "標準的な敷地条件の仮定値",
}


def run_legal_check(
    requirements: RequirementBaseline,
    plans: list[FloorPlan],
    llm,
    site_info: Optional[SiteInfo] = None,
) -> LegalCheckOutput:
    """住宅要件書と間取り案から法規チェックを実行する

    site_info を渡した場合は敷地照会（LLM 呼び出し）をスキップする。
    自律修正ループの2周目のように、同一敷地で再判定する場合に使う。
    """
    site = site_info if site_info is not None else lookup_site(requirements, llm)
    return check_plans(plans, site)


# ─────────────────────────────────────────
# ユーザー向け応答の生成（ORC-4 翻訳）
# ─────────────────────────────────────────

def _format_site_info(site: SiteInfo) -> str:
    rows = [
        ("所在地", site.address or "未確定"),
        ("敷地面積", f"{site.site_area_sqm:.1f}㎡（約{site.site_area_sqm / 3.305785:.1f}坪）" if site.site_area_sqm else "未確定"),
        ("用途地域", site.zoning or "未確認"),
        ("指定建ぺい率", f"{site.building_coverage_ratio:.0f}%" if site.building_coverage_ratio is not None else "未確認"),
        ("指定容積率", f"{site.floor_area_ratio:.0f}%" if site.floor_area_ratio is not None else "未確認"),
        ("前面道路幅員", f"{site.road_width_m}m" if site.road_width_m is not None else "未確認"),
        ("防火地域", site.fire_zone or "未確認"),
    ]
    lines = [f"- {label}：{value}" for label, value in rows]
    lines.append(f"- データ出典：{_SOURCE_LABEL.get(site.source, site.source)}")
    if site.assumptions:
        lines.append("- 前提・仮定：" + " / ".join(site.assumptions))
    return "\n".join(lines)


def _shared_flag_items(output: LegalCheckOutput) -> set[str]:
    """全案に共通する要確認フラグ（＝敷地に対する制限）の項目名"""
    if not output.checks:
        return set()
    common = {f.item for f in output.checks[0].manual_flags}
    for check in output.checks[1:]:
        common &= {f.item for f in check.manual_flags}
    return common


def _format_plan_check(check: PlanLegalCheck, shared_items: set[str]) -> str:
    lines = [f"**案{check.plan_index}：{check.plan_concept}** — {_STATUS_MARK.get(check.status, check.status)}"]
    for item in check.items:
        mark = {"ok": "✅", "ng": "⚠️", "unknown": "❓"}.get(item.status, "・")
        detail = f"{item.actual} / 上限 {item.limit}" if item.actual else "判定材料が不足"
        margin = f"（{item.margin}）" if item.margin else ""
        lines.append(f"- {mark} {item.item}：{detail}{margin}　〔{item.basis}〕")
    # この案だけに該当する要確認フラグ（間取りデータの矛盾など）は案ごとに示す
    for flag in check.manual_flags:
        if flag.item not in shared_items:
            mark = _SEVERITY_MARK.get(flag.severity, "・")
            lines.append(f"- {mark} **{flag.item}** — {flag.reason}")
    lines.append(f"- {check.summary}")
    return "\n".join(lines)


def build_legal_reply(output: LegalCheckOutput, include_disclaimer: bool = True) -> str:
    """法規チェック結果をユーザー向けの日本語セクションに整形する

    include_disclaimer=False は、オーケストレーターが応答全体の末尾に統合の免責表示を
    付与する場合（compose ノード）に使う。構造化データ側の
    `LegalCheckOutput.disclaimer` は常に免責文を保持するため LAW-4 は満たされる。
    """
    if not output.checks:
        return "法規チェックの対象となる間取り案がありませんでした。"

    parts = [
        "### ⚖️ 法規チェック結果（参考判定）",
        "",
        "**判定に用いた敷地条件**",
        _format_site_info(output.site_info),
        "",
        "**自動判定（建ぺい率・容積率・高さ）**",
    ]
    shared_items = _shared_flag_items(output)
    for check in output.checks:
        parts.append("")
        parts.append(_format_plan_check(check, shared_items))

    # 敷地に対する制限は全案で共通のため1回だけ提示する
    shared_flags = [f for f in output.checks[0].manual_flags if f.item in shared_items]
    if shared_flags:
        parts.extend(["", "**要確認項目（専門家の確認が必要／自動判定の対象外）**"])
        for flag in shared_flags:
            mark = _SEVERITY_MARK.get(flag.severity, "・")
            parts.append(f"- {mark} **{flag.item}** — {flag.reason}　〔{flag.basis}〕")

    parts.extend(["", output.summary])
    if include_disclaimer:
        parts.extend(["", f"> {output.disclaimer}"])
    return "\n".join(parts)


# ─────────────────────────────────────────
# 自律修正ループへの引き渡し
# ─────────────────────────────────────────

def build_replan_constraints(output: LegalCheckOutput) -> Optional[str]:
    """不適合があった場合、間取り生成AIに渡す制約条件テキストを組み立てる

    不適合が無ければ None を返す（＝再生成しない）。
    """
    ng_checks = [c for c in output.checks if c.status == "ng"]
    if not ng_checks:
        return None

    site = output.site_info
    lines = ["## 法規チェックで不適合となったため、以下の制約を満たすように修正してください"]

    if site.site_area_sqm:
        lines.append(f"- 敷地面積: {site.site_area_sqm:.1f}㎡")
    if site.zoning:
        lines.append(f"- 用途地域: {site.zoning}")
    if site.building_coverage_ratio is not None and site.site_area_sqm:
        max_footprint = site.site_area_sqm * site.building_coverage_ratio / 100
        lines.append(
            f"- 建築面積の上限: {max_footprint:.1f}㎡"
            f"（建ぺい率 {site.building_coverage_ratio:.0f}%）"
            "　※1階床面積がこの範囲に収まるようにしてください"
        )

    # 容積率の上限は判定時に採用した値（基準容積率を含む）をそのまま使う
    for item in ng_checks[0].items:
        if item.item == "容積率" and item.limit and site.site_area_sqm:
            limit_pct = float(item.limit.split("%")[0])
            lines.append(
                f"- 延床面積の上限: {site.site_area_sqm * limit_pct / 100:.1f}㎡"
                f"（容積率 {limit_pct:.0f}%）"
            )
            break

    # 高さ制限は判定時と同じ解決順（敷地の指定値 → 用途地域の法定値）で求める
    height_limit = (
        site.height_limit_m
        if site.height_limit_m is not None
        else get_rule(site.zoning).absolute_height_limit_m
    )
    if height_limit is not None:
        max_floors = 2 if height_limit <= 10 else 3
        lines.append(
            f"- 建物高さの上限: {height_limit:.0f}m"
            f"　※{max_floors}階建て以下にしてください（階高3m＋屋根2mで概算）"
        )

    lines.append("")
    lines.append("### 各案の不適合内容")
    for check in ng_checks:
        for item in check.items:
            if item.status == "ng":
                lines.append(f"- 案{check.plan_index}（{check.plan_concept}）: {item.item} — {item.message}")

    lines.append("")
    lines.append("上記の上限を守りつつ、元の3案のコンセプト（コスパ重視／広さ重視／収納重視）は維持してください。")
    return "\n".join(lines)
