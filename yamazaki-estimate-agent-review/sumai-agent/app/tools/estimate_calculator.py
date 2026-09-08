"""再現可能な決定論的概算見積計算ツール。LLMに金額計算をさせない。"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from app.schemas.estimate import (
    CostLine,
    EstimateRequest,
    EstimateResult,
    SavingsOption,
)

SQM_PER_TSUBO = 3.305785
DISCLAIMER = (
    "本結果は公的統計を基準にした研究用モックの概算です。個別敷地、地域差、"
    "設計・申請、地盤改良、外構、諸費用、税等を反映していません。"
    "契約・発注には使用せず、建築士・施工会社の正式見積を確認してください。"
)


@lru_cache(maxsize=1)
def load_rates() -> dict:
    path = Path(__file__).parents[1] / "data" / "estimate_rates.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _round_10k(value: float) -> int:
    return int(round(value / 10_000) * 10_000)


def calculate_estimate(req: EstimateRequest) -> EstimateResult:
    rates = load_rates()
    tsubo = req.floor_area_sqm / SQM_PER_TSUBO
    structure_factor = rates["structure_factors"][req.structure]
    floor_factor = rates["floor_factors"][str(req.floors)]
    grade_factor = rates["grade_factors"][req.grade]

    base = tsubo * rates["base_yen_per_tsubo"]
    adjusted = base * structure_factor * floor_factor * grade_factor
    lines = [
        CostLine(
            label="建物建設費（補正後）",
            amount_yen=_round_10k(adjusted),
            calculation=(
                f"{tsubo:.1f}坪 × {rates['base_yen_per_tsubo']:,}円/坪 × "
                f"構造{structure_factor:.2f} × 階数{floor_factor:.2f} × 仕様{grade_factor:.2f}"
            ),
        )
    ]

    equipment_total = 0.0
    selected_equipment = rates["equipment"]
    for code in req.equipment:
        item = selected_equipment[code]
        amount = item.get("fixed_yen", item.get("yen_per_tsubo", 0) * tsubo)
        equipment_total += amount
        calc = (
            f"固定額 {amount:,.0f}円"
            if "fixed_yen" in item
            else f"{tsubo:.1f}坪 × {item['yen_per_tsubo']:,}円/坪"
        )
        lines.append(CostLine(label=item["label"], amount_yen=_round_10k(amount), calculation=calc))

    expected = adjusted + equipment_total
    uncertainty = rates["uncertainty_rate"]
    low = _round_10k(expected * (1 - uncertainty))
    expected_rounded = _round_10k(expected)
    high = _round_10k(expected * (1 + uncertainty))

    within_budget = None
    budget_gap = None
    if req.budget_yen is not None:
        within_budget = high <= req.budget_yen
        budget_gap = req.budget_yen - high

    savings = []
    if req.grade != "economy":
        economy_cost = base * structure_factor * floor_factor * rates["grade_factors"]["economy"]
        savings.append(SavingsOption(
            label="仕様グレードをエコノミーへ変更",
            estimated_savings_yen=max(0, _round_10k(adjusted - economy_cost)),
            note="設備・仕上げの優先順位を施工会社と確認してください。",
        ))
    savings.append(SavingsOption(
        label="延床面積を5%縮小",
        estimated_savings_yen=_round_10k(adjusted * 0.05),
        note="廊下・収納・個室面積を一律に削らず、生活動線を再設計します。",
    ))
    for code in req.equipment:
        line = next(line for line in lines if line.label == selected_equipment[code]["label"])
        savings.append(SavingsOption(
            label=f"{line.label}を別工事または将来設置へ変更",
            estimated_savings_yen=line.amount_yen,
            note="初期費用は下がりますが、後付け可否と追加工事費を確認してください。",
        ))
    savings.sort(key=lambda x: x.estimated_savings_yen, reverse=True)

    source = rates["source"]
    return EstimateResult(
        floor_area_sqm=round(req.floor_area_sqm, 2),
        floor_area_tsubo=round(tsubo, 2),
        low_yen=low,
        expected_yen=expected_rounded,
        high_yen=high,
        budget_yen=req.budget_yen,
        within_budget=within_budget,
        budget_gap_yen=budget_gap,
        lines=lines,
        savings_options=savings,
        assumptions=[
            source["scope"],
            f"基準単価は{rates['base_yen_per_tsubo']:,}円/坪、概算幅は±{uncertainty:.0%}。",
            "補正係数と設備単価はデモ用静的データであり、地域・時点補正は未実装。",
        ],
        source_name=source["name"],
        source_url=source["url"],
        source_fiscal_year=source["fiscal_year"],
        disclaimer=DISCLAIMER,
    )
