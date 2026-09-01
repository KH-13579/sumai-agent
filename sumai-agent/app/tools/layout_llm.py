"""間取り構造化 — 実験的LLM座標生成経路（SUMAI_LAYOUT_MODE=llm）

決定論エンジン（layout_engine.build_geometry）の経路からは一切importされない。
qwen2.5:3bなどに座標（矩形パッキング）を直接生成させた場合の崩壊度を実測する
ための独立した実験モジュール。フットプリント・動線帯（玄関/階段/廊下）は
決定論エンジンの結果をそのまま使い、LLMには「残り矩形への部屋の配置」だけを
担わせる。検算NGなら1回リトライし、なお不足していれば決定論配置へフォールバック。
"""
from __future__ import annotations

import logging
from typing import List

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.schemas.geometry import (
    CIRCULATION_TYPES,
    BuildingGeometry,
    FloorGeometry,
    RoomBox,
    RoomSpec,
    SiteBoundary,
)
from app.tools.geometry_check import GeometryCheckResult, run_geometry_check
from app.tools.layout_engine import build_geometry, derive_adjacencies

logger = logging.getLogger(__name__)

_FILL_RATE_OK_THRESHOLD = 90.0


class _LLMRoomBox(BaseModel):
    room_id: str
    floor: int
    cx: float
    cy: float
    w: int = Field(gt=0)
    h: int = Field(gt=0)


class _LLMLayoutOutput(BaseModel):
    rooms: List[_LLMRoomBox]


def _build_prompt(baseline: BuildingGeometry, specs_by_id: dict[str, RoomSpec]) -> str:
    lines = ["以下の各階の配置可能領域(グリッド単位、910mm/セル、南西角が原点)に、"
             "指定された部屋を重ならないように配置してください。",
             "中心座標cx,cyと幅w,高さhをグリッド単位の整数で出力し、"
             "面積 w*h*0.8281m2 が target_area_m2 に近くなるようにしてください。"]

    for floor_geo in baseline.floors:
        residual_x = floor_geo.footprint_x
        residual_y = floor_geo.footprint_y
        residual_w = floor_geo.footprint_w - 2
        residual_h = floor_geo.footprint_h
        lines.append(
            f"\n### {floor_geo.floor}階: 配置可能領域 x∈[{residual_x},{residual_x + residual_w}), "
            f"y∈[{residual_y},{residual_y + residual_h})"
        )
        for room_id, spec in specs_by_id.items():
            if spec.floor == floor_geo.floor:
                lines.append(
                    f"- room_id={room_id}, floor={spec.floor}, label={spec.label}, "
                    f"target_area_m2={spec.target_area_m2}"
                )
    return "\n".join(lines)


def _to_room_boxes(
    llm_output: _LLMLayoutOutput,
    specs_by_id: dict[str, RoomSpec],
    baseline_boxes_by_id: dict[str, RoomBox],
    floor_geo: FloorGeometry,
    notes: list[str],
) -> list[RoomBox]:
    llm_by_id = {b.room_id: b for b in llm_output.rooms if b.floor == floor_geo.floor}
    result: list[RoomBox] = []

    for room_id, spec in specs_by_id.items():
        if spec.floor != floor_geo.floor:
            continue
        llm_box = llm_by_id.get(room_id)
        if llm_box is None:
            fallback = baseline_boxes_by_id[room_id]
            notes.append(f"'{spec.label}'はLLM未出力のため決定論配置で補完しました")
            result.append(fallback)
            continue

        w, h = max(1, llm_box.w), max(1, llm_box.h)
        x = round(llm_box.cx - w / 2)
        y = round(llm_box.cy - h / 2)
        max_x = floor_geo.footprint_x + floor_geo.footprint_w - 2 - w
        max_y = floor_geo.footprint_y + floor_geo.footprint_h - h
        x = min(max(x, floor_geo.footprint_x), max(floor_geo.footprint_x, max_x))
        y = min(max(y, floor_geo.footprint_y), max(floor_geo.footprint_y, max_y))

        result.append(
            RoomBox(
                room_id=room_id, room_type=spec.room_type, label=spec.label,
                x=x, y=y, w=w, h=h, floor=spec.floor, target_area_m2=spec.target_area_m2,
            )
        )
    return result


def _rebuild_geometry(
    baseline: BuildingGeometry,
    llm_output: _LLMLayoutOutput,
    specs_by_id: dict[str, RoomSpec],
    notes: list[str],
) -> BuildingGeometry:
    floors: list[FloorGeometry] = []
    for floor_geo in baseline.floors:
        hall_boxes = [r for r in floor_geo.rooms if r.room_type in CIRCULATION_TYPES]
        baseline_boxes_by_id = {r.room_id: r for r in floor_geo.rooms if r.room_type not in CIRCULATION_TYPES}

        placed_boxes = _to_room_boxes(llm_output, specs_by_id, baseline_boxes_by_id, floor_geo, notes)
        all_boxes = hall_boxes + placed_boxes
        adjacencies = derive_adjacencies(all_boxes, floor_geo.floor)

        floors.append(
            FloorGeometry(
                floor=floor_geo.floor,
                footprint_x=floor_geo.footprint_x, footprint_y=floor_geo.footprint_y,
                footprint_w=floor_geo.footprint_w, footprint_h=floor_geo.footprint_h,
                rooms=all_boxes, adjacencies=adjacencies,
            )
        )

    return BuildingGeometry(
        site=baseline.site,
        floors=floors,
        inter_floor_adjacencies=baseline.inter_floor_adjacencies,
        layout_source="llm",
        notes=list(notes),
    )


def _feedback_prompt(check: GeometryCheckResult) -> str:
    overlap_desc = ", ".join(f"{o.a}-{o.b}(F{o.floor})" for o in check.overlaps) or "なし"
    return (
        f"前回の配置には問題がありました。重なり: {overlap_desc} / "
        f"充足率: {check.fill_rate_pct}% / フットプリント外: {check.out_of_footprint}。"
        "重なりを解消し、指定領域内に収まるよう再配置してください。"
    )


def generate_geometry_via_llm(
    specs: List[RoomSpec], site: SiteBoundary, llm: ChatOllama
) -> BuildingGeometry:
    """LLMに座標(矩形パッキング)を直接生成させる実験経路。NGなら決定論へフォールバック"""
    baseline = build_geometry(specs, site)
    specs_by_id: dict[str, RoomSpec] = {}
    for spec in specs:
        if spec.room_type not in CIRCULATION_TYPES:
            specs_by_id[spec.room_id] = spec

    system = SystemMessage(content="あなたは住宅の部屋配置を担当するAIです。指定領域内に重ならないよう部屋を配置してください。")
    prompt = _build_prompt(baseline, specs_by_id)
    structured_llm = llm.with_structured_output(_LLMLayoutOutput)

    messages = [system, HumanMessage(content=prompt)]
    notes: list[str] = []

    for attempt in range(2):
        try:
            llm_output: _LLMLayoutOutput = structured_llm.invoke(messages)
        except Exception as e:
            logger.warning("LLM座標生成に失敗しました(attempt=%d): %s", attempt, e)
            break

        candidate = _rebuild_geometry(baseline, llm_output, specs_by_id, list(notes))
        check = run_geometry_check(candidate)
        if not check.overlaps and check.fill_rate_pct >= _FILL_RATE_OK_THRESHOLD:
            return candidate

        if attempt == 0:
            messages.append(HumanMessage(content=_feedback_prompt(check)))

    fallback = baseline.model_copy(deep=True)
    fallback.layout_source = "deterministic_fallback"
    fallback.notes.append("LLM座標生成が検算NGのため決定論エンジンにフォールバックしました")
    return fallback
