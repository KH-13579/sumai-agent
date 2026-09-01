"""FastAPI ルーター — 機械可読アーティファクト（KH案 FR-07）とSVG配信"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException, Response

from app.api.chat import _get_graph
from app.tools.plan_metrics import (
    building_area_m2,
    building_coverage_pct,
    floor_area_ratio_pct,
    floor_count,
    max_height_m,
    site_area_m2,
    to_tsubo,
    total_floor_area_m2,
)
from app.tools.svg_renderer import render_floor_svg

router = APIRouter()

SCHEMA_VERSION = "1.0"
DISCLAIMER = (
    "本アーティファクトはAIによる概算・参考プランです。"
    "詳細な設計・法規確認・正確な見積は、建築士やハウスメーカーにご相談ください。"
)
DATA_SOURCES = ["スマイエージェント 決定論的レイアウトエンジン", "ユーザーヒアリング入力"]


def _get_session_state(session_id: str) -> dict[str, Any]:
    graph = _get_graph()
    config = {"configurable": {"thread_id": session_id}}
    snapshot = graph.get_state(config)
    if not snapshot or not snapshot.values:
        raise HTTPException(status_code=404, detail="セッションが見つかりません")
    return snapshot.values


def _build_artifact(session_id: str) -> dict[str, Any]:
    values = _get_session_state(session_id)
    plans = values.get("floor_plans") or []
    if not plans:
        raise HTTPException(status_code=404, detail="間取り案がまだ生成されていません")

    plan_payloads = []
    for plan in plans:
        geometry = plan.geometry
        metrics = None
        if geometry is not None:
            metrics = {
                "building_area_m2": building_area_m2(geometry),
                "total_floor_area_m2": total_floor_area_m2(geometry),
                "floor_count": floor_count(geometry),
                "max_height_m": max_height_m(geometry),
                "site_area_m2": site_area_m2(geometry),
                "site_area_tsubo": to_tsubo(site_area_m2(geometry)),
                "building_coverage_pct": building_coverage_pct(geometry),
                "floor_area_ratio_pct": floor_area_ratio_pct(geometry),
            }
        plan_payloads.append({
            "concept": plan.concept,
            "total_floor_area": plan.total_floor_area,
            "floors": plan.floors,
            "rooms": [r.model_dump() for r in plan.rooms],
            "rationale": plan.rationale,
            "estimated_cost": plan.estimated_cost,
            "geometry": geometry.model_dump() if geometry is not None else None,
            "metrics": metrics,
            "check": plan.check.model_dump() if plan.check is not None else None,
        })

    requirements = values.get("requirements")

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "requirements": requirements.model_dump() if requirements is not None else None,
        "plans": plan_payloads,
        "data_sources": DATA_SOURCES,
        "disclaimer": DISCLAIMER,
    }


@router.get("/artifact/{session_id}")
async def get_artifact(session_id: str) -> dict[str, Any]:
    return _build_artifact(session_id)


@router.get("/artifact/{session_id}/download")
async def download_artifact(session_id: str) -> Response:
    artifact = _build_artifact(session_id)
    body = json.dumps(artifact, ensure_ascii=False, indent=2)
    return Response(
        content=body,
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="artifact_{session_id}.json"'},
    )


@router.get("/plan-svg/{session_id}/{plan_index}/{floor}")
async def get_plan_svg(session_id: str, plan_index: int, floor: int) -> Response:
    values = _get_session_state(session_id)
    plans = values.get("floor_plans") or []
    if plan_index < 0 or plan_index >= len(plans):
        raise HTTPException(status_code=404, detail="指定の案が見つかりません")

    plan = plans[plan_index]
    if plan.geometry is None:
        raise HTTPException(status_code=404, detail="この案にはジオメトリがありません")

    try:
        svg = render_floor_svg(plan.geometry, floor)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return Response(content=svg, media_type="image/svg+xml")
