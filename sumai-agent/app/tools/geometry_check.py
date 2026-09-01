"""間取り構造化 — 幾何検算（KH案 §8.3 / §9.2.2）

重なり・隙間・フットプリント内包・敷地内包・面積誤差を機械的に検算する。
決定論エンジンの出力ではoverlaps=[]・gap=0・fill_rate=100%になるのが
期待値であり、これが回帰ガードとして機能する。LLM座標経路(SUMAI_LAYOUT_MODE=llm)
ではこの結果が自律修正ループ・フォールバックの判定材料になる。
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from app.schemas.geometry import BuildingGeometry, CELL_AREA_M2, FloorGeometry, RoomBox

ASPECT_WARNING_THRESHOLD = 3.0
MIN_ROOM_AREA_M2 = 1.66  # 約1畳


class OverlapItem(BaseModel):
    a: str
    b: str
    floor: int
    overlap_area_m2: float


class GeometryCheckResult(BaseModel):
    ok: bool
    overlaps: list[OverlapItem] = Field(default_factory=list)
    gap_area_m2: float = 0.0
    out_of_footprint: list[str] = Field(default_factory=list)
    out_of_site: list[str] = Field(default_factory=list)
    area_errors: dict[str, float] = Field(default_factory=dict)
    total_area_error_pct: float = 0.0
    fill_rate_pct: float = 100.0
    warnings: list[str] = Field(default_factory=list)


def _rect_overlap_area(a: RoomBox, b: RoomBox) -> float:
    overlap_w = min(a.x + a.w, b.x + b.w) - max(a.x, b.x)
    overlap_h = min(a.y + a.h, b.y + b.h) - max(a.y, b.y)
    if overlap_w <= 0 or overlap_h <= 0:
        return 0.0
    return round(overlap_w * overlap_h * CELL_AREA_M2, 2)


def _check_overlaps(rooms: list[RoomBox], floor_no: int) -> list[OverlapItem]:
    result: list[OverlapItem] = []
    for i, a in enumerate(rooms):
        for b in rooms[i + 1:]:
            area = _rect_overlap_area(a, b)
            if area > 0:
                result.append(OverlapItem(a=a.room_id, b=b.room_id, floor=floor_no, overlap_area_m2=area))
    return result


def _check_coverage(floor_geo: FloorGeometry) -> tuple[int, int, list[str]]:
    """フットプリント内の被覆セル数・全セル数・フットプリントをはみ出す room_id を返す"""
    fx, fy = floor_geo.footprint_x, floor_geo.footprint_y
    fw, fh = floor_geo.footprint_w, floor_geo.footprint_h
    covered = [[False] * fw for _ in range(fh)]
    out_of_footprint: list[str] = []

    for room in floor_geo.rooms:
        rx, ry = room.x - fx, room.y - fy
        if rx < 0 or ry < 0 or rx + room.w > fw or ry + room.h > fh:
            out_of_footprint.append(room.room_id)
            continue
        for gy in range(ry, ry + room.h):
            for gx in range(rx, rx + room.w):
                covered[gy][gx] = True

    total_cells = fw * fh
    covered_cells = sum(1 for row in covered for c in row if c)
    return covered_cells, total_cells, out_of_footprint


def _check_site_containment(building: BuildingGeometry, floor_geo: FloorGeometry) -> list[str]:
    site = building.site
    min_x, min_y = site.setback_grid, site.setback_grid
    max_x = site.width_grid - site.setback_grid
    max_y = site.depth_grid - site.setback_grid

    violations: list[str] = []
    for room in floor_geo.rooms:
        if room.x < min_x or room.y < min_y or room.x + room.w > max_x or room.y + room.h > max_y:
            violations.append(room.room_id)
    return violations


def run_geometry_check(building: BuildingGeometry) -> GeometryCheckResult:
    overlaps: list[OverlapItem] = []
    out_of_footprint: list[str] = []
    out_of_site: list[str] = []
    area_errors: dict[str, float] = {}
    warnings: list[str] = []

    total_covered_cells = 0
    total_cells = 0
    target_sum = 0.0
    actual_sum = 0.0

    for floor_geo in building.floors:
        overlaps.extend(_check_overlaps(floor_geo.rooms, floor_geo.floor))

        covered_cells, cell_count, oof = _check_coverage(floor_geo)
        out_of_footprint.extend(f"{rid}(F{floor_geo.floor})" for rid in oof)
        total_covered_cells += covered_cells
        total_cells += cell_count

        out_of_site.extend(
            f"{rid}(F{floor_geo.floor})" for rid in _check_site_containment(building, floor_geo)
        )

        for room in floor_geo.rooms:
            key = f"{room.room_id}(F{floor_geo.floor})"
            if room.target_area_m2 > 0:
                error_pct = round((room.area_m2 - room.target_area_m2) / room.target_area_m2 * 100, 1)
                area_errors[key] = error_pct
            target_sum += room.target_area_m2
            actual_sum += room.area_m2

            if room.aspect > ASPECT_WARNING_THRESHOLD:
                warnings.append(f"{room.label}(F{floor_geo.floor})のアスペクト比が{room.aspect}:1と細長すぎます")
            if room.area_m2 < MIN_ROOM_AREA_M2:
                warnings.append(f"{room.label}(F{floor_geo.floor})の面積が{room.area_m2}m2と狭小です")

    total_area_error_pct = (
        round((actual_sum - target_sum) / target_sum * 100, 1) if target_sum > 0 else 0.0
    )
    gap_area_m2 = round(max(0, total_cells - total_covered_cells) * CELL_AREA_M2, 2)
    fill_rate_pct = round(total_covered_cells / total_cells * 100, 1) if total_cells else 100.0

    warnings.extend(building.notes)

    ok = not overlaps and not out_of_footprint and not out_of_site and fill_rate_pct >= 99.9

    return GeometryCheckResult(
        ok=ok,
        overlaps=overlaps,
        gap_area_m2=gap_area_m2,
        out_of_footprint=out_of_footprint,
        out_of_site=out_of_site,
        area_errors=area_errors,
        total_area_error_pct=total_area_error_pct,
        fill_rate_pct=fill_rate_pct,
        warnings=warnings,
    )
