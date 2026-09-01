"""間取り構造化 — 実績値メトリクス（他機能への提供IF）

法規チェック・見積もり機能が必要とする面積・高さの実績値をここに一元化する。
上限値（建ぺい率・容積率の許容値、高さ制限など）との比較判定はここでは行わない
— それは法規チェック機能側の責務。
"""
from __future__ import annotations

from app.schemas.geometry import BuildingGeometry, FLOOR_HEIGHT_M, ROOF_RISE_M, TSUBO_M2


def building_area_m2(building: BuildingGeometry) -> float:
    """建築面積（1階フットプリント面積）"""
    if not building.floors:
        return 0.0
    ground_floor = min(building.floors, key=lambda f: f.floor)
    return ground_floor.footprint_area_m2


def total_floor_area_m2(building: BuildingGeometry) -> float:
    """延床面積（全階フットプリント面積の合計）"""
    return round(sum(f.footprint_area_m2 for f in building.floors), 2)


def floor_count(building: BuildingGeometry) -> int:
    return len(building.floors)


def max_height_m(building: BuildingGeometry) -> float:
    """最高高さ（階高×階数 + 屋根）"""
    return round(floor_count(building) * FLOOR_HEIGHT_M + ROOF_RISE_M, 2)


def site_area_m2(building: BuildingGeometry) -> float:
    return building.site.area_m2


def building_coverage_pct(building: BuildingGeometry) -> float:
    """建ぺい率の実績値（%）。上限との比較は行わない"""
    site_area = site_area_m2(building)
    if site_area <= 0:
        return 0.0
    return round(building_area_m2(building) / site_area * 100, 1)


def floor_area_ratio_pct(building: BuildingGeometry) -> float:
    """容積率の実績値（%）。上限との比較は行わない"""
    site_area = site_area_m2(building)
    if site_area <= 0:
        return 0.0
    return round(total_floor_area_m2(building) / site_area * 100, 1)


def to_tsubo(area_m2: float) -> float:
    return round(area_m2 / TSUBO_M2, 2)


def from_tsubo(tsubo: float) -> float:
    return round(tsubo * TSUBO_M2, 2)
