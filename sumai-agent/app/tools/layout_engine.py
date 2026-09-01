"""間取り構造化 — 決定論的レイアウトエンジン（KH案 §9.2.1/§8.3）

LLMは部屋タイプ・目標面積・階のみを決める（RoomSpec）。座標は
再帰二分割（BSP）で決定論的に計算し、重なり0・隙間0を構造的に保証する。
"""
from __future__ import annotations

import math
from typing import Optional

from app.data.site_presets import SITE_PRESETS, get_preset_by_key
from app.schemas.geometry import (
    HALL_WIDTH_GRID,
    CELL_AREA_M2,
    CIRCULATION_TYPES,
    Adjacency,
    BuildingGeometry,
    FloorGeometry,
    RoomBox,
    RoomSpec,
    SiteBoundary,
)

FLOOR1_ZONES: tuple[str, ...] = (
    "LDK", "リビング", "ダイニング", "和室", "キッチン", "パントリー",
    "洗面脱衣", "浴室", "トイレ", "収納", "その他",
)
FLOOR2_ZONES: tuple[str, ...] = (
    "主寝室", "WIC", "子供部屋", "寝室", "書斎", "トイレ", "収納",
    "バルコニー", "その他",
)

_MIN_FOOTPRINT_WIDTH_GRID = 6
_MIN_FOOTPRINT_DEPTH_GRID = 5
_ASPECT_TARGET = 1.25  # 幅:奥行


def infer_floor_for_room_type(room_type: str) -> int:
    """room_typeからLLMが階を出さなかった場合のフォールバック階を推定する"""
    if room_type in FLOOR2_ZONES and room_type not in FLOOR1_ZONES:
        return 2
    return 1


def _zone_index(room_type: str, floor_no: int) -> int:
    zones = FLOOR1_ZONES if floor_no == 1 else FLOOR2_ZONES
    try:
        return zones.index(room_type)
    except ValueError:
        return len(zones)


def _footprint_size(area_m2: float) -> tuple[int, int]:
    width = round(math.sqrt(area_m2 * _ASPECT_TARGET / CELL_AREA_M2))
    width = max(width, _MIN_FOOTPRINT_WIDTH_GRID)
    residual_width = max(width - HALL_WIDTH_GRID, 1)
    depth = round(area_m2 / (residual_width * CELL_AREA_M2))
    depth = max(depth, _MIN_FOOTPRINT_DEPTH_GRID)
    return width, depth


def _reserve_hall_band(
    floor_no: int, floor_count: int, width: int, depth: int
) -> tuple[list[tuple[str, str, int, int, int, int]], tuple[int, int, int, int]]:
    """circulation部屋(タイプ, room_id, x, y, w, h)と、残り矩形(x,y,w,h)を返す"""
    hall_x = width - HALL_WIDTH_GRID
    boxes: list[tuple[str, str, int, int, int, int]] = []

    if floor_count == 1:
        boxes.append(("玄関", f"entrance_f{floor_no}", hall_x, 0, HALL_WIDTH_GRID, 2))
        if depth > 2:
            boxes.append(("廊下", f"corridor_f{floor_no}", hall_x, 2, HALL_WIDTH_GRID, depth - 2))
    elif floor_no == 1:
        boxes.append(("玄関", f"entrance_f{floor_no}", hall_x, 0, HALL_WIDTH_GRID, 2))
        boxes.append(("階段", f"stair_f{floor_no}", hall_x, 2, HALL_WIDTH_GRID, 2))
        if depth > 4:
            boxes.append(("廊下", f"corridor_f{floor_no}", hall_x, 4, HALL_WIDTH_GRID, depth - 4))
    else:
        boxes.append(("ホール", f"hall_f{floor_no}_lower", hall_x, 0, HALL_WIDTH_GRID, 2))
        boxes.append(("階段", f"stair_f{floor_no}", hall_x, 2, HALL_WIDTH_GRID, 2))
        if depth > 4:
            boxes.append(("ホール", f"hall_f{floor_no}_upper", hall_x, 4, HALL_WIDTH_GRID, depth - 4))

    residual_rect = (0, 0, width - HALL_WIDTH_GRID, depth)
    return boxes, residual_rect


def _split_rect(
    rect: tuple[int, int, int, int], frac: float
) -> Optional[tuple[tuple[int, int, int, int], tuple[int, int, int, int]]]:
    x, y, w, h = rect
    if w >= h and w >= 2:
        axis_len = w
        cut = max(1, min(axis_len - 1, round(frac * axis_len)))
        return (x, y, cut, h), (x + cut, y, w - cut, h)
    if h >= 2:
        axis_len = h
        cut = max(1, min(axis_len - 1, round(frac * axis_len)))
        return (x, y, w, cut), (x, y + cut, w, h - cut)
    if w >= 2:
        axis_len = w
        cut = max(1, min(axis_len - 1, round(frac * axis_len)))
        return (x, y, cut, h), (x + cut, y, w - cut, h)
    return None


def _bsp(
    rect: tuple[int, int, int, int],
    rooms: list[RoomSpec],
    notes: list[str],
) -> list[tuple[RoomSpec, int, int, int, int]]:
    if not rooms:
        return []
    if len(rooms) == 1:
        x, y, w, h = rect
        return [(rooms[0], x, y, w, h)]

    total = sum(r.target_area_m2 for r in rooms)
    cum = 0.0
    k = len(rooms) - 1
    for i in range(1, len(rooms)):
        cum += rooms[i - 1].target_area_m2
        if cum >= total / 2:
            k = i
            break

    frac = sum(r.target_area_m2 for r in rooms[:k]) / total
    split = _split_rect(rect, frac)

    if split is None:
        merged = list(rooms)
        smallest_idx = min(range(len(merged)), key=lambda i: merged[i].target_area_m2)
        victim = merged.pop(smallest_idx)
        neighbor_idx = min(smallest_idx, len(merged) - 1) if smallest_idx > 0 else 0
        neighbor = merged[neighbor_idx]
        merged[neighbor_idx] = neighbor.model_copy(
            update={"target_area_m2": neighbor.target_area_m2 + victim.target_area_m2}
        )
        notes.append(f"{victim.label}を面積不足のため{neighbor.label}に統合しました")
        return _bsp(rect, merged, notes)

    rect_a, rect_b = split
    return _bsp(rect_a, rooms[:k], notes) + _bsp(rect_b, rooms[k:], notes)


def derive_adjacencies(rooms: list[RoomBox], floor_no: int) -> list[Adjacency]:
    """矩形の辺共有から隣接グラフを導出する（layout_llm.py からも利用される公開関数）"""
    adjacencies: list[Adjacency] = []
    for i, a in enumerate(rooms):
        for b in rooms[i + 1:]:
            touch_x = (a.x + a.w == b.x) or (b.x + b.w == a.x)
            touch_y = (a.y + a.h == b.y) or (b.y + b.h == a.y)
            overlap_y = min(a.y + a.h, b.y + b.h) - max(a.y, b.y)
            overlap_x = min(a.x + a.w, b.x + b.w) - max(a.x, b.x)

            shared = 0
            if touch_x and overlap_y >= 1:
                shared = overlap_y
            elif touch_y and overlap_x >= 1:
                shared = overlap_x
            else:
                continue

            is_circulation = a.room_type in CIRCULATION_TYPES or b.room_type in CIRCULATION_TYPES
            relation = "door" if (shared >= 2 or is_circulation) else "adjacent"
            adjacencies.append(
                Adjacency(a=a.room_id, b=b.room_id, relation=relation, shared_len_grid=shared, floor=floor_no)
            )
    return adjacencies


def _place_building_on_site(
    site: SiteBoundary, width: int, max_depth: int
) -> tuple[int, int, Optional[str]]:
    """敷地内での建物原点(footprint_x, footprint_y)を返す。収まらなければnote付き"""
    setback = site.setback_grid
    buildable_width = site.width_grid - 2 * setback
    x_offset = setback + max(0, (buildable_width - width) // 2)
    y_offset = setback

    note = None
    if width > buildable_width or (y_offset + max_depth) > (site.depth_grid - setback):
        note = (
            f"敷地({site.preset_key})に建物が収まらない可能性があります "
            f"(建物 {width}x{max_depth}グリッド / 敷地内利用可能領域 "
            f"{buildable_width}x{site.depth_grid - 2 * setback}グリッド)"
        )
    return x_offset, y_offset, note


def _select_fitting_site(
    site: SiteBoundary, width: int, max_depth: int, notes: list[str]
) -> tuple[SiteBoundary, int, int]:
    ordered = sorted(SITE_PRESETS.values(), key=lambda s: s.area_m2)
    candidates = [site] + [p for p in ordered if p.area_tsubo > site.area_tsubo]

    for candidate in candidates:
        x_offset, y_offset, note = _place_building_on_site(candidate, width, max_depth)
        if note is None:
            if candidate.preset_key != site.preset_key:
                notes.append(
                    f"敷地プリセットを{site.preset_key}から{candidate.preset_key}へ自動的に繰り上げました"
                )
            return candidate, x_offset, y_offset

    # どのプリセットにも収まらない場合は最大プリセットのまま警告のみ残す
    x_offset, y_offset, note = _place_building_on_site(candidates[-1], width, max_depth)
    if note:
        notes.append(note)
    return candidates[-1], x_offset, y_offset


def build_geometry(specs: list[RoomSpec], site: SiteBoundary) -> BuildingGeometry:
    """部屋の意図(RoomSpec)から座標付きの建物ジオメトリを決定論的に構築する"""
    notes: list[str] = []

    by_floor: dict[int, list[RoomSpec]] = {}
    for spec in specs:
        if spec.room_type in CIRCULATION_TYPES:
            notes.append(f"'{spec.label}'は動線室のためレイアウトエンジンが自動配置する部屋と重複し除外しました")
            continue
        by_floor.setdefault(spec.floor, []).append(spec)

    if not by_floor:
        by_floor[1] = []

    floor_count = max(by_floor.keys())
    sizing_floor = min(by_floor.keys())
    sizing_area = sum(r.target_area_m2 for r in by_floor[sizing_floor]) or 30.0
    width, _ = _footprint_size(sizing_area)

    per_floor_depth: dict[int, int] = {}
    for floor_no in range(1, floor_count + 1):
        floor_specs = by_floor.get(floor_no, [])
        floor_area = sum(r.target_area_m2 for r in floor_specs) or 20.0
        _, depth = _footprint_size(floor_area)
        per_floor_depth[floor_no] = depth

    max_depth = max(per_floor_depth.values())
    fitted_site, footprint_x, footprint_y = _select_fitting_site(site, width, max_depth, notes)

    floors: list[FloorGeometry] = []
    stair_boxes_by_floor: dict[int, RoomBox] = {}

    for floor_no in range(1, floor_count + 1):
        depth = per_floor_depth[floor_no]
        floor_specs = sorted(
            by_floor.get(floor_no, []),
            key=lambda r: _zone_index(r.room_type, floor_no),
        )

        hall_boxes, residual_rect = _reserve_hall_band(floor_no, floor_count, width, depth)

        room_boxes: list[RoomBox] = []
        for room_type, room_id, lx, ly, lw, lh in hall_boxes:
            box = RoomBox(
                room_id=room_id, room_type=room_type, label=room_type,
                x=footprint_x + lx, y=footprint_y + ly, w=lw, h=lh,
                floor=floor_no, target_area_m2=round(lw * lh * CELL_AREA_M2, 2),
            )
            room_boxes.append(box)
            if room_type == "階段":
                stair_boxes_by_floor[floor_no] = box

        if floor_specs:
            placements = _bsp(residual_rect, floor_specs, notes)
            for spec, lx, ly, lw, lh in placements:
                room_boxes.append(
                    RoomBox(
                        room_id=spec.room_id, room_type=spec.room_type, label=spec.label,
                        x=footprint_x + lx, y=footprint_y + ly, w=lw, h=lh,
                        floor=floor_no, target_area_m2=spec.target_area_m2,
                    )
                )

        adjacencies = derive_adjacencies(room_boxes, floor_no)
        floors.append(
            FloorGeometry(
                floor=floor_no,
                footprint_x=footprint_x, footprint_y=footprint_y,
                footprint_w=width, footprint_h=depth,
                rooms=room_boxes, adjacencies=adjacencies,
            )
        )

    inter_floor_adjacencies: list[Adjacency] = []
    for floor_no in range(1, floor_count):
        lower = stair_boxes_by_floor.get(floor_no)
        upper = stair_boxes_by_floor.get(floor_no + 1)
        if lower and upper:
            inter_floor_adjacencies.append(
                Adjacency(a=lower.room_id, b=upper.room_id, relation="stair",
                          shared_len_grid=HALL_WIDTH_GRID, floor=floor_no)
            )

    return BuildingGeometry(
        site=fitted_site,
        floors=floors,
        inter_floor_adjacencies=inter_floor_adjacencies,
        layout_source="deterministic",
        notes=notes,
    )
