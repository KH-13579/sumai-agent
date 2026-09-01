"""間取り構造化 — 決定論エンジンの開発用プレビュースクリプト

Ollama/LLM不要。固定の部屋リストで平屋/2階建て × 敷地サイズを組み合わせて
geometryを生成し、build/preview/ にSVGを書き出し、幾何検算結果を表示する。
pytestの一部ではない（手動確認用）。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.data.site_presets import SITE_PRESETS
from app.schemas.geometry import RoomSpec
from app.tools.geometry_check import run_geometry_check
from app.tools.layout_engine import build_geometry
from app.tools.svg_renderer import render_floor_svg

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "build" / "preview"


def _two_story_specs() -> list[RoomSpec]:
    return [
        RoomSpec(room_id="ldk", room_type="LDK", label="LDK", target_area_m2=28.0, floor=1),
        RoomSpec(room_id="wa", room_type="和室", label="和室", target_area_m2=8.0, floor=1),
        RoomSpec(room_id="senmen", room_type="洗面脱衣", label="洗面脱衣", target_area_m2=4.0, floor=1),
        RoomSpec(room_id="furo", room_type="浴室", label="浴室", target_area_m2=4.0, floor=1),
        RoomSpec(room_id="toilet1", room_type="トイレ", label="トイレ", target_area_m2=1.8, floor=1),
        RoomSpec(room_id="shunou1", room_type="収納", label="収納", target_area_m2=2.0, floor=1),
        RoomSpec(room_id="shu_bed", room_type="主寝室", label="主寝室", target_area_m2=13.0, floor=2),
        RoomSpec(room_id="wic", room_type="WIC", label="WIC", target_area_m2=3.0, floor=2),
        RoomSpec(room_id="kids1", room_type="子供部屋", label="子供部屋1", target_area_m2=10.0, floor=2),
        RoomSpec(room_id="kids2", room_type="子供部屋", label="子供部屋2", target_area_m2=10.0, floor=2),
        RoomSpec(room_id="toilet2", room_type="トイレ", label="トイレ", target_area_m2=1.8, floor=2),
        RoomSpec(room_id="balcony", room_type="バルコニー", label="バルコニー", target_area_m2=5.0, floor=2),
    ]


def _single_story_specs() -> list[RoomSpec]:
    return [
        RoomSpec(room_id="ldk", room_type="LDK", label="LDK", target_area_m2=24.0, floor=1),
        RoomSpec(room_id="shu_bed", room_type="主寝室", label="主寝室", target_area_m2=12.0, floor=1),
        RoomSpec(room_id="kids1", room_type="子供部屋", label="子供部屋", target_area_m2=8.0, floor=1),
        RoomSpec(room_id="senmen", room_type="洗面脱衣", label="洗面脱衣", target_area_m2=4.0, floor=1),
        RoomSpec(room_id="furo", room_type="浴室", label="浴室", target_area_m2=4.0, floor=1),
        RoomSpec(room_id="toilet1", room_type="トイレ", label="トイレ", target_area_m2=1.8, floor=1),
        RoomSpec(room_id="shunou1", room_type="収納", label="収納", target_area_m2=3.0, floor=1),
    ]


SCENARIOS = {
    "2階建て_35坪": (_two_story_specs, "35坪"),
    "2階建て_30坪": (_two_story_specs, "30坪"),
    "2階建て_50坪": (_two_story_specs, "50坪"),
    "平屋_40坪": (_single_story_specs, "40坪"),
}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"{'シナリオ':20} {'overlaps':9} {'gap_m2':8} {'out_fp':7} {'out_site':9} {'fill%':7} ok")
    print("-" * 80)

    determinism_probe = {}

    for name, (spec_fn, site_key) in SCENARIOS.items():
        site = SITE_PRESETS[site_key]
        specs = spec_fn()
        building = build_geometry(specs, site)
        check = run_geometry_check(building)

        print(
            f"{name:20} {len(check.overlaps):<9} {check.gap_area_m2:<8} "
            f"{len(check.out_of_footprint):<7} {len(check.out_of_site):<9} "
            f"{check.fill_rate_pct:<7} {check.ok}"
        )
        if check.warnings:
            for w in check.warnings:
                print(f"    warning: {w}")

        for floor_geo in building.floors:
            svg = render_floor_svg(building, floor_geo.floor)
            out_path = OUTPUT_DIR / f"{name}_F{floor_geo.floor}.svg"
            out_path.write_text(svg, encoding="utf-8")

            # 決定論性チェック: 同一入力で2回実行してSVGが完全一致するか
            svg2 = render_floor_svg(build_geometry(spec_fn(), site), floor_geo.floor)
            determinism_probe[f"{name}_F{floor_geo.floor}"] = svg == svg2

        stair_positions = set()
        for floor_geo in building.floors:
            for room in floor_geo.rooms:
                if room.room_type == "階段":
                    stair_positions.add((room.x - floor_geo.footprint_x, room.y - floor_geo.footprint_y, room.w, room.h))
        if len(stair_positions) > 1:
            print(f"    !! 階段位置が階間で不一致: {stair_positions}")

    print("-" * 80)
    all_deterministic = all(determinism_probe.values())
    print(f"SVG再現性（同一入力で完全一致）: {'OK' if all_deterministic else 'NG'}")
    if not all_deterministic:
        for k, v in determinism_probe.items():
            if not v:
                print(f"    NG: {k}")

    print(f"\nSVG出力先: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
