"""間取り構造化 — 敷地プリセット（KH案 §8.3 敷地内自動配置の入力）

実測GIS連携は行わず、整形地（矩形）のプリセットを静的に持つ。
グリッド単位は app.schemas.geometry.GRID_MM（910mm）に準拠。
"""
from __future__ import annotations

import re
from typing import Optional

from app.schemas.geometry import SiteBoundary

# 坪数ラベル → SiteBoundary。widths/depthsは正方形〜やや長方形の整形地で近似。
SITE_PRESETS: dict[str, SiteBoundary] = {
    "30坪": SiteBoundary(
        preset_key="30坪", width_grid=11, depth_grid=11,
        road_side="south", road_width_m=4.0, setback_grid=1,
    ),
    "35坪": SiteBoundary(
        preset_key="35坪", width_grid=12, depth_grid=12,
        road_side="south", road_width_m=5.0, setback_grid=1,
    ),
    "40坪": SiteBoundary(
        preset_key="40坪", width_grid=13, depth_grid=12,
        road_side="south", road_width_m=6.0, setback_grid=1,
    ),
    "50坪": SiteBoundary(
        preset_key="50坪", width_grid=14, depth_grid=14,
        road_side="south", road_width_m=6.0, setback_grid=1,
    ),
    "60坪": SiteBoundary(
        preset_key="60坪", width_grid=16, depth_grid=15,
        road_side="south", road_width_m=6.0, setback_grid=1,
    ),
}

DEFAULT_PRESET_KEY = "35坪"

_TSUBO_PATTERN = re.compile(r"(\d+)\s*坪")


def select_site_preset(land_info: Optional[str]) -> SiteBoundary:
    """land_info（自由記述）から最も近い坪数の敷地プリセットを選ぶ。

    一致する数値が取れない場合は既定プリセット(35坪)を返す。
    """
    if land_info:
        match = _TSUBO_PATTERN.search(land_info)
        if match:
            target_tsubo = int(match.group(1))
            best_key = min(
                SITE_PRESETS,
                key=lambda k: abs(SITE_PRESETS[k].area_tsubo - target_tsubo),
            )
            return SITE_PRESETS[best_key]
    return SITE_PRESETS[DEFAULT_PRESET_KEY]


def get_preset_by_key(key: str) -> Optional[SiteBoundary]:
    return SITE_PRESETS.get(key)
