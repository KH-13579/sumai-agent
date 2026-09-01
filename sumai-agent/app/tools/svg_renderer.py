"""間取り構造化 — SVG描画（KH案 §8.3 / FR-08 / NFR-05）

BuildingGeometryから階ごとに1枚のSVGを生成する純関数。LLMはこの描画結果を
読み書きしない（KH案 §9.2.1）。
"""
from __future__ import annotations

from xml.sax.saxutils import escape

from app.schemas.geometry import GRID_M, BuildingGeometry, FloorGeometry

PX_PER_GRID = 32
PADDING_PX = 56
ROAD_MARGIN_GRID = 1.5
DISCLAIMER_TEXT = "本図は参考プランです。詳細は専門家にご確認ください。"

ROOM_COLORS: dict[str, str] = {
    "LDK": "#fde2b8", "リビング": "#fde2b8", "ダイニング": "#fde2b8",
    "キッチン": "#ffd8a8", "パントリー": "#ffe8c2", "和室": "#e9d8a6",
    "主寝室": "#d8c7f0", "寝室": "#d8c7f0", "子供部屋": "#d8c7f0", "書斎": "#c9b6ec",
    "浴室": "#bde0fe", "洗面脱衣": "#bde0fe", "トイレ": "#a2d2ff",
    "玄関": "#d9d9d9", "ホール": "#d9d9d9", "階段": "#bdbdbd", "廊下": "#d9d9d9",
    "収納": "#c7e9c0", "WIC": "#c7e9c0", "バルコニー": "#b7e4c7",
    "その他": "#eeeeee",
}
DEFAULT_ROOM_COLOR = "#eeeeee"
MIN_LABEL_CELLS = 3


def _esc(text: str) -> str:
    return escape(text)


def _find_floor(building: BuildingGeometry, floor: int) -> FloorGeometry:
    for f in building.floors:
        if f.floor == floor:
            return f
    raise ValueError(f"floor {floor} not found in building geometry")


def render_floor_svg(building: BuildingGeometry, floor: int, *, show_site: bool = True) -> str:
    floor_geo = _find_floor(building, floor)
    site = building.site

    building_min_x = floor_geo.footprint_x - 1
    building_max_x = floor_geo.footprint_x + floor_geo.footprint_w + 1
    building_min_y = floor_geo.footprint_y - 1
    building_max_y = floor_geo.footprint_y + floor_geo.footprint_h + 1

    margin = {"north": 0.0, "south": 0.0, "east": 0.0, "west": 0.0}
    if show_site:
        margin[site.road_side] = ROAD_MARGIN_GRID
        # 建物が敷地からはみ出す場合でも部屋が描画範囲外にならないよう、
        # キャンバスは敷地と建物フットプリントの両方を包含するサイズにする
        # （敷地内包チェックNGの案でも、はみ出し具合が見えることが重要）
        min_x = min(-margin["west"], building_min_x)
        max_x = max(site.width_grid + margin["east"], building_max_x)
        min_y = min(-margin["south"], building_min_y)
        max_y = max(site.depth_grid + margin["north"], building_max_y)
    else:
        min_x, max_x = building_min_x, building_max_x
        min_y, max_y = building_min_y, building_max_y

    scale = PX_PER_GRID
    canvas_w = (max_x - min_x) * scale + 2 * PADDING_PX
    canvas_h = (max_y - min_y) * scale + 2 * PADDING_PX

    def to_px(gx: float, gy: float) -> tuple[float, float]:
        px = PADDING_PX + (gx - min_x) * scale
        py = PADDING_PX + (max_y - gy) * scale
        return px, py

    def rect_px(x: int, y: int, w: int, h: int) -> tuple[float, float, float, float]:
        px0, py0 = to_px(x, y + h)
        return px0, py0, w * scale, h * scale

    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {canvas_w:.1f} {canvas_h:.1f}" '
        f'width="{canvas_w:.0f}" height="{canvas_h:.0f}" font-family="sans-serif">'
    )
    parts.append(f'<rect x="0" y="0" width="{canvas_w:.1f}" height="{canvas_h:.1f}" fill="#ffffff" />')

    if show_site:
        sx, sy, sw, sh = rect_px(0, 0, site.width_grid, site.depth_grid)
        parts.append(
            f'<rect x="{sx:.1f}" y="{sy:.1f}" width="{sw:.1f}" height="{sh:.1f}" '
            f'fill="none" stroke="#888888" stroke-width="1.5" stroke-dasharray="6,4" />'
        )
        parts.append(
            f'<text x="{sx:.1f}" y="{sy - 8:.1f}" font-size="11" fill="#666666">'
            f'敷地: {site.preset_key} 約{site.area_tsubo}坪（{site.area_m2:.1f}m2）</text>'
        )
        parts.append(_render_road(site, to_px))

    bx, by, bw, bh = rect_px(floor_geo.footprint_x, floor_geo.footprint_y, floor_geo.footprint_w, floor_geo.footprint_h)
    parts.append(
        f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw:.1f}" height="{bh:.1f}" '
        f'fill="none" stroke="#333333" stroke-width="3" />'
    )

    for room in floor_geo.rooms:
        rx, ry, rw, rh = rect_px(room.x, room.y, room.w, room.h)
        color = ROOM_COLORS.get(room.room_type, DEFAULT_ROOM_COLOR)
        parts.append(
            f'<rect x="{rx:.1f}" y="{ry:.1f}" width="{rw:.1f}" height="{rh:.1f}" '
            f'fill="{color}" stroke="#555555" stroke-width="1" />'
        )
        if room.w * room.h >= MIN_LABEL_CELLS:
            cx_px, cy_px = to_px(room.cx, room.cy)
            parts.append(
                f'<text x="{cx_px:.1f}" y="{cy_px - 4:.1f}" font-size="12" text-anchor="middle" fill="#222222">'
                f'{_esc(room.label)}</text>'
            )
            parts.append(
                f'<text x="{cx_px:.1f}" y="{cy_px + 12:.1f}" font-size="10" text-anchor="middle" fill="#555555">'
                f'{room.area_tatami}畳 / {room.area_m2}m2</text>'
            )

    parts.append(_render_north_arrow(canvas_w))
    parts.append(_render_scale_bar(canvas_h, scale))
    parts.append(
        f'<text x="{PADDING_PX}" y="{canvas_h - 10:.1f}" font-size="10" fill="#999999">{_esc(DISCLAIMER_TEXT)}</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def _render_road(site, to_px) -> str:
    """siteのroad_side側に接道の帯（幅員ラベル付き）を描画する"""
    side = site.road_side
    label = f"接道 幅員{site.road_width_m}m"

    if side == "south":
        gx0, gx1, gy0, gy1 = 0, site.width_grid, -ROAD_MARGIN_GRID, 0
    elif side == "north":
        gx0, gx1, gy0, gy1 = 0, site.width_grid, site.depth_grid, site.depth_grid + ROAD_MARGIN_GRID
    elif side == "west":
        gx0, gx1, gy0, gy1 = -ROAD_MARGIN_GRID, 0, 0, site.depth_grid
    else:  # east
        gx0, gx1, gy0, gy1 = site.width_grid, site.width_grid + ROAD_MARGIN_GRID, 0, site.depth_grid

    x0, y0 = to_px(gx0, gy1)  # 北西角（pxは最小）
    x1, y1 = to_px(gx1, gy0)  # 南東角（pxは最大）
    width_px, height_px = x1 - x0, y1 - y0

    return (
        f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{width_px:.1f}" height="{height_px:.1f}" '
        f'fill="#f0f0f0" />'
        f'<text x="{x0 + 4:.1f}" y="{y0 + height_px / 2:.1f}" font-size="10" fill="#777777">{_esc(label)}</text>'
    )


def _render_north_arrow(canvas_w: float) -> str:
    ax = canvas_w - PADDING_PX - 10
    ay = PADDING_PX
    return (
        f'<g transform="translate({ax:.1f},{ay:.1f})">'
        f'<line x1="0" y1="24" x2="0" y2="0" stroke="#333333" stroke-width="2" />'
        f'<polygon points="-5,6 5,6 0,-4" fill="#333333" />'
        f'<text x="0" y="38" font-size="12" text-anchor="middle" fill="#333333">N</text>'
        f'</g>'
    )


def _render_scale_bar(canvas_h: float, scale: float) -> str:
    length_px = (1.0 / GRID_M) * scale
    x0 = PADDING_PX
    y0 = canvas_h - 28
    return (
        f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x0 + length_px:.1f}" y2="{y0:.1f}" '
        f'stroke="#333333" stroke-width="2" />'
        f'<text x="{x0:.1f}" y="{y0 - 4:.1f}" font-size="10" fill="#333333">1m</text>'
    )
