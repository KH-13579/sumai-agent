"""間取り構造化 — SVG描画（KH案 §8.3 / FR-08 / NFR-05）

BuildingGeometryから階ごとに1枚のSVGを生成する純関数。LLMはこの描画結果を
読み書きしない（KH案 §9.2.1）。

見た目は実際の不動産サイトにある「間取り図」（線画＋建具記号）に近づけている。
ドアの開口・スイング記号は layout_engine.derive_adjacencies が既に計算済みの
隣接関係（relation="door"）をそのまま使っており、新しいデータもLLM呼び出しの
追加も必要としない（決定論的な後処理のみ）。
"""
from __future__ import annotations

from xml.sax.saxutils import escape

from app.schemas.geometry import GRID_M, CIRCULATION_TYPES, BuildingGeometry, FloorGeometry, RoomBox

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
# トイレの下限（1.6m2 ≒ 1.93セル。planning_agent._ROOM_AREA_BOUNDS_M2 と揃える）より
# 小さいと、色だけで名前が表示されない部屋ができてしまう（何の部屋か分からない）。
# 部屋タイプの最小想定サイズより確実に低い値にする。
MIN_LABEL_CELLS = 1.5

WALL_COLOR = "#3a3a3a"
FIXTURE_COLOR = "#55606b"
WINDOW_COLOR = "#3f7fbf"

# 窓を描く部屋タイプ（居室系のみ。トイレ・収納・階段・廊下・玄関等は対象外）
WINDOW_ROOM_TYPES = frozenset({
    "LDK", "リビング", "ダイニング", "キッチン", "和室",
    "主寝室", "寝室", "子供部屋", "書斎", "浴室",
})
# 建具記号を描くための最低面積（セル数）。同様にトイレの下限より確実に低くする
MIN_FIXTURE_CELLS = 1.8


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
    base_canvas_h = (max_y - min_y) * scale + 2 * PADDING_PX

    # 凡例：色だけで部屋の種類が分からない、という指摘への対応。
    # この階に実際にある部屋タイプだけを表示する（出現順・重複除去）。
    room_types_in_use: list[str] = []
    seen_types: set[str] = set()
    for room in floor_geo.rooms:
        if room.room_type not in seen_types:
            seen_types.add(room.room_type)
            room_types_in_use.append(room.room_type)
    legend_svg, legend_h = _render_legend(room_types_in_use, canvas_w, base_canvas_h)
    canvas_h = base_canvas_h + legend_h

    def to_px(gx: float, gy: float) -> tuple[float, float]:
        px = PADDING_PX + (gx - min_x) * scale
        py = PADDING_PX + (max_y - gy) * scale
        return px, py

    def rect_px(x: float, y: float, w: float, h: float) -> tuple[float, float, float, float]:
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
        f'fill="none" stroke="{WALL_COLOR}" stroke-width="4" />'
    )

    room_boxes: dict[str, tuple[float, float, float, float]] = {}
    for room in floor_geo.rooms:
        rx, ry, rw, rh = rect_px(room.x, room.y, room.w, room.h)
        room_boxes[room.room_id] = (rx, ry, rw, rh)
        color = ROOM_COLORS.get(room.room_type, DEFAULT_ROOM_COLOR)
        parts.append(
            f'<rect x="{rx:.1f}" y="{ry:.1f}" width="{rw:.1f}" height="{rh:.1f}" '
            f'fill="{color}" stroke="{WALL_COLOR}" stroke-width="1.4" />'
        )
        fixture_svg = ""
        if room.w * room.h >= MIN_FIXTURE_CELLS:
            fixture_svg = _render_fixture(room.room_type, rx, ry, rw, rh)
            if fixture_svg:
                parts.append(fixture_svg)
        if room.w * room.h >= MIN_LABEL_CELLS:
            parts.append(_render_room_text(room, rx, ry, rw, rh, has_fixture=bool(fixture_svg)))
        parts.append(_render_windows(room, floor_geo, to_px))

    parts.append(_render_doors(floor_geo, room_boxes, to_px))

    parts.append(_render_north_arrow(canvas_w))
    parts.append(_render_scale_bar(base_canvas_h, scale))
    parts.append(
        f'<text x="{PADDING_PX}" y="{base_canvas_h - 10:.1f}" font-size="10" fill="#999999">{_esc(DISCLAIMER_TEXT)}</text>'
    )
    parts.append(legend_svg)
    parts.append("</svg>")
    return "".join(parts)


# ─── 部屋ラベル（幅に応じた自動縮小・折り返し） ───────────────────
# 浴室・トイレ等は1グリッド幅（約91cm=32px）しかなく、"洗面脱衣"のような
# 4文字ラベルや"2.0畳 / 3.31m2"のような面積表記が固定フォントサイズだと
# 隣の部屋やアイコンにはみ出す。部屋の実際のピクセル幅に収まるようフォントサイズを
# 下げ、それでも収まらなければ2行に折り返す。

_CJK_CODEPOINT_MIN = 0x2E80  # この値以上をおおよそ全角文字とみなす
_LABEL_FONT_SIZES: tuple[int, ...] = (12, 11, 10, 9, 8)


def _char_width(ch: str, font_size: float) -> float:
    # 全角（CJK）文字はほぼ正方形、半角の英数字・記号は概ね半分の幅
    return font_size * (0.98 if ord(ch) >= _CJK_CODEPOINT_MIN else 0.56)


def _text_width(text: str, font_size: float) -> float:
    return sum(_char_width(ch, font_size) for ch in text)


def _fit_font_size(text: str, max_width: float) -> int:
    for size in _LABEL_FONT_SIZES:
        if _text_width(text, size) <= max_width:
            return size
    return _LABEL_FONT_SIZES[-1]


def _wrap_label(text: str) -> list[str]:
    """1行に収まらないラベルを2行に折り返す（"洗面脱衣"→"洗面"/"脱衣"）"""
    if len(text) <= 2:
        return [text]
    mid = (len(text) + 1) // 2
    return [text[:mid], text[mid:]]


# ─── 駐車場の実寸表示 ─────────────────────────────────────────
# 駐車場は room_type="その他"（他の物置等と同じ扱い）になるため、ラベル文字列で判定する。
# 「◯畳/◯m2」という表記は面積の感覚が掴みにくいので、実寸(m)と一般的な駐車枠の目安
# （国交省の資料等でよく使われる幅・奥行の目安）から車格を推定して表示する。
_PARKING_LABEL_KEYWORDS = ("駐車場", "カーポート", "駐車")

# (最小幅m, 最小奥行m, 表示ラベル) を大きい方から順に判定する
_PARKING_SIZE_CLASSES: tuple[tuple[float, float, str], ...] = (
    (2.5, 5.3, "SUV・ミニバン可"),
    (2.3, 5.0, "普通車可"),
    (2.0, 3.6, "軽自動車可"),
)


def _is_parking_room(room: RoomBox) -> bool:
    return any(kw in room.label for kw in _PARKING_LABEL_KEYWORDS)


def _parking_size_lines(room: RoomBox) -> list[str]:
    """駐車場の実寸(m)と、目安となる車格を2行に分けて返す（参考情報・保証しない）"""
    width_m = round(min(room.w, room.h) * GRID_M, 1)
    depth_m = round(max(room.w, room.h) * GRID_M, 1)
    car_class = "やや手狭"
    for min_w, min_d, label in _PARKING_SIZE_CLASSES:
        if width_m >= min_w and depth_m >= min_d:
            car_class = label
            break
    return [f"{width_m}m×{depth_m}m", f"{car_class}の目安"]


def _render_room_text(room: RoomBox, rx: float, ry: float, rw: float, rh: float, *, has_fixture: bool) -> str:
    max_w = rw * 0.94
    label = room.label

    lines = [label]
    size = _fit_font_size(label, max_w)
    if _text_width(label, size) > max_w:
        lines = _wrap_label(label)
        size = min(_fit_font_size(line, max_w) for line in lines)

    if _is_parking_room(room):
        # 「◯畳/◯m2」は駐車場では直感的でないため、実寸(m)と車格の目安に置き換える
        info_lines = _parking_size_lines(room)
    else:
        info_lines = [f"{room.area_tatami}畳 / {room.area_m2}m2"]

    area_size = max(7, size - 2)
    # 情報の優先度が低いため、幅に収まらない行は削り、それでも1行も残らない・
    # 高さが足りない場合は丸ごと省略する（ラベル本体を優先し、潰れた文字は出さない）
    fitted_info_lines = [line for line in info_lines if _text_width(line, area_size) <= max_w]
    min_h_grid = (1.5 if has_fixture else 1.05) + 0.4 * max(0, len(fitted_info_lines) - 1)
    if not fitted_info_lines or rh < PX_PER_GRID * min_h_grid:
        fitted_info_lines = []

    line_gap = size + 3
    area_gap = area_size + 2
    block_h = (len(lines) - 1) * line_gap + len(fitted_info_lines) * area_gap
    cx = rx + rw / 2

    if has_fixture:
        top = ry + rh - block_h - 8  # アイコンは上側、ラベルは部屋下部に寄せる
    else:
        top = ry + rh / 2 - block_h / 2 + size * 0.35

    parts: list[str] = []
    y = top
    for line in lines:
        parts.append(
            f'<text x="{cx:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="middle" fill="#222222">'
            f'{_esc(line)}</text>'
        )
        y += line_gap
    for line in fitted_info_lines:
        parts.append(
            f'<text x="{cx:.1f}" y="{y:.1f}" font-size="{area_size}" text-anchor="middle" fill="#555555">'
            f'{_esc(line)}</text>'
        )
        y += area_gap
    return "".join(parts)


# ─── 建具・設備記号 ─────────────────────────────────────────
# 実際の間取り図に近づけるため、部屋タイプごとに簡易的な記号を描く。
# いずれも座標計算のみの純粋なSVG生成で、外部画像・LLMは使わない。

def _render_fixture(room_type: str, rx: float, ry: float, rw: float, rh: float) -> str:
    if room_type == "トイレ":
        return _fixture_toilet(rx, ry, rw, rh)
    if room_type == "浴室":
        return _fixture_bath(rx, ry, rw, rh)
    if room_type == "洗面脱衣":
        return _fixture_sink(rx, ry, rw, rh)
    if room_type in ("主寝室", "寝室", "子供部屋"):
        return _fixture_bed(rx, ry, rw, rh, room_type)
    if room_type == "階段":
        return _fixture_stairs(rx, ry, rw, rh)
    if room_type in ("収納", "WIC"):
        return _fixture_closet(rx, ry, rw, rh)
    return ""


def _fixture_toilet(rx: float, ry: float, rw: float, rh: float) -> str:
    short = min(rw, rh)
    pad = short * 0.16
    w = short * 0.34
    tank_h = w * 0.4
    bowl_h = w * 0.6
    x0 = rx + rw - pad - w
    y0 = ry + pad
    return (
        f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{w:.1f}" height="{tank_h:.1f}" '
        f'fill="none" stroke="{FIXTURE_COLOR}" stroke-width="1" />'
        f'<ellipse cx="{x0 + w / 2:.1f}" cy="{y0 + tank_h + bowl_h / 2:.1f}" rx="{w / 2:.1f}" ry="{bowl_h / 2:.1f}" '
        f'fill="none" stroke="{FIXTURE_COLOR}" stroke-width="1" />'
    )


def _fixture_bath(rx: float, ry: float, rw: float, rh: float) -> str:
    pad = min(rw, rh) * 0.14
    bw = rw - 2 * pad
    bh = min(rh * 0.5, rw * 0.65)
    x0, y0 = rx + pad, ry + pad
    return (
        f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{bw:.1f}" height="{bh:.1f}" rx="7" ry="7" '
        f'fill="none" stroke="{FIXTURE_COLOR}" stroke-width="1.2" />'
    )


def _fixture_sink(rx: float, ry: float, rw: float, rh: float) -> str:
    pad = min(rw, rh) * 0.16
    w = rw - 2 * pad
    h = min(rh * 0.3, w * 0.5)
    x0, y0 = rx + pad, ry + pad
    cx, cy, r = x0 + w * 0.28, y0 + h / 2, h * 0.32
    return (
        f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{w:.1f}" height="{h:.1f}" '
        f'fill="none" stroke="{FIXTURE_COLOR}" stroke-width="1" />'
        f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="none" stroke="{FIXTURE_COLOR}" stroke-width="1" />'
    )


# ベッドは部屋の広さに対する割合ではなく、現実の寸法（グリッド単位）で固定する。
# 割合ベースだと同じ「シングルベッド」でも部屋の広さ次第で巨大化・矮小化してしまい、
# サイズ感がおかしく見える。ドアの実寸91cm運用と同じ考え方。
_BED_SIZE_GRID: dict[str, tuple[float, float]] = {
    "double": (1.6, 2.15),   # ダブルベッド 約145×195cm（主寝室向け）
    "single": (1.05, 2.0),   # シングルベッド 約95×180cm（寝室・子供部屋向け）
}
_BED_PILLOW_DEPTH_GRID = 0.35


def _bed_shape(rx: float, ry: float, rw: float, rh: float, bed_type: str) -> str:
    bw, bl = (v * PX_PER_GRID for v in _BED_SIZE_GRID[bed_type])
    margin = PX_PER_GRID * 0.12

    if bw + margin * 2 <= rw and bl + margin * 2 <= rh:
        vertical, w_draw, h_draw = True, bw, bl
    elif bl + margin * 2 <= rw and bw + margin * 2 <= rh:
        vertical, w_draw, h_draw = False, bl, bw
    else:
        return ""  # 実寸のベッドが物理的に入らない部屋には無理に描かない

    x0, y0 = rx + margin, ry + margin
    pillow_depth = _BED_PILLOW_DEPTH_GRID * PX_PER_GRID
    parts = [
        f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{w_draw:.1f}" height="{h_draw:.1f}" rx="4" ry="4" '
        f'fill="#ffffff" fill-opacity="0.85" stroke="{FIXTURE_COLOR}" stroke-width="1" />'
    ]
    if vertical:
        parts.append(
            f'<rect x="{x0 + 2:.1f}" y="{y0 + 2:.1f}" width="{w_draw - 4:.1f}" height="{pillow_depth:.1f}" '
            f'fill="#eef1f4" stroke="{FIXTURE_COLOR}" stroke-width="0.6" />'
        )
    else:
        parts.append(
            f'<rect x="{x0 + 2:.1f}" y="{y0 + 2:.1f}" width="{pillow_depth:.1f}" height="{h_draw - 4:.1f}" '
            f'fill="#eef1f4" stroke="{FIXTURE_COLOR}" stroke-width="0.6" />'
        )
    return "".join(parts)


def _fixture_bed(rx: float, ry: float, rw: float, rh: float, room_type: str) -> str:
    bed_type = "double" if room_type == "主寝室" else "single"
    svg = _bed_shape(rx, ry, rw, rh, bed_type)
    if not svg and bed_type == "double":
        svg = _bed_shape(rx, ry, rw, rh, "single")  # ダブルが入らなければシングルに格下げ
    return svg


def _fixture_stairs(rx: float, ry: float, rw: float, rh: float) -> str:
    vertical = rh >= rw
    steps = max(4, int((rh if vertical else rw) // (min(rw, rh) * 0.22)))
    lines = []
    if vertical:
        for i in range(1, steps):
            y = ry + rh * i / steps
            lines.append(f'<line x1="{rx:.1f}" y1="{y:.1f}" x2="{rx + rw:.1f}" y2="{y:.1f}" stroke="{FIXTURE_COLOR}" stroke-width="0.8" />')
        ax = rx + rw / 2
        lines.append(f'<line x1="{ax:.1f}" y1="{ry + rh * 0.85:.1f}" x2="{ax:.1f}" y2="{ry + rh * 0.15:.1f}" stroke="{FIXTURE_COLOR}" stroke-width="1" />')
        lines.append(f'<polygon points="{ax - 4:.1f},{ry + rh * 0.22:.1f} {ax + 4:.1f},{ry + rh * 0.22:.1f} {ax:.1f},{ry + rh * 0.12:.1f}" fill="{FIXTURE_COLOR}" />')
    else:
        for i in range(1, steps):
            x = rx + rw * i / steps
            lines.append(f'<line x1="{x:.1f}" y1="{ry:.1f}" x2="{x:.1f}" y2="{ry + rh:.1f}" stroke="{FIXTURE_COLOR}" stroke-width="0.8" />')
        ay = ry + rh / 2
        lines.append(f'<line x1="{rx + rw * 0.15:.1f}" y1="{ay:.1f}" x2="{rx + rw * 0.85:.1f}" y2="{ay:.1f}" stroke="{FIXTURE_COLOR}" stroke-width="1" />')
        lines.append(f'<polygon points="{rx + rw * 0.78:.1f},{ay - 4:.1f} {rx + rw * 0.78:.1f},{ay + 4:.1f} {rx + rw * 0.88:.1f},{ay:.1f}" fill="{FIXTURE_COLOR}" />')
    return "".join(lines)


def _fixture_closet(rx: float, ry: float, rw: float, rh: float) -> str:
    # 収納・WICは対角線（X）で表す（不動産の間取り図での慣習的な表記）
    return (
        f'<line x1="{rx:.1f}" y1="{ry:.1f}" x2="{rx + rw:.1f}" y2="{ry + rh:.1f}" stroke="{FIXTURE_COLOR}" stroke-width="0.8" />'
        f'<line x1="{rx + rw:.1f}" y1="{ry:.1f}" x2="{rx:.1f}" y2="{ry + rh:.1f}" stroke="{FIXTURE_COLOR}" stroke-width="0.8" />'
    )


# ─── 窓 ─────────────────────────────────────────────────────

def _exterior_sides(room: RoomBox, floor_geo: FloorGeometry) -> list[str]:
    """部屋のどの辺が建物の外周（外壁）に接しているかを返す"""
    sides = []
    if room.x == floor_geo.footprint_x:
        sides.append("west")
    if room.x + room.w == floor_geo.footprint_x + floor_geo.footprint_w:
        sides.append("east")
    if room.y == floor_geo.footprint_y:
        sides.append("south")
    if room.y + room.h == floor_geo.footprint_y + floor_geo.footprint_h:
        sides.append("north")
    return sides


def _render_windows(room: RoomBox, floor_geo: FloorGeometry, to_px) -> str:
    if room.room_type not in WINDOW_ROOM_TYPES:
        return ""
    parts = []
    for side in _exterior_sides(room, floor_geo):
        if side in ("west", "east"):
            gx = room.x if side == "west" else room.x + room.w
            gy0, gy1 = room.y + room.h * 0.25, room.y + room.h * 0.75
            (px, py0), (_, py1) = to_px(gx, gy0), to_px(gx, gy1)
            parts.append(
                f'<line x1="{px:.1f}" y1="{py0:.1f}" x2="{px:.1f}" y2="{py1:.1f}" '
                f'stroke="{WINDOW_COLOR}" stroke-width="3.5" />'
            )
        else:
            gy = room.y if side == "south" else room.y + room.h
            gx0, gx1 = room.x + room.w * 0.25, room.x + room.w * 0.75
            (px0, py), (px1, _) = to_px(gx0, gy), to_px(gx1, gy)
            parts.append(
                f'<line x1="{px0:.1f}" y1="{py:.1f}" x2="{px1:.1f}" y2="{py:.1f}" '
                f'stroke="{WINDOW_COLOR}" stroke-width="3.5" />'
            )
    return "".join(parts)


# ─── ドア（開口部・スイング記号） ─────────────────────────────
# layout_engine.derive_adjacencies が既に判定済みの relation="door" を使う。
# 新しい幾何データは持たず、部屋の矩形からその場で壁の共有区間を再計算する。

def _door_openings(floor_geo: FloorGeometry) -> list[dict]:
    rooms_by_id = {r.room_id: r for r in floor_geo.rooms}
    openings: list[dict] = []
    for adj in floor_geo.adjacencies:
        if adj.relation != "door":
            continue
        a, b = rooms_by_id.get(adj.a), rooms_by_id.get(adj.b)
        if a is None or b is None:
            continue

        touch_x = (a.x + a.w == b.x) or (b.x + b.w == a.x)
        touch_y = (a.y + a.h == b.y) or (b.y + b.h == a.y)

        if touch_x:
            wall_x = a.x + a.w if a.x + a.w == b.x else b.x + b.w
            lo, hi = max(a.y, b.y), min(a.y + a.h, b.y + b.h)
            if hi - lo < 0.6:
                continue
            door_len = min(1.0, hi - lo)
            mid = (lo + hi) / 2
            swing_room = _swing_target(a, b)
            direction = -1 if swing_room.x + swing_room.w == wall_x else 1
            openings.append({
                "axis": "v", "const": wall_x, "start": mid - door_len / 2, "end": mid + door_len / 2,
                "direction": direction, "door_len": door_len,
            })
        elif touch_y:
            wall_y = a.y + a.h if a.y + a.h == b.y else b.y + b.h
            lo, hi = max(a.x, b.x), min(a.x + a.w, b.x + b.w)
            if hi - lo < 0.6:
                continue
            door_len = min(1.0, hi - lo)
            mid = (lo + hi) / 2
            swing_room = _swing_target(a, b)
            direction = -1 if swing_room.y + swing_room.h == wall_y else 1
            openings.append({
                "axis": "h", "const": wall_y, "start": mid - door_len / 2, "end": mid + door_len / 2,
                "direction": direction, "door_len": door_len,
            })
    return openings


def _swing_target(a: RoomBox, b: RoomBox) -> RoomBox:
    """ドアがどちら側に開くかを選ぶ。廊下側ではなく居室側に開くのが自然"""
    a_circ, b_circ = a.room_type in CIRCULATION_TYPES, b.room_type in CIRCULATION_TYPES
    if a_circ and not b_circ:
        return b
    if b_circ and not a_circ:
        return a
    return b


def _render_doors(floor_geo: FloorGeometry, room_boxes: dict[str, tuple[float, float, float, float]], to_px) -> str:
    parts: list[str] = []
    for opening in _door_openings(floor_geo):
        if opening["axis"] == "v":
            wall_x = opening["const"]
            (hx, hy) = to_px(wall_x, opening["start"])
            (_, oy) = to_px(wall_x, opening["end"])
            gap_top, gap_bottom = min(hy, oy), max(hy, oy)
            parts.append(
                f'<rect x="{hx - 3:.1f}" y="{gap_top:.1f}" width="6" height="{gap_bottom - gap_top:.1f}" fill="#ffffff" />'
            )
            tip_x = hx + opening["direction"] * opening["door_len"] * PX_PER_GRID
            hinge, tip, other = (hx, hy), (tip_x, hy), (hx, oy)
        else:
            wall_y = opening["const"]
            (hx, hy) = to_px(opening["start"], wall_y)
            (ox, _) = to_px(opening["end"], wall_y)
            gap_left, gap_right = min(hx, ox), max(hx, ox)
            parts.append(
                f'<rect x="{gap_left:.1f}" y="{hy - 3:.1f}" width="{gap_right - gap_left:.1f}" height="6" fill="#ffffff" />'
            )
            tip_y = hy - opening["direction"] * opening["door_len"] * PX_PER_GRID
            hinge, tip, other = (hx, hy), (hx, tip_y), (ox, hy)
        parts.append(_door_symbol(hinge, tip, other))
    return "".join(parts)


def _door_symbol(hinge: tuple[float, float], tip: tuple[float, float], other: tuple[float, float]) -> str:
    (hx, hy), (tx, ty), (ox, oy) = hinge, tip, other
    r = ((tx - hx) ** 2 + (ty - hy) ** 2) ** 0.5
    if r < 1:
        return ""
    cross = (tx - hx) * (oy - hy) - (ty - hy) * (ox - hx)
    sweep = 1 if cross > 0 else 0
    return (
        f'<line x1="{hx:.1f}" y1="{hy:.1f}" x2="{tx:.1f}" y2="{ty:.1f}" stroke="{WALL_COLOR}" stroke-width="1.2" />'
        f'<path d="M {tx:.1f} {ty:.1f} A {r:.1f} {r:.1f} 0 0 {sweep} {ox:.1f} {oy:.1f}" '
        f'fill="none" stroke="{WALL_COLOR}" stroke-width="0.8" stroke-dasharray="2,2" />'
    )


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


# ─── 凡例 ─────────────────────────────────────────────────
# 色だけで部屋の種類が判別できない、という指摘への対応。この階に実際にある
# 部屋タイプだけを、色スウォッチ＋ラベルとして横並び・折り返しで描画する。

_LEGEND_SWATCH_SIZE = 12
_LEGEND_FONT_SIZE = 10
_LEGEND_ROW_HEIGHT = 20
_LEGEND_ITEM_GAP = 16


def _render_legend(room_types: list[str], canvas_w: float, y_start: float) -> tuple[str, float]:
    """凡例のSVG断片と、消費した高さ(px)を返す

    部屋ラベルの自動縮小（_text_width）と同じ考え方で、キャンバス幅に収まる分だけ
    1行に詰め、収まらなければ次の行へ折り返す。
    """
    if not room_types:
        return "", 0.0

    max_x = canvas_w - PADDING_PX
    parts: list[str] = []
    x = float(PADDING_PX)
    y = y_start + _LEGEND_ROW_HEIGHT
    rows = 1

    for room_type in room_types:
        item_w = _LEGEND_SWATCH_SIZE + 4 + _text_width(room_type, _LEGEND_FONT_SIZE) + _LEGEND_ITEM_GAP
        if x + item_w > max_x and x > PADDING_PX:
            x = float(PADDING_PX)
            y += _LEGEND_ROW_HEIGHT
            rows += 1
        color = ROOM_COLORS.get(room_type, DEFAULT_ROOM_COLOR)
        swatch_y = y - _LEGEND_SWATCH_SIZE + 2
        parts.append(
            f'<rect x="{x:.1f}" y="{swatch_y:.1f}" width="{_LEGEND_SWATCH_SIZE}" height="{_LEGEND_SWATCH_SIZE}" '
            f'fill="{color}" stroke="{WALL_COLOR}" stroke-width="0.6" />'
        )
        parts.append(
            f'<text x="{x + _LEGEND_SWATCH_SIZE + 4:.1f}" y="{y:.1f}" font-size="{_LEGEND_FONT_SIZE}" fill="#555555">'
            f'{_esc(room_type)}</text>'
        )
        x += item_w

    total_h = rows * _LEGEND_ROW_HEIGHT + 8
    return "".join(parts), total_h
