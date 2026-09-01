"""間取り構造化 — 座標・幾何スキーマ（KH案 §9.2.1 準拠）

グリッドは910mm半間モジュール。全ての矩形は敷地南西角を原点とする
グローバル座標系（グリッド単位・整数）で格納し、中心座標はcomputed
propertyとして公開する。
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, computed_field

# --- 定数 --------------------------------------------------------------

GRID_MM: int = 910
GRID_M: float = GRID_MM / 1000.0
CELL_AREA_M2: float = GRID_M * GRID_M
TATAMI_M2: float = CELL_AREA_M2 * 2  # 1畳 = 2セル
TSUBO_M2: float = 3.30578

FLOOR_HEIGHT_M: float = 2.9
ROOF_RISE_M: float = 1.2

HALL_WIDTH_GRID: int = 2

# 延床面積のうち動線(玄関/階段/廊下/ホール)が占める概算比率。
# LLMが出す各部屋のarea_m2は申告総面積とスケールがずれることがあるため、
# 部屋面積の合計をこの比率を引いた分にリスケールして整合させる。
CIRCULATION_AREA_RATIO: float = 0.15

ROOM_TYPES: tuple[str, ...] = (
    "LDK", "リビング", "ダイニング", "キッチン", "パントリー",
    "主寝室", "寝室", "子供部屋", "書斎", "和室",
    "浴室", "洗面脱衣", "トイレ",
    "玄関", "ホール", "階段", "廊下",
    "収納", "WIC", "バルコニー", "その他",
)

CIRCULATION_TYPES: frozenset[str] = frozenset({"玄関", "ホール", "階段", "廊下"})

ROOM_TYPE_ALIASES: dict[str, str] = {
    "リビングダイニングキッチン": "LDK",
    "リビング・ダイニング・キッチン": "LDK",
    "洋室": "寝室",
    "洗面所": "洗面脱衣",
    "脱衣室": "洗面脱衣",
    "洗面室": "洗面脱衣",
    "ウォークインクローゼット": "WIC",
    "walk-in closet": "WIC",
    "納戸": "収納",
    "クローゼット": "収納",
    "ワークスペース": "書斎",
    "土間": "玄関",
    "玄関ホール": "玄関",
    "駐車場": "その他",
    "カーポート": "その他",
    "テラス": "バルコニー",
    "ベランダ": "バルコニー",
}


def normalize_room_type(raw: str) -> tuple[str, Optional[str]]:
    """LLM出力の部屋タイプ表記をROOM_TYPES語彙へ正規化する。

    Returns:
        (正規化後の room_type, 警告メッセージ or None)
    """
    if raw in ROOM_TYPES:
        return raw, None
    aliased = ROOM_TYPE_ALIASES.get(raw)
    if aliased is not None:
        return aliased, None
    return "その他", f"未知の部屋タイプ '{raw}' を 'その他' に正規化しました"


# --- エンジン入力 --------------------------------------------------------

class RoomSpec(BaseModel):
    """レイアウトエンジンへの入力（座標を持たない部屋の意図）"""

    room_id: str
    room_type: str = Field(description="ROOM_TYPESのいずれか")
    label: str = Field(description="表示名（例: 主寝室, 子供部屋1）")
    target_area_m2: float = Field(gt=0)
    floor: int = Field(ge=1, description="1階=1, 2階=2, ...")


# --- 幾何出力 ------------------------------------------------------------

class RoomBox(BaseModel):
    """座標付き部屋（グリッド単位・整数、南西角＋幅高）"""

    room_id: str
    room_type: str
    label: str
    x: int
    y: int
    w: int = Field(gt=0)
    h: int = Field(gt=0)
    floor: int
    target_area_m2: float

    @computed_field  # type: ignore[misc]
    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @computed_field  # type: ignore[misc]
    @property
    def cy(self) -> float:
        return self.y + self.h / 2

    @computed_field  # type: ignore[misc]
    @property
    def area_m2(self) -> float:
        return round(self.w * self.h * CELL_AREA_M2, 2)

    @computed_field  # type: ignore[misc]
    @property
    def area_tatami(self) -> float:
        return round(self.w * self.h / 2, 1)

    @computed_field  # type: ignore[misc]
    @property
    def aspect(self) -> float:
        long_side = max(self.w, self.h)
        short_side = min(self.w, self.h)
        return round(long_side / short_side, 2)


class Adjacency(BaseModel):
    a: str
    b: str
    relation: Literal["adjacent", "door", "stair"]
    shared_len_grid: int
    floor: int


class SiteBoundary(BaseModel):
    """敷地境界（グリッド単位・整数、南西角=(0,0)）"""

    preset_key: str
    width_grid: int = Field(gt=0)
    depth_grid: int = Field(gt=0)
    road_side: Literal["north", "south", "east", "west"]
    road_width_m: float
    setback_grid: int = 1

    @computed_field  # type: ignore[misc]
    @property
    def width_m(self) -> float:
        return round(self.width_grid * GRID_M, 2)

    @computed_field  # type: ignore[misc]
    @property
    def depth_m(self) -> float:
        return round(self.depth_grid * GRID_M, 2)

    @computed_field  # type: ignore[misc]
    @property
    def area_m2(self) -> float:
        return round(self.width_grid * self.depth_grid * CELL_AREA_M2, 2)

    @computed_field  # type: ignore[misc]
    @property
    def area_tsubo(self) -> float:
        return round(self.area_m2 / TSUBO_M2, 1)


class FloorGeometry(BaseModel):
    floor: int
    footprint_x: int
    footprint_y: int
    footprint_w: int
    footprint_h: int
    rooms: list[RoomBox] = Field(default_factory=list)
    adjacencies: list[Adjacency] = Field(default_factory=list)

    @computed_field  # type: ignore[misc]
    @property
    def footprint_area_m2(self) -> float:
        return round(self.footprint_w * self.footprint_h * CELL_AREA_M2, 2)


class BuildingGeometry(BaseModel):
    schema_version: str = "1.0"
    grid_mm: int = GRID_MM
    floor_height_m: float = FLOOR_HEIGHT_M
    roof_rise_m: float = ROOF_RISE_M
    site: SiteBoundary
    floors: list[FloorGeometry] = Field(default_factory=list)
    inter_floor_adjacencies: list[Adjacency] = Field(default_factory=list)
    layout_source: Literal["deterministic", "llm", "deterministic_fallback"] = "deterministic"
    notes: list[str] = Field(default_factory=list)
