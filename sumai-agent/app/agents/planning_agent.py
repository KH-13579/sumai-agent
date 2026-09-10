"""間取り生成AI — 住宅要件から複数の間取り案を生成するエージェント

座標はLLMに出させない。LLMは部屋タイプ・目標面積・階のみを構造化出力し、
座標は決定論的レイアウトエンジン（app.tools.layout_engine）が計算する。
"""
from __future__ import annotations

import logging
import os
import re
from typing import List, Optional

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

from app.data.site_presets import get_preset_by_key, select_site_preset
from app.schemas.geometry import CIRCULATION_AREA_RATIO, ROOM_TYPES, TATAMI_M2, TSUBO_M2, RoomSpec, normalize_room_type
from app.schemas.requirements import RequirementBaseline
from app.schemas.floorplan import FloorPlan, Room, PlanningOutput
from app.tools.geometry_check import run_geometry_check
from app.tools.layout_engine import build_geometry, infer_floor_for_room_type
from app.tools.layout_llm import generate_geometry_via_llm

logger = logging.getLogger(__name__)

_ROOM_TYPE_LIST = "、".join(ROOM_TYPES)
_TOTAL_AREA_M2_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*㎡")
_TOTAL_AREA_TSUBO_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*坪")
_FLOOR_COUNT_PATTERN = re.compile(r"(\d+)\s*階建て")
_RESCALE_WARN_THRESHOLD_LOW = 0.5
_RESCALE_WARN_THRESHOLD_HIGH = 2.0
# 一般的な戸建て住宅の延床面積として現実的な範囲(約12坪〜約121坪)。
# 小型モデルはtotal_floor_area自体を大きく誤ることがあるため、この範囲外の
# 申告値は信頼せず、部屋面積合計の方をそのまま採用する。
_PLAUSIBLE_TOTAL_AREA_MIN_M2 = 40.0
_PLAUSIBLE_TOTAL_AREA_MAX_M2 = 400.0


def _format_area_tatami(area_m2: float) -> str:
    """area_m2から表示用の畳数テキストを生成する

    LLMにこのテキスト自体を生成させると、英語混じりの不自然な文字列に
    なることがある（実測で確認済み）ため、常に数値area_m2から合成する。
    """
    return f"{round(area_m2 / TATAMI_M2, 1)}畳"


# 坪単価の概算レンジ（木造想定）。プロンプト中の説明文と揃えること。
_TSUBO_UNIT_PRICE_LOW = 60
_TSUBO_UNIT_PRICE_HIGH = 80


def _format_estimated_cost(total_floor_area_m2: float) -> str:
    """延床面積(m2)から概算費用を坪単価ベースで機械的に算出する

    estimated_costをLLMの自由記述に任せると、プロンプトのfew-shot例文
    「2,400〜3,000万円（建物本体。坪単価65万円前後）」をそのまま出力してしまい、
    面積が異なる案でも費用が一言一句同じになりかねない。
    _format_area_tatami と同じ理由で、費用も常に確定した延床面積から
    計算し、LLMの出力は使わない。
    """
    tsubo = total_floor_area_m2 / TSUBO_M2
    low = round(tsubo * _TSUBO_UNIT_PRICE_LOW / 100) * 100
    high = round(tsubo * _TSUBO_UNIT_PRICE_HIGH / 100) * 100
    return f"{low:,.0f}〜{high:,.0f}万円（建物本体。坪単価{_TSUBO_UNIT_PRICE_LOW}〜{_TSUBO_UNIT_PRICE_HIGH}万円想定）"

# 部屋タイプ別の面積レンジ(m2)。下のプロンプト文言（_ROOM_SIZE_REFERENCE）と
# 必ず同じ数値を使うこと。「必ずこの範囲に収める」とプロンプトで指示しても、
# 小型モデルは守れないことがある（主寝室が20畳＝約33m2になる、トイレが4m2超に
# なる等）。指示が効かない領域なので、_rooms_to_specs で確実にクランプする。
_ROOM_AREA_BOUNDS_M2: dict[str, tuple[float, float]] = {
    "LDK": (25.0, 42.0), "リビング": (13.0, 20.0), "ダイニング": (10.0, 13.0),
    "キッチン": (7.0, 10.0), "パントリー": (2.0, 3.0), "和室": (7.0, 13.0),
    "主寝室": (10.0, 17.0), "寝室": (7.0, 13.0), "子供部屋": (7.0, 10.0), "書斎": (5.0, 10.0),
    "浴室": (3.0, 5.0), "洗面脱衣": (3.0, 5.0), "トイレ": (1.6, 2.0),
    "収納": (2.0, 5.0), "WIC": (2.0, 5.0), "バルコニー": (3.0, 7.0),
}


# 延床面積への合わせ込み(_rescale_specs_to_target)の後段でも同じ上限で
# 厳密にクランプすると、コンセプトごとの延床面積の違い（コスパ重視は小さめ／
# 広さ重視は大きめ）が部屋サイズに反映されず、3案の部屋構成がほぼ同じ数値に
# 収束してしまう（副作用として概算費用まで同額になる）。
# 再スケール後は許容範囲を広げ、「明らかにおかしい」極端な値だけを補正する。
_POST_RESCALE_TOLERANCE = 1.4


def _clamp_room_area(room_type: str, area_m2: float, *, tolerance: float = 1.0) -> float:
    """部屋タイプ別の目安レンジ(_ROOM_AREA_BOUNDS_M2)に収まるよう補正する

    tolerance>1.0 を渡すと上限・下限を緩め、多少の逸脱は許容する（再スケール後用）。
    """
    bounds = _ROOM_AREA_BOUNDS_M2.get(room_type)
    if not bounds:
        return area_m2
    lo, hi = bounds
    return min(max(area_m2, lo / tolerance), hi * tolerance)


_ROOM_SIZE_REFERENCE = """## 部屋タイプ別の一般的な広さの目安（必ずこの範囲に収める）
- LDK: 15〜25畳（約25〜42m2）
- リビング: 8〜12畳（約13〜20m2）
- ダイニング: 6〜8畳（約10〜13m2）
- キッチン: 4〜6畳（約7〜10m2）
- パントリー: 1〜2畳（約2〜3m2）
- 主寝室: 6〜10畳（約10〜17m2）
- 寝室: 4.5〜8畳（約7〜13m2）
- 子供部屋: 4.5〜6畳（約7〜10m2）
- 書斎: 3〜6畳（約5〜10m2）
- 和室: 4.5〜8畳（約7〜13m2）
- 浴室: 2〜3畳（約3〜5m2）
- 洗面脱衣: 2〜3畳（約3〜5m2）
- トイレ: 1畳（約1.6〜2m2）
- 収納・WIC: 1〜3畳（約2〜5m2）
- バルコニー: 2〜4畳（約3〜7m2）
（1畳 ≒ 1.66m2 が目安）"""

PLANNING_SYSTEM_PROMPT = f"""あなたは住宅設計の専門家AIです。
ユーザーの住宅要件定義書をもとに、コンセプトの異なる3つの間取り案を提案します。

## 生成する3案のコンセプト
1. **コスパ重視案** — 予算内で最大限の機能を実現
2. **広さ重視案** — LDKや主要室の広さを優先
3. **収納・機能重視案** — 収納量・生活動線・使い勝手を優先

## 各案に含める情報
- concept: コンセプト名
- total_floor_area: 延床面積の目安（㎡と坪）。**一般的な戸建ての延床面積は25〜45坪（約80〜150m2）程度です。
  この範囲を大きく外れる値（100坪超や15坪未満など）は書かないこと**
- floors: 階数構成（例: 2階建て, 平屋）
- rooms: 主要な部屋一覧。各部屋には以下を必ず付与する
  - name: 表示名（例: LDK, 主寝室, 子供部屋1）
  - note: 補足（採光・用途など、なければnull）
  - room_type: 次の語彙から必ず1つ選ぶ（{_ROOM_TYPE_LIST}）
  - area_m2: 面積の目安（数値, m2）。下記の「部屋タイプ別の一般的な広さの目安」の範囲に収めること
  - floor: 所属階（1階=1, 2階=2 の整数）。2階建てなら主寝室・子供部屋・書斎などは必ずfloor=2にする

{_ROOM_SIZE_REFERENCE}

- layout_description: 間取りの全体説明（動線・採光・階構成）
- rationale: ユーザー要望への適合根拠
- estimated_cost: 概算費用レンジ（坪単価ベース: 木造60〜80万円/坪として概算）

## rationale・layout_descriptionの整合性（重要）
rationale・layout_descriptionでは、**この案のroomsに実際に含めた部屋・設備だけ**を
根拠として挙げること。例えば書斎を勧める理由に使うなら、その案のroomsに書斎を
必ず含める。roomsに入れていない部屋（書斎・和室など）を「設置可能」「対応」のように
言及してはならない（実際には無いのに「ある」と誤解させるため）。

## ユーザーが明示的に希望した部屋・設備の扱い（重要）
ユーザーが「書斎が欲しい」のように名指しで希望した部屋・設備は、**3案のうち
最低2案には反映する**こと（3案とも面積の都合で難しい場合のみ1案に絞ってよいが、
その場合は理由をrationaleで具体的に説明する）。名指しの希望を1案にしか反映しない
まま特に説明もない、という状態は避ける。

## 出力フォーマット（必ずJSON形式で返す）
{{
  "plans": [
    {{
      "concept": "コスパ重視案",
      "total_floor_area": "約100㎡（約30坪）",
      "floors": "2階建て",
      "rooms": [
        {{"name": "LDK", "area": "18畳", "note": "南向き・吹き抜けなし"}},
        {{"name": "主寝室", "area": "8畳", "note": "ウォークインクローゼット付き"}},
        {{"name": "子供部屋", "area": "6畳×1", "note": "将来仕切り対応"}},
        {{"name": "浴室・洗面", "area": "標準サイズ", "note": null}},
        {{"name": "トイレ", "area": "2箇所", "note": "各階"}},
        {{"name": "駐車場", "area": "1台", "note": "カーポート"}}
      ],
      "layout_description": "1階にLDK・浴室・洗面・トイレ・収納。2階に主寝室・子供部屋・トイレ。家事動線を重視したコンパクト設計。",
      "rationale": "予算3500万円以内で実現しやすい標準仕様。維持費も抑えられ、子育て世代に最適。",
      "estimated_cost": "2,400〜3,000万円（建物本体。坪単価65万円前後）"
    }}
  ],
  "summary": "3案の比較サマリー文（200字程度）"
}}

## 面積の書き方（厳守）
後段の法規チェック（建ぺい率・容積率）がこの数値を使って計算するため、以下を必ず守る。

- `total_floor_area` は **「約NNN㎡（約NN坪）」の形式のみ**。階数など他の情報を混ぜない
- **1坪 = 約3.31㎡**。坪で要望された広さは㎡に換算して書く（例: 35坪 → 約116㎡。「約35㎡」と書くのは誤り）
- **3案の延床面積は同じ値にしない**。コンセプトに応じて差をつける
  （目安: コスパ重視は要望より約10%小さめ／広さ重視は約10〜20%大きめ／収納重視はほぼ要望どおり）
- 延床面積は主要な部屋の面積合計を**必ず上回る**ようにする（廊下・階段・水回りを含むため）
- 一般的な戸建ての延床面積は **80〜150㎡（24〜45坪）** 程度。この範囲を大きく外れる場合は要望を読み違えている
- `floors` には階数のみを書く（例: "2階建て"）。面積を書かない
- 部屋の広さは「18畳」「6畳×2」のように畳数で書く（1畳 ≈ 1.62㎡）

## 注意事項
- 玄関・階段・廊下・ホールは配置エンジンが自動的に確保するため、roomsには含めなくてよい
- 概算費用は必ず「概算・専門家確認を推奨」の前提で提示
- 法規の厳密な判定は後段の法規チェックAIが行う。ここでは「参考プランです」と明示する
"""


class _LLMRoom(BaseModel):
    name: str = Field(description="部屋名（例: LDK, 主寝室, 子供部屋）")
    note: Optional[str] = Field(None, description="補足（採光・用途など）")
    room_type: str = Field(description="部屋タイプ（指定語彙から選択）")
    area_m2: float = Field(gt=0, description="面積の目安（数値, m2）")
    floor: int = Field(1, ge=1, description="所属階（1階=1, 2階=2, ...）")


class _LLMFloorPlan(BaseModel):
    concept: str
    total_floor_area: str
    floors: str
    rooms: List[_LLMRoom]
    layout_description: str
    rationale: str
    estimated_cost: Optional[str] = None


# ユーザーが名指しで要望した場合に、最低2/3案への反映を保証する「任意扱いになりがちな」
# 部屋タイプ。LDK・主寝室・トイレ等の基本部屋は本来どの案にも入るはずなので対象外にする。
_OPTIONAL_ROOM_TYPES = ("書斎", "和室", "WIC", "パントリー", "バルコニー")
_MIN_PLANS_WITH_REQUESTED_ROOM = 2


def _detect_requested_room_types(requirements: RequirementBaseline) -> List[str]:
    """要件書の自由記述欄から、名指しで要望された部屋タイプを検出する"""
    text = "".join(filter(None, [
        requirements.notes, requirements.lifestyle_flow,
        requirements.desired_size, requirements.storage_needs,
    ]))
    return [rt for rt in _OPTIONAL_ROOM_TYPES if rt in text]


def _inject_room(plan: "_LLMFloorPlan", room_type: str, lo: float, hi: float) -> None:
    declared_floor_count = _parse_declared_floor_count(plan.floors)
    floor = 2 if declared_floor_count >= 2 else 1
    plan.rooms.append(_LLMRoom(
        name=room_type, note=None, room_type=room_type,
        area_m2=round((lo + hi) / 2, 1), floor=floor,
    ))


def _ensure_requested_rooms_present(plans: List["_LLMFloorPlan"], requirements: RequirementBaseline) -> None:
    """名指しで要望された部屋タイプが、最低2/3案には入るよう保証する

    プロンプトで「複数案に反映すること」と指示しても、小型モデルは1案にしか
    反映しないことがある。部屋の有無は機械的に判定できるので、プロンプトの
    指示だけに頼らず決定論的に補う。
    """
    if len(plans) < 2:
        return
    target_count = min(_MIN_PLANS_WITH_REQUESTED_ROOM, len(plans))
    for room_type in _detect_requested_room_types(requirements):
        lo, hi = _ROOM_AREA_BOUNDS_M2.get(room_type, (5.0, 8.0))
        has_room = [any(r.room_type == room_type for r in p.rooms) for p in plans]
        mentions = [room_type in (p.layout_description + p.rationale) for p in plans]

        needed = target_count - sum(has_room)
        if needed > 0:
            # 部屋は無いのに説明文だけでその部屋の存在を主張している案があれば、
            # 文章と部屋リストの不整合を解消できるようそちらを優先して補う。
            missing_indices = [i for i, already in enumerate(has_room) if not already]
            missing_indices.sort(key=lambda i: not mentions[i])
            for i in missing_indices[:needed]:
                _inject_room(plans[i], room_type, lo, hi)
                has_room[i] = True

        # 上記で必要数を満たしてもなお、部屋が無いのに文章だけがその部屋の存在を
        # 主張している案が残っていれば、差別化よりも「書いてあることと実際の
        # 部屋が食い違わない」ことを優先し、そちらにも追加する（小型モデルは
        # 3案すべての説明文に同じ部屋を書いてしまうことがあり、優先順位付け
        # だけでは解消しきれないため）。
        for i, (already, mentioned) in enumerate(zip(has_room, mentions)):
            if not already and mentioned:
                _inject_room(plans[i], room_type, lo, hi)


class _LLMPlanningOutput(BaseModel):
    plans: List[_LLMFloorPlan] = Field(description="生成した間取り案（3案）")
    summary: str = Field(description="3案の比較サマリー")


def _rooms_to_specs(rooms: List[_LLMRoom]) -> tuple[List[RoomSpec], List[str]]:
    """LLM出力の部屋一覧をRoomSpecへ変換する。正規化の警告をnotesとして返す"""
    specs: List[RoomSpec] = []
    warnings: List[str] = []
    for idx, room in enumerate(rooms):
        room_type, warning = normalize_room_type(room.room_type)
        if warning:
            warnings.append(warning)
        floor = room.floor if room.floor and room.floor >= 1 else infer_floor_for_room_type(room_type)
        specs.append(
            RoomSpec(
                room_id=f"room{idx}",
                room_type=room_type,
                label=room.name,
                target_area_m2=_clamp_room_area(room_type, room.area_m2),
                floor=floor,
            )
        )
    return specs, warnings


def _parse_declared_floor_count(floors_text: str) -> int:
    """floors文字列（例: '2階建て', '平屋'）から階数を読み取る。既定は1"""
    if "平屋" in floors_text:
        return 1
    match = _FLOOR_COUNT_PATTERN.search(floors_text)
    if match:
        return max(1, int(match.group(1)))
    return 1


def _reassign_floors_if_needed(
    specs: List[RoomSpec], declared_floor_count: int
) -> tuple[List[RoomSpec], List[str]]:
    """申告階数と実際に使われている階が矛盾する場合、部屋タイプから階を再推定する

    小型モデルでは「2階建て」と宣言しつつ全部屋をfloor=1で出すことがある。
    その場合、寝室系ゾーン(FLOOR2_ZONES)の部屋を階上に振り直す。
    """
    warnings: List[str] = []
    if declared_floor_count <= 1 or not specs:
        return specs, warnings

    if max(s.floor for s in specs) >= declared_floor_count:
        return specs, warnings

    warnings.append(
        f"申告階数({declared_floor_count}階建て)に対し全部屋がfloor="
        f"{max(s.floor for s in specs)}に集中していたため、部屋タイプから階を再推定しました"
    )
    reassigned = [
        s.model_copy(update={"floor": min(infer_floor_for_room_type(s.room_type), declared_floor_count)})
        for s in specs
    ]
    return reassigned, warnings


def _parse_target_total_area_m2(total_floor_area: str) -> Optional[float]:
    """total_floor_area文字列（例: '約100㎡（約30坪）'）から数値を抜き出す"""
    match = _TOTAL_AREA_M2_PATTERN.search(total_floor_area)
    if match:
        return float(match.group(1))
    match = _TOTAL_AREA_TSUBO_PATTERN.search(total_floor_area)
    if match:
        return float(match.group(1)) * TSUBO_M2
    return None


def _rescale_specs_to_target(
    specs: List[RoomSpec], target_total_floor_area_m2: Optional[float]
) -> tuple[List[RoomSpec], List[str]]:
    """LLMの部屋面積合計を、申告延床面積(動線分を除く)に合わせてリスケールする

    LLMはtotal_floor_areaとarea_m2のスケールが数割〜数倍ずれることがある
    （数値の精密さに課題があるという既知の傾向）。相対的な部屋の大小関係は
    信頼し、絶対スケールだけをユーザーが実際に指定した延床面積に合わせる。
    """
    warnings: List[str] = []
    if not specs or target_total_floor_area_m2 is None or target_total_floor_area_m2 <= 0:
        return specs, warnings

    if not (_PLAUSIBLE_TOTAL_AREA_MIN_M2 <= target_total_floor_area_m2 <= _PLAUSIBLE_TOTAL_AREA_MAX_M2):
        warnings.append(
            f"申告延床面積({round(target_total_floor_area_m2, 1)}m2)が現実的な住宅の規模から"
            "外れていたため無視し、部屋面積の合計をそのまま採用しました"
        )
        return specs, warnings

    current_sum = sum(s.target_area_m2 for s in specs)
    if current_sum <= 0:
        return specs, warnings

    room_target_total = target_total_floor_area_m2 * (1 - CIRCULATION_AREA_RATIO)
    scale = room_target_total / current_sum

    if scale < _RESCALE_WARN_THRESHOLD_LOW or scale > _RESCALE_WARN_THRESHOLD_HIGH:
        warnings.append(
            f"LLMの部屋面積合計({round(current_sum, 1)}m2)が申告延床面積"
            f"({round(target_total_floor_area_m2, 1)}m2)と大きくずれていたため"
            f"{round(scale, 2)}倍にスケール補正しました"
        )

    # 延床面積への合わせ込み（scale）は全部屋に一律で掛かるため、クランプ済みの
    # 部屋がここで再び目安レンジの外に押し戻されることがある。再度クランプして
    # 「延床面積の帳尻合わせで主寝室だけ異常に大きくなる」を確実に防ぐ。
    rescaled = [
        s.model_copy(update={
            "target_area_m2": _clamp_room_area(
                s.room_type, round(s.target_area_m2 * scale, 2), tolerance=_POST_RESCALE_TOLERANCE
            )
        })
        for s in specs
    ]
    return rescaled, warnings


def _select_site(requirements: RequirementBaseline):
    override_key = os.getenv("SUMAI_SITE_PRESET", "").strip()
    if override_key:
        preset = get_preset_by_key(override_key)
        if preset is not None:
            return preset
        logger.warning("SUMAI_SITE_PRESET='%s' は未知のプリセットのため要件から自動選択します", override_key)
    return select_site_preset(requirements.land_info)


def _build_floor_plan(llm_plan: _LLMFloorPlan, requirements: RequirementBaseline, llm: ChatOllama) -> FloorPlan:
    specs, normalize_warnings = _rooms_to_specs(llm_plan.rooms)

    declared_floor_count = _parse_declared_floor_count(llm_plan.floors)
    specs, reassign_warnings = _reassign_floors_if_needed(specs, declared_floor_count)

    target_total_m2 = _parse_target_total_area_m2(llm_plan.total_floor_area)
    specs, rescale_warnings = _rescale_specs_to_target(specs, target_total_m2)

    area_by_room_id = {s.room_id: s.target_area_m2 for s in specs}
    floor_by_room_id = {s.room_id: s.floor for s in specs}

    rooms = [
        Room(
            name=r.name,
            area=_format_area_tatami(area_by_room_id.get(f"room{idx}", r.area_m2)),
            note=r.note,
            room_type=normalize_room_type(r.room_type)[0],
            area_m2=area_by_room_id.get(f"room{idx}", r.area_m2),
            floor=floor_by_room_id.get(f"room{idx}", r.floor),
        )
        for idx, r in enumerate(llm_plan.rooms)
    ]

    geometry = None
    check = None
    layout_mode = os.getenv("SUMAI_LAYOUT_MODE", "deterministic").strip().lower()
    try:
        site = _select_site(requirements)
        if layout_mode == "llm":
            geometry = generate_geometry_via_llm(specs, site, llm)
        else:
            geometry = build_geometry(specs, site)
        geometry.notes.extend(normalize_warnings)
        geometry.notes.extend(reassign_warnings)
        geometry.notes.extend(rescale_warnings)
        check = run_geometry_check(geometry)
    except Exception:
        logger.exception("案「%s」のジオメトリ生成に失敗しました。テキスト提案のみ返します", llm_plan.concept)

    # 概算費用は「申告延床面積」（コンセプトごとに意図的に差をつけている数値）を
    # 優先して使う。部屋面積合計から逆算する方式だと、_clamp_room_area の許容範囲に
    # 収まるよう複数案が同じような値に補正され、費用まで同額になってしまうため、
    # 申告値が非現実的な場合のみ部屋面積合計にフォールバックする。
    if target_total_m2 is not None and _PLAUSIBLE_TOTAL_AREA_MIN_M2 <= target_total_m2 <= _PLAUSIBLE_TOTAL_AREA_MAX_M2:
        estimated_total_floor_area_m2 = target_total_m2
    else:
        total_room_area_m2 = sum(r.area_m2 for r in rooms)
        estimated_total_floor_area_m2 = total_room_area_m2 / (1 - CIRCULATION_AREA_RATIO) if total_room_area_m2 > 0 else None
    estimated_cost = (
        _format_estimated_cost(estimated_total_floor_area_m2)
        if estimated_total_floor_area_m2
        else llm_plan.estimated_cost
    )

    return FloorPlan(
        concept=llm_plan.concept,
        total_floor_area=llm_plan.total_floor_area,
        floors=llm_plan.floors,
        rooms=rooms,
        layout_description=llm_plan.layout_description,
        rationale=llm_plan.rationale,
        estimated_cost=estimated_cost,
        geometry=geometry,
        check=check,
    )


def run_planning(
    requirements: RequirementBaseline,
    llm: ChatOllama,
    legal_constraints: str | None = None,
) -> PlanningOutput:
    """住宅要件書をもとに間取り3案を生成する

    legal_constraints は法規チェックAIからの修正指示（自律修正ループの2周目）。
    渡された場合は建ぺい率・容積率・高さの上限を守るようプロンプトに制約を追加する。
    """
    req_summary = f"""
## 住宅要件書
- 家族構成: {requirements.family_structure or "不明"}
- 予算: {requirements.budget or "不明"}
- 土地: {requirements.land_info or "不明"}
- 希望の広さ・部屋数: {requirements.desired_size or "不明"}
- 好みのデザイン: {requirements.preferred_design or "未指定"}
- 重視する生活動線: {requirements.lifestyle_flow or "未指定"}
- 収納の希望: {requirements.storage_needs or "未指定"}
- その他の要望: {requirements.notes or "なし"}

上記の要件をもとに、コンセプトの異なる3つの間取り案を提案してください。
"""

    if legal_constraints:
        req_summary += "\n" + legal_constraints + "\n"

    messages = [
        SystemMessage(content=PLANNING_SYSTEM_PROMPT),
        HumanMessage(content=req_summary),
    ]

    structured_llm = llm.with_structured_output(_LLMPlanningOutput, include_raw=True)
    try:
        raw_result = structured_llm.invoke(messages)
    except Exception as e:
        # LLM呼び出し自体（接続エラー・タイムアウト等）の失敗。生出力は取得できない
        logger.error("間取り生成のLLM呼び出しに失敗しました: %s", e)
        return PlanningOutput(
            plans=[],
            summary="間取り案の生成中にエラーが発生しました。もう一度お試しください。",
        )

    result: Optional[_LLMPlanningOutput] = raw_result.get("parsed")
    if result is None:
        # JSON Schema制約下でも、フィールド制約違反などでパースが失敗することがある。
        # 次回同じ現象を診断できるよう、実際のモデル出力をログに残す。
        raw_message = raw_result.get("raw")
        raw_content = str(getattr(raw_message, "content", ""))[:2000]
        logger.error(
            "間取り生成の構造化出力パースに失敗しました: parsing_error=%s raw_content=%s",
            raw_result.get("parsing_error"), raw_content,
        )
        return PlanningOutput(
            plans=[],
            summary="間取り案の生成中にエラーが発生しました。もう一度お試しください。",
        )

    _ensure_requested_rooms_present(result.plans, requirements)

    plans: List[FloorPlan] = []
    for p in result.plans:
        try:
            plans.append(_build_floor_plan(p, requirements, llm))
        except Exception:
            logger.exception("案「%s」の構築に失敗したためスキップします", p.concept)

    return PlanningOutput(plans=plans, summary=result.summary)
