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

## 注意事項
- 玄関・階段・廊下・ホールは配置エンジンが自動的に確保するため、roomsには含めなくてよい
- 概算費用は必ず「概算・専門家確認を推奨」の前提で提示
- 法規チェックは行わず「参考プランです」と明示する
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
                target_area_m2=room.area_m2,
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

    rescaled = [s.model_copy(update={"target_area_m2": round(s.target_area_m2 * scale, 2)}) for s in specs]
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

    return FloorPlan(
        concept=llm_plan.concept,
        total_floor_area=llm_plan.total_floor_area,
        floors=llm_plan.floors,
        rooms=rooms,
        layout_description=llm_plan.layout_description,
        rationale=llm_plan.rationale,
        estimated_cost=llm_plan.estimated_cost,
        geometry=geometry,
        check=check,
    )


def run_planning(requirements: RequirementBaseline, llm: ChatOllama) -> PlanningOutput:
    """住宅要件書をもとに間取り3案を生成する（座標は決定論エンジンが付与）"""
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

    plans: List[FloorPlan] = []
    for p in result.plans:
        try:
            plans.append(_build_floor_plan(p, requirements, llm))
        except Exception:
            logger.exception("案「%s」の構築に失敗したためスキップします", p.concept)

    return PlanningOutput(plans=plans, summary=result.summary)
