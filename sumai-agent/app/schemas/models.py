"""スマイエージェント — データスキーマ定義"""
from __future__ import annotations
from typing import Optional, List, Literal
from pydantic import BaseModel, Field


# ─────────────────────────────────────────
# ヒアリングAI 出力スキーマ
# ─────────────────────────────────────────

class RequirementBaseline(BaseModel):
    """ヒアリングAI が作成・随時更新する住宅要件書"""
    family_structure: Optional[str] = Field(None, description="家族構成")
    budget: Optional[str] = Field(None, description="予算（総予算・建物予算）")
    land_info: Optional[str] = Field(None, description="土地の有無・所在地・広さ・条件")
    preferred_design: Optional[str] = Field(None, description="好みのデザイン・テイスト")
    desired_size: Optional[str] = Field(None, description="希望の広さ・延床面積・部屋数")
    lifestyle_flow: Optional[str] = Field(None, description="重視する生活動線")
    storage_needs: Optional[str] = Field(None, description="収納の希望")
    notes: Optional[str] = Field(None, description="その他の要望")
    is_complete: bool = Field(False, description="間取り生成に進める最低限が揃ったか")
    missing_fields: List[str] = Field(default_factory=list, description="不足している重要項目")


class HearingOutput(BaseModel):
    """ヒアリングAI の出力"""
    requirements: RequirementBaseline
    follow_up_question: Optional[str] = Field(None, description="追質問（不足時のみ）")


# ─────────────────────────────────────────
# 間取り生成AI 出力スキーマ
# ─────────────────────────────────────────

class Room(BaseModel):
    name: str = Field(description="部屋名（例: LDK, 主寝室, 子供部屋）")
    area: str = Field(description="広さの目安（例: 20畳, 8畳）")
    note: Optional[str] = Field(None, description="補足（採光・用途など）")


class FloorPlan(BaseModel):
    concept: str = Field(description="コンセプト（例: コスパ重視, 広さ重視, 収納重視）")
    total_floor_area: str = Field(description="延床面積の目安（例: 約110㎡／約33坪）")
    floors: str = Field(description="階数構成（例: 2階建て, 平屋）")
    rooms: List[Room] = Field(description="主要な部屋一覧")
    layout_description: str = Field(description="間取りの全体説明（動線・採光・階構成）")
    rationale: str = Field(description="ユーザー要望への適合根拠")
    estimated_cost: Optional[str] = Field(None, description="概算費用レンジ（坪単価ベース）")


class PlanningOutput(BaseModel):
    """間取り生成AI の出力"""
    plans: List[FloorPlan] = Field(description="生成した間取り案（3案）")
    summary: str = Field(description="3案の比較サマリー")


# ─────────────────────────────────────────
# チャット API スキーマ
# ─────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id: str = Field(description="セッションID")
    message: str = Field(description="ユーザーのメッセージ")


class ChatResponse(BaseModel):
    reply: str = Field(description="自然言語の応答（ユーザー向け）")
    requirements: Optional[RequirementBaseline] = Field(None, description="現在の住宅要件書")
    floor_plans: Optional[List[FloorPlan]] = Field(None, description="生成した間取り案")
    stage: Literal["hearing", "planning", "follow_up"] = Field(description="現在のフェーズ")
    done: bool = Field(False, description="間取り提示まで完了したか")
