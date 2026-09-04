"""スマイエージェント — 間取り生成AI 出力スキーマ"""
from __future__ import annotations
from typing import Optional, List
from pydantic import BaseModel, Field


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
