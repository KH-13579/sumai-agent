"""スマイエージェント — チャット API スキーマ"""
from __future__ import annotations
from typing import Optional, List, Literal
from pydantic import BaseModel, Field

from app.schemas.requirements import RequirementBaseline
from app.schemas.floorplan import FloorPlan
from app.schemas.maker import MakerRecommendation


class ChatRequest(BaseModel):
    session_id: str = Field(description="セッションID")
    message: str = Field(description="ユーザーのメッセージ")


class ChatResponse(BaseModel):
    reply: str = Field(description="自然言語の応答（ユーザー向け）")
    requirements: Optional[RequirementBaseline] = Field(None, description="現在の住宅要件書")
    floor_plans: Optional[List[FloorPlan]] = Field(None, description="生成した間取り案")
    maker_recommendations: Optional[List[MakerRecommendation]] = Field(None, description="推薦ハウスメーカー・ポータル")
    stage: Literal["hearing", "planning", "maker", "follow_up"] = Field(description="現在のフェーズ")
    done: bool = Field(False, description="間取り提示まで完了したか")
