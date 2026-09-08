"""スマイエージェント — ヒアリングAI 出力スキーマ"""
from __future__ import annotations
from typing import Optional, List
from pydantic import BaseModel, Field


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
