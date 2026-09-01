"""スマイエージェント — メーカー推薦AI 出力スキーマ"""
from __future__ import annotations
from typing import Optional, List
from pydantic import BaseModel, Field


class MakerRecommendation(BaseModel):
    """1社分の推薦情報"""
    rank: int = Field(description="推薦順位（1〜3）")
    name: str = Field(description="メーカー・ポータル名")
    type: str = Field(description="'builder'（注文住宅メーカー）or 'portal'（情報ポータル）")
    reason: str = Field(description="このユーザーへの推薦理由（要件との照合根拠）")
    strengths: List[str] = Field(default_factory=list, description="主な強み")
    price_band: str = Field(description="価格帯")
    best_for: List[str] = Field(default_factory=list, description="どんなニーズに向くか")
    website: str = Field(description="公式サイトURL")
    caution: Optional[str] = Field(None, description="注意点・デメリット（あれば）")


class MakerRecommendationOutput(BaseModel):
    """メーカー推薦AI の出力"""
    recommendations: List[MakerRecommendation] = Field(
        description="推薦メーカー・ポータル（最大3件）"
    )
    summary: str = Field(description="推薦全体のサマリー・補足説明")
