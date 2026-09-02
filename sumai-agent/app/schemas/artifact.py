"""スマイエージェント — 機械可読アーティファクト

要件定義書 v1.0 §16「機械可読アーティファクト」に対応する統合出力。
1セッションで各エージェントが生成した結果を1つの JSON にまとめ、
ハウスメーカー側システムや下流エージェント（見積AI）が
そのまま読み込める形で提供する。

下流エージェントを追加する際は、対応するスロットをここに1行追加する。
"""
from __future__ import annotations
from datetime import datetime
from typing import Any, Optional, List
from pydantic import BaseModel, Field

from app.schemas.floorplan import FloorPlan
from app.schemas.legal import PlanLegalCheck, SiteInfo
from app.schemas.maker import MakerRecommendation
from app.schemas.requirements import RequirementBaseline


class SumaiArtifact(BaseModel):
    """セッションの成果物一式（機械可読形式）"""
    schema_version: str = Field("1.0", description="本アーティファクトのスキーマ版")
    session_id: str = Field(description="セッションID")
    generated_at: datetime = Field(description="生成日時")

    # ── MVP ──
    requirements: Optional[RequirementBaseline] = Field(None, description="住宅要件書（ヒアリングAI）")
    floor_plans: List[FloorPlan] = Field(default_factory=list, description="間取り案（間取り生成AI）")
    site_info: Optional[SiteInfo] = Field(None, description="敷地・法規制パラメータ（法規チェックAI）")
    legal_checks: List[PlanLegalCheck] = Field(default_factory=list, description="法規チェック結果（法規チェックAI）")
    maker_recommendation: Optional[List[MakerRecommendation]] = Field(None, description="メーカー推薦（メーカー推薦AI）")

    # ── Phase 2（スロットのみ先に定義）──
    estimate: Optional[dict[str, Any]] = Field(None, description="概算見積（見積AI／Phase 2）")

    disclaimer: str = Field(description="免責表示（NFR-05／LAW-4）")
