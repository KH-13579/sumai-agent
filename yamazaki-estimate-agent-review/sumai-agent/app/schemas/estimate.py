"""概算見積ツールの入出力スキーマ。"""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


StructureType = Literal["wood", "steel", "rc"]
GradeType = Literal["economy", "standard", "premium"]
EquipmentType = Literal["solar_4kw", "central_air", "zeh_insulation"]


class EstimateRequest(BaseModel):
    floor_area_sqm: float = Field(gt=0, le=1000, description="延床面積（平方メートル）")
    structure: StructureType = Field("wood", description="構造")
    floors: int = Field(2, ge=1, le=3, description="階数")
    grade: GradeType = Field("standard", description="仕様グレード")
    equipment: List[EquipmentType] = Field(default_factory=list, description="追加設備")
    budget_yen: Optional[int] = Field(None, gt=0, description="建物予算（円）")

    @model_validator(mode="after")
    def reject_duplicate_equipment(self):
        if len(self.equipment) != len(set(self.equipment)):
            raise ValueError("equipment に同じ設備を重複指定できません")
        return self


class CostLine(BaseModel):
    label: str
    amount_yen: int
    calculation: str


class SavingsOption(BaseModel):
    label: str
    estimated_savings_yen: int
    note: str


class EstimateResult(BaseModel):
    floor_area_sqm: float
    floor_area_tsubo: float
    low_yen: int
    expected_yen: int
    high_yen: int
    budget_yen: Optional[int]
    within_budget: Optional[bool]
    budget_gap_yen: Optional[int]
    lines: List[CostLine]
    savings_options: List[SavingsOption]
    assumptions: List[str]
    source_name: str
    source_url: str
    source_fiscal_year: int
    disclaimer: str
