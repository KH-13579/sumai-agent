"""スマイエージェント — 法規チェックAI 出力スキーマ

要件定義書 v1.0 §7.6（LAW-1〜4）に対応する。
- LAW-2: 自動判定できる項目 → LegalCheckItem
- LAW-3: 人間が確認すべき項目 → ManualCheckFlag（要確認フラグ）
- LAW-4: 適法性を保証しない旨 → LegalCheckOutput.disclaimer
"""
from __future__ import annotations
from typing import Optional, List, Literal
from pydantic import BaseModel, Field

# 判定ステータス。ng（不適合） > unknown（判定材料不足） > ok（適合）の優先順で総合判定する
CheckStatus = Literal["ok", "ng", "unknown"]

# 敷地情報の取得元。デモの説明可能性（どこまでが実データか）を担保するために保持する
SiteSource = Literal["user_input", "preset", "reinfolib", "assumed"]


class SiteInfo(BaseModel):
    """敷地・法規制パラメータ（敷地照会ツール／LLM抽出で埋める）

    要件定義書 §7.7「上級者向け」入力項目および §9.1 の GIS 系データソースに対応。
    未取得の項目は None のままにし、推測で埋めた場合は assumptions に必ず記録する。
    """
    address: Optional[str] = Field(None, description="所在地（住所・地番）")
    site_area_sqm: Optional[float] = Field(None, description="敷地面積（㎡）")
    zoning: Optional[str] = Field(None, description="用途地域（例: 第一種低層住居専用地域）")
    building_coverage_ratio: Optional[float] = Field(None, description="指定建ぺい率（%）")
    floor_area_ratio: Optional[float] = Field(None, description="指定容積率（%）")
    road_width_m: Optional[float] = Field(None, description="前面道路の幅員（m）")
    fire_zone: Optional[str] = Field(None, description="防火地域／準防火地域／指定なし")
    height_limit_m: Optional[float] = Field(None, description="絶対高さ制限（m）。低層住居専用地域等のみ")
    has_land: bool = Field(False, description="土地を所有・特定済みか（false なら仮定値による参考判定）")
    source: SiteSource = Field("assumed", description="敷地情報の取得元")
    assumptions: List[str] = Field(default_factory=list, description="判定に用いた仮定・前提")


class LegalCheckItem(BaseModel):
    """自動判定できた法規項目（LAW-2）"""
    item: str = Field(description="項目名（例: 建ぺい率, 容積率, 絶対高さ制限）")
    status: CheckStatus = Field(description="判定結果")
    actual: Optional[str] = Field(None, description="実測・推定値の表示文字列")
    limit: Optional[str] = Field(None, description="上限値の表示文字列")
    margin: Optional[str] = Field(None, description="余裕／超過の量")
    basis: str = Field(description="根拠法令（例: 建築基準法 第53条）")
    message: str = Field(description="判定の説明文（前提・仮定を含む）")


class ManualCheckFlag(BaseModel):
    """自動判定できず専門家確認が必要な項目（LAW-3 要確認フラグ）"""
    item: str = Field(description="確認項目名（例: 道路斜線制限）")
    reason: str = Field(description="自動判定できない理由・確認が必要な理由")
    basis: str = Field(description="根拠法令・出典")
    severity: Literal["high", "medium", "low"] = Field("medium", description="確認の優先度")


class PlanLegalCheck(BaseModel):
    """間取り1案に対する法規チェック結果"""
    plan_index: int = Field(description="対象の間取り案番号（1始まり）")
    plan_concept: str = Field(description="対象の間取り案コンセプト")
    status: CheckStatus = Field(description="自動判定項目の総合結果")
    items: List[LegalCheckItem] = Field(default_factory=list, description="自動判定項目（LAW-2）")
    manual_flags: List[ManualCheckFlag] = Field(default_factory=list, description="要確認フラグ（LAW-3）")
    summary: str = Field("", description="この案の判定サマリー")


class LegalCheckOutput(BaseModel):
    """法規チェックAI の出力"""
    site_info: SiteInfo = Field(description="判定に用いた敷地・法規制パラメータ")
    checks: List[PlanLegalCheck] = Field(default_factory=list, description="間取り案ごとの判定結果")
    summary: str = Field("", description="全案を通した総括")
    disclaimer: str = Field("", description="免責表示（LAW-4／NFR-05）")
    references: List[str] = Field(
        default_factory=list,
        description="参照した法令条文。e-Gov 法令API による条文取得（Phase 2/3）の受け口",
    )
