"""グラフ状態の定義とエージェント間の受け渡し契約

各専門エージェントは「自分の構造化結果を state の専用スロットに書き込み、ユーザー向けの
説明文は reply_sections に1セクション追加する」という共通の作法に従う。
最終的な応答文の組み立て（ORC-4 翻訳）は compose ノードが一括で行うため、
エージェントを追加しても応答の体裁は崩れない。
"""
from __future__ import annotations

from typing import Annotated, Any, List, Optional, Sequence, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from app.schemas.floorplan import FloorPlan
from app.schemas.legal import PlanLegalCheck, SiteInfo
from app.schemas.maker import MakerRecommendation
from app.schemas.requirements import RequirementBaseline


class ReplySection(TypedDict):
    """1エージェントぶんのユーザー向け応答セクション"""
    agent: str      # エージェント識別子（"planning" / "legal" / ...）
    title: str      # セクション見出し（ログ・デバッグ用。本文には含めない）
    markdown: str   # 本文（markdown）


def merge_reply_sections(
    old: Optional[Sequence[ReplySection]], new: Optional[Sequence[ReplySection]]
) -> List[ReplySection]:
    """reply_sections のリデューサー

    None を渡すとリセットする。MemorySaver は状態をターンを越えて保持するため、
    ターン開始時にオーケストレーターがリセットしないと前ターンの応答が積み重なる。
    """
    if new is None:
        return []
    return list(old or []) + list(new)


def section(agent: str, title: str, markdown: str) -> List[ReplySection]:
    """ノードの戻り値にそのまま入れられる形でセクションを1件作る

        return {"reply_sections": section("legal", "法規チェック結果", text), ...}
    """
    return [ReplySection(agent=agent, title=title, markdown=markdown)]


class SumaiState(TypedDict, total=False):
    """LangGraph のグラフ状態

    エージェントを追加するときは、結果の格納先スロットをここに1行追加する。
    """
    messages: Annotated[List[BaseMessage], add_messages]

    # ── 各エージェントの構造化結果 ──
    requirements: Optional[RequirementBaseline]          # ヒアリングAI
    floor_plans: Optional[List[FloorPlan]]               # 間取り生成AI
    site_info: Optional[SiteInfo]                        # 法規チェックAI（敷地照会）
    legal_checks: Optional[List[PlanLegalCheck]]         # 法規チェックAI（判定結果）
    estimate: Optional[dict[str, Any]]                   # Phase 2: 見積AI
    maker_recommendation: Optional[List[MakerRecommendation]]  # メーカー推薦AI

    # ── 応答の組み立て ──
    stage: str                                           # 現在のフェーズ（ChatResponse.stage）
    reply: str                                           # compose ノードが確定させる最終応答
    reply_sections: Annotated[List[ReplySection], merge_reply_sections]
    done: bool                                           # 間取り提示まで到達したか

    # ── エージェント間の指示（生成→検査→修正の受け渡し）──
    legal_constraints: Optional[str]                      # 法規チェック→間取り生成への修正指示

    # ── ループ停止性（NFR-07）──
    hearing_turns: int                                   # ヒアリングを繰り返した回数
    legal_retry: int                                     # 法規NGによる間取り再生成の回数
