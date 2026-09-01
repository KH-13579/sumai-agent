"""エージェントパイプライン — 専門エージェントの直列実行と拡張点

要件充足後に走る専門エージェント群を「順序付きのステップ列」として宣言的に定義し、
LangGraph のノードと条件付きエッジを自動生成する。エージェントを追加する際に
グラフの分岐を書き換える必要はなく、ステップ列に1件追加するだけで済む。

    STEPS = [
        AgentStep("planning", "planning", planning_node),
        AgentStep("legal",    "legal",    legal_node),
        AgentStep("estimate", "estimate", estimate_node),   # ← 追加はこの1行
    ]
    add_pipeline_to_graph(builder, STEPS, terminal="compose")

- `is_enabled` … その回のリクエストで実行するかどうか（条件付きスキップ）
- `route_override` … 次ステップの代わりに任意のノードへ飛ばす（生成→検査→修正の逆流）
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from app.agents.state import SumaiState


def _always_enabled(state: SumaiState) -> bool:
    return True


def _no_override(state: SumaiState) -> Optional[str]:
    return None


@dataclass(frozen=True)
class AgentStep:
    """パイプラインを構成する1エージェント"""

    name: str
    """LangGraph のノード名。route_override の飛び先としても使われる"""

    stage: str
    """ChatResponse.stage に出す値。UI のフロー表示に使う"""

    run: Callable[[SumaiState], dict]
    """ノード関数。state を受け取り、更新する差分 dict を返す"""

    is_enabled: Callable[[SumaiState], bool] = _always_enabled
    """False を返すとこのステップを飛ばして次のステップへ進む"""

    route_override: Callable[[SumaiState], Optional[str]] = _no_override
    """ノード名を返すとそこへ遷移する。修正ループの逆流に使う（None なら次ステップ）"""


def resolve_next(
    steps: Sequence[AgentStep], current_index: int, state: SumaiState, terminal: str
) -> str:
    """current_index の次に実行すべきノード名を求める

    実行可能な後続ステップが無ければ terminal を返す。
    """
    for step in steps[current_index + 1 :]:
        if step.is_enabled(state):
            return step.name
    return terminal


def first_enabled(steps: Sequence[AgentStep], state: SumaiState, terminal: str) -> str:
    """パイプラインの入口となるノード名を求める（前段ノードのルーターから使う）"""
    return resolve_next(steps, -1, state, terminal)


def _make_router(
    steps: Sequence[AgentStep], index: int, terminal: str
) -> Callable[[SumaiState], str]:
    """1ステップぶんのルーター関数を作る"""
    step = steps[index]

    def route(state: SumaiState) -> str:
        override = step.route_override(state)
        if override:
            return override
        return resolve_next(steps, index, state, terminal)

    return route


def add_pipeline_to_graph(
    builder, steps: Sequence[AgentStep], terminal: str
) -> None:
    """StateGraph にステップ列のノードと条件付きエッジを登録する

    ルーターは「後続ステップ名のいずれか」または terminal を返すため、
    遷移先マップには全ステップ名と terminal を含める（逆流にも対応するため）。
    """
    for step in steps:
        builder.add_node(step.name, step.run)

    targets = {step.name: step.name for step in steps}
    targets[terminal] = terminal

    for index, step in enumerate(steps):
        builder.add_conditional_edges(step.name, _make_router(steps, index, terminal), targets)
