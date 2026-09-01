"""間取り生成AI — 住宅要件から複数の間取り案を生成するエージェント"""
from __future__ import annotations

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
import json

from app.schemas.requirements import RequirementBaseline
from app.schemas.floorplan import FloorPlan, Room, PlanningOutput
from app.tools.area_utils import normalize_total_floor_area

PLANNING_SYSTEM_PROMPT = """あなたは住宅設計の専門家AIです。
ユーザーの住宅要件定義書をもとに、コンセプトの異なる3つの間取り案を提案します。

## 生成する3案のコンセプト
1. **コスパ重視案** — 予算内で最大限の機能を実現
2. **広さ重視案** — LDKや主要室の広さを優先
3. **収納・機能重視案** — 収納量・生活動線・使い勝手を優先

## 各案に含める情報
- concept: コンセプト名
- total_floor_area: 延床面積の目安（㎡と坪）
- floors: 階数構成
- rooms: 主要な部屋一覧（部屋名・広さ・補足）
- layout_description: 間取りの全体説明（動線・採光・階構成）
- rationale: ユーザー要望への適合根拠
- estimated_cost: 概算費用レンジ（坪単価ベース: 木造60〜80万円/坪として概算）

## 出力フォーマット（必ずJSON形式で返す）
{
  "plans": [
    {
      "concept": "コスパ重視案",
      "total_floor_area": "約100㎡（約30坪）",
      "floors": "2階建て",
      "rooms": [
        {"name": "LDK", "area": "18畳", "note": "南向き・吹き抜けなし"},
        {"name": "主寝室", "area": "8畳", "note": "ウォークインクローゼット付き"},
        {"name": "子供部屋", "area": "6畳×1", "note": "将来仕切り対応"},
        {"name": "浴室・洗面", "area": "標準サイズ", "note": null},
        {"name": "トイレ", "area": "2箇所", "note": "各階"},
        {"name": "駐車場", "area": "1台", "note": "カーポート"}
      ],
      "layout_description": "1階にLDK・浴室・洗面・トイレ・収納。2階に主寝室・子供部屋・トイレ。家事動線を重視したコンパクト設計。",
      "rationale": "予算3500万円以内で実現しやすい標準仕様。維持費も抑えられ、子育て世代に最適。",
      "estimated_cost": "2,400〜3,000万円（建物本体。坪単価65万円前後）"
    }
  ],
  "summary": "3案の比較サマリー文（200字程度）"
}

## 面積の書き方（厳守）
後段の法規チェック（建ぺい率・容積率）がこの数値を使って計算するため、以下を必ず守る。

- `total_floor_area` は **「約NNN㎡（約NN坪）」の形式のみ**。階数など他の情報を混ぜない
- **1坪 = 約3.31㎡**。坪で要望された広さは㎡に換算して書く（例: 35坪 → 約116㎡。「約35㎡」と書くのは誤り）
- **3案の延床面積は同じ値にしない**。コンセプトに応じて差をつける
  （目安: コスパ重視は要望より約10%小さめ／広さ重視は約10〜20%大きめ／収納重視はほぼ要望どおり）
- 延床面積は主要な部屋の面積合計を**必ず上回る**ようにする（廊下・階段・水回りを含むため）
- 一般的な戸建ての延床面積は **80〜150㎡（24〜45坪）** 程度。この範囲を大きく外れる場合は要望を読み違えている
- `floors` には階数のみを書く（例: "2階建て"）。面積を書かない
- 部屋の広さは「18畳」「6畳×2」のように畳数で書く（1畳 ≈ 1.62㎡）

## 注意事項
- 必ず有効なJSONのみを返す（前後に余分なテキストは不要）
- 概算費用は必ず「概算・専門家確認を推奨」の前提で提示
- 法規の厳密な判定は後段の法規チェックAIが行う。ここでは「参考プランです」と明示する
"""


def run_planning(
    requirements: RequirementBaseline,
    llm: ChatOllama,
    legal_constraints: str | None = None,
) -> PlanningOutput:
    """住宅要件書をもとに間取り3案を生成する

    legal_constraints は法規チェックAIからの修正指示（自律修正ループの2周目）。
    渡された場合は建ぺい率・容積率・高さの上限を守るようプロンプトに制約を追加する。
    """
    req_summary = f"""
## 住宅要件書
- 家族構成: {requirements.family_structure or "不明"}
- 予算: {requirements.budget or "不明"}
- 土地: {requirements.land_info or "不明"}
- 希望の広さ・部屋数: {requirements.desired_size or "不明"}
- 好みのデザイン: {requirements.preferred_design or "未指定"}
- 重視する生活動線: {requirements.lifestyle_flow or "未指定"}
- 収納の希望: {requirements.storage_needs or "未指定"}
- その他の要望: {requirements.notes or "なし"}

上記の要件をもとに、コンセプトの異なる3つの間取り案を提案してください。
"""

    if legal_constraints:
        req_summary += "\n" + legal_constraints + "\n"

    messages = [
        SystemMessage(content=PLANNING_SYSTEM_PROMPT),
        HumanMessage(content=req_summary),
    ]

    # JSON解析に失敗した場合は1回だけリトライする
    data = None
    for _attempt in range(2):
        response = llm.invoke(messages)
        raw_text = response.content
        try:
            text = raw_text.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(lines[1:-1])
            data = json.loads(text)
            break
        except json.JSONDecodeError:
            data = None

    if data is None:
        return PlanningOutput(
            plans=[],
            summary="間取り案の生成中にエラーが発生しました。もう一度お試しください。",
        )

    plans = []
    corrections: list[str] = []
    for p in data.get("plans", []):
        rooms = [
            Room(
                name=r.get("name", ""),
                area=r.get("area", ""),
                note=r.get("note"),
            )
            for r in p.get("rooms", [])
        ]

        # LLM は坪の数値を㎡欄に書くことがある（例: 35坪 → "約35㎡"）。
        # 後段の法規チェックが面積を数値として使うため、ここで決定論的に補正する。
        concept = p.get("concept", "")
        total_area, correction = normalize_total_floor_area(
            p.get("total_floor_area", ""), [r.area for r in rooms]
        )
        if correction:
            corrections.append(f"{concept}: {correction}")

        plans.append(
            FloorPlan(
                concept=concept,
                total_floor_area=total_area or "",
                floors=p.get("floors", ""),
                rooms=rooms,
                layout_description=p.get("layout_description", ""),
                rationale=p.get("rationale", ""),
                estimated_cost=p.get("estimated_cost"),
            )
        )

    summary = data.get("summary", "")
    if corrections:
        # 補正した事実は隠さず出力に残す（説明可能性の担保）
        summary += "\n\n※ 面積表記の自動補正: " + " / ".join(corrections)

    return PlanningOutput(
        plans=plans,
        summary=summary,
    )
