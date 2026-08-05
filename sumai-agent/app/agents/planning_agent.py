"""間取り生成AI — 住宅要件から複数の間取り案を生成するエージェント"""
from __future__ import annotations

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
import json

from app.schemas.requirements import RequirementBaseline
from app.schemas.floorplan import FloorPlan, Room, PlanningOutput

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

## 注意事項
- 必ず有効なJSONのみを返す（前後に余分なテキストは不要）
- 概算費用は必ず「概算・専門家確認を推奨」の前提で提示
- 法規チェックは行わず「参考プランです」と明示する
"""


def run_planning(requirements: RequirementBaseline, llm: ChatOllama) -> PlanningOutput:
    """住宅要件書をもとに間取り3案を生成する"""
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
    for p in data.get("plans", []):
        rooms = [
            Room(
                name=r.get("name", ""),
                area=r.get("area", ""),
                note=r.get("note"),
            )
            for r in p.get("rooms", [])
        ]
        plans.append(
            FloorPlan(
                concept=p.get("concept", ""),
                total_floor_area=p.get("total_floor_area", ""),
                floors=p.get("floors", ""),
                rooms=rooms,
                layout_description=p.get("layout_description", ""),
                rationale=p.get("rationale", ""),
                estimated_cost=p.get("estimated_cost"),
            )
        )

    return PlanningOutput(
        plans=plans,
        summary=data.get("summary", ""),
    )
