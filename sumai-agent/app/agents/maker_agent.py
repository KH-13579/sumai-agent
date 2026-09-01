"""メーカー推薦AI — 住宅要件・間取り案からハウスメーカーを推薦するエージェント"""
from __future__ import annotations

import json
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

from app.schemas.requirements import RequirementBaseline
from app.schemas.floorplan import FloorPlan
from app.schemas.maker import MakerRecommendation, MakerRecommendationOutput
from app.data.demo_data import DEMO_MAKERS

MAKER_RECOMMENDATION_PROMPT = """あなたは住宅メーカー選定の専門AIコンサルタントです。
ユーザーの住宅要件・間取りプランをもとに、最適なハウスメーカー・不動産ポータルを最大3件推薦します。

## 推薦の優先基準（重要度順）
1. **予算（価格帯）との整合** — 坪単価がユーザーの総予算・建物予算に収まるか
2. **工法・構造の要件** — 木造希望 / 耐震重視 / バリアフリーなど
3. **土地状況** — 土地なし（ポータルも併用）/ 狭小地 / 広い土地
4. **ライフスタイルの優先事項** — 大収納希望→ミサワホーム、スマートホーム→パナソニックなど
5. **家族構成・将来像** — 子育て・バリアフリー・二世帯など

## 出力フォーマット（必ずJSONのみで返す）
{
  "recommendations": [
    {
      "rank": 1,
      "name": "メーカー名",
      "type": "builder",
      "reason": "このユーザーにこのメーカーを推薦する具体的な理由（要件と照合して2〜3文）",
      "strengths": ["強み1", "強み2"],
      "price_band": "価格帯の説明",
      "best_for": ["向いているニーズ1", "向いているニーズ2"],
      "website": "https://...",
      "caution": "注意点・デメリット（なければnull）"
    }
  ],
  "summary": "3件の推薦全体のまとめと、次のアクション（来場予約・カタログ請求）の提案（150字程度）"
}

## 注意事項
- 必ず有効なJSONのみを返す（前後に余分なテキストは不要）
- recommendationsは1〜3件（状況に応じて最適な件数を選ぶ）
- 土地がない場合は必ずSUUMOかLIFULL HOME'S等のポータルを1件含める
- 理由は「なぜこのユーザーに合うか」を具体的に述べる（「高品質だから」だけはNG）
- 予算が低い場合は無理にプレミアムメーカーを推薦しない
"""


def _format_maker_catalog() -> str:
    """メーカーデータをLLMに渡すテキスト形式に整形する"""
    lines = ["## 選定可能なメーカー・ポータル一覧\n"]
    for m in DEMO_MAKERS:
        m_type = "注文住宅メーカー" if m["type"] == "builder" else "不動産情報ポータル"
        lines.append(f"### {m['name']}（{m_type}）")
        if m.get("construction_method"):
            lines.append(f"- 工法: {m['construction_method']}")
        lines.append(f"- 価格帯: {m['price_band']}")
        lines.append(f"- 強み: {', '.join(m['strengths'])}")
        lines.append(f"- こんな人向け: {', '.join(m['best_for'])}")
        lines.append(f"- 公式サイト: {m['website']}")
        lines.append("")
    return "\n".join(lines)


def _format_requirements(req: RequirementBaseline) -> str:
    """要件書をLLM用テキストに整形"""
    return f"""## ユーザーの住宅要件
- 家族構成: {req.family_structure or "不明"}
- 予算: {req.budget or "不明"}
- 土地: {req.land_info or "不明"}
- 希望の広さ・部屋数: {req.desired_size or "不明"}
- 好みのデザイン: {req.preferred_design or "未指定"}
- 重視する生活動線: {req.lifestyle_flow or "未指定"}
- 収納の希望: {req.storage_needs or "未指定"}
- その他の要望: {req.notes or "なし"}"""


def _format_plans(plans: list[FloorPlan]) -> str:
    """間取り案のサマリーをLLM用テキストに整形"""
    if not plans:
        return ""
    lines = ["\n## 提案済み間取り案のサマリー"]
    for i, p in enumerate(plans, 1):
        lines.append(f"- 案{i}（{p.concept}）: {p.total_floor_area}・{p.floors}・概算{p.estimated_cost or '要確認'}")
    return "\n".join(lines)


def run_maker_recommendation(
    requirements: RequirementBaseline,
    plans: list[FloorPlan],
    llm: ChatOllama,
) -> MakerRecommendationOutput:
    """住宅要件・間取り案からメーカーを推薦する"""
    maker_catalog = _format_maker_catalog()
    req_text = _format_requirements(requirements)
    plans_text = _format_plans(plans)

    user_content = f"""{maker_catalog}

{req_text}{plans_text}

上記の要件と間取り案をもとに、最適なハウスメーカー・情報ポータルを最大3件推薦してください。
"""

    messages = [
        SystemMessage(content=MAKER_RECOMMENDATION_PROMPT),
        HumanMessage(content=user_content),
    ]

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
        # フォールバック: 静的データから上位3件を返す
        fallback = [
            MakerRecommendation(
                rank=i + 1,
                name=m["name"],
                type=m["type"],
                reason="要件との詳細マッチングが行えませんでした。参考としてご確認ください。",
                strengths=m["strengths"],
                price_band=m["price_band"],
                best_for=m["best_for"],
                website=m["website"],
                caution=None,
            )
            for i, m in enumerate(DEMO_MAKERS[:3])
        ]
        return MakerRecommendationOutput(
            recommendations=fallback,
            summary="AIによる詳細推薦の生成に失敗しました。上記は参考として表示しています。",
        )

    recs = []
    for r in data.get("recommendations", []):
        recs.append(
            MakerRecommendation(
                rank=r.get("rank", len(recs) + 1),
                name=r.get("name", ""),
                type=r.get("type", "builder"),
                reason=r.get("reason", ""),
                strengths=r.get("strengths", []),
                price_band=r.get("price_band", ""),
                best_for=r.get("best_for", []),
                website=r.get("website", ""),
                caution=r.get("caution"),
            )
        )

    return MakerRecommendationOutput(
        recommendations=recs,
        summary=data.get("summary", ""),
    )
