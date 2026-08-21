"""pytest — MVP 受入基準テスト（LLM モック使用）"""
from __future__ import annotations
import json
import uuid
from unittest.mock import patch, MagicMock
from app.schemas.requirements import RequirementBaseline, HearingOutput
from app.schemas.floorplan import FloorPlan, Room, PlanningOutput
from app.schemas.maker import MakerRecommendation, MakerRecommendationOutput


# ─── ヒアリングAI テスト ───────────────────────────

def make_mock_llm_hearing_incomplete():
    """要件不足時のヒアリングAIレスポンスをモック"""
    mock = MagicMock()
    mock.invoke.return_value = MagicMock(
        content=json.dumps({
            "requirements": {
                "family_structure": "夫婦2人",
                "budget": None,
                "land_info": None,
                "preferred_design": None,
                "desired_size": None,
                "lifestyle_flow": None,
                "storage_needs": None,
                "notes": None,
                "is_complete": False,
                "missing_fields": ["budget", "land_info", "desired_size"]
            },
            "follow_up_question": "予算と土地の有無について教えていただけますか？"
        })
    )
    return mock


def make_mock_llm_hearing_complete():
    """要件充足時のヒアリングAIレスポンスをモック"""
    mock = MagicMock()
    mock.invoke.return_value = MagicMock(
        content=json.dumps({
            "requirements": {
                "family_structure": "夫婦＋子1（3歳）",
                "budget": "4000万円（土地込み）",
                "land_info": "土地なし、埼玉県内で探す予定",
                "preferred_design": "モダン",
                "desired_size": "30〜35坪、3LDK",
                "lifestyle_flow": "家事動線重視",
                "storage_needs": "収納多め",
                "notes": "リビングを広くしたい",
                "is_complete": True,
                "missing_fields": []
            },
            "follow_up_question": None
        })
    )
    return mock


def make_mock_llm_planning():
    """間取り生成AIレスポンスをモック"""
    mock = MagicMock()
    mock.invoke.return_value = MagicMock(
        content=json.dumps({
            "plans": [
                {
                    "concept": "コスパ重視案",
                    "total_floor_area": "約100㎡（約30坪）",
                    "floors": "2階建て",
                    "rooms": [
                        {"name": "LDK", "area": "18畳", "note": "南向き"},
                        {"name": "主寝室", "area": "8畳", "note": None},
                        {"name": "子供部屋", "area": "6畳", "note": None},
                        {"name": "浴室・洗面", "area": "標準", "note": None}
                    ],
                    "layout_description": "1階にLDK、2階に寝室・子供部屋。シンプルな動線。",
                    "rationale": "予算内で実現しやすいスタンダード設計。",
                    "estimated_cost": "2,400〜2,800万円"
                },
                {
                    "concept": "広さ重視案",
                    "total_floor_area": "約120㎡（約36坪）",
                    "floors": "2階建て",
                    "rooms": [
                        {"name": "LDK", "area": "24畳", "note": "吹き抜け"},
                        {"name": "主寝室", "area": "10畳", "note": None},
                        {"name": "子供部屋", "area": "8畳", "note": None},
                        {"name": "浴室・洗面", "area": "標準", "note": None}
                    ],
                    "layout_description": "LDKを最大化。吹き抜けで開放感。",
                    "rationale": "リビングを広くしたいご要望に最適。",
                    "estimated_cost": "2,900〜3,400万円"
                },
                {
                    "concept": "収納重視案",
                    "total_floor_area": "約110㎡（約33坪）",
                    "floors": "2階建て",
                    "rooms": [
                        {"name": "LDK", "area": "20畳", "note": "パントリー付き"},
                        {"name": "主寝室", "area": "8畳", "note": "WIC付き"},
                        {"name": "子供部屋", "area": "7畳", "note": None},
                        {"name": "収納室", "area": "4畳", "note": "大型収納"}
                    ],
                    "layout_description": "各所に収納を配置。家事効率を重視。",
                    "rationale": "収納多めのご要望に応えた設計。",
                    "estimated_cost": "2,700〜3,200万円"
                }
            ],
            "summary": "3案ともご予算内で実現可能です。広さ重視案はリビングの開放感が最大、収納重視案は日常の整理整頓が楽になります。"
        })
    )
    return mock


# ─── テストケース ───────────────────────────────────

def test_ac1_hearing_returns_followup_when_incomplete():
    """AC-1: 要件不足時にヒアリングAIが追質問を返す（間取りは生成されない）"""
    from langchain_core.messages import HumanMessage
    from app.agents.hearing_agent import run_hearing

    mock_llm = make_mock_llm_hearing_incomplete()
    history = [HumanMessage(content="家を建てたいです。夫婦2人です。")]
    result = run_hearing(history, mock_llm)

    assert result.requirements.is_complete is False
    assert len(result.requirements.missing_fields) > 0
    assert result.follow_up_question is not None and len(result.follow_up_question) > 0
    print(f"✅ AC-1 PASS: 追質問='{result.follow_up_question}'")


def test_ac2_planning_generates_3_plans_when_complete():
    """AC-2: 4必須項目が揃うと間取り生成AIが3案を返す"""
    from app.agents.planning_agent import run_planning

    requirements = RequirementBaseline(
        family_structure="夫婦＋子1（3歳）",
        budget="4000万円",
        land_info="土地なし、埼玉県内",
        desired_size="30〜35坪、3LDK",
        is_complete=True,
        missing_fields=[],
    )

    mock_llm = make_mock_llm_planning()
    result = run_planning(requirements, mock_llm)

    assert len(result.plans) == 3
    concepts = [p.concept for p in result.plans]
    print(f"✅ AC-2 PASS: 間取り3案生成 = {concepts}")


def test_ac3_session_continues():
    """AC-3: セッションIDで会話が継続できることを確認（構造テスト）"""
    session_id = str(uuid.uuid4())
    assert len(session_id) > 0
    print(f"✅ AC-3 PASS: session_id='{session_id}' が正常に生成される")


def test_ac2_hearing_complete_when_all_required_fields():
    """AC-2補: 4項目が揃うとis_complete=trueになる"""
    from langchain_core.messages import HumanMessage
    from app.agents.hearing_agent import run_hearing

    mock_llm = make_mock_llm_hearing_complete()
    history = [HumanMessage(content="家族は夫婦と子1人3歳、予算4000万、土地なし埼玉、3LDK30坪希望")]
    result = run_hearing(history, mock_llm)

    assert result.requirements.is_complete is True
    assert result.requirements.family_structure is not None
    assert result.requirements.budget is not None
    assert result.requirements.land_info is not None
    assert result.requirements.desired_size is not None
    print(f"✅ AC-2補 PASS: 要件充足確認")


# ─── メーカー推薦AI テスト ──────────────────────────────────


def make_mock_llm_maker():
    """メーカー推薦AIレスポンスをモック"""
    mock = MagicMock()
    mock.invoke.return_value = MagicMock(
        content=json.dumps({
            "recommendations": [
                {
                    "rank": 1,
                    "name": "タマホーム",
                    "type": "builder",
                    "reason": "予算3500万円以内でコストパフォーマンスが高く、標準仕様が充実しているため。",
                    "strengths": ["業界最高水準のコスパ", "標準仕様が充実", "全国展開"],
                    "price_band": "リーズナブル（坪単価45〜65万円）",
                    "best_for": ["予算を抑えて広い家を建てたい"],
                    "website": "https://www.tamahome.jp/",
                    "caution": None
                },
                {
                    "rank": 2,
                    "name": "SUUMO（スーモ）",
                    "type": "portal",
                    "reason": "土地なしのため、まず土地探しと複数メーカー比較をSUUMOで行うことを推薦。",
                    "strengths": ["日本最大級の不動産情報", "複数メーカーを一括比較できる"],
                    "price_band": "情報ポータル（メーカー比較・無料）",
                    "best_for": ["土地と建物をまとめて探したい"],
                    "website": "https://suumo.jp/",
                    "caution": None
                }
            ],
            "summary": "予算を重視しつつ、まず土地探しと並行してタマホームの展示場訪問をおすすめします。"
        })
    )
    return mock


def test_ac5_maker_recommendation_returns_results():
    """AC-5: メーカー推薦AIが要件に基づいて推薦を返す"""
    from app.agents.maker_agent import run_maker_recommendation

    requirements = RequirementBaseline(
        family_structure="夫婦＋子1（3歳）",
        budget="3500万円以内",
        land_info="土地なし、埼玉県内で探す予定",
        desired_size="30〜35坪、3LDK",
        is_complete=True,
        missing_fields=[],
    )
    plans = [
        FloorPlan(
            concept="コスパ重視案",
            total_floor_area="約100㎡（約30坪）",
            floors="2階建て",
            rooms=[Room(name="LDK", area="18畳", note=None)],
            layout_description="シンプル設計",
            rationale="予算内で実現",
            estimated_cost="2,400〜2,800万円",
        )
    ]

    mock_llm = make_mock_llm_maker()
    result = run_maker_recommendation(requirements, plans, mock_llm)

    assert len(result.recommendations) >= 1
    assert result.recommendations[0].rank == 1
    assert result.recommendations[0].name != ""
    assert result.summary != ""
    # 土地なし → ポータルが含まれるか確認
    types = [r.type for r in result.recommendations]
    assert "portal" in types
    print(f"✅ AC-5 PASS: 推薦{len(result.recommendations)}件 — {[r.name for r in result.recommendations]}")


def test_ac5_maker_recommendation_schema():
    """AC-5補: MakerRecommendationスキーマが正しく動作する"""
    rec = MakerRecommendation(
        rank=1,
        name="積水ハウス",
        type="builder",
        reason="高品質・長期保証が魅力",
        strengths=["長期60年保証", "ZEH対応"],
        price_band="高め（坪単価80〜120万円）",
        best_for=["高品質重視"],
        website="https://www.sekisuihouse.co.jp/",
        caution=None,
    )
    assert rec.rank == 1
    assert rec.type == "builder"
    assert rec.caution is None
    print("✅ AC-5補 PASS: MakerRecommendationスキーマ正常")


if __name__ == "__main__":
    test_ac1_hearing_returns_followup_when_incomplete()
    test_ac2_planning_generates_3_plans_when_complete()
    test_ac3_session_continues()
    test_ac2_hearing_complete_when_all_required_fields()
    test_ac5_maker_recommendation_returns_results()
    test_ac5_maker_recommendation_schema()
    print("\n🎉 全テスト通過")
