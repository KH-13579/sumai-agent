"""ヒアリングAI — ユーザー要望の深掘り・構造化エージェント"""
from __future__ import annotations

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
import json

from app.schemas.models import RequirementBaseline, HearingOutput

HEARING_SYSTEM_PROMPT = """あなたは住宅の専門的なヒアリングAIです。
ユーザーの住宅購入・建設の要望を丁寧に深掘りし、設計に必要な要件を構造化します。

## あなたの役割
- ユーザーの発言から住宅要件を構造化する
- 不足している重要情報を1〜2項目に絞って親しみやすく追質問する
- 読み取れない項目は推測で埋めず、nullのままにする

## 間取り生成に必要な4必須項目
1. family_structure（家族構成）
2. budget（予算）
3. land_info（土地の有無・場所）
4. desired_size（希望の広さ・部屋数）

この4項目が揃った場合のみ is_complete = true とする。

## 出力フォーマット（必ずJSON形式で返す）
{
  "requirements": {
    "family_structure": "（取得できた情報 or null）",
    "budget": "（取得できた情報 or null）",
    "land_info": "（取得できた情報 or null）",
    "preferred_design": "（取得できた情報 or null）",
    "desired_size": "（取得できた情報 or null）",
    "lifestyle_flow": "（取得できた情報 or null）",
    "storage_needs": "（取得できた情報 or null）",
    "notes": "（取得できた情報 or null）",
    "is_complete": false,
    "missing_fields": ["不足項目1", "不足項目2"]
  },
  "follow_up_question": "（is_complete=falseの場合のみ。1〜2項目に絞った追質問文。is_complete=trueの場合はnull）"
}

## 注意事項
- 必ず有効なJSONのみを返す（前後に余分なテキストは不要）
- 親しみやすく、分かりやすい日本語で質問する
- ユーザーが不安にならないよう、専門用語は避ける
"""


def run_hearing(conversation_history: list, llm: ChatOllama) -> HearingOutput:
    """会話履歴からヒアリングAIを実行し、要件を構造化する"""
    messages = [SystemMessage(content=HEARING_SYSTEM_PROMPT)] + conversation_history

    response = llm.invoke(messages)
    raw_text = response.content

    # JSON抽出
    try:
        # コードブロックの除去
        text = raw_text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1])
        data = json.loads(text)
    except json.JSONDecodeError:
        # フォールバック: 不完全なレスポンスへの対応
        return HearingOutput(
            requirements=RequirementBaseline(
                is_complete=False,
                missing_fields=["family_structure", "budget", "land_info", "desired_size"],
            ),
            follow_up_question="申し訳ありません。もう少し詳しく教えていただけますか？家族構成や予算、希望の場所などを教えてください。",
        )

    req_data = data.get("requirements", {})
    requirements = RequirementBaseline(
        family_structure=req_data.get("family_structure"),
        budget=req_data.get("budget"),
        land_info=req_data.get("land_info"),
        preferred_design=req_data.get("preferred_design"),
        desired_size=req_data.get("desired_size"),
        lifestyle_flow=req_data.get("lifestyle_flow"),
        storage_needs=req_data.get("storage_needs"),
        notes=req_data.get("notes"),
        is_complete=req_data.get("is_complete", False),
        missing_fields=req_data.get("missing_fields", []),
    )

    return HearingOutput(
        requirements=requirements,
        follow_up_question=data.get("follow_up_question"),
    )
