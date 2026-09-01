"""ヒアリングAI — ユーザー要望の深掘り・構造化エージェント"""
from __future__ import annotations

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
import json

from app.schemas.requirements import RequirementBaseline, HearingOutput

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

## 既知情報・未定回答の扱い（重要）
- 「現在判明している情報」として提示された項目は、ユーザーが新しい情報を言わない限り値を維持し、再度質問しない
- ユーザーが「未定」「わからない」「まだ」「決めていない」等と回答した項目は、値をnullに戻さず、文字列 "未定" として記録する（＝聞いた上での未定回答も取得済みとして扱う）
- 同じ項目について、直前までに追質問済みであれば繰り返し聞かない

## 注意事項
- 必ず有効なJSONのみを返す（前後に余分なテキストは不要）
- 親しみやすく、分かりやすい日本語で質問する
- ユーザーが不安にならないよう、専門用語は避ける
"""

_FIELD_LABELS = {
    "family_structure": "家族構成",
    "budget": "予算",
    "land_info": "土地の有無・場所",
    "preferred_design": "好みのデザイン",
    "desired_size": "希望の広さ・部屋数",
    "lifestyle_flow": "重視する生活動線",
    "storage_needs": "収納の希望",
    "notes": "その他の要望",
}


# LLM は「読み取れない項目は null」という指示に反して、"未定" のような
# プレースホルダ文字列を入れてくることがある（実測で発生）。これを値として扱うと
# 「項目は埋まっているが中身が無い」状態のまま間取り生成に進んでしまうため、未取得に戻す。
_PLACEHOLDER_VALUES = {
    "未定", "未確認", "未指定", "不明", "なし", "特になし", "特にない",
    "null", "none", "n/a", "-", "―", "？", "?", "",
}


def _clean_value(value):
    """プレースホルダ文字列を None に正規化する"""
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if stripped.lower() in _PLACEHOLDER_VALUES:
        return None
    return stripped or None


def _format_known_requirements(known: RequirementBaseline | None) -> str:
    """既知の要件をプロンプト注入用のテキストに整形する"""
    if known is None:
        return ""
    lines = []
    for field, label in _FIELD_LABELS.items():
        value = getattr(known, field, None)
        if value is not None:
            lines.append(f"- {label}: {value}")
    return "\n".join(lines)


def run_hearing(
    conversation_history: list,
    llm: ChatOllama,
    known_requirements: RequirementBaseline | None = None,
) -> HearingOutput:
    """会話履歴からヒアリングAIを実行し、要件を構造化する"""
    system_content = HEARING_SYSTEM_PROMPT
    known_summary = _format_known_requirements(known_requirements)
    if known_summary:
        system_content += (
            "\n\n## 現在判明している情報（再度聞かないこと）\n" + known_summary
        )

    messages = [SystemMessage(content=system_content)] + conversation_history

    # JSON解析に失敗した場合は1回だけリトライする
    data = None
    for _attempt in range(2):
        response = llm.invoke(messages)
        raw_text = response.content
        try:
            # コードブロックの除去
            text = raw_text.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(lines[1:-1])
            data = json.loads(text)
            break
        except json.JSONDecodeError:
            data = None

    if data is None:
        # フォールバック: リトライしても解析できない場合
        return HearingOutput(
            requirements=RequirementBaseline(
                is_complete=False,
                missing_fields=["family_structure", "budget", "land_info", "desired_size"],
            ),
            follow_up_question="申し訳ありません。もう少し詳しく教えていただけますか？家族構成や予算、希望の場所などを教えてください。",
        )

    req_data = data.get("requirements", {})
    requirements = RequirementBaseline(
        family_structure=_clean_value(req_data.get("family_structure")),
        budget=_clean_value(req_data.get("budget")),
        land_info=_clean_value(req_data.get("land_info")),
        preferred_design=_clean_value(req_data.get("preferred_design")),
        desired_size=_clean_value(req_data.get("desired_size")),
        lifestyle_flow=_clean_value(req_data.get("lifestyle_flow")),
        storage_needs=_clean_value(req_data.get("storage_needs")),
        notes=_clean_value(req_data.get("notes")),
        is_complete=req_data.get("is_complete", False),
        missing_fields=req_data.get("missing_fields", []),
    )

    return HearingOutput(
        requirements=requirements,
        follow_up_question=data.get("follow_up_question"),
    )
