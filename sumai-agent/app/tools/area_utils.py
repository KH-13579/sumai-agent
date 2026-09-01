"""面積表記のパースと正規化（間取り生成AI・法規チェックAI の共通ユーティリティ）

LLM が出力する自由文の面積表記（「約110㎡（約33坪）」「30坪」「18畳」）を数値に変換する。
また、LLM が坪の数値を㎡欄に書いてしまう単位取り違えを決定論的に検出・補正する
（要件定義書 KH案 §9.2.2 の「再アライメント＝ルールベースでの崩れの補正」に相当）。
"""
from __future__ import annotations

import re
import unicodedata
from typing import Optional, Sequence

# 1坪 = 400/121 ㎡
TSUBO_TO_SQM = 3.305785

# 1畳 ≈ 1.62㎡（中京間）
JO_TO_SQM = 1.62

# 部屋面積の合計が延床面積を上回ることは構造上あり得ないが、
# 畳の換算幅を考慮して 5% の余裕を持たせる
ROOM_SUM_TOLERANCE = 1.05

# 戸建ての延床面積として妥当とみなす範囲（㎡）。単位補正の妥当性判断に使う
PLAUSIBLE_TOTAL_MIN = 40.0
PLAUSIBLE_TOTAL_MAX = 400.0

# 面積が読み取れない部屋が多い場合に、単位取り違えを疑う延床面積のしきい値（㎡）
_SUSPICIOUS_TOTAL_MAX = 50.0
_SUSPICIOUS_ROOM_COUNT = 4


def parse_area_sqm(text: Optional[str]) -> Optional[float]:
    """「約100㎡（約30坪）」「30坪」等の自由文から面積（㎡）を取り出す

    ㎡ 表記を優先し、無ければ坪表記を㎡へ換算する。単位が全く無い場合は㎡とみなす。
    数値が見つからない場合は None（＝判定材料不足）を返す。
    """
    if text is None:
        return None
    normalized = unicodedata.normalize("NFKC", str(text)).replace(",", "")

    sqm = re.search(r"(\d+(?:\.\d+)?)\s*(?:m2|m²|㎡|平米|平方メートル)", normalized, re.IGNORECASE)
    if sqm:
        return float(sqm.group(1))

    tsubo = re.search(r"(\d+(?:\.\d+)?)\s*坪", normalized)
    if tsubo:
        return round(float(tsubo.group(1)) * TSUBO_TO_SQM, 1)

    bare = re.search(r"(\d+(?:\.\d+)?)", normalized)
    if bare:
        return float(bare.group(1))
    return None


def parse_floor_count(text: Optional[str]) -> Optional[int]:
    """「2階建て」「平屋」等の自由文から階数を取り出す。読み取れない場合は None"""
    if text is None:
        return None
    normalized = unicodedata.normalize("NFKC", str(text))
    if "平屋" in normalized or "平家" in normalized:
        return 1
    match = re.search(r"(\d+)\s*階", normalized)
    if match:
        floors = int(match.group(1))
        # 「1階にLDK」のような部位の記述と区別できないため、常識的な範囲に丸める
        return floors if 1 <= floors <= 5 else None
    return None


def parse_room_area_sqm(area_text: Optional[str]) -> Optional[float]:
    """部屋の広さ表記から面積（㎡）を推定する

    「18畳」「6畳×2」「20㎡」に対応。「標準サイズ」「2箇所」のように面積として
    読み取れない表記は None を返す（合計から除外する）。
    """
    if area_text is None:
        return None
    normalized = unicodedata.normalize("NFKC", str(area_text)).replace(",", "")

    jo = re.search(r"(\d+(?:\.\d+)?)\s*畳", normalized)
    if jo:
        area = float(jo.group(1)) * JO_TO_SQM
        # 「6畳×2」のような部屋数の指定を反映する
        multiplier = re.search(r"畳\s*[×x*]\s*(\d+)", normalized)
        if multiplier:
            area *= int(multiplier.group(1))
        return round(area, 1)

    sqm = re.search(r"(\d+(?:\.\d+)?)\s*(?:m2|m²|㎡|平米)", normalized, re.IGNORECASE)
    if sqm:
        return float(sqm.group(1))
    return None


def sum_room_areas_sqm(area_texts: Sequence[Optional[str]]) -> float:
    """面積として読み取れた部屋の合計（㎡）。延床面積の下限の目安になる"""
    return round(sum(filter(None, (parse_room_area_sqm(t) for t in area_texts))), 1)


def normalize_total_floor_area(
    total_text: Optional[str], room_area_texts: Sequence[Optional[str]]
) -> tuple[Optional[str], Optional[str]]:
    """延床面積の単位取り違え（坪の数値を㎡欄に書く）を検出して補正する

    LLM は「35坪」という要望に対して `約35㎡（約10.5坪）` のように、坪の数値を
    そのまま㎡欄へ書いてしまうことがある（実機の qwen2.5:14b で再現）。
    この誤りは「部屋面積の合計が延床面積を超える」という形で必ず現れるため、
    数値を坪として解釈し直したときに整合し、かつ戸建てとして妥当な範囲に収まる場合のみ補正する。

    戻り値は (補正後の表記, 補正内容の説明)。補正しない場合は (元の表記, None)。
    """
    if not total_text:
        return total_text, None

    total = parse_area_sqm(total_text)
    if total is None or total <= 0:
        return total_text, None

    room_sum = sum_room_areas_sqm(room_area_texts)

    # 部屋面積が読み取れた場合は「合計が延床を超えているか」で判断する
    if room_sum > 0:
        if room_sum <= total * ROOM_SUM_TOLERANCE:
            return total_text, None
        suspicious = True
    else:
        # 部屋面積が読み取れない場合は、延床が極端に小さいことを手がかりにする
        suspicious = total < _SUSPICIOUS_TOTAL_MAX and len(room_area_texts) >= _SUSPICIOUS_ROOM_COUNT

    if not suspicious:
        return total_text, None

    # 数値を坪として解釈し直す
    corrected = round(total * TSUBO_TO_SQM, 1)
    if not (PLAUSIBLE_TOTAL_MIN <= corrected <= PLAUSIBLE_TOTAL_MAX):
        return total_text, None
    if room_sum > 0 and room_sum > corrected * ROOM_SUM_TOLERANCE:
        # 坪として解釈しても矛盾が解消しない場合は補正しない（法規チェック側で保留にする）
        return total_text, None

    note = (
        f"延床面積の単位を補正しました（「{total_text}」→ 約{corrected:.0f}㎡）。"
        f"部屋面積の合計 {room_sum:.1f}㎡ と矛盾していたため、{total:.0f} を坪として解釈しています。"
        if room_sum > 0
        else f"延床面積の単位を補正しました（「{total_text}」→ 約{corrected:.0f}㎡）。"
             f"戸建てとして小さすぎるため、{total:.0f} を坪として解釈しています。"
    )
    return f"約{corrected:.0f}㎡（約{total:.0f}坪）", note
