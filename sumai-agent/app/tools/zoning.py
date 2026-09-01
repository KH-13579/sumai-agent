"""用途地域マスタ（建築基準法の法定値を静的テーブル化）

要件定義書 v1.0 §9.1 では用途地域・建ぺい率・容積率は不動産情報ライブラリ API から
取得する方針だが、指定値が取得できない場合の既定値および法定の制限（絶対高さ・斜線・
日影・外壁後退の適用有無）は本テーブルで保持する。

参照条文:
- 第52条（容積率／前面道路幅員による制限）
- 第53条（建ぺい率）
- 第54条（外壁の後退距離）
- 第55条（低層住居専用地域等内における建築物の高さの限度）
- 第56条（道路斜線・北側斜線）／第56条の2（日影規制・別表第四）
- 別表第二（用途制限）
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ZoningRule:
    """用途地域ごとの法規パラメータ"""
    name: str
    default_bcr: float                          # 指定がない場合に仮定する建ぺい率（%）
    default_far: float                          # 指定がない場合に仮定する容積率（%）
    far_road_coefficient: float                 # 前面道路幅員に掛ける法定係数（第52条2項）
    absolute_height_limit_m: Optional[float]    # 絶対高さ制限（第55条）。該当なしは None
    has_north_slope: bool                       # 北側斜線制限（第56条1項3号）
    has_shadow_regulation: bool                 # 日影規制（第56条の2・別表第四）
    has_wall_setback: bool                      # 外壁後退距離の定め（第54条）
    residential_allowed: bool = True            # 住宅の建築可否（別表第二）
    note: str = ""


# 住居系は道路幅員係数 4/10、その他は 6/10（第52条2項）。
# 特定行政庁の指定により住居系でも 6/10 となる区域があるが、安全側（厳しい側）に 4/10 を用いる。
_RESIDENTIAL_COEF = 0.4
_OTHER_COEF = 0.6

ZONING_RULES: dict[str, ZoningRule] = {
    "第一種低層住居専用地域": ZoningRule(
        "第一種低層住居専用地域", 50, 100, _RESIDENTIAL_COEF, 10.0, True, True, True,
        note="低層住宅専用。絶対高さ10m（都市計画により12m）・北側斜線・外壁後退の定めあり",
    ),
    "第二種低層住居専用地域": ZoningRule(
        "第二種低層住居専用地域", 50, 100, _RESIDENTIAL_COEF, 10.0, True, True, True,
        note="低層住宅中心。絶対高さ10m（都市計画により12m）・北側斜線・外壁後退の定めあり",
    ),
    "田園住居地域": ZoningRule(
        "田園住居地域", 50, 100, _RESIDENTIAL_COEF, 10.0, True, True, True,
        note="農地と低層住宅の混在。低層住居専用地域と同等の制限",
    ),
    "第一種中高層住居専用地域": ZoningRule(
        "第一種中高層住居専用地域", 60, 200, _RESIDENTIAL_COEF, None, True, True, False,
        note="中高層住宅専用。絶対高さ制限はないが北側斜線・日影規制あり",
    ),
    "第二種中高層住居専用地域": ZoningRule(
        "第二種中高層住居専用地域", 60, 200, _RESIDENTIAL_COEF, None, True, True, False,
        note="中高層住宅中心。北側斜線・日影規制あり",
    ),
    "第一種住居地域": ZoningRule(
        "第一種住居地域", 60, 200, _RESIDENTIAL_COEF, None, False, True, False,
        note="住居環境保護。北側斜線は適用外だが日影規制あり",
    ),
    "第二種住居地域": ZoningRule(
        "第二種住居地域", 60, 200, _RESIDENTIAL_COEF, None, False, True, False,
    ),
    "準住居地域": ZoningRule(
        "準住居地域", 60, 200, _RESIDENTIAL_COEF, None, False, True, False,
        note="道路沿道の業務と住居の調和",
    ),
    "近隣商業地域": ZoningRule(
        "近隣商業地域", 80, 200, _OTHER_COEF, None, False, True, False,
    ),
    "商業地域": ZoningRule(
        "商業地域", 80, 400, _OTHER_COEF, None, False, False, False,
    ),
    "準工業地域": ZoningRule(
        "準工業地域", 60, 200, _OTHER_COEF, None, False, True, False,
    ),
    "工業地域": ZoningRule(
        "工業地域", 60, 200, _OTHER_COEF, None, False, False, False,
        note="住宅は建築可能だが住環境の保護は図られない",
    ),
    "工業専用地域": ZoningRule(
        "工業専用地域", 60, 200, _OTHER_COEF, None, False, False, False,
        residential_allowed=False,
        note="住宅は建築できない（別表第二）",
    ),
}

# 用途地域が特定できない場合に用いる安全側の既定値。
# 住居系の一般的な指定（建ぺい60%／容積200%）を仮定し、仮定であることを必ず明示する。
UNKNOWN_ZONING = ZoningRule(
    "用途地域不明", 60, 200, _RESIDENTIAL_COEF, None, False, True, False,
    note="用途地域が特定できないため住居系の一般的な指定を仮定",
)

# ユーザー表記のゆれを正規名称へ寄せるための別名表
_ALIASES: dict[str, str] = {
    "一低層": "第一種低層住居専用地域",
    "二低層": "第二種低層住居専用地域",
    "低層住居専用地域": "第一種低層住居専用地域",
    "一中高": "第一種中高層住居専用地域",
    "二中高": "第二種中高層住居専用地域",
    "中高層住居専用地域": "第一種中高層住居専用地域",
    "住居地域": "第一種住居地域",
    "商業": "商業地域",
    "近商": "近隣商業地域",
    "準工": "準工業地域",
}

def normalize_zoning(text: Optional[str]) -> Optional[str]:
    """自由文中の用途地域表記を ZONING_RULES のキーに正規化する

    「第1種低層住居専用地域」「一低層」等の表記ゆれを吸収する。
    判定できない場合は None を返す（推測で埋めない）。
    """
    if not text:
        return None
    # 全角英数を半角に寄せ、記号・空白を除去してから照合する
    normalized = unicodedata.normalize("NFKC", str(text))
    normalized = re.sub(r"[\s　・（）()]", "", normalized)
    # 「第1種」→「第一種」
    for arabic, kanji in (("1", "一"), ("2", "二"), ("3", "三")):
        normalized = normalized.replace(f"第{arabic}種", f"第{kanji}種")

    for name in ZONING_RULES:
        if name in normalized:
            return name
    # 「第一種低層住居専用」のように「地域」が欠けている表記
    for name in ZONING_RULES:
        if name.replace("地域", "") in normalized:
            return name
    for alias, name in _ALIASES.items():
        if alias in normalized:
            return name
    return None


def get_rule(zoning: Optional[str]) -> ZoningRule:
    """用途地域名から法規パラメータを取得する（不明時は UNKNOWN_ZONING）"""
    name = normalize_zoning(zoning)
    if name is None:
        return UNKNOWN_ZONING
    return ZONING_RULES[name]
