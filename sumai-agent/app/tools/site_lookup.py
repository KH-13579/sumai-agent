"""敷地照会ツール — 住所・自由文から敷地・法規制パラメータを取得する（LAW-1）

取得の優先順（上にあるものを優先し、欠けた項目を下位で補完する）:
  1. user_input … ユーザー発話から LLM が抽出した値
  2. reinfolib  … 不動産情報ライブラリ API（REINFOLIB_API_KEY 設定時のみ）
  3. preset     … デモ対象自治体のプリセット（要件定義書 §12「対象自治体を1つに限定」）
  4. assumed    … 標準的な敷地条件の仮定値（NFR-06 オフライン再生の担保）

外部 API が使えない環境でもデモを完走できるよう、API 呼び出しの失敗は例外を投げずに
下位のフォールバックへ落とす。どこまでが実データかは SiteInfo.source と assumptions に残す。
"""
from __future__ import annotations

import json
import os
from typing import Any, Optional

from app.schemas.legal import SiteInfo
from app.schemas.requirements import RequirementBaseline
from app.tools.zoning import get_rule, normalize_zoning

SITE_EXTRACTION_PROMPT = """あなたは住宅の敷地情報を整理する専門AIです。
ユーザーの住宅要件書から、法規チェックに必要な敷地情報だけを抽出します。

## 抽出する項目
- address: 所在地（都道府県・市区町村・地番など、書かれている範囲で）
- site_area_sqm: 敷地面積（数値。㎡単位に換算する。「50坪」なら 165.3）
- zoning: 用途地域（例: 第一種低層住居専用地域）
- road_width_m: 前面道路の幅員（数値。m単位）
- fire_zone: 防火地域 / 準防火地域 / 指定なし のいずれか
- has_land: 土地を所有または特定済みなら true、これから探すなら false

## 重要なルール
- 書かれていない項目は必ず null にする。推測で埋めてはいけない
- 「土地はまだ持っていない」「これから探す」→ has_land: false
- 「所有している」「相続した」「自宅を建て替え」→ has_land: true
- 建物の延床面積（希望の広さ）と敷地面積を混同しないこと。敷地面積は土地の広さのみ

## 出力フォーマット（必ずJSONのみを返す）
{
  "address": null,
  "site_area_sqm": null,
  "zoning": null,
  "road_width_m": null,
  "fire_zone": null,
  "has_land": false
}
"""

# デモ対象自治体のプリセット（要件定義書 §12：法規の全国対応はスコープ外、対象自治体を1つに限定）
# 実運用では不動産情報ライブラリ API から取得する値を、オフラインデモ用に固定化したもの。
PRESET_SITES: dict[str, dict[str, Any]] = {
    "さいたま市": {
        "zoning": "第一種低層住居専用地域",
        "building_coverage_ratio": 50.0,
        "floor_area_ratio": 100.0,
        "road_width_m": 4.0,
        "fire_zone": "準防火地域",
    },
    "川口市": {
        "zoning": "第一種住居地域",
        "building_coverage_ratio": 60.0,
        "floor_area_ratio": 200.0,
        "road_width_m": 6.0,
        "fire_zone": "準防火地域",
    },
}

# 土地が未確定の場合に用いる標準的な分譲地の想定（首都圏近郊の一般的な条件）
DEFAULT_ASSUMED_SITE: dict[str, Any] = {
    "site_area_sqm": 132.0,          # 約40坪
    "zoning": "第一種住居地域",
    "building_coverage_ratio": 60.0,
    "floor_area_ratio": 200.0,
    "road_width_m": 4.0,
    "fire_zone": None,
}


# ─────────────────────────────────────────
# 1. ユーザー発話からの抽出（LLM）
# ─────────────────────────────────────────

def _requirements_digest(requirements: RequirementBaseline) -> str:
    """敷地情報が含まれ得る要件項目だけをプロンプトへ渡す"""
    return f"""## 住宅要件書（敷地に関する記述）
- 土地情報: {requirements.land_info or "記載なし"}
- 希望の広さ・部屋数: {requirements.desired_size or "記載なし"}
- その他の要望: {requirements.notes or "記載なし"}

上記から敷地情報を抽出してJSONで返してください。
"""


def _coerce_float(value: Any) -> Optional[float]:
    """LLM が文字列で返した数値も受け取れるようにする"""
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result > 0 else None


def extract_site_from_text(requirements: RequirementBaseline, llm) -> dict[str, Any]:
    """LLM で要件書の自由文から敷地情報を構造化抽出する

    JSON 解析に失敗した場合は空 dict を返し、後続のフォールバックに委ねる。
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    messages = [
        SystemMessage(content=SITE_EXTRACTION_PROMPT),
        HumanMessage(content=_requirements_digest(requirements)),
    ]

    # ヒアリングAI・間取り生成AIと同様、解析失敗時は1回だけリトライする
    for _attempt in range(2):
        try:
            response = llm.invoke(messages)
        except Exception:
            return {}
        text = str(response.content).strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1])
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        return {
            "address": data.get("address") or None,
            "site_area_sqm": _coerce_float(data.get("site_area_sqm")),
            "zoning": normalize_zoning(data.get("zoning")),
            "road_width_m": _coerce_float(data.get("road_width_m")),
            "fire_zone": data.get("fire_zone") or None,
            "has_land": bool(data.get("has_land", False)),
        }
    return {}


# ─────────────────────────────────────────
# 2. 不動産情報ライブラリ API（LAW-1）
# ─────────────────────────────────────────

def fetch_from_reinfolib(address: str) -> Optional[dict[str, Any]]:
    """不動産情報ライブラリ API から用途地域・建ぺい率・容積率を照会する

    REINFOLIB_API_KEY が未設定の場合は None を返す（デモ既定はオフライン動作）。

    注意: 同 API の用途地域データはタイル配信（GeoJSON/PBF）であり、住所から点照会するには
    ジオコーディングと point-in-polygon が必要になる。エンドポイントとレスポンス項目名は
    技術検証（スパイク）で確定させる前提とし、ここでは環境変数で差し替えられる形にしている。
    想定外のレスポンスは None を返して下位のフォールバックへ落とす。
    """
    api_key = os.getenv("REINFOLIB_API_KEY", "").strip()
    if not api_key or not address:
        return None

    base_url = os.getenv("REINFOLIB_API_BASE", "https://www.reinfolib.mlit.go.jp/ex-api/external")
    endpoint = os.getenv("REINFOLIB_ZONING_ENDPOINT", "/XKT001")

    try:
        import httpx

        response = httpx.get(
            f"{base_url.rstrip('/')}{endpoint}",
            params={"address": address},
            headers={"Ocp-Apim-Subscription-Key": api_key},
            timeout=5.0,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception:
        # ネットワーク断・API 障害・仕様変更のいずれでもデモを止めない（NFR-06）
        return None

    return _map_reinfolib_payload(payload)


# API のレスポンス項目名の候補（仕様確定までの揺れを吸収する）
_REINFOLIB_FIELD_MAP: dict[str, tuple[str, ...]] = {
    "zoning": ("youto_chiiki", "use_district", "youto", "zoning"),
    "building_coverage_ratio": ("kenpeiritsu", "building_coverage_ratio", "bcr"),
    "floor_area_ratio": ("yousekiritsu", "floor_area_ratio", "far"),
    "fire_zone": ("bouka_chiiki", "fire_prevention_district", "fire_zone"),
}


def _map_reinfolib_payload(payload: Any) -> Optional[dict[str, Any]]:
    """API レスポンスから SiteInfo に相当する項目を取り出す"""
    record: Any = payload
    # {"features": [{"properties": {...}}]} / {"data": [{...}]} 形式に対応する
    if isinstance(record, dict):
        for key in ("features", "data", "items", "results"):
            value = record.get(key)
            if isinstance(value, list) and value:
                record = value[0]
                break
    if isinstance(record, dict) and isinstance(record.get("properties"), dict):
        record = record["properties"]
    if not isinstance(record, dict):
        return None

    result: dict[str, Any] = {}
    for target, candidates in _REINFOLIB_FIELD_MAP.items():
        for candidate in candidates:
            if candidate in record and record[candidate] not in (None, ""):
                result[target] = record[candidate]
                break

    if "zoning" in result:
        result["zoning"] = normalize_zoning(result["zoning"])
    for numeric in ("building_coverage_ratio", "floor_area_ratio"):
        if numeric in result:
            result[numeric] = _coerce_float(result[numeric])

    cleaned = {k: v for k, v in result.items() if v is not None}
    return cleaned or None


# ─────────────────────────────────────────
# 3. プリセット照会
# ─────────────────────────────────────────

def lookup_preset(address: Optional[str]) -> Optional[dict[str, Any]]:
    """住所文字列にデモ対象自治体名が含まれていればプリセット値を返す"""
    if not address:
        return None
    for municipality, values in PRESET_SITES.items():
        if municipality in address:
            return dict(values)
    return None


# ─────────────────────────────────────────
# 敷地照会の本体
# ─────────────────────────────────────────

# 「土地を持っていない」ことを示す語句（LLM 抽出が has_land を誤った場合の補正に使う）
_NO_LAND_KEYWORDS = ("土地なし", "土地はなし", "持っていな", "所有していな", "これから探", "探す予定", "土地未定")
_HAS_LAND_KEYWORDS = ("所有", "相続", "建て替え", "建替", "自宅の敷地", "土地あり", "土地は所有")


def _infer_has_land(requirements: RequirementBaseline, extracted: dict[str, Any]) -> bool:
    """土地を所有・特定済みかを推定する"""
    text = " ".join(filter(None, [requirements.land_info, requirements.notes]))
    if any(keyword in text for keyword in _NO_LAND_KEYWORDS):
        return False
    if any(keyword in text for keyword in _HAS_LAND_KEYWORDS):
        return True
    # 敷地面積が読み取れているなら特定済みとみなす
    if extracted.get("site_area_sqm"):
        return True
    return bool(extracted.get("has_land", False))


def lookup_site(requirements: RequirementBaseline, llm) -> SiteInfo:
    """住宅要件書から法規チェック用の敷地情報を組み立てる

    ユーザー入力 → 外部API → プリセット → 用途地域の法定既定値 → 仮定値 の順に補完し、
    仮定で埋めた内容はすべて assumptions に記録する（説明可能性の担保）。
    """
    extracted = extract_site_from_text(requirements, llm)
    has_land = _infer_has_land(requirements, extracted)

    values: dict[str, Any] = {
        "address": extracted.get("address"),
        "site_area_sqm": extracted.get("site_area_sqm"),
        "zoning": extracted.get("zoning"),
        "road_width_m": extracted.get("road_width_m"),
        "fire_zone": extracted.get("fire_zone"),
        "building_coverage_ratio": None,
        "floor_area_ratio": None,
    }
    assumptions: list[str] = []
    source = "user_input" if any(v is not None for v in values.values()) else "assumed"

    def fill(patch: dict[str, Any], label: str, new_source: Optional[str] = None) -> None:
        """未取得の項目のみを補完し、補完した項目を記録する"""
        filled = [key for key, value in patch.items() if key in values and values.get(key) is None and value is not None]
        if not filled:
            return
        for key in filled:
            values[key] = patch[key]
        if new_source:
            nonlocal source
            source = new_source
        assumptions.append(f"{label}: {', '.join(_FIELD_LABELS.get(k, k) for k in filled)}")

    # 2. 外部 API（キー未設定・障害時は None）
    api_values = fetch_from_reinfolib(values["address"] or "")
    if api_values:
        fill(api_values, "不動産情報ライブラリAPIから取得", new_source="reinfolib")

    # 3. デモ対象自治体のプリセット
    preset = lookup_preset(values["address"])
    if preset:
        fill(preset, "対象自治体プリセットから補完", new_source="preset" if source == "assumed" else None)

    # 4. 用途地域が判明していれば法定の一般的な指定値で建ぺい率・容積率を補完
    if values["zoning"]:
        rule = get_rule(values["zoning"])
        fill(
            {"building_coverage_ratio": rule.default_bcr, "floor_area_ratio": rule.default_far},
            f"{rule.name}の一般的な指定値を仮定",
        )

    # 5. 最終フォールバック（標準的な分譲地を想定）
    fill(DEFAULT_ASSUMED_SITE, "標準的な敷地条件を仮定")

    if not has_land:
        assumptions.insert(0, "土地が未確定のため、標準的な敷地条件による参考判定")

    return SiteInfo(
        address=values["address"],
        site_area_sqm=values["site_area_sqm"],
        zoning=values["zoning"],
        building_coverage_ratio=values["building_coverage_ratio"],
        floor_area_ratio=values["floor_area_ratio"],
        road_width_m=values["road_width_m"],
        fire_zone=values["fire_zone"],
        height_limit_m=get_rule(values["zoning"]).absolute_height_limit_m,
        has_land=has_land,
        source=source,  # type: ignore[arg-type]
        assumptions=assumptions,
    )


_FIELD_LABELS = {
    "address": "所在地",
    "site_area_sqm": "敷地面積",
    "zoning": "用途地域",
    "building_coverage_ratio": "指定建ぺい率",
    "floor_area_ratio": "指定容積率",
    "road_width_m": "前面道路幅員",
    "fire_zone": "防火地域",
}
