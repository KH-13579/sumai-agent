"""LLM 応答のキャッシュ（要件定義書 NFR-06 オフライン再生 / NFR-01 デモ再現性）

CPU 推論では間取り生成1回に十数分〜数十分かかるため、デモをそのまま実演すると
待ち時間が成立しない。そこで LLM の応答を一度だけ実機で取得してファイルに保存し、
2回目以降はそのキャッシュを返す。ネットワーク断・Ollama 停止でもデモが完走し、
かつ同じ入力に対して常に同じ応答になる（＝再現性が保証される）。

動作モードは環境変数 SUMAI_LLM_CACHE で切り替える:

    off     … 何もしない（既定）。常に実機のLLMを呼ぶ
    auto    … キャッシュがあれば返し、無ければ実機を呼んで保存する（推奨）
    replay  … キャッシュのみを使う。無ければ例外（Ollama 不要でデモできることの検証用）
    record  … 常に実機を呼び、キャッシュを上書きする（作り直し）

キャッシュキーはモデル名・JSONモード・全メッセージ列のハッシュ。会話履歴ごと
含めるため、同じ手順を踏めば必ず命中し、手順が変われば命中しない。
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional, Sequence

from langchain_core.messages import AIMessage, BaseMessage

# キャッシュファイルの置き場所（リポジトリ直下 fixtures/llm/）
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "fixtures" / "llm"

VALID_MODES = ("off", "auto", "replay", "record")


class LLMCacheMiss(RuntimeError):
    """replay モードでキャッシュが見つからなかった場合に送出する"""


def cache_mode() -> str:
    """環境変数から動作モードを読む。未知の値は off として扱う"""
    mode = os.getenv("SUMAI_LLM_CACHE", "off").strip().lower()
    return mode if mode in VALID_MODES else "off"


def cache_dir() -> Path:
    """キャッシュファイルの置き場所。SUMAI_LLM_CACHE_DIR で差し替え可能"""
    override = os.getenv("SUMAI_LLM_CACHE_DIR", "").strip()
    return Path(override) if override else DEFAULT_CACHE_DIR


def _serialize_messages(messages: Sequence[BaseMessage]) -> List[dict]:
    """キー計算とキャッシュの可読性のためにメッセージ列を素の dict にする"""
    return [
        {"role": getattr(m, "type", m.__class__.__name__), "content": str(m.content)}
        for m in messages
    ]


def compute_key(model: str, json_mode: bool, messages: Sequence[BaseMessage]) -> str:
    """キャッシュを一意に特定するキー

    モデル名を含めるため、モデルを変更するとキャッシュは命中しなくなる（＝作り直しが必要）。
    別モデルの応答を黙って返してしまう事故を防ぐための意図的な設計。
    """
    payload = {
        "model": model,
        "json_mode": json_mode,
        "messages": _serialize_messages(messages),
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class CachedChatModel:
    """ChatOllama を包んで invoke() の応答をキャッシュする

    ノード側が使っているのは invoke(messages) -> .content を持つオブジェクト、
    という最小の約束だけなので、その1点だけを代理する。
    未知の属性は内側のモデルへ委譲するため、既存コードからは透過的に見える。
    """

    def __init__(
        self,
        inner: Any,
        model: str,
        json_mode: bool = False,
        mode: Optional[str] = None,
        directory: Optional[Path] = None,
    ) -> None:
        self._inner = inner
        self._model = model
        self._json_mode = json_mode
        self._mode = mode if mode is not None else cache_mode()
        self._dir = directory if directory is not None else cache_dir()

    # ─── キャッシュファイルの読み書き ──────────────

    def _path(self, key: str) -> Path:
        return self._dir / f"{key[:16]}.json"

    def _load(self, key: str) -> Optional[str]:
        path = self._path(key)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            # 壊れたキャッシュは無いものとして扱う（デモを止めない）
            return None
        # ファイル名は短縮ハッシュなので、完全一致を必ず確認する
        if data.get("key") != key:
            return None
        response = data.get("response")
        return response if isinstance(response, str) else None

    def _save(self, key: str, messages: Sequence[BaseMessage], response: str, elapsed: float) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        record = {
            "key": key,
            "model": self._model,
            "json_mode": self._json_mode,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_sec": round(elapsed, 1),
            # プロンプトも残す。何を録ったのか後から確認・レビューできるようにする
            "messages": _serialize_messages(messages),
            "response": response,
        }
        try:
            self._path(key).write_text(
                json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except OSError:
            # 保存に失敗しても応答自体は返せるため、デモは続行させる
            pass

    # ─── 代理する唯一のメソッド ──────────────────

    def invoke(self, messages: Sequence[BaseMessage], **kwargs: Any) -> AIMessage:
        if self._mode == "off":
            return self._inner.invoke(messages, **kwargs)

        key = compute_key(self._model, self._json_mode, messages)

        if self._mode in ("auto", "replay"):
            cached = self._load(key)
            if cached is not None:
                return AIMessage(content=cached)
            if self._mode == "replay":
                raise LLMCacheMiss(
                    f"キャッシュが見つかりません（key={key[:16]}, model={self._model}）。"
                    "SUMAI_LLM_CACHE=auto で一度実行してキャッシュを作成してください。"
                )

        started = time.monotonic()
        response = self._inner.invoke(messages, **kwargs)
        elapsed = time.monotonic() - started

        content = str(response.content)
        if self._mode in ("auto", "record"):
            self._save(key, messages, content, elapsed)
        return response

    def __getattr__(self, name: str) -> Any:
        """invoke 以外は内側の ChatOllama へ委譲する"""
        return getattr(self._inner, name)


# ─── 運用補助 ───────────────────────────────

def cache_stats(directory: Optional[Path] = None) -> dict:
    """キャッシュの件数と、キャッシュによって短縮された累計時間を返す（不足の確認用）"""
    target = directory if directory is not None else cache_dir()
    if not target.exists():
        return {"count": 0, "total_elapsed_sec": 0.0, "models": [], "directory": str(target)}

    count = 0
    total = 0.0
    models: set[str] = set()
    for path in sorted(target.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        count += 1
        total += float(data.get("elapsed_sec") or 0.0)
        if data.get("model"):
            models.add(str(data["model"]))

    return {
        "count": count,
        "total_elapsed_sec": round(total, 1),
        "models": sorted(models),
        "directory": str(target),
    }
