"""pytest — LLM 応答キャッシュのテスト（NFR-06 オフライン再生 / NFR-01 再現性）

Ollama を使わず、実機LLMの代わりに呼び出し回数を数えるスタブを差し込んで検証する。
「キャッシュ利用時に実機を1度も呼ばない」ことが確認できれば、オフラインでデモが成立する。
"""
from __future__ import annotations

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.tools.llm_cache import (
    CachedChatModel,
    LLMCacheMiss,
    cache_stats,
    compute_key,
)

MESSAGES = [
    SystemMessage(content="あなたは住宅AIコンシェルジュです。"),
    HumanMessage(content="さいたま市の土地に家を建てたいです。"),
]


class StubLLM:
    """実機LLMの代役。invoke の呼び出し回数を数える"""

    def __init__(self, content: str = '{"plans": []}') -> None:
        self.content = content
        self.calls = 0
        self.model_name = "stub-model"

    def invoke(self, messages, **kwargs):
        self.calls += 1
        return AIMessage(content=self.content)


def _wrap(inner, tmp_path, mode, json_mode=True):
    return CachedChatModel(
        inner, model="qwen2.5:14b", json_mode=json_mode, mode=mode, directory=tmp_path
    )


# ─── キーの性質 ────────────────────────────────

def test_key_is_stable_for_same_input():
    assert compute_key("qwen2.5:14b", True, MESSAGES) == compute_key("qwen2.5:14b", True, MESSAGES)


@pytest.mark.parametrize("model,json_mode,messages", [
    ("qwen2.5:7b", True, MESSAGES),                                   # モデル違い
    ("qwen2.5:14b", False, MESSAGES),                                 # JSONモード違い
    ("qwen2.5:14b", True, MESSAGES + [HumanMessage(content="追記")]),  # 会話履歴違い
])
def test_key_differs_when_input_differs(model, json_mode, messages):
    """モデル・モード・会話履歴のいずれかが変わればキーは変わる

    特にモデル名を含めるのが重要。14b で録った応答を 7b のデモで黙って
    返してしまう事故を防ぐ。
    """
    assert compute_key(model, json_mode, messages) != compute_key("qwen2.5:14b", True, MESSAGES)


# ─── モードごとの挙動 ───────────────────────────

def test_off_mode_always_calls_live_llm(tmp_path):
    stub = StubLLM()
    llm = _wrap(stub, tmp_path, "off")

    llm.invoke(MESSAGES)
    llm.invoke(MESSAGES)

    assert stub.calls == 2
    assert list(tmp_path.glob("*.json")) == []   # キャッシュも作らない


def test_auto_mode_records_then_replays(tmp_path):
    """auto は1回目に実機を呼んで保存し、2回目以降は実機を呼ばない"""
    stub = StubLLM(content='{"plans": ["保存された応答"]}')
    llm = _wrap(stub, tmp_path, "auto")

    first = llm.invoke(MESSAGES)
    assert stub.calls == 1
    assert len(list(tmp_path.glob("*.json"))) == 1

    second = llm.invoke(MESSAGES)
    assert stub.calls == 1, "キャッシュがあるのに実機を呼んでいる"
    assert second.content == first.content


def test_replay_mode_never_calls_live_llm(tmp_path):
    """replay はキャッシュのみで応答する（＝Ollama 停止中でもデモが成立する）"""
    recorder = _wrap(StubLLM(content="キャッシュ済み"), tmp_path, "record")
    recorder.invoke(MESSAGES)

    stub = StubLLM()
    player = _wrap(stub, tmp_path, "replay")
    result = player.invoke(MESSAGES)

    assert stub.calls == 0
    assert result.content == "キャッシュ済み"


def test_replay_mode_fails_loudly_on_miss(tmp_path):
    """キャッシュが無ければ黙って実機に落ちず、その場で失敗する（デモ前検証のため）"""
    stub = StubLLM()
    llm = _wrap(stub, tmp_path, "replay")

    with pytest.raises(LLMCacheMiss) as exc:
        llm.invoke(MESSAGES)

    assert stub.calls == 0
    assert "SUMAI_LLM_CACHE=auto" in str(exc.value)   # 復旧方法を示す


def test_record_mode_overwrites_existing(tmp_path):
    """record はキャッシュがあっても実機を呼び直して上書きする（作り直し用）"""
    _wrap(StubLLM(content="古い応答"), tmp_path, "record").invoke(MESSAGES)

    stub = StubLLM(content="新しい応答")
    _wrap(stub, tmp_path, "record").invoke(MESSAGES)
    assert stub.calls == 1

    replayed = _wrap(StubLLM(), tmp_path, "replay").invoke(MESSAGES)
    assert replayed.content == "新しい応答"


# ─── 壊れたキャッシュ・異常系 ────────────────────

def test_corrupted_recording_is_treated_as_miss(tmp_path):
    """壊れたキャッシュでデモを止めない（auto なら実機にフォールバックする）"""
    key = compute_key("qwen2.5:14b", True, MESSAGES)
    (tmp_path / f"{key[:16]}.json").write_text("{壊れたJSON", encoding="utf-8")

    stub = StubLLM(content="実機の応答")
    result = _wrap(stub, tmp_path, "auto").invoke(MESSAGES)

    assert stub.calls == 1
    assert result.content == "実機の応答"


def test_hash_collision_is_rejected(tmp_path):
    """短縮ハッシュのファイル名が衝突しても、完全一致キーで弾く"""
    key = compute_key("qwen2.5:14b", True, MESSAGES)
    (tmp_path / f"{key[:16]}.json").write_text(
        json.dumps({"key": "別のキー", "response": "他人の応答"}), encoding="utf-8"
    )

    stub = StubLLM(content="正しい応答")
    result = _wrap(stub, tmp_path, "auto").invoke(MESSAGES)

    assert stub.calls == 1
    assert result.content == "正しい応答"


def test_recording_contains_prompt_for_review(tmp_path):
    """キャッシュにはプロンプトも残す（何を保存したのか後から確認できるように）"""
    _wrap(StubLLM(content="応答本文"), tmp_path, "record").invoke(MESSAGES)

    data = json.loads(next(tmp_path.glob("*.json")).read_text(encoding="utf-8"))
    assert data["model"] == "qwen2.5:14b"
    assert data["json_mode"] is True
    assert data["response"] == "応答本文"
    assert [m["role"] for m in data["messages"]] == ["system", "human"]
    assert "さいたま市" in data["messages"][1]["content"]
    assert "recorded_at" in data and "elapsed_sec" in data


def test_unknown_attributes_delegate_to_inner(tmp_path):
    """invoke 以外は内側のモデルへ透過する"""
    stub = StubLLM()
    assert _wrap(stub, tmp_path, "auto").model_name == "stub-model"


# ─── 運用補助 ───────────────────────────────

def test_cache_stats_counts_and_sums(tmp_path):
    _wrap(StubLLM(content="a"), tmp_path, "record").invoke(MESSAGES)
    _wrap(StubLLM(content="b"), tmp_path, "record", json_mode=False).invoke(MESSAGES)

    stats = cache_stats(tmp_path)
    assert stats["count"] == 2
    assert stats["models"] == ["qwen2.5:14b"]
    assert stats["total_elapsed_sec"] >= 0.0


def test_cache_stats_on_missing_directory(tmp_path):
    stats = cache_stats(tmp_path / "存在しない")
    assert stats["count"] == 0 and stats["models"] == []


# ─── オーケストレーターへの組み込み ─────────────────

def test_get_llm_wraps_only_when_cache_enabled(monkeypatch):
    """SUMAI_LLM_CACHE=off（既定）では素の ChatOllama、有効時はラッパーを返す"""
    from app.agents.orchestrator import _get_llm

    monkeypatch.delenv("SUMAI_LLM_CACHE", raising=False)
    assert not isinstance(_get_llm(json_mode=True), CachedChatModel)

    monkeypatch.setenv("SUMAI_LLM_CACHE", "auto")
    wrapped = _get_llm(json_mode=True)
    assert isinstance(wrapped, CachedChatModel)

    # 未知の値は off として扱う（誤設定でデモが止まらないように）
    monkeypatch.setenv("SUMAI_LLM_CACHE", "typo")
    assert not isinstance(_get_llm(json_mode=True), CachedChatModel)


def test_temperature_is_applied_and_falls_back(monkeypatch):
    """SUMAI_TEMPERATURE が実際にLLMへ渡る（NFR-01 ばらつき抑制）"""
    from app.agents.orchestrator import _get_llm, _temperature

    monkeypatch.setenv("SUMAI_TEMPERATURE", "0.1")
    assert _temperature() == pytest.approx(0.1)
    assert _get_llm().temperature == pytest.approx(0.1)

    monkeypatch.setenv("SUMAI_TEMPERATURE", "数値でない")
    assert _temperature() == pytest.approx(0.3)
