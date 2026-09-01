"""デモシナリオの LLM 応答をキャッシュに保存する（NFR-06 オフライン再生）

ブラウザに張り付いて数十分待つ代わりに、このスクリプトを一度流しておけば
デモ当日はキャッシュから即座に応答が返る。所要時間も計測して出すため、
モデル比較（既定の qwen2.5:3b で足りるか、デモは 7b / 14b にするか）のベンチにも使える。

使い方:

    # 全ペルソナ分を作成（初回は時間がかかる。裏で流しておく）
    ./venv/Scripts/python.exe scripts/record_demo.py

    # ペルソナCだけ作成（法規チェックの見せ場）
    ./venv/Scripts/python.exe scripts/record_demo.py --persona C

    # モデルを指定して計測（.env を書き換えずに比較できる）
    ./venv/Scripts/python.exe scripts/record_demo.py --persona C --model qwen2.5:14b

    # キャッシュだけでデモが完走するか検証する（Ollama を止めてから実行）
    ./venv/Scripts/python.exe scripts/record_demo.py --persona C --verify

--verify は SUMAI_LLM_CACHE=replay で走らせるため、キャッシュが1つでも欠けていれば
その場で失敗する。デモ前の最終確認はこれで行う。
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# リポジトリ直下を import パスに追加（どこから起動しても app を解決できるように）
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv

load_dotenv(REPO_ROOT / ".env")

# ヒアリングが未完のときに送る定型返答。キャッシュを命中させるため、デモ当日も
# この文言をそのまま使う必要がある（別の言い方をすると命中しない）。
PROCEED_MESSAGE = "提案に移ってください。"

# 1セッションあたりの最大ターン数（MAX_HEARING_TURNS + 提案ターンの余裕）
MAX_TURNS = 5


def _personas(selected: str | None) -> list[dict]:
    from app.data.demo_data import DEMO_PRESETS

    if not selected:
        return DEMO_PRESETS
    # "C" / "ペルソナC" / "c" のいずれでも指定できるようにする
    needle = selected.strip().upper().replace("ペルソナ", "")
    matched = [p for p in DEMO_PRESETS if needle in p["name"].upper()]
    if not matched:
        names = " / ".join(p["name"] for p in DEMO_PRESETS)
        raise SystemExit(f"ペルソナ '{selected}' が見つかりません。指定可能: {names}")
    return matched


def _run_session(persona: dict, session_id: str) -> tuple[bool, list[tuple[str, float]]]:
    """1ペルソナ分の会話を完走させ、(完走したか, [(送信内容, 所要秒)]) を返す"""
    from langchain_core.messages import HumanMessage

    from app.agents.orchestrator import build_graph

    graph, _ = build_graph()
    config = {"configurable": {"thread_id": session_id}}

    timings: list[tuple[str, float]] = []
    message = persona["preset_message"]

    for turn in range(1, MAX_TURNS + 1):
        label = "プリセット発話" if turn == 1 else PROCEED_MESSAGE
        print(f"  [ターン{turn}] {label[:28]}… ", end="", flush=True)

        started = time.monotonic()
        result = graph.invoke({"messages": [HumanMessage(content=message)]}, config=config)
        elapsed = time.monotonic() - started
        timings.append((label, elapsed))

        stage = result.get("stage", "?")
        plans = result.get("floor_plans") or []
        checks = result.get("legal_checks") or []
        print(f"{elapsed:7.1f}秒  stage={stage} 間取り={len(plans)}案 法規={len(checks)}件")

        if result.get("done") and plans:
            return True, timings
        message = PROCEED_MESSAGE

    return False, timings


def main() -> int:
    parser = argparse.ArgumentParser(description="デモシナリオの LLM 応答をキャッシュに保存する")
    parser.add_argument("--persona", help="対象ペルソナ（A / B / C）。省略時は全件")
    parser.add_argument("--model", help="使用モデル（省略時は .env の SUMAI_MODEL）")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="キャッシュのみで完走するか検証する（SUMAI_LLM_CACHE=replay。Ollama を停止して実行）",
    )
    args = parser.parse_args()

    if args.model:
        os.environ["SUMAI_MODEL"] = args.model
    os.environ["SUMAI_LLM_CACHE"] = "replay" if args.verify else "auto"

    from app.tools.llm_cache import LLMCacheMiss, cache_stats

    model = os.getenv("SUMAI_MODEL", "qwen2.5:3b")
    mode = os.environ["SUMAI_LLM_CACHE"]
    before = cache_stats()

    print("=" * 68)
    print(f"{'キャッシュの検証' if args.verify else 'キャッシュ作成'}  モデル={model}  モード={mode}")
    print(f"保存先: {before['directory']}（既存 {before['count']} 件）")
    print("=" * 68)

    personas = _personas(args.persona)
    results: list[tuple[str, bool, float]] = []

    for persona in personas:
        print(f"\n▼ {persona['name']}")
        # セッションIDにモデル名を含め、モデルを変えたときに前回の状態を引き継がないようにする
        session_id = f"record-{model}-{persona['name']}"
        try:
            completed, timings = _run_session(persona, session_id)
        except LLMCacheMiss as e:
            print(f"\n  ❌ キャッシュが不足しています: {e}")
            results.append((persona["name"], False, 0.0))
            continue

        total = sum(t for _, t in timings)
        results.append((persona["name"], completed, total))
        print(f"  合計 {total:.1f}秒（{total / 60:.1f}分）  完走={'OK' if completed else 'NG'}")

    after = cache_stats()
    print("\n" + "=" * 68)
    print("結果")
    print("=" * 68)
    for name, completed, total in results:
        print(f"  {'✅' if completed else '❌'} {name:<28} {total:7.1f}秒")

    added = after["count"] - before["count"]
    print(f"\nキャッシュ: {after['count']} 件（今回 +{added} 件）")
    print(f"キャッシュ作成に費やした実機時間の累計: {after['total_elapsed_sec'] / 60:.1f}分")
    print("→ キャッシュ利用時はこの時間が 0 になります（SUMAI_LLM_CACHE=auto または replay）")

    if all(ok for _, ok, _ in results):
        print("\n✅ 全ペルソナが完走しました")
        return 0
    print("\n❌ 完走しなかったペルソナがあります")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
