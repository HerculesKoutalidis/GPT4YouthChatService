"""
run_pipeline.py
===============
Runs every question in eval/questions.jsonl through your REAL RAG pipeline
(src.engine.rag_engine.ChatEngine) and records, per question:

    - the chunks that were retrieved (text + payload metadata + score)
    - the final answer the LLM produced

Output: eval/results/<run_name>.jsonl  (one JSON object per question)

This file is what judge.py scores. Run it ONCE per RAG version so you can
compare versions later (baseline vs after each roadmap step).

--------------------------------------------------------------------------
HOW TO RUN
--------------------------------------------------------------------------
Recommended — inside the streamlit container (has all deps + can reach the
qdrant/vllm containers by service name, and your repo is mounted at /app):

    docker compose exec streamlit-ui python -m eval.run_pipeline --run-name baseline_v0.2

From the host instead (needs deps installed locally + qdrant/vllm ports
published, which they are): make sure VLLM_API_KEY is exported, then:

    python -m eval.run_pipeline --run-name baseline_v0.2

Notes:
  * Retrieval is deterministic, so the chunks recorded here match what the
    LLM saw when generating the answer.
  * This does NOT need an Anthropic key — that's only for judge.py.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

# Make "src" importable no matter where we're launched from.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.engine.rag_engine import ChatEngine  # noqa: E402

HERE = Path(__file__).resolve().parent
QUESTIONS = HERE / "questions.jsonl"
RESULTS_DIR = HERE / "results"


def load_questions():
    items = []
    with QUESTIONS.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def retrieve_chunks(engine: ChatEngine, query: str, limit: int):
    """
    Mirror ChatEngine.get_context() but keep the results STRUCTURED
    (text + metadata + score) instead of collapsing them into one string.
    Uses the engine's own encoder / qdrant client / collection so it stays
    in sync with the real pipeline.
    """
    vec = engine.encoder.encode(query).tolist()
    hits = engine.qdrant.search(
        collection_name=engine.collection_name,
        query_vector=vec,
        limit=limit,
    )
    chunks = []
    for rank, h in enumerate(hits, start=1):
        payload = h.payload or {}
        chunks.append({
            "rank": rank,
            "score": float(h.score) if h.score is not None else None,
            "file_name": payload.get("file_name", ""),
            "title": payload.get("title", ""),
            "text": (payload.get("text", "") or "").strip(),
        })
    return chunks


def generate_answer(engine: ChatEngine, question: str) -> str:
    """Single-turn: fresh system message, then this question. Drain the stream."""
    messages = [{"role": "system", "content": engine.instructions}]
    stream = engine.get_llm_response(messages, question)
    parts = []
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            parts.append(delta)
    return "".join(parts).strip()


def run_one(engine: ChatEngine, question: str, fallback_limit: int):
    """Generate the answer, then record the context the engine ACTUALLY used.

    New engine (small-to-big) exposes engine.last_sources — the parent sections
    fed to the LLM. We record those, so the judge scores what the model saw.
    Old/other engines without last_sources fall back to a direct child search."""
    answer = generate_answer(engine, question)
    src = getattr(engine, "last_sources", None)
    if src:
        retrieved = [{
            "rank": s.get("n"),
            "score": None,
            "file_name": s.get("file_name", ""),
            "title": s.get("title", ""),
            "text": s.get("text", ""),
            "matched_excerpt": s.get("matched_excerpt", ""),
        } for s in src]
    else:
        retrieved = retrieve_chunks(engine, question, fallback_limit)
    return retrieved, answer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default=None,
                    help="Label for this run, e.g. baseline_v0.2 or after_step1. "
                         "Defaults to a timestamp.")
    ap.add_argument("--limit", type=int, default=None,
                    help="How many chunks to RECORD per question. "
                         "Default = your config rag.top_k. Lower it (e.g. 10) to "
                         "keep result files small; the answer still uses the real pipeline.")
    ap.add_argument("--sleep", type=float, default=0.0,
                    help="Seconds to pause between questions (be gentle on the shared GPU).")
    args = ap.parse_args()

    run_name = args.run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{run_name}.jsonl"

    print("Loading ChatEngine (this loads the encoder + connects to Qdrant/vLLM)...")
    engine = ChatEngine()
    record_limit = args.limit or engine.config["rag"].get("child_top_k") or 10
    print(f"Recording top {record_limit} retrieved chunks per question.")

    questions = load_questions()
    print(f"Running {len(questions)} questions -> {out_path}\n")

    with out_path.open("w", encoding="utf-8") as out:
        for i, q in enumerate(questions, start=1):
            qid, text = q["id"], q["question"]
            print(f"[{i:>2}/{len(questions)}] {qid} ...", end=" ", flush=True)
            rec = {**q}
            try:
                rec["retrieved"], rec["answer"] = run_one(engine, text, record_limit)
                rec["error"] = None
                print(f"ok ({len(rec['retrieved'])} ctx, {len(rec['answer'])} chars)")
            except Exception as e:
                rec["retrieved"] = rec.get("retrieved", [])
                rec["answer"] = ""
                rec["error"] = f"{type(e).__name__}: {e}"
                print(f"ERROR: {rec['error']}")
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            if args.sleep:
                time.sleep(args.sleep)

    print(f"\nDone. Wrote {out_path}")
    print("Next:  python -m eval.judge --run", run_name)


if __name__ == "__main__":
    main()
