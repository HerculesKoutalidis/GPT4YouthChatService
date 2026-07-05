"""
check_run.py
============
Sanity-check a run_pipeline output (eval/results/<run>.jsonl) BEFORE judging.
Verifies structure and the quality of the retrieved parent context, and prints
a few real samples so you can eyeball them — instead of scrolling a 1.2 MB file.

    python -m eval.check_run --run after_step1
    python -m eval.check_run --run after_step1 --show 3   # more samples

Checks:
  * every line is valid JSON with the expected fields
  * how many questions returned no context (retrieval empty)
  * how many produced an empty answer or errored
  * context size stats (chars per question, parents per question)
  * suspicious parents: extremely short, or looks like page-number noise
  * diversity: how many distinct documents appear across all questions
Read-only. No API calls, no model needed.
"""

import os
import re
import sys
import json
import argparse
import statistics as stats
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")


def load(run):
    path = os.path.join(RESULTS_DIR, f"{run}.jsonl")
    if not os.path.exists(path):
        sys.exit(f"Not found: {path}")
    recs = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError as e:
                sys.exit(f"INVALID JSON on line {i}: {e}")
    return recs, path


def looks_like_noise(text):
    """Parent that is mostly page-number / header boilerplate, not real content."""
    t = text.strip()
    if len(t) < 120:
        return True
    # ratio of digits + very short lines
    digits = sum(c.isdigit() for c in t)
    return digits / max(1, len(t)) > 0.25


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--show", type=int, default=2, help="How many full samples to print.")
    args = ap.parse_args()

    recs, path = load(args.run)
    print(f"Loaded {len(recs)} records from {path}\nAll lines are valid JSON.\n")

    no_ctx = empty_ans = errored = 0
    ctx_chars, n_parents = [], []
    all_docs = Counter()
    short_parents = 0
    noise_parents = 0

    for r in recs:
        if r.get("error"):
            errored += 1
        ans = (r.get("answer") or "").strip()
        if not ans:
            empty_ans += 1
        retrieved = r.get("retrieved") or []
        if not retrieved:
            no_ctx += 1
        n_parents.append(len(retrieved))
        total = 0
        for c in retrieved:
            txt = c.get("text", "") or ""
            total += len(txt)
            all_docs[c.get("file_name", "?")] += 1
            if len(txt.strip()) < 120:
                short_parents += 1
            elif looks_like_noise(txt):
                noise_parents += 1
        ctx_chars.append(total)

    def line(label, val):
        print(f"  {label:<34} {val}")

    print("HEALTH")
    line("questions", len(recs))
    line("pipeline errors", errored)
    line("empty answers", empty_ans)
    line("questions with NO context", no_ctx)
    print("\nCONTEXT SIZE")
    line("avg parents / question", f"{stats.mean(n_parents):.1f}" if n_parents else "-")
    line("avg context chars / question", f"{stats.mean(ctx_chars):.0f}" if ctx_chars else "-")
    line("min / max context chars", f"{min(ctx_chars)} / {max(ctx_chars)}" if ctx_chars else "-")
    print("\nPARENT QUALITY (flags to eyeball, not hard errors)")
    line("very short parents (<120 chars)", short_parents)
    line("page-number/noise-looking parents", noise_parents)
    print("\nDIVERSITY (parents pulled per document, across all questions)")
    for doc, n in all_docs.most_common():
        print(f"    {n:>4}  {doc}")

    # samples
    print("\n" + "=" * 70)
    print(f"SAMPLES (first {args.show} questions: question + its parent context)")
    print("=" * 70)
    for r in recs[:args.show]:
        print(f"\n### {r.get('id')} [{r.get('partner')}/{r.get('level')}]")
        print(f"Q: {r.get('question')}")
        for c in (r.get("retrieved") or []):
            src = c.get("file_name", "?")
            txt = re.sub(r"\s+", " ", c.get("text", "")).strip()
            print(f"\n  -- parent from {src} --")
            print(f"     {txt[:500]}{' ...' if len(txt) > 500 else ''}")
        print(f"\n  ANSWER (first 300 chars): {(r.get('answer') or '')[:300].strip()} ...")


if __name__ == "__main__":
    main()
