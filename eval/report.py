# This file collects scores in a table, compares runs
"""
report.py
=========
Aggregates one or more *_judged.jsonl files into a readable table so you can
see, at a glance, whether a RAG change helped — and by how much.

    # single run
    python -m eval.report baseline_v0.2

    # compare runs side by side (baseline first, then each later step)
    python -m eval.report baseline_v0.2 after_step1 after_step2

Pass run names WITHOUT the "_judged" suffix (it's added automatically).
Pure stdlib — no extra installs.
"""

import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"

CRITERIA = ["context_usefulness", "groundedness", "specificity",
            "source_attribution", "actionability"]


def load_judged(run: str):
    path = RESULTS_DIR / f"{run}_judged.jsonl"
    if not path.exists():
        sys.exit(f"Not found: {path}  (did you run judge.py --run {run} ?)")
    recs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            recs.append(json.loads(line))
    return recs


def summarise(recs):
    """Return means per criterion (overall + per level) and a few counters."""
    buckets = defaultdict(list)          # key -> list of scores
    refusals = 0
    judged = 0
    errors = 0
    for r in recs:
        s = r.get("scores", {})
        if not s or "error" in s:
            errors += 1
            continue
        judged += 1
        if s.get("refusal"):
            refusals += 1
        for c in CRITERIA:
            if isinstance(s.get(c), (int, float)):
                buckets[("all", c)].append(s[c])
                buckets[(r.get("level", "?"), c)].append(s[c])
    means = {k: (sum(v) / len(v)) for k, v in buckets.items() if v}
    return {"means": means, "judged": judged, "errors": errors,
            "refusals": refusals, "n": len(recs)}


def fmt(x):
    return f"{x:.2f}" if isinstance(x, float) else str(x)


def print_table(runs, summaries):
    col_w = 16
    header = "criterion".ljust(24) + "".join(r[:col_w].ljust(col_w) for r in runs)
    print("\n" + header)
    print("-" * len(header))

    def row(label, key):
        cells = []
        prev = None
        for r in runs:
            m = summaries[r]["means"].get(key)
            if m is None:
                cells.append("-".ljust(col_w))
            else:
                delta = ""
                if prev is not None:
                    d = m - prev
                    delta = f" ({'+' if d >= 0 else ''}{d:.2f})"
                cells.append((fmt(m) + delta).ljust(col_w))
                prev = m
        print(label.ljust(24) + "".join(cells))

    print("OVERALL (mean 1-5)")
    for c in CRITERIA:
        row("  " + c, ("all", c))

    for level in ("experienced", "less_experienced"):
        print(f"\n{level.upper()}")
        for c in CRITERIA:
            row("  " + c, (level, c))

    print("\nMETA")
    for label, field in [("  questions", "n"), ("  judged ok", "judged"),
                         ("  judge errors", "errors"), ("  refusals", "refusals")]:
        cells = "".join(fmt(summaries[r][field]).ljust(col_w) for r in runs)
        print(label.ljust(24) + cells)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run names (without _judged suffix)")
    args = ap.parse_args()

    summaries = {r: summarise(load_judged(r)) for r in args.runs}
    print_table(args.runs, summaries)

    if len(args.runs) >= 2:
        print("\n(Deltas in parentheses are vs the previous column — left to right.)")


if __name__ == "__main__":
    main()