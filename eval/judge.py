"""
judge.py
========
Reference-free LLM-as-judge scoring for a run produced by run_pipeline.py.

It sends each {question, retrieved chunks, answer} to a STRONG model (Claude,
via the Anthropic API) and asks it to score the answer on a rubric built
directly from the CAZALLA testers' complaints. No manual relevance labels
needed — that's the whole point.

Input : eval/results/<run>.jsonl
Output: eval/results/<run>_judged.jsonl   (same records + a "scores" field)

--------------------------------------------------------------------------
SETUP (once)
--------------------------------------------------------------------------
    pip install anthropic
    export ANTHROPIC_API_KEY=sk-ant-...      # from console.anthropic.com

    # This is a pay-as-you-go API key (separate from your Claude Pro plan).
    # Cost here is tiny: ~70 questions x one short call each.

--------------------------------------------------------------------------
RUN
--------------------------------------------------------------------------
    python -m eval.judge --run baseline_v0.2
    # optionally: --model claude-opus-4-8   (stronger judge, a bit pricier)

Do NOT judge with your own Llama-3.2-3B — you'd be grading a model with
itself. Use a clearly stronger, independent judge.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

try:
    import anthropic
except ImportError:
    sys.exit("Missing dependency. Run:  pip install anthropic")

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"

# Default judge model. Bump to claude-opus-4-8 for maximum reliability.
DEFAULT_MODEL = "claude-sonnet-4-6"

# Keep judge cost/latency bounded, but scale the per-source budget to how
# many sources there are -- fewer, larger sources (parent sections) each get
# more room, otherwise the truly-matched span can fall outside what's shown
# while the LLM itself saw the whole thing.
MAX_CHUNKS_TO_JUDGE = 12
TOTAL_CONTEXT_BUDGET = 9000       # total chars of source text shown to the judge
MIN_CHUNK_CHAR_CAP = 700
EXCERPT_CHAR_CAP = 500            # the matched span is short; rarely needs cutting


RUBRIC = """You are a strict evaluator of a RAG chatbot that helps EU youth workers.
The chatbot retrieves passages from a library of youth-work manuals (energisers,
non-formal education toolkits, Erasmus+ guides, anti-bullying/anti-racism handbooks,
etc.) and uses them to help a youth worker DESIGN and GENERATE content: workshops,
icebreakers, debriefs, project plans. The retrieved passages are raw material /
inspiration, NOT direct answers.

Score the ANSWER on each criterion from 1 (poor) to 5 (excellent). Judge only
from what you are given.

1. context_usefulness — Did the retrieved passages actually contain useful, on-topic
   raw material a youth worker could build on for THIS question? (5 = highly relevant
   passages; 1 = irrelevant/empty.)

2. groundedness — Are the concrete specifics in the answer (named methods, activities,
   structures, facts) actually supported by the retrieved passages, rather than
   invented? Penalise unsupported specifics and hallucinated facts/programmes.

3. specificity — Is the answer concretely useful and youth-work-specific, or is it
   generic filler that any LLM would produce WITHOUT retrieval? (5 = names real
   methods/activities and gives concrete detail; 1 = vague boilerplate.) This is the
   testers' single biggest complaint — score it harshly.

4. source_attribution — When drawing on retrieved material, does the answer point to
   its sources (document names / links) so the user can go deeper? (5 = clear, correct
   attribution where warranted; 1 = none when it clearly should have.) If the question
   genuinely needs no sourcing, score 3 as neutral.

5. actionability — Is it something a youth worker could run as-is: clear structure,
   timings, materials, step-by-step instructions, a real debrief? (5 = ready to use;
   1 = abstract, missing the practical detail.)

Also set:
   refusal — true if the answer refuses / declines the request, else false.
   (Note: legitimate youth-work topics like sexual-health education should NOT be
   refused; a refusal there is a FALSE refusal and should tank actionability.)

Return ONLY a JSON object, no prose, no markdown fences:
{"context_usefulness": int, "groundedness": int, "specificity": int,
 "source_attribution": int, "actionability": int, "refusal": bool,
 "rationale": "one or two sentences"}"""


def build_user_content(rec: dict) -> str:
    chunks = rec.get("retrieved", [])[:MAX_CHUNKS_TO_JUDGE]
    n = max(1, len(chunks))
    per_chunk_cap = max(MIN_CHUNK_CHAR_CAP, TOTAL_CONTEXT_BUDGET // n)

    lines = []
    for c in chunks:
        src = c.get("file_name") or c.get("title") or "unknown"
        full_text = c.get("text", "") or ""
        excerpt = (c.get("matched_excerpt") or "").strip()

        block = [f"--- source: {src} (score={c.get('score')}) ---"]
        if excerpt and excerpt != full_text.strip():
            # This is the exact span that earned retrieval -- show it in full
            # (it's short), THEN a capped slice of the wider section it lives
            # in, so the judge sees what the LLM saw without being blind to
            # where in a long parent the relevant part actually is.
            block.append(f"[Matched excerpt -- why this was retrieved]\n{excerpt[:EXCERPT_CHAR_CAP]}")
            block.append(f"[Surrounding section, truncated]\n{full_text[:per_chunk_cap]}")
        else:
            block.append(full_text[:per_chunk_cap])
        lines.append("\n".join(block))

    context_block = "\n\n".join(lines) if lines else "(no chunks retrieved)"
    answer = rec.get("answer", "") or "(empty answer)"
    return (
        f"QUESTION:\n{rec['question']}\n\n"
        f"RETRIEVED PASSAGES (top {len(chunks)}):\n{context_block}\n\n"
        f"CHATBOT ANSWER:\n{answer}"
    )


def parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{"):]
    start, end = text.find("{"), text.rfind("}")
    return json.loads(text[start:end + 1])


def judge_record(client, model, rec, retries=2):
    content = build_user_content(rec)
    for attempt in range(retries + 1):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=400,
                system=RUBRIC,
                messages=[{"role": "user", "content": content}],
            )
            raw = "".join(b.text for b in resp.content if b.type == "text")
            return parse_json(raw)
        except Exception as e:
            if attempt < retries:
                time.sleep(2 * (attempt + 1))
                continue
            return {"error": f"{type(e).__name__}: {e}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="Run name, e.g. baseline_v0.2")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()

    in_path = RESULTS_DIR / f"{args.run}.jsonl"
    out_path = RESULTS_DIR / f"{args.run}_judged.jsonl"
    if not in_path.exists():
        sys.exit(f"Not found: {in_path}\nRun run_pipeline.py first.")

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY first (export ANTHROPIC_API_KEY=sk-ant-...).")

    client = anthropic.Anthropic()
    records = [json.loads(l) for l in in_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"Judging {len(records)} records with {args.model} -> {out_path}\n")

    with out_path.open("w", encoding="utf-8") as out:
        for i, rec in enumerate(records, start=1):
            print(f"[{i:>2}/{len(records)}] {rec['id']} ...", end=" ", flush=True)
            if rec.get("error"):
                rec["scores"] = {"error": "pipeline_error"}
                print("skipped (pipeline error)")
            else:
                rec["scores"] = judge_record(client, args.model, rec)
                print(rec["scores"].get("error", "ok"))
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()

    print(f"\nDone. Wrote {out_path}")
    print("Next:  python -m eval.report", args.run)


if __name__ == "__main__":
    main()
