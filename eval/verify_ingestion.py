"""
verify_ingestion.py
===================
Health / integrity check for the two-collection RAG store. Answers "is the
data stored CORRECTLY?", not just "what's there".

    docker compose exec -w /app streamlit-ui python -m eval.verify_ingestion

    # end-to-end small-to-big preview for a query (child hit -> its parent):
    docker compose exec -w /app streamlit-ui python -m eval.verify_ingestion \
        --search "how to run a debriefing after an activity"

Checks per collection:
  * point count
  * payload completeness: % of points where each key is non-empty
  * children: every parent_id resolves to a real parent (no orphans)
  * page sanity: page_start <= page_end, both > 0
  * per-document counts (parents / children)
Read-only.
"""

import os
import sys
import argparse
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from qdrant_client import QdrantClient

IS_DOCKER = os.environ.get("RUNNING_IN_DOCKER", "false").lower() == "true"
QDRANT_URL = "http://qdrant-gptforyouth:6333" if IS_DOCKER else "http://localhost:6343"

CHILD = "gpt4youth_docs"
PARENT = "gpt4youth_parents"

# keys we expect to be populated on a child
CHILD_KEYS = ["document_id", "file_name", "title", "text", "parent_id",
              "section_path", "chunk_index", "tags", "summary", "who_it_helps"]


def scroll_all(qc, name, with_vectors=False):
    out, offset = [], None
    while True:
        pts, offset = qc.scroll(name, limit=1000, offset=offset,
                                with_payload=True, with_vectors=with_vectors)
        out.extend(pts)
        if offset is None:
            break
    return out


def pct(n, d):
    return f"{(100.0 * n / d):.0f}%" if d else "-"


def completeness(points, keys):
    filled = {k: 0 for k in keys}
    for p in points:
        pl = p.payload or {}
        for k in keys:
            v = pl.get(k)
            if v not in (None, "", [], {}):
                filled[k] += 1
    n = len(points)
    print(f"\nPayload completeness ({n} points):")
    for k in keys:
        flag = "" if filled[k] == n else "  <-- some empty"
        print(f"  {k:<15} {pct(filled[k], n):>4}{flag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--search", default=None, help="Preview child->parent expansion for a query.")
    args = ap.parse_args()

    qc = QdrantClient(url=QDRANT_URL)
    cols = [c.name for c in qc.get_collections().collections]
    for needed in (CHILD, PARENT):
        if needed not in cols:
            sys.exit(f"Missing collection: {needed}. Found: {cols}")

    children = scroll_all(qc, CHILD)
    parents = scroll_all(qc, PARENT)
    print(f"Children: {len(children)}   Parents: {len(parents)}")

    completeness(children, CHILD_KEYS)

    # orphan check: every child.parent_id exists in parents
    parent_ids = {p.id for p in parents}
    parent_pid_field = {(p.payload or {}).get("parent_id") for p in parents}
    orphans = [c for c in children
               if (c.payload or {}).get("parent_id") not in parent_ids
               and (c.payload or {}).get("parent_id") not in parent_pid_field]
    print(f"\nOrphan children (parent_id with no matching parent): {len(orphans)}")
    for c in orphans[:3]:
        print(f"   e.g. child {c.id} -> parent_id {(c.payload or {}).get('parent_id')}")

    # page sanity
    bad_pages = 0
    for p in children + parents:
        pl = p.payload or {}
        ps, pe = pl.get("page_start"), pl.get("page_end")
        if ps is not None and pe is not None and (ps < 1 or pe < ps):
            bad_pages += 1
    print(f"Points with inconsistent page range: {bad_pages}")

    # per-document breakdown
    dc = defaultdict(lambda: [0, 0])
    for c in children:
        dc[(c.payload or {}).get("file_name", "?")][0] += 1
    for p in parents:
        dc[(p.payload or {}).get("file_name", "?")][1] += 1
    print("\nPer-document (children / parents):")
    for fn in sorted(dc):
        ch, pa = dc[fn]
        print(f"  {ch:>5} / {pa:<4}  {fn}")

    verdict = "OK" if (not orphans and bad_pages == 0) else "CHECK ABOVE"
    print(f"\nIntegrity: {verdict}")

    # optional: show the small-to-big expansion for a real query
    if args.search:
        from sentence_transformers import SentenceTransformer
        enc = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        vec = enc.encode(args.search).tolist()
        hits = qc.query_points(CHILD, query=vec, limit=5, with_payload=True).points
        print(f"\n{'='*70}\nSMALL-TO-BIG PREVIEW: {args.search!r}\n{'='*70}")
        for rank, h in enumerate(hits, 1):
            pl = h.payload or {}
            pid = pl.get("parent_id")
            par = qc.retrieve(PARENT, ids=[pid], with_payload=True)
            ptext = (par[0].payload.get("text", "") if par else "")[:200]
            print(f"\n#{rank} score={h.score:.3f}  {pl.get('file_name')} | {pl.get('section_path','')[:60]}")
            print(f"   CHILD : {pl.get('text','')[:120].strip()}")
            print(f"   PARENT: {ptext.strip()}...")


if __name__ == "__main__":
    main()
