"""
inspect_qdrant.py
=================
Quick read-only overview of what's actually inside your Qdrant so you can
see collections, vector config, point count, payload schema (keys + types),
and a few sample points. Optionally runs a test search.

Run inside the streamlit container (has qdrant-client + reaches the db):

    # list everything + inspect the configured collection
    docker compose exec -w /app streamlit-ui python -m eval.inspect_qdrant

    # inspect a specific collection
    docker compose exec -w /app streamlit-ui python -m eval.inspect_qdrant --collection eu_job_market

    # also run a test semantic search (uses the same encoder as the app)
    docker compose exec -w /app streamlit-ui python -m eval.inspect_qdrant --search "icebreakers for a youth workshop"

Read-only: it never writes or deletes anything.
"""

import os
import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qdrant_client import QdrantClient  # noqa: E402

IS_DOCKER = os.environ.get("RUNNING_IN_DOCKER", "false").lower() == "true"
QDRANT_URL = "http://qdrant-gptforyouth:6333" if IS_DOCKER else "http://localhost:6343"


def type_name(v):
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int):
        return "int"
    if isinstance(v, float):
        return "float"
    if isinstance(v, str):
        return "str"
    if isinstance(v, list):
        inner = type_name(v[0]) if v else "?"
        return f"list[{inner}]"
    if isinstance(v, dict):
        return "dict"
    return type(v).__name__


def short(v, n=90):
    s = str(v).replace("\n", " ")
    return s if len(s) <= n else s[:n] + " ..."


def inspect_collection(client, name):
    print(f"\n{'='*70}\nCOLLECTION: {name}\n{'='*70}")
    info = client.get_collection(name)

    # vector config (handles single or named vectors)
    vectors = info.config.params.vectors
    print("Vector config:")
    if hasattr(vectors, "size"):
        print(f"  size={vectors.size}  distance={vectors.distance}")
    elif isinstance(vectors, dict):
        for vname, vp in vectors.items():
            print(f"  [{vname}] size={vp.size}  distance={vp.distance}")
    else:
        print(f"  {vectors}")

    count = client.count(name, exact=True).count
    print(f"Points: {count}")

    # sample a few points to derive the payload schema
    points, _ = client.scroll(name, limit=5, with_payload=True, with_vectors=False)
    if not points:
        print("No points stored yet.")
        return

    schema = {}
    for p in points:
        for k, v in (p.payload or {}).items():
            schema.setdefault(k, type_name(v))
    print("\nPayload keys (type):")
    for k in sorted(schema):
        print(f"  - {k:<16} {schema[k]}")

    print("\nSample point payloads:")
    for i, p in enumerate(points[:3], start=1):
        print(f"  [{i}] id={p.id}")
        for k, v in (p.payload or {}).items():
            print(f"        {k:<14}= {short(v)}")


def test_search(client, name, query):
    print(f"\n{'='*70}\nTEST SEARCH: {query!r}\n{'='*70}")
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    vec = enc.encode(query).tolist()
    resp = client.query_points(collection_name=name, query=vec, limit=5, with_payload=True)
    for rank, h in enumerate(resp.points, start=1):
        pl = h.payload or {}
        src = pl.get("file_name") or pl.get("title") or "?"
        print(f"  #{rank}  score={h.score:.4f}  src={src}")
        print(f"        {short(pl.get('text',''), 120)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", default=None,
                    help="Collection to inspect. Default: read rag.collection_name from config.")
    ap.add_argument("--search", default=None, help="Optional test query.")
    args = ap.parse_args()

    client = QdrantClient(url=QDRANT_URL)

    print(f"Connected to {QDRANT_URL}")
    cols = [c.name for c in client.get_collections().collections]
    print(f"Collections: {cols or '(none)'}")

    target = args.collection
    if target is None:
        try:
            from src.engine.rag_engine import config
            target = config["rag"]["collection_name"]
        except Exception:
            target = cols[0] if cols else None

    if not target:
        sys.exit("No collection to inspect.")
    if target not in cols:
        sys.exit(f"Collection {target!r} not found. Available: {cols}")

    inspect_collection(client, target)
    if args.search:
        test_search(client, target, args.search)


if __name__ == "__main__":
    main()
