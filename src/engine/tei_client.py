"""
tei_client.py  (Step 2) — thin HTTP client for the two TEI services.
====================================================================
Used by BOTH ingestion (host .venv) and the live engine (Streamlit
container), so ingest-time and query-time embeddings come from the SAME
bge-m3 model — this is what prevents a dense-vector mismatch.

Endpoints (Hugging Face Text Embeddings Inference):
  POST {EMBED_URL}/embed
      {"inputs": [...], "normalize": true, "truncate": true}  -> [[...], ...]
  POST {RERANK_URL}/rerank
      {"query": q, "texts": [...]}  -> [{"index": i, "score": s}, ...]  (best-first)

URLs auto-switch like rag_engine.py already does for vLLM:
  in Docker  -> reach sibling containers by service name on internal port 80
  on host    -> reach the published ports on localhost (8090 / 8091)
Override with env vars TEI_EMBED_URL / TEI_RERANK_URL if ever needed.
"""

import os
import requests

IS_DOCKER = os.environ.get("RUNNING_IN_DOCKER", "false").lower() == "true"

EMBED_URL = os.environ.get(
    "TEI_EMBED_URL",
    "http://tei-embeddings:80" if IS_DOCKER else "http://localhost:8090",
)
RERANK_URL = os.environ.get(
    "TEI_RERANK_URL",
    "http://tei-reranker:80" if IS_DOCKER else "http://localhost:8091",
)

_TIMEOUT = 120

# TEI's default --max-client-batch-size is 32, so keep client batches <= 32
# unless you raise that flag in docker-compose.yml.
_DEFAULT_BATCH = 32


def embed(texts, batch_size=_DEFAULT_BATCH, normalize=True, truncate=True):
    """Embed a string or list[str] -> list[list[float]] (1024-dim, bge-m3).

    Sends in client-side batches so a full-corpus ingest doesn't POST 6900
    chunks in one giant request. TEI also batches server-side on the GPU."""
    single = isinstance(texts, str)
    items = [texts] if single else list(texts)
    out = []
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        r = requests.post(
            f"{EMBED_URL}/embed",
            json={"inputs": batch, "normalize": normalize, "truncate": truncate},
            timeout=_TIMEOUT,
        )
        r.raise_for_status()
        out.extend(r.json())
    return out[0] if single else out


def rerank(query, texts, batch_size=_DEFAULT_BATCH, truncate=True):
    """Cross-encoder scores for (query, text) pairs.
    Returns list of {"index": int, "score": float}, sorted best-first, where
    `index` maps back to the position in the `texts` you passed in.

    Chunked in batches of <=32 because TEI's default --max-client-batch-size is
    32; bge-reranker scores are per-pair absolute, so merging across batches and
    re-sorting by score is valid."""
    texts = list(texts)
    if not texts:
        return []
    merged = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        r = requests.post(
            f"{RERANK_URL}/rerank",
            json={"query": query, "texts": batch, "truncate": truncate},
            timeout=_TIMEOUT,
        )
        r.raise_for_status()
        for item in r.json():
            merged.append({"index": item["index"] + i, "score": item["score"]})
    merged.sort(key=lambda x: x["score"], reverse=True)
    return merged


def health():
    """Reachability check for both services (used by the smoke test below)."""
    status = {}
    for name, url in (("embeddings", EMBED_URL), ("reranker", RERANK_URL)):
        try:
            resp = requests.get(f"{url}/health", timeout=5)
            resp.raise_for_status()
            status[name] = "OK"
        except Exception as e:
            status[name] = f"DOWN ({e})"
    return status


if __name__ == "__main__":
    # Quick smoke test:  python3 tei_client.py
    print("Health:", health())
    v = embed("icebreaker activity for teenagers")
    print(f"Embedding dim: {len(v)} (expected 1024)")
    ranked = rerank(
        "icebreaker for teens",
        ["a debrief method for reflection",
         "an energizer game for 13-18 year olds"],
    )
    print("Rerank (best-first):", ranked)
