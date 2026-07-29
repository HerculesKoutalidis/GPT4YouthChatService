"""
ingest_pdfs.py  (v3 — hybrid: dense bge-m3 + sparse BM25, parent/child + catalog metadata)
==========================================================================================
Builds the RAG knowledge base into TWO Qdrant collections:

  gpt4youth_docs     child chunks (~180 tokens)  -> what we SEARCH (hybrid)
  gpt4youth_parents  parent sections (~1000 tok) -> what we FEED the LLM later

Step 3 (hybrid): the child collection now holds TWO named vectors per point:
  "dense"  : bge-m3 1024-dim (semantic), via the tei-embeddings service (GPU)
  "sparse" : BM25 term weights, via FastEmbed (CPU, no model weights needed)
The sparse side uses Qdrant's server-side IDF modifier, so document-frequency
statistics live in Qdrant itself — nothing to precompute or keep in sync.
Sparse vectors are built from the SAME contextual text as dense (header + chunk),
so exact-term matches on titles/tags also work.

The parent collection is unchanged (fetched by id, never searched).

Prereqs:
  1) Qdrant running:            docker compose up -d qdrant
  2) TEI embeddings running:    docker compose up -d tei-embeddings
  3) Catalog built & reviewed:  python3 src/ingestion/build_catalog.py
  4) PDFs in data/raw_pdfs/
  5) fastembed installed:       pip install fastembed

Run (from project root, Qdrant on localhost:6343, TEI on localhost:8090):
  python3 src/ingestion/ingest_pdfs.py --reset    # REQUIRED once for v3:
                                                  # the child collection schema
                                                  # changes to named vectors.

Re-running is safe: each document's old points are deleted before re-upsert
(deterministic IDs + per-doc cleanup => no duplicates, no orphans).
"""

import os
import sys
import uuid
import argparse
import logging

# Make the project root importable so we can share src/engine/tei_client.py
# (ingestion normally runs with src/ingestion on sys.path, not the root).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct, VectorParams, Distance, Filter, FieldCondition, MatchValue,
    SparseVectorParams, SparseVector, Modifier,
)
from transformers import AutoTokenizer
from fastembed import SparseTextEmbedding

import pdf_processing as pp
from catalog import CATALOG_COLUMNS, SUPPORTED_LANGS, load_catalog
from src.engine.tei_client import embed as tei_embed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ingest")

# --- Paths / config ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_PDF_DIR = os.path.join(BASE_DIR, "data", "raw_pdfs")
CATALOG_PATH = os.path.join(BASE_DIR, "data", "metadata", "docs_catalog.xlsx")
QDRANT_URL = "http://localhost:6343"

CHILD_COLLECTION = "gpt4youth_docs"
PARENT_COLLECTION = "gpt4youth_parents"

# Named-vector labels on the child collection (must match rag_engine.py).
DENSE_NAME = "dense"
SPARSE_NAME = "sparse"

# BM25 sparse encoder (FastEmbed, CPU). Term weighting only; IDF is applied
# server-side by Qdrant thanks to Modifier.IDF on the collection.
SPARSE_MODEL_ID = "Qdrant/bm25"

# Ingestion-time knobs (chunk sizing) — unchanged since Step 1 on purpose:
# the only Step 3 variable is the added sparse vector.
CHILD_TOKENS, CHILD_OVERLAP = 180, 30      # small, precise chunks (what we search)
PARENT_TOKENS, PARENT_OVERLAP = 1000, 120  # not embedded for search; just context

# Must match the --model-id of the `tei-embeddings` service in docker-compose.yml.
# Used ONLY to load the tokenizer locally for token-based chunk sizing + token_count;
# the actual dense vectors come from the TEI service, not from this.
EMBED_MODEL_ID = os.environ.get("EMBED_MODEL_ID", "BAAI/bge-m3")

NS = uuid.NAMESPACE_URL


def det_id(*parts) -> str:
    return str(uuid.uuid5(NS, ":".join(str(p) for p in parts)))


# ---------------------------------------------------------------------------
# Pure builder (no IO / no network) — unit-testable
# ---------------------------------------------------------------------------
def build_points_for_doc(file_name, pages, toc, meta, child_splitter,
                         parent_splitter, encode_fn, sparse_encode_fn=None,
                         tokenizer=None):
    """Return (child_points, parent_points).

    Children get {"dense": [...], "sparse": SparseVector(...)} named vectors
    (sparse omitted when sparse_encode_fn is None); parents keep a single
    unnamed dense vector as before."""
    full_text, starts, page_nums = pp.join_pages(pages)
    doc_title = meta.get("title") or os.path.splitext(file_name)[0]
    tags_list = pp.parse_tags(meta.get("tags"))
    tags_text = ", ".join(tags_list)
    document_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, file_name))

    doc_meta = {
        "document_id": document_id,
        "file_name": file_name,
        "title": doc_title,
        "summary": meta.get("summary", "") or "",
        "tags": tags_list,
        "country": meta.get("country", "") or "",
        "language": (meta.get("language", "") or "").lower(),
        "doc_type": meta.get("doc_type", "") or "",
        "who_it_helps": meta.get("who_it_helps", "") or "",
        "how_it_helps": meta.get("how_it_helps", "") or "",
        "source_link": meta.get("source_link", "") or "",
        "year": str(meta.get("year", "") or ""),
    }

    parents = pp.split_hierarchical(
        full_text, parent_splitter, child_splitter,
        starts=starts, page_nums=page_nums, toc=toc, doc_title=doc_title,
    )

    # ---- parents (unchanged: single unnamed dense vector, fetched by id) ----
    parent_points, parent_texts = [], []
    for par in parents:
        pid = det_id(file_name, "parent", par.index)
        parent_texts.append(par.text)
        parent_points.append({
            "id": pid,
            "payload": {
                "document_id": document_id,
                "file_name": file_name,
                "parent_id": pid,
                "parent_index": par.index,
                "text": par.text,
                "section_path": par.section_path,
                "page_start": par.page_start,
                "page_end": par.page_end,
                "title": doc_title,
                "source_link": doc_meta["source_link"],
            },
        })
    parent_vecs = encode_fn(parent_texts) if parent_texts else []
    for pt, v in zip(parent_points, parent_vecs):
        pt["vector"] = v

    # ---- children (hybrid: dense + sparse, both from the contextual text) ----
    child_points, child_embed_texts = [], []
    for par in parents:
        pid = det_id(file_name, "parent", par.index)
        header = pp.contextual_header(doc_title, par.section_path, tags_text)
        for c in par.children:
            cid = det_id(file_name, "child", c.index)
            embed_text = f"{header}\n\n{c.text}" if header else c.text
            child_embed_texts.append(embed_text)
            tok = len(tokenizer.encode(c.text)) if tokenizer else None
            payload = {
                **doc_meta,
                "chunk_index": c.index,
                "text": c.text,               # raw text (for display / citation)
                "parent_id": pid,
                "section_path": c.section_path,
                "page_start": c.page_start,
                "page_end": c.page_end,
                "token_count": tok,
            }
            child_points.append({"id": cid, "payload": payload})

    dense_vecs = encode_fn(child_embed_texts) if child_embed_texts else []
    sparse_vecs = (sparse_encode_fn(child_embed_texts)
                   if (sparse_encode_fn and child_embed_texts) else None)

    for i, (pt, dv) in enumerate(zip(child_points, dense_vecs)):
        vec = {DENSE_NAME: dv}
        if sparse_vecs is not None:
            sv = sparse_vecs[i]
            vec[SPARSE_NAME] = SparseVector(
                indices=sv.indices.tolist(), values=sv.values.tolist())
        pt["vector"] = vec

    return child_points, parent_points


# ---------------------------------------------------------------------------
# Qdrant IO
# ---------------------------------------------------------------------------
def ensure_child_collection(qc, name, dim, reset):
    """Child collection: named dense vector + named sparse vector (IDF)."""
    if reset and qc.collection_exists(name):
        log.info(f"Dropping collection {name}")
        qc.delete_collection(name)
    if not qc.collection_exists(name):
        log.info(f"Creating collection {name} "
                 f"(dense[{DENSE_NAME}]={dim} Cosine + sparse[{SPARSE_NAME}] IDF)")
        qc.create_collection(
            name,
            vectors_config={DENSE_NAME: VectorParams(size=dim, distance=Distance.COSINE)},
            sparse_vectors_config={SPARSE_NAME: SparseVectorParams(modifier=Modifier.IDF)},
        )


def ensure_parent_collection(qc, name, dim, reset):
    """Parent collection: unchanged single unnamed dense vector."""
    if reset and qc.collection_exists(name):
        log.info(f"Dropping collection {name}")
        qc.delete_collection(name)
    if not qc.collection_exists(name):
        log.info(f"Creating collection {name} (size={dim}, Cosine)")
        qc.create_collection(name, vectors_config=VectorParams(size=dim, distance=Distance.COSINE))


def delete_doc(qc, name, document_id):
    qc.delete(collection_name=name, points_selector=Filter(
        must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
    ))


def to_points(items):
    return [PointStruct(id=i["id"], vector=i["vector"], payload=i["payload"]) for i in items]


# Qdrant caps a single HTTP request at 32MB; upsert in batches (1024-dim dense
# + sparse per point adds up fast on big PDFs).
UPSERT_BATCH = 256


def upsert_in_batches(qc, name, points, batch=UPSERT_BATCH):
    for i in range(0, len(points), batch):
        qc.upsert(name, points=points[i:i + batch])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true", help="Drop & recreate both collections first.")
    ap.add_argument("--allow-missing", action="store_true",
                    help="Ingest PDFs not in the catalog (minimal metadata) instead of skipping.")
    args = ap.parse_args()

    if not os.path.isdir(RAW_PDF_DIR):
        raise SystemExit(f"PDF dir not found: {RAW_PDF_DIR}")
    catalog = load_catalog(CATALOG_PATH)
    if not catalog and not args.allow_missing:
        raise SystemExit(f"No catalog at {CATALOG_PATH}. Run build_catalog.py first "
                         f"(or pass --allow-missing).")

    # Tokenizer is only for chunk sizing / token_count — dense vectors come from TEI.
    log.info(f"Loading tokenizer for {EMBED_MODEL_ID} (CPU, chunk sizing only)...")
    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_ID)

    # Ask the TEI embeddings service for the dense vector size (single source of truth).
    log.info("Probing tei-embeddings for the dense vector dimension...")
    try:
        dim = len(tei_embed("dimension probe"))
    except Exception as e:
        raise SystemExit(
            f"Cannot reach tei-embeddings ({e}).\n"
            f"Start it with:  docker compose up -d tei-embeddings"
        )
    log.info(f"TEI dense dim = {dim}")

    log.info(f"Loading sparse encoder {SPARSE_MODEL_ID} (FastEmbed, CPU)...")
    sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL_ID)

    def sparse_encode_fn(texts):
        # BM25 term weights per chunk; Qdrant applies IDF server-side.
        return list(sparse_model.embed(texts))

    child_sp = pp.make_token_splitter(tokenizer, CHILD_TOKENS, CHILD_OVERLAP)
    parent_sp = pp.make_token_splitter(tokenizer, PARENT_TOKENS, PARENT_OVERLAP)

    def encode_fn(texts):
        # bge-m3 via TEI (GPU). tei_embed batches internally (<=32/request).
        return tei_embed(texts)

    qc = QdrantClient(url=QDRANT_URL)
    ensure_child_collection(qc, CHILD_COLLECTION, dim, args.reset)
    ensure_parent_collection(qc, PARENT_COLLECTION, dim, args.reset)

    pdfs = sorted(f for f in os.listdir(RAW_PDF_DIR) if f.lower().endswith(".pdf"))
    tot_c = tot_p = 0
    for fn in pdfs:
        meta = catalog.get(fn)
        if meta is None:
            if not args.allow_missing:
                log.warning(f"SKIP {fn}: not in catalog (run build_catalog.py).")
                continue
            meta = {c: "" for c in CATALOG_COLUMNS}
            meta["file_name"] = fn
            meta["title"] = os.path.splitext(fn)[0]

        lang = (meta.get("language") or "").lower()
        if lang and lang not in SUPPORTED_LANGS:
            log.warning(f"SKIP {fn}: language '{lang}' not in SUPPORTED_LANGS.")
            continue

        try:
            pages, toc = pp.extract_pages(os.path.join(RAW_PDF_DIR, fn))
        except Exception as e:
            log.error(f"FAIL {fn}: {e}")
            continue

        children, parents = build_points_for_doc(
            fn, pages, toc, meta, child_sp, parent_sp, encode_fn,
            sparse_encode_fn=sparse_encode_fn, tokenizer=tokenizer,
        )
        document_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, fn))
        delete_doc(qc, CHILD_COLLECTION, document_id)     # clean re-ingest
        delete_doc(qc, PARENT_COLLECTION, document_id)
        if parents:
            upsert_in_batches(qc, PARENT_COLLECTION, to_points(parents))
        if children:
            upsert_in_batches(qc, CHILD_COLLECTION, to_points(children))
        tot_c += len(children)
        tot_p += len(parents)
        log.info(f"OK {fn}: {len(parents)} parents / {len(children)} children "
                 f"[{meta.get('metadata_source','')}]")

    log.info(f"DONE. {tot_p} parents in '{PARENT_COLLECTION}', "
             f"{tot_c} children in '{CHILD_COLLECTION}' (dense+sparse).")
    log.info("Next (Step 3.2): hybrid RRF retrieval in rag_engine.py "
             "(prefetch dense+sparse -> fusion -> rerank -> parents).")


if __name__ == "__main__":
    main()
