"""
ingest_pdfs.py  (v2 — hierarchical parent/child + catalog metadata)
===================================================================
Builds the RAG knowledge base into TWO Qdrant collections:

  gpt4youth_docs     child chunks (~180 tokens)  -> what we SEARCH
  gpt4youth_parents  parent sections (~1000 tok) -> what we FEED the LLM later

Each child carries: document_id, parent_id, section_path, page range,
chunk text (raw) + the full doc-level metadata from data/metadata/docs_catalog.xlsx.
The vector is built from a *contextual* version of the child (title/section/tags
prepended) so an isolated chunk still retrieves well — but the raw text is what
gets stored and shown.

Prereqs:
  1) Qdrant running:            docker compose up -d qdrant
  2) Catalog built & reviewed:  python src/ingestion/build_catalog.py
  3) PDFs in data/raw_pdfs/

Run (from project root, Qdrant reachable on localhost:6343):
  python src/ingestion/ingest_pdfs.py            # create-if-missing + upsert
  python src/ingestion/ingest_pdfs.py --reset    # drop & rebuild both collections

Re-running is safe: each document's old points are deleted before re-upsert
(deterministic IDs + per-doc cleanup => no duplicates, no orphans).

Step 1 stays on all-MiniLM-L6-v2 (dense only). Sparse vectors + bge-m3 arrive
in Step 2 (which will recreate these collections).
"""

import os
import uuid
import argparse
import logging

from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct, VectorParams, Distance, Filter, FieldCondition, MatchValue,
)
from sentence_transformers import SentenceTransformer

import pdf_processing as pp
from catalog import CATALOG_COLUMNS, SUPPORTED_LANGS, load_catalog

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ingest")

# --- Paths / config ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_PDF_DIR = os.path.join(BASE_DIR, "data", "raw_pdfs")
CATALOG_PATH = os.path.join(BASE_DIR, "data", "metadata", "docs_catalog.xlsx")
QDRANT_URL = "http://localhost:6343"

CHILD_COLLECTION = "gpt4youth_docs"
PARENT_COLLECTION = "gpt4youth_parents"

CHILD_TOKENS, CHILD_OVERLAP = 180, 30      # fits MiniLM's 256-token limit (+ header)
PARENT_TOKENS, PARENT_OVERLAP = 1000, 120  # not embedded for search; just context

NS = uuid.NAMESPACE_URL


def det_id(*parts) -> str:
    return str(uuid.uuid5(NS, ":".join(str(p) for p in parts)))


# ---------------------------------------------------------------------------
# Pure builder (no IO / no network) — unit-testable
# ---------------------------------------------------------------------------
def build_points_for_doc(file_name, pages, toc, meta, child_splitter,
                         parent_splitter, encode_fn, tokenizer=None):
    """Return (child_points, parent_points) as (id, vector, payload) tuples."""
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

    # ---- parents ----
    parent_points, parent_texts, parent_ids = [], [], []
    for par in parents:
        pid = det_id(file_name, "parent", par.index)
        parent_ids.append(pid)
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

    # ---- children (embedded with contextual header) ----
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
    child_vecs = encode_fn(child_embed_texts) if child_embed_texts else []
    for pt, v in zip(child_points, child_vecs):
        pt["vector"] = v

    return child_points, parent_points


# ---------------------------------------------------------------------------
# Qdrant IO
# ---------------------------------------------------------------------------
def ensure_collection(qc, name, dim, reset):
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

    log.info("Loading all-MiniLM-L6-v2 (CPU)...")
    encoder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    dim = encoder.get_sentence_embedding_dimension()
    child_sp = pp.make_token_splitter(encoder.tokenizer, CHILD_TOKENS, CHILD_OVERLAP)
    parent_sp = pp.make_token_splitter(encoder.tokenizer, PARENT_TOKENS, PARENT_OVERLAP)

    def encode_fn(texts):
        return [v.tolist() for v in encoder.encode(texts, batch_size=64, show_progress_bar=False)]

    qc = QdrantClient(url=QDRANT_URL)
    ensure_collection(qc, CHILD_COLLECTION, dim, args.reset)
    ensure_collection(qc, PARENT_COLLECTION, dim, args.reset)

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
            log.warning(f"SKIP {fn}: language '{lang}' not supported in Step 1.")
            continue

        try:
            pages, toc = pp.extract_pages(os.path.join(RAW_PDF_DIR, fn))
        except Exception as e:
            log.error(f"FAIL {fn}: {e}")
            continue

        children, parents = build_points_for_doc(
            fn, pages, toc, meta, child_sp, parent_sp, encode_fn, tokenizer=encoder.tokenizer,
        )
        document_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, fn))
        delete_doc(qc, CHILD_COLLECTION, document_id)     # clean re-ingest
        delete_doc(qc, PARENT_COLLECTION, document_id)
        if parents:
            qc.upsert(PARENT_COLLECTION, points=to_points(parents))
        if children:
            qc.upsert(CHILD_COLLECTION, points=to_points(children))
        tot_c += len(children)
        tot_p += len(parents)
        log.info(f"OK {fn}: {len(parents)} parents / {len(children)} children "
                 f"[{meta.get('metadata_source','')}]")

    log.info(f"DONE. {tot_p} parents in '{PARENT_COLLECTION}', "
             f"{tot_c} children in '{CHILD_COLLECTION}'.")
    log.info("Next: point config.yaml rag.collection_name -> 'gpt4youth_docs', "
             "then wire small-to-big retrieval in rag_engine.py.")


if __name__ == "__main__":
    main()
