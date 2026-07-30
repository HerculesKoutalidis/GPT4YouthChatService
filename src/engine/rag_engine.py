import os, yaml
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Prefetch, FusionQuery, Fusion, SparseVector

from .tei_client import embed as tei_embed, rerank as tei_rerank


# --------------------------------------------------
# Environment-aware service endpoints
# --------------------------------------------------
IS_DOCKER = os.environ.get("RUNNING_IN_DOCKER", "false").lower() == "true"

VLLM_API_BASE = (
    "http://vllm-engine:8005/v1"
    if IS_DOCKER
    else "http://localhost:8005/v1"
)

QDRANT_URL = (
    "http://qdrant-gptforyouth:6333"
    if IS_DOCKER
    else "http://localhost:6343"
)

# Named-vector labels on the child collection (must match ingest_pdfs.py).
DENSE_NAME = "dense"
SPARSE_NAME = "sparse"
SPARSE_MODEL_ID = "Qdrant/bm25"

# --------------------------------------------------
# Configuration loader
# --------------------------------------------------

def _deep_merge(base: dict, override: dict) -> dict:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config():
    """Load config.yaml, optionally overridden by config.local.yaml."""
    base_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    local_path = os.path.join(os.path.dirname(__file__), "..", "config.local.yaml")

    with open(base_path, "r") as f:
        config = yaml.safe_load(f)

    if os.path.exists(local_path):
        with open(local_path, "r") as f:
            local_config = yaml.safe_load(f)
            _deep_merge(config, local_config)
    return config

config = load_config()


# --------------------------------------------------
# Helpers
# --------------------------------------------------
def _format_pages(ps, pe):
    if ps and pe:
        return f"p.{ps}" if ps == pe else f"pp.{ps}-{pe}"
    if ps:
        return f"p.{ps}"
    return ""


# --------------------------------------------------
# Chat Engine
# --------------------------------------------------

class ChatEngine:
    """
    RAG pipeline (Step 3 — hybrid small-to-big + rerank):

    User query
        -> rewrite to a standalone query (LLM, if multi-turn)
        -> dense embed (bge-m3, GPU via tei-embeddings)
           + sparse embed (BM25 via FastEmbed, CPU)
        -> Qdrant Query API on CHILD collection (gpt4youth_docs):
             prefetch dense (retrieve_top_k) + prefetch sparse (sparse_top_k)
             -> server-side RRF fusion -> hybrid_top_k candidates
           (use_sparse: false falls back to dense-only search)
        -> rerank (bge-reranker-v2-m3, tei-reranker)     [cross-encoder re-order]
        -> dedupe hits to their PARENTS (per-doc cap for diversity)
        -> fetch PARENT sections (gpt4youth_parents)     [rich context]
        -> build prompt + deterministic Sources list
        -> vLLM / Qwen3-14B-AWQ  (streaming, thinking disabled)
    """

    def __init__(self):
        self.config = config

        self.client = OpenAI(
            api_key=os.environ.get("VLLM_API_KEY", ""),
            base_url=VLLM_API_BASE)

        self.qdrant = QdrantClient(url=QDRANT_URL)

        rag = self.config["rag"]
        self.child_collection = rag.get("collection_name", "gpt4youth_docs")
        self.parent_collection = rag.get("parent_collection_name", "gpt4youth_parents")
        self.retrieve_top_k = rag.get("retrieve_top_k", 50)   # dense prefetch size
        self.sparse_top_k = rag.get("sparse_top_k", 50)       # sparse prefetch size
        self.hybrid_top_k = rag.get("hybrid_top_k", 50)       # fused list size -> reranker
        self.use_sparse = rag.get("use_sparse", True)
        self.use_reranker = rag.get("use_reranker", True)
        self.max_parents = rag.get("max_parents", 4)
        self.max_parents_per_doc = rag.get("max_parents_per_doc", 2)
        self.use_query_rewrite = rag.get("use_query_rewrite", True)
        self.rewrite_max_tokens = rag.get("rewrite_max_tokens", 80)

        # Sparse query encoder (BM25, CPU, tiny). Loaded lazily so the engine
        # still starts (dense-only) if fastembed isn't installed.
        self._sparse_model = None
        if self.use_sparse:
            try:
                from fastembed import SparseTextEmbedding
                self._sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL_ID)
            except Exception as e:
                print(f"Sparse encoder unavailable ({e}); falling back to dense-only.")
                self.use_sparse = False

        self.model_name = self.config["llm"]["model_name"]
        self.instructions = self.config["system_instructions"]

        # populated on every retrieval, rendered by the UI as citations
        self.last_sources = []
        self.last_context = ""

    # ---------------------------------------------------------------
    # Retrieval: hybrid child search -> rerank -> parent expansion
    # ---------------------------------------------------------------
    def _search_children(self, query: str):
        """Return child hits: hybrid (dense+sparse, RRF) or dense-only."""
        qvec = tei_embed(query)  # bge-m3, 1024-dim (GPU via tei-embeddings)

        if self.use_sparse and self._sparse_model is not None:
            sq = next(iter(self._sparse_model.query_embed(query)))
            return self.qdrant.query_points(
                collection_name=self.child_collection,
                prefetch=[
                    Prefetch(query=qvec, using=DENSE_NAME,
                             limit=self.retrieve_top_k),
                    Prefetch(query=SparseVector(indices=sq.indices.tolist(),
                                                values=sq.values.tolist()),
                             using=SPARSE_NAME,
                             limit=self.sparse_top_k),
                ],
                query=FusionQuery(fusion=Fusion.RRF),
                limit=self.hybrid_top_k,
                with_payload=True,
            ).points

        return self.qdrant.query_points(
            collection_name=self.child_collection,
            query=qvec,
            using=DENSE_NAME,
            limit=self.retrieve_top_k,
            with_payload=True,
        ).points

    def get_context(self, query: str) -> str:
        self.last_sources = []
        self.last_context = ""
        try:
            child_hits = self._search_children(query)
            if not child_hits:
                return ""

            # --- rerank stage: cross-encoder re-orders the fused candidates ---
            # bge-reranker scores (query, child_text) jointly, which is far more
            # accurate than fusion rank alone. If the reranker is unreachable we
            # fall back to the fused order rather than losing context entirely.
            if self.use_reranker:
                try:
                    texts = [(h.payload or {}).get("text", "") for h in child_hits]
                    ranked = tei_rerank(query, texts)  # [{"index","score"}...] best-first
                    child_hits = [child_hits[r["index"]] for r in ranked]
                except Exception as e:
                    print(f"Rerank failed, falling back to fused order: {e}")

            # pick unique parents by (reranked) child order, with a per-document
            # cap so one big manual can't monopolise the context. Keep the matched
            # child text too -- the exact span that earned this parent's inclusion.
            chosen, per_doc, matched_excerpt = [], {}, {}
            for h in child_hits:
                pl = h.payload or {}
                pid = pl.get("parent_id")
                fname = pl.get("file_name", "")
                if not pid or pid in chosen:
                    continue
                if per_doc.get(fname, 0) >= self.max_parents_per_doc:
                    continue
                chosen.append(pid)
                matched_excerpt[pid] = pl.get("text", "").strip()
                per_doc[fname] = per_doc.get(fname, 0) + 1
                if len(chosen) >= self.max_parents:
                    break

            if not chosen:
                return ""

            records = self.qdrant.retrieve(
                collection_name=self.parent_collection,
                ids=chosen,
                with_payload=True,
            )
            by_id = {r.id: r for r in records}

            blocks, sources = [], []
            for i, pid in enumerate(chosen, start=1):
                rec = by_id.get(pid)
                if not rec:
                    continue
                pl = rec.payload or {}
                title = pl.get("title", "") or pl.get("file_name", "")
                pages = _format_pages(pl.get("page_start"), pl.get("page_end"))
                label = f"[{i}] {title}" + (f", {pages}" if pages else "")
                blocks.append(f"{label}\n{pl.get('text','').strip()}")
                sources.append({
                    "n": i,
                    "title": title,
                    "file_name": pl.get("file_name", ""),
                    "pages": pages,
                    "section_path": pl.get("section_path", ""),
                    "source_link": pl.get("source_link", ""),
                    "text": pl.get("text", "").strip(),
                    "matched_excerpt": matched_excerpt.get(pid, ""),
                })

            self.last_sources = sources
            self.last_context = "\n\n".join(blocks)
            return self.last_context

        except Exception as e:
            print(f"RAG retrieval failed (Qdrant/TEI offline?): {e}")
            return ""

    def format_sources(self) -> str:
        """Deterministic Markdown 'Sources' block for the UI to append after the
        streamed answer. Guarantees citations regardless of the model."""
        if not self.last_sources:
            return ""
        lines = ["", "---", "**Sources**"]
        for s in self.last_sources:
            line = f"{s['n']}. {s['title']}"
            if s["pages"]:
                line += f" ({s['pages']})"
            if s["source_link"]:
                line += f" — {s['source_link']}"
            lines.append(line)
        return "\n".join(lines)

    def _rewrite_query(self, messages: list, current_prompt: str) -> str:
        """Turn a possibly-context-dependent last message (e.g. "yes", "the
        second one") into a standalone search query, using the conversation.

        Uses the same vLLM model (no extra infra). Deterministic (temp 0),
        non-streamed, thinking disabled. Falls back to the plain concatenation
        if anything goes wrong or there\'s no real history to fold in."""
        history = [m for m in messages[1:] if m["role"] in ("user", "assistant")]
        # Nothing to disambiguate against -> concatenation (== old behaviour).
        if not history:
            return self._build_retrieval_query(messages, current_prompt)

        convo = "\n".join(
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
            for m in history[-(2 * self.config['llm']['max_history']):]
        )
        rewrite_instructions = (
            "You rewrite the user\'s latest message into a single, self-contained "
            "search query for a document retrieval system about EU youth work, "
            "Erasmus+ and non-formal education. Resolve references (pronouns, "
            "\"yes\", \"the first one\") using the conversation. Keep the user\'s "
            "own key terms and acronyms. Output ONLY the query, no quotes, no preamble."
        )
        user_block = (
            f"Conversation so far:\n{convo}\n\n"
            f"Latest user message:\n{current_prompt}\n\n"
            "Standalone search query:"
        )
        try:
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": rewrite_instructions},
                    {"role": "user", "content": user_block},
                ],
                temperature=0.0,
                max_tokens=self.rewrite_max_tokens,
                stream=False,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            rewritten = (resp.choices[0].message.content or "").strip()
            # Guard against a degenerate/empty rewrite.
            if len(rewritten) < 3:
                return self._build_retrieval_query(messages, current_prompt)
            return rewritten
        except Exception as e:
            print(f"Query rewrite failed, using concatenation: {e}")
            return self._build_retrieval_query(messages, current_prompt)

    def _build_retrieval_query(self, messages: list, current_prompt: str) -> str:
        recent = [m["content"] for m in messages[-4:] if m["role"] != "system"]
        recent.append(current_prompt)
        return " ".join(recent)

    # ---------------------------------------------------------------
    # Generation
    # ---------------------------------------------------------------
    def get_llm_response(self, messages, prompt):
        if self.use_query_rewrite:
            retrieval_query = self._rewrite_query(messages, prompt)
        else:
            retrieval_query = self._build_retrieval_query(messages, prompt)
        context_data = self.get_context(retrieval_query)

        enhanced_prompt = (
            "Use the retrieved material below to give concrete, youth-work-specific help. "
            "The material comes from real youth-work manuals; draw specific methods, "
            "activities and structures from it rather than giving generic advice. "
            "When a method or fact comes from the material, cite its source naturally (e.g. "
            "\"the Compass manual suggests...\"). If you go beyond the material, present it as "
            "general best practice WITHOUT naming a manual, and never invent figures, page "
            "numbers, or rules that aren't in the material.\n\n"
            f"Retrieved material:\n{context_data if context_data else '(nothing retrieved)'}\n\n"
            f"User question:\n{prompt}\n\n"
            "If the material doesn't cover the question, say so briefly and fall back to "
            "general best practice, framed as professional advice."
        )

        max_history = self.config["llm"]["max_history"]
        system_message = messages[:1]
        conversation_history = messages[1:-1][-(2 * max_history):]

        api_messages = (
            system_message
            + conversation_history
            + [{"role": "user", "content": enhanced_prompt}]
        )

        return self.client.chat.completions.create(
            model=self.model_name,
            messages=api_messages,
            temperature=self.config["llm"]["temperature"],
            max_tokens=self.config["llm"]["max_tokens"],
            stream=True,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},  # Qwen3 only
        )
