import os, yaml
from openai import OpenAI
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


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
    RAG pipeline (Step 1 — small-to-big):

    User query
        -> embed (all-MiniLM, CPU)
        -> search CHILD collection (gpt4youth_docs)      [precise, small chunks]
        -> dedupe hits to their PARENTS (with per-doc cap for diversity)
        -> fetch PARENT sections (gpt4youth_parents)     [rich context]
        -> build prompt + deterministic Sources list
        -> vLLM / Llama-3.2-3B  (streaming)
    """

    def __init__(self):
        self.config = config

        self.client = OpenAI(
            api_key=os.environ.get("VLLM_API_KEY", ""),
            base_url=VLLM_API_BASE)

        self.qdrant = QdrantClient(url=QDRANT_URL)
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

        rag = self.config["rag"]
        self.child_collection = rag.get("collection_name", "gpt4youth_docs")
        self.parent_collection = rag.get("parent_collection_name", "gpt4youth_parents")
        self.child_top_k = rag.get("child_top_k", 18)
        self.max_parents = rag.get("max_parents", 4)
        self.max_parents_per_doc = rag.get("max_parents_per_doc", 2)

        self.model_name = self.config["llm"]["model_name"]
        self.instructions = self.config["system_instructions"]

        # populated on every retrieval, rendered by the UI as citations
        self.last_sources = []
        self.last_context = ""

    # ---------------------------------------------------------------
    # Retrieval: child search -> parent expansion (small-to-big)
    # ---------------------------------------------------------------
    def get_context(self, query: str) -> str:
        self.last_sources = []
        self.last_context = ""
        try:
            qvec = self.encoder.encode(query).tolist()
            child_hits = self.qdrant.query_points(
                collection_name=self.child_collection,
                query=qvec,
                limit=self.child_top_k,
                with_payload=True,
            ).points

            # pick unique parents in order of best child score, with a
            # per-document cap so one big manual can't monopolise the context.
            # Keep the matched child text too -- it's the exact span that
            # earned this parent's inclusion, which the judge needs to see.
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
            print(f"RAG retrieval failed (Qdrant offline?): {e}")
            return ""

    def format_sources(self) -> str:
        """Deterministic Markdown 'Sources' block for the UI to append after the
        streamed answer. Guarantees citations regardless of the 3B model."""
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

    def _build_retrieval_query(self, messages: list, current_prompt: str) -> str:
        recent = [m["content"] for m in messages[-4:] if m["role"] != "system"]
        recent.append(current_prompt)
        return " ".join(recent)

    # ---------------------------------------------------------------
    # Generation
    # ---------------------------------------------------------------
    def get_llm_response(self, messages, prompt):
        retrieval_query = self._build_retrieval_query(messages, prompt)
        context_data = self.get_context(retrieval_query)

        enhanced_prompt = (
            "Use the retrieved material below to give concrete, youth-work-specific help. "
            "The material comes from real youth-work manuals; draw specific methods, "
            "activities and structures from it rather than giving generic advice. "
            "When you use a source, refer to it naturally (e.g. \"the Compass manual "
            "suggests...\"). Never invent details that aren't supported by the material.\n\n"
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
        )
