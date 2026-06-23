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

def load_config():
    """
    Load the default configuration and optionally
    override it with values from config.local.yaml.
    """

    base_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "config.yaml")

    local_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "config.local.yaml")

    with open(base_path, "r") as f:
        config = yaml.safe_load(f)

    if os.path.exists(local_path):
        with open(local_path, "r") as f:
            local_config = yaml.safe_load(f)
            config.update(local_config)
    return config

config = load_config()


# --------------------------------------------------
# Chat Engine
# --------------------------------------------------

class ChatEngine:
    """
    Main RAG pipeline:

    User Query
        ↓
    Embedding
        ↓
    Qdrant Retrieval
        ↓
    Prompt Construction
        ↓
    vLLM Inference
    """

    def __init__(self):

        self.config = config

        self.client = OpenAI(
            api_key=os.environ.get("VLLM_API_KEY", ""),
            base_url=VLLM_API_BASE)

        self.qdrant = QdrantClient(url=QDRANT_URL)

        self.encoder = SentenceTransformer(
            "all-MiniLM-L6-v2",
            device="cpu")

        self.collection_name = self.config["rag"]["collection_name"]
        self.model_name = self.config["llm"]["model_name"]
        self.instructions = self.config["system_instructions"]

    def get_context(self, query: str, limit=None) -> str:
        """
        Retrieve the most relevant chunks from Qdrant.
        Returns a formatted context block ready for prompting.
        """

        try:
            if limit is None:
                limit = self.config["rag"]["top_k"]
            query_vector = self.encoder.encode(query).tolist()
            search_results = self.qdrant.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit
            )

            context_chunks = []

            for idx, hit in enumerate(search_results, start=1):
                text = hit.payload.get("text", "").strip()
                if text:
                    context_chunks.append(
                        f"[Document {idx}]\n{text}"
                    )

            return "\n\n".join(context_chunks)

        except Exception as e:

            print(
                f"RAG Search failed "
                f"(Qdrant probably offline): {e}"
            )

            # Allow the chatbot to continue
            # even if retrieval fails.
            return ""

    def get_llm_response(self, messages, prompt):
        """
        Build the final prompt and send it to vLLM.
        Only the most recent conversation turns are
        included in order to control context size.
        """
        # -----------------------------
        # Retrieve supporting documents
        # -----------------------------

        context_data = self.get_context(prompt)

        # -----------------------------
        # Build RAG prompt
        # -----------------------------

        enhanced_prompt = (
            "Use the retrieved information below when it is relevant to the user's question.\n\n"
            f"Retrieved Context:\n{context_data}\n\n"
            f"User Question:\n{prompt}\n\n"
            "Instructions:\n"
            "- Use the retrieved context whenever it is relevant.\n"
            "- If the context does not contain the answer, rely on your general knowledge.\n"
            "- Do not claim information exists in the context if it does not."
        )

        # -----------------------------
        # Keep only recent conversation
        # -----------------------------

        max_history = self.config["llm"]["max_history"]

        system_message = messages[:1]

        conversation_history = messages[1:-1]

        # Keep the latest N user-assistant exchanges
        conversation_history = conversation_history[
            -(2 * max_history):
        ]

        api_messages = (
            system_message + conversation_history
            + [ {
                    "role": "user",
                    "content": enhanced_prompt
                }
            ])

        # -----------------------------
        # Call vLLM
        # -----------------------------

        return self.client.chat.completions.create(
            model=self.model_name,
            messages=api_messages,
            temperature=self.config["llm"]["temperature"],
            max_tokens=self.config["llm"]["max_tokens"],
            stream=True
        )