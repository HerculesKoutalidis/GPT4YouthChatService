import os
from openai import OpenAI
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# Logic for environment-based config
IS_DOCKER = os.environ.get("RUNNING_IN_DOCKER", "false").lower() == "true"
VLLM_API_BASE = "http://vllm-engine:8005/v1" if IS_DOCKER else "http://localhost:8005/v1"
QDRANT_URL = "http://qdrant-gptforyouth:6343" if IS_DOCKER else "http://localhost:6343"

class ChatEngine:
    def __init__(self):
        self.client = OpenAI(api_key=os.environ.get("VLLM_API_KEY", ""), base_url=VLLM_API_BASE)
        self.qdrant = QdrantClient(url=QDRANT_URL)
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        self.collection_name = "eu_job_market"
        self.model_name = "meta-llama/Llama-3.2-3B-Instruct"

    # In rag_engine.py -> get_context method
    def get_context(self, query: str, limit: int = 3) -> str:
        try:
            # If Qdrant isn't running, this will fail immediately
            vector = self.encoder.encode(query).tolist()
            search_result = self.qdrant.search(
                collection_name=self.collection_name, 
                query_vector=vector, 
                limit=limit
            )
            return "\n\n".join([hit.payload.get("text", "") for hit in search_result])
        except Exception as e:
            print(f"RAG Search failed (Qdrant probably offline): {e}")
            return "" # Returns empty context so the LLM still works!

    def get_llm_response(self, messages, prompt):
        context_data = self.get_context(prompt)
        enhanced_prompt = f"Context: {context_data}\n\nQuestion: {prompt}"
        
        # Prepare the messages for the API (replacing last user message with context)
        api_messages = messages[:-1] + [{"role": "user", "content": enhanced_prompt}]
        
        return self.client.chat.completions.create(
            model=self.model_name,
            messages=api_messages,
            temperature=0.7,
            max_tokens=2048,
            stream=True
        )