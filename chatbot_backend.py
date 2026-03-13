import streamlit as st , os
from openai import OpenAI
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from PIL import Image 

# --- Configuration ---
# Check if we are running inside a Docker container
# 'RUNNING_IN_DOCKER' is a variable we will define in our docker-compose file
IS_DOCKER = os.environ.get("RUNNING_IN_DOCKER", "false").lower() == "true"

if IS_DOCKER:
    # Docker internal network addresses
    VLLM_API_BASE = "http://vllm-engine:8005/v1"
    QDRANT_URL = "http://qdrant-db:6333"
else:
    # Local server addresses
    VLLM_API_BASE = "http://localhost:8005/v1"
    QDRANT_URL = "http://localhost:6333"

# Common settings
VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "") #"super-secret-admin-key"
COLLECTION_NAME = "eu_job_market"
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"

# --- AVATARS CONFIGURATION ---
USER_AVATAR = "./images/user_image.jpg"
ASSISTANT_AVATAR = "./images/business_woman.png"

# --- Page Config ---
st.set_page_config(page_title="GPT4Youth Assistant", layout="wide")
 
# --- CUSTOM HEADER ---
try:
    col_empty_left, col_center, col_empty_right = st.columns([1, 4, 1])
    with col_center:
        st.markdown("<div style='text-align: center; margin-bottom: 20px;'><h1>GPT4Youth Chat</h1></div>", unsafe_allow_html=True)
        learning_img = Image.open("./images/digitaleducation.png")
        st.markdown("<style>div.stImage {text-align: center; display: block; margin-left: auto; margin-right: auto; width: 100%;}</style>", unsafe_allow_html=True)
        st.image(learning_img, width=600) 
except FileNotFoundError:
    st.title("🇪🇺 EU Education & Job Market Bot")

# --- Singleton Resources ---
@st.cache_resource
def get_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

@st.cache_resource
def get_qdrant_client():
    return QdrantClient(url=QDRANT_URL)

@st.cache_resource
def get_llm_client():
    return OpenAI(api_key=VLLM_API_KEY, base_url=VLLM_API_BASE)

encoder = get_embedding_model()
qdrant = get_qdrant_client()
client = get_llm_client()

# --- Session State (Memory & Logic) ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are a helpful assistant specialized in the EU Job Market and Education sectors."}
    ]

# Tracks if the user is currently editing their last prompt
if "editing_last" not in st.session_state:
    st.session_state.editing_last = False

# --- RAG Logic ---
def get_context(query: str, limit: int = 3) -> str:
    try:
        vector = encoder.encode(query).tolist()
        search_result = qdrant.search(collection_name=COLLECTION_NAME, query_vector=vector, limit=limit)
        return "\n\n".join([hit.payload.get("text", "") for hit in search_result])
    except:
        return ""

def process_query(prompt):
    """Encapsulated logic to handle RAG + vLLM call with Sliding Window Memory"""
    context_data = get_context(prompt)
    current_message_with_context = f"Context: {context_data}\n\nQuestion: {prompt}"
    
    # --- SLIDING WINDOW MEMORY ---
    # We keep the System Prompt (index 0) and the last 20 messages.
    # This ensures the bot remembers the last ~10-15 full exchanges.
    MAX_HISTORY = 70
    if len(st.session_state.messages) > MAX_HISTORY:
        # Keep system prompt + the last 20 messages
        history_to_send = [st.session_state.messages[0]] + st.session_state.messages[-MAX_HISTORY:]
    else:
        history_to_send = st.session_state.messages
    
    # Replace the very last user message in the payload with the one containing RAG context
    api_messages = history_to_send[:-1] + [{"role": "user", "content": current_message_with_context}]
    
    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        message_placeholder = st.empty()
        full_response = ""
        try:
            stream = client.chat.completions.create(
                model=MODEL_NAME,
                messages=api_messages,
                temperature=0.7,
                max_tokens=2048, # Per user request
                stream=True
            )
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    full_response += chunk.choices[0].delta.content
                    message_placeholder.markdown(full_response + "")
            message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
        except Exception as e:
            st.error(f"Connection Error: {e}")

# --- UI Layout ---

# 1. Identify the last user message index
last_user_idx = None
for i in range(len(st.session_state.messages) - 1, -1, -1):
    if st.session_state.messages[i]["role"] == "user":
        last_user_idx = i
        break

# 2. Display chat history
for i, message in enumerate(st.session_state.messages):
    if message["role"] == "system":
        continue
    
    avatar_path = USER_AVATAR if message["role"] == "user" else ASSISTANT_AVATAR
    
    with st.chat_message(message["role"], avatar=avatar_path):
        if i == last_user_idx and st.session_state.editing_last:
            with st.form(key="edit_last_prompt_form"):
                edited_text = st.text_area("Edit your prompt:", value=message["content"])
                col1, col2 = st.columns([1, 5])
                if col1.form_submit_button("Update"):
                    st.session_state.messages[i]["content"] = edited_text
                    st.session_state.messages = st.session_state.messages[:i+1]
                    st.session_state.editing_last = False
                    st.rerun()
                if col2.form_submit_button("Cancel"):
                    st.session_state.editing_last = False
                    st.rerun()
        else:
            st.markdown(message["content"])
            if i == last_user_idx and not st.session_state.editing_last:
                if st.button("Change", key=f"btn_edit_{i}"):
                    st.session_state.editing_last = True
                    st.rerun()

# 3. Handle processing if the last message is a user prompt
if len(st.session_state.messages) > 0 and st.session_state.messages[-1]["role"] == "user" and not st.session_state.editing_last:
    process_query(st.session_state.messages[-1]["content"])

# 4. Standard Input Field (Footer)
if prompt := st.chat_input("Ask anything about EU jobs or universities..."):
    st.session_state.editing_last = False
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.rerun()