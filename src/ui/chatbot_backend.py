import os
import streamlit as st
from PIL import Image 
from src.engine.rag_engine import ChatEngine 

# --- Page Config (MUST BE FIRST) ---
st.set_page_config(page_title="GPT4Youth Assistant", layout="wide")

# --- Configuration & Assets ---
# Find the absolute path to the directory where this script (chatbot_backend.py) lives
UI_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Configuration & Assets ---
#Chatbot UI images
USER_AVATAR = os.path.join(UI_DIR, "images", "user_image.jpg")
ASSISTANT_AVATAR = os.path.join(UI_DIR, "images", "business_woman.png")
LEARNING_IMG_PATH = os.path.join(UI_DIR, "images", "digitaleducation.png")

# --- Initialize Engine ---
@st.cache_resource
def load_engine():
    return ChatEngine()

engine = load_engine()

# --- Core UI Logic ---
def process_query(prompt):
    """Handles the UI updates while waiting for the engine's stream."""
    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        message_placeholder = st.empty()
        full_response = ""
        try:
            # Call the decoupled engine
            stream = engine.get_llm_response(st.session_state.messages, prompt)
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
                    message_placeholder.markdown(full_response + "")
            message_placeholder.markdown(full_response)
            
            # Save the final response to memory
            st.session_state.messages.append({"role": "assistant", "content": full_response})
        except Exception as e:
            st.error(f"Engine Error: {e}")

# --- CUSTOM HEADER ---
try:
    col_empty_left, col_center, col_empty_right = st.columns([1, 4, 1])
    with col_center:
        st.markdown("<div style='text-align: center; margin-bottom: 20px;'><h1>GPT4Youth Chat</h1></div>", unsafe_allow_html=True)
        learning_img = Image.open(LEARNING_IMG_PATH)
        st.markdown("<style>div.stImage {text-align: center; display: block; margin-left: auto; margin-right: auto; width: 100%;}</style>", unsafe_allow_html=True)
        st.image(learning_img, width=600) 
except FileNotFoundError:
    st.title("🇪🇺 EU Education & Job Market Bot")

# --- Session State (Memory & Logic) ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are a helpful assistant specialized in the EU Job Market and Education sectors."}
    ]

if "editing_last" not in st.session_state:
    st.session_state.editing_last = False

# --- UI Layout: Chat History ---

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

# --- Trigger Processing ---
if len(st.session_state.messages) > 0 and st.session_state.messages[-1]["role"] == "user" and not st.session_state.editing_last:
    process_query(st.session_state.messages[-1]["content"])

# --- User Input Field ---
if prompt := st.chat_input("Ask anything about EU jobs or universities..."):
    st.session_state.editing_last = False
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.rerun()