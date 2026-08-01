import os
import streamlit as st
from PIL import Image
from src.engine.rag_engine import ChatEngine
from src.engine.rag_engine import config

# chat_store.py lives next to this file (src/ui/). Streamlit puts the main
# script's directory on sys.path, so a plain import works. If your setup runs
# everything as the `src` package, change this to: from src.ui import chat_store as store
import chat_store as store

# --- Page Config (MUST BE FIRST) ---
st.set_page_config(page_title="GPT4Youth Assistant", layout="wide")

# --- Configuration & Assets ---
UI_DIR = os.path.dirname(os.path.abspath(__file__))
USER_AVATAR = os.path.join(UI_DIR, "images", "user_image.jpg")
ASSISTANT_AVATAR = os.path.join(UI_DIR, "images", "business_woman.png")
LEARNING_IMG_PATH = os.path.join(UI_DIR, "images", "digitaleducation.png")

TITLE_DISPLAY_LEN = 20  # sidebar shows first ~20 chars + "…"


# --- Initialize Engine (once) ---
@st.cache_resource
def load_engine():
    return ChatEngine()


# --- Initialize DB (once) ---
@st.cache_resource
def init_store():
    store.init_db()
    return True


engine = load_engine()
init_store()


# --- Identity seam ------------------------------------------------------------
# TEMPORARY until Keycloak / oauth2-proxy is in front of the app (Step 6).
# Once oauth2-proxy is wired, it injects the identity as a request header
# (typically X-Auth-Request-Email / X-Forwarded-Email). st.context.headers
# (public since Streamlit v1.37.0) exposes those, so this function becomes the
# ONLY place that changes — the rest of the app already keys everything on
# whatever string this returns.
def get_current_user() -> str:
    try:
        headers = getattr(st.context, "headers", None)
        if headers:
            for h in ("X-Auth-Request-Email", "X-Forwarded-Email", "X-Forwarded-User"):
                val = headers.get(h)
                if val:
                    return val
    except Exception:
        pass
    if os.environ.get("GPT4YOUTH_DEV_USER"):
        return os.environ["GPT4YOUTH_DEV_USER"]
    return st.session_state.get("dev_user_field", "dev_user")


# --- Title helpers ------------------------------------------------------------
def make_title(first_user_message: str) -> str:
    """LLM-generated conversation title if the engine supports it; else the raw
    first message. Both get truncated for display by _short()."""
    text = (first_user_message or "").strip()
    try:
        if hasattr(engine, "generate_title"):
            llm_title = engine.generate_title(text)
            if llm_title and llm_title.strip():
                return llm_title.strip()
    except Exception:
        pass
    return text or "New chat"


def _short(title: str, n: int = TITLE_DISPLAY_LEN) -> str:
    title = (title or "Untitled").strip()
    return title if len(title) <= n else title[:n].rstrip() + "…"


# --- Session helpers ----------------------------------------------------------
def _fresh_messages():
    return [{"role": "system", "content": config["system_instructions"]}]


def start_new_chat():
    """Reset to an empty, unsaved conversation."""
    st.session_state.messages = _fresh_messages()
    st.session_state.conversation_id = None
    st.session_state.editing_last = False
    st.session_state.needs_title = False


def load_chat(conv_id: str):
    """Load a past conversation into the main view. System prompt is re-prepended
    from CURRENT config, so prompt fixes apply retroactively."""
    st.session_state.messages = _fresh_messages() + store.get_messages(conv_id)
    st.session_state.conversation_id = conv_id
    st.session_state.editing_last = False
    st.session_state.needs_title = False


# --- Core UI Logic ------------------------------------------------------------
def process_query(prompt):
    """Stream the engine's reply, append deterministic citations, persist it."""
    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        message_placeholder = st.empty()
        full_response = ""
        try:
            stream = engine.get_llm_response(st.session_state.messages, prompt)
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
                    message_placeholder.markdown(full_response + "▌")

            # append deterministic citations (guaranteed, not model-dependent)
            sources_md = engine.format_sources()
            if sources_md:
                full_response += "\n" + sources_md

            message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})

            # persist the assistant turn (conversation was created on the user turn)
            if st.session_state.conversation_id:
                store.append_message(st.session_state.conversation_id, "assistant", full_response)
        except Exception as e:
            st.error(f"Engine Error: {e}")


# --- Session State (init) -----------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = _fresh_messages()
if "editing_last" not in st.session_state:
    st.session_state.editing_last = False
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None
if "needs_title" not in st.session_state:
    st.session_state.needs_title = False
if "dev_user_field" not in st.session_state:
    st.session_state.dev_user_field = "dev_user"

user_id = get_current_user()

# --- SIDEBAR: conversations ---------------------------------------------------
with st.sidebar:
    st.markdown("## 💬 Conversations")

    if st.button("➕  New chat", use_container_width=True, type="primary"):
        start_new_chat()
        st.rerun()

    st.divider()

    conversations = store.list_conversations(user_id)
    if not conversations:
        st.caption("No saved conversations yet.")

    for conv in conversations:
        cid = conv["id"]
        is_active = cid == st.session_state.conversation_id
        full_title = conv["title"] or "Untitled"

        col_open, col_menu = st.columns([6, 1])

        # Open / load this conversation. Full title on hover; truncated label.
        if col_open.button(
            ("🟢 " if is_active else "") + _short(full_title),
            key=f"open_{cid}",
            use_container_width=True,
            help=full_title,
        ):
            load_chat(cid)
            st.rerun()

        # "⋮" menu: Rename / Delete (extensible later).
        with col_menu.popover("⋮", use_container_width=True):
            new_name = st.text_input(
                "Rename chat",
                value=full_title,
                key=f"rename_input_{cid}",
                max_chars=80,
            )
            c1, c2 = st.columns(2)
            if c1.button("💾 Save", key=f"rename_btn_{cid}", use_container_width=True):
                store.rename_conversation(cid, new_name)
                st.rerun()
            if c2.button("🗑 Delete", key=f"del_{cid}", use_container_width=True):
                store.delete_conversation(cid)
                if is_active:
                    start_new_chat()
                st.rerun()

    # --- Dev-only user switcher (REMOVE once Keycloak provides identity) ---
    st.divider()
    with st.expander("🔧 Dev: current user", expanded=False):
        st.text_input(
            "user id (temporary — replaced by Keycloak login)",
            key="dev_user_field",
        )
        st.caption(f"Acting as: **{user_id}**")


# --- CUSTOM HEADER (centered image, aligned with the title) ---
try:
    col_empty_left, col_center, col_empty_right = st.columns([1, 4, 1])
    with col_center:
        st.markdown(
            "<div style='text-align:center; margin-bottom:20px;'><h1>GPT4Youth Chat</h1></div>",
            unsafe_allow_html=True,
        )
        _l, _mid, _r = st.columns([1, 2, 1])
        with _mid:
            st.image(LEARNING_IMG_PATH, use_container_width=True)
except FileNotFoundError:
    st.title("🇪🇺 EU Education & Job Market Bot")


# --- UI Layout: Chat History --------------------------------------------------
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
                    st.session_state.messages = st.session_state.messages[: i + 1]
                    # sync DB: this user message is at seq (i - 1) because the
                    # system prompt at session index 0 is NOT stored. Remove it
                    # and everything after, then re-add the edited text; the
                    # regenerated assistant turn is persisted by process_query.
                    if st.session_state.conversation_id:
                        store.truncate_from(st.session_state.conversation_id, i - 1)
                        store.append_message(
                            st.session_state.conversation_id, "user", edited_text
                        )
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

# --- Trigger Processing of last user's message --------------------------------
if (
    len(st.session_state.messages) > 0
    and st.session_state.messages[-1]["role"] == "user"
    and not st.session_state.editing_last
):
    process_query(st.session_state.messages[-1]["content"])

    # Auto-title a brand-new conversation from its first user message, AFTER the
    # first answer (so the title call never delays the reply). Falls back to the
    # truncated first message if the engine has no generate_title().
    if st.session_state.needs_title and st.session_state.conversation_id:
        first_user = next(
            (m["content"] for m in st.session_state.messages if m["role"] == "user"), ""
        )
        store.rename_conversation(st.session_state.conversation_id, make_title(first_user))
        st.session_state.needs_title = False
        st.rerun()

# --- User Input Field ---------------------------------------------------------
if prompt := st.chat_input("Ask anything about EU jobs or universities..."):
    st.session_state.editing_last = False
    # Create the conversation lazily on the FIRST user message (empty chats
    # never hit the DB). Provisional title = first message; upgraded to an
    # LLM title after the first answer (see needs_title above).
    if st.session_state.conversation_id is None:
        st.session_state.conversation_id = store.create_conversation(user_id, title=prompt)
        st.session_state.needs_title = True
    store.append_message(st.session_state.conversation_id, "user", prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.rerun()
