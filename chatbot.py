#%%
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st
from PIL import Image
#from vector import retriever

#%%
model = OllamaLLM(model="llama3.2:3b", base_url="http://ollama:11434") 
#model = OllamaLLM(model="llama3.1:8b-instruct-q5_1")


template = """
Here is the question to answer: {question} 
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

# 1️⃣ Initialize chat history
if "history" not in st.session_state:
    st.session_state.history = []

user_image, assistant_image = "./images/user_image.jpg", "./images/business_woman.png" #images of user and chatbot dialogue avatars 
# Open your local image
learning_img = Image.open("./images/digitaleducation.png")
#blended_learning.png
#st.title("📖 Job Assistant")

# Create two columns: one for the image, one for the title
col1, col2 = st.columns([1, 4])
with col1:
    st.image(learning_img, width=350)
with col2:
    st.markdown("<h1>GPT4Youth Assistant</h1>", unsafe_allow_html=True)

# Render previous messages with avatars
for msg in st.session_state.history:
    with st.chat_message(msg["role"], avatar=msg["avatar"]):
        st.write(msg["content"])

# Capture new input
if user_input := st.chat_input("Ask about anything concerning training, education and occupational topics"):
    # 🧑‍🍳 user message
    st.session_state.history.append({
        "role": "user",
        "content": user_input,
        "avatar": user_image
    })
    with st.chat_message("user", avatar=user_image):
        st.write(user_input)

    # 🤖 assistant response
    #reviews = retriever.invoke(user_input)
    response = chain.invoke({ "question": user_input})
    st.session_state.history.append({
        "role": "assistant",
        "content": response,
        "avatar": assistant_image
    })
    with st.chat_message("assistant", avatar= assistant_image):
        st.write(response)
# %%
