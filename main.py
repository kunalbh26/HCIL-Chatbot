# main.py

import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import time

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="HCIL IT Assistant Chatbot",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded"
)

# -------------------------------
# Custom Styling
# -------------------------------
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

# -------------------------------
# Load Model (cached)
# -------------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# -------------------------------
# Sidebar
# -------------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-title">HCIL</div>', unsafe_allow_html=True)
    if st.button("⏪ Collapse Sidebar"):
        st.experimental_set_query_params(sidebar="collapsed")

    st.markdown("### 📂 Upload Knowledge Base")
    uploaded_file = st.file_uploader(
        "Upload Excel file with 'questions' and 'answers' columns",
        type=["xlsx"]
    )
    st.caption("🚗 Made with ❤️ for Honda Cars India")

# -------------------------------
# Session Initialization
# -------------------------------
for key in ['knowledge_base_loaded', 'messages', 'feedback_request']:
    if key not in st.session_state:
        st.session_state[key] = False if key == 'knowledge_base_loaded' else []

# -------------------------------
# Load Knowledge Base
# -------------------------------
if uploaded_file and not st.session_state.knowledge_base_loaded:
    try:
        df = pd.read_excel(uploaded_file)
        if {'questions', 'answers'}.issubset(df.columns):
            embeddings = model.encode(df['questions'].tolist())
            nn_model = NearestNeighbors(n_neighbors=1, metric='cosine').fit(np.array(embeddings))
            st.session_state.update({
                'df': df,
                'nn_model': nn_model,
                'knowledge_base_loaded': True,
                'messages': [],
                'feedback_request': False
            })
            st.success("✅ Knowledge base loaded! Let's go!")
            st.rerun()
        else:
            st.error("Missing 'questions' and/or 'answers' columns.")
    except Exception as e:
        st.error(f"Error loading file: {e}")

# -------------------------------
# Bot Logic
# -------------------------------
def get_bot_response(user_query):
    greetings = ["hi", "hello", "hey", "how are you"]
    exit_cmds = ["bye", "quit", "exit", "end"]

    query_lower = user_query.strip().lower()

    if any(greet in query_lower for greet in greetings):
        return "Hi! 👋 How can I help you today?"

    if any(exit_word in query_lower for exit_word in exit_cmds):
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Thank you for chatting. **Mata ne!** 👋 (see you later)"
        })
        st.session_state.feedback_request = False
        time.sleep(1)
        st.session_state.messages = []
        st.rerun()

    query_embed = model.encode([user_query])
    distances, indices = st.session_state.nn_model.kneighbors(query_embed)
    best_idx = indices[0][0]
    distance = distances[0][0]

    if distance > 0.4:  # Confidence threshold
        return "Hmm 🤔 I’m not sure I understand that. Could you rephrase your question?"
    else:
        return st.session_state.df.iloc[best_idx]['answers']

# -------------------------------
# App Title
# -------------------------------
st.markdown("""
<div class="persistent-header">
    HCIL IT Assistant Chatbot
</div>
""", unsafe_allow_html=True)

# -------------------------------
# Main Chat UI
# -------------------------------
if not st.session_state.knowledge_base_loaded:
    st.info("Please upload a valid knowledge base to begin chatting.")
else:
    if not st.session_state.messages:
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Hi there! I'm your IT helpdesk assistant. Ask me anything!"
        })

    for message in st.session_state.messages:
        bubble_class = "bot-message" if message["role"] == "assistant" else "user-message"
        st.markdown(f"""
            <div class="message-container {bubble_class}">
                <div class="bubble">{message['content']}</div>
            </div>
        """, unsafe_allow_html=True)

    # Feedback buttons
    if st.session_state.get('feedback_request'):
        col1, col2 = st.columns(2)
        with col1:
            if st.button("👍"):
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "Glad to help! 😊 Ask me more or type 'bye' to exit."
                })
                st.session_state.feedback_request = False
                st.rerun()
        with col2:
            if st.button("👎"):
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "Sorry about that. 😓 Try rephrasing your question!"
                })
                st.session_state.feedback_request = False
                st.rerun()

    # Chat Input
    if user_input := st.chat_input("Ask your IT question..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        response = get_bot_response(user_input)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.feedback_request = True
        st.rerun()
