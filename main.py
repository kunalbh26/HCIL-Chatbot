import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import time
import base64

# --------------------------------
# Page Configuration
# --------------------------------
st.set_page_config(
    page_title="HCIL IT Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------
# Load Custom CSS
# --------------------------------
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

# --------------------------------
# Load Model (Cached)
# --------------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# --------------------------------
# Helper Functions
# --------------------------------
def get_bot_response(user_query, df, nn_model, model):
    # Handle greetings and exit
    greetings = {"hi": "Hello there! 👋 How can I help you today?", 
                 "hello": "Hi! Need help with something?", 
                 "how are you": "I'm here and ready to assist you! 💻", 
                 "hey": "Hey! How can I support you today?"}

    exits = ["bye", "exit", "quit", "end"]

    user_query_lower = user_query.lower().strip()

    if user_query_lower in greetings:
        return greetings[user_query_lower]

    if user_query_lower in exits:
        st.session_state.reset = True
        return "Thank you for chatting with me! Have a great day! 👋"

    with st.spinner("Thinking... 🤔"):
        query_embed = model.encode([user_query])
        distances, indices = nn_model.kneighbors(query_embed)
        best_idx = indices[0][0]
        response = df.iloc[best_idx]['answers']
        time.sleep(1)
    return response

# --------------------------------
# Sidebar Setup
# --------------------------------
with st.sidebar:
    st.markdown("""
        <div class="sidebar-header">HCIL</div>
        <div class="sidebar-toggle" onclick="toggleSidebar()">⬅️</div>
    """, unsafe_allow_html=True)

    with st.expander("📂 Upload Knowledge Base", expanded=True):
        uploaded_file = st.file_uploader(
            "Upload an Excel File",
            type=["xlsx"],
            help="Ensure file has 'questions' and 'answers' columns."
        )

    st.caption("Crafted with ❤️ for HCIL")

# --------------------------------
# Initialize Session State
# --------------------------------
for key in ['knowledge_base_loaded', 'messages', 'feedback_request', 'reset']:
    if key not in st.session_state:
        st.session_state[key] = False if key == 'knowledge_base_loaded' else [] if key == 'messages' else False

# --------------------------------
# Load Knowledge Base
# --------------------------------
if uploaded_file and not st.session_state.knowledge_base_loaded:
    with st.spinner("🚀 Loading knowledge base..."):
        try:
            df = pd.read_excel(uploaded_file)
            if not {'questions', 'answers'}.issubset(df.columns):
                st.error("❌ Excel must have 'questions' and 'answers' columns.")
            else:
                st.session_state.df = df
                embeddings = model.encode(df['questions'].tolist())
                nn_model = NearestNeighbors(n_neighbors=1, metric='cosine', algorithm='brute')
                nn_model.fit(np.array(embeddings))

                st.session_state.nn_model = nn_model
                st.session_state.knowledge_base_loaded = True
                st.success("✅ Knowledge base loaded!")
                time.sleep(1)
                st.rerun()
        except Exception as e:
            st.error(f"❌ Failed to load knowledge base: {e}")

# --------------------------------
# Main Chat UI
# --------------------------------
if not st.session_state.knowledge_base_loaded:
    st.markdown("""
        <div class="center-title">
            HCIL IT Assistant Chatbot
        </div>
    """, unsafe_allow_html=True)
    st.info("Please upload a knowledge base to begin.")
else:
    st.markdown("""
        <div class="chat-header">
            <div class="avatar-header">🤖</div>
            <div class="header-info">
                <span class="header-title">HCIL IT Helpdesk</span>
                <span class="header-status">Online</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    if st.session_state.reset:
        st.session_state.messages = []
        st.session_state.feedback_request = False
        st.session_state.reset = False

    if not st.session_state.messages:
        st.session_state.messages.append({"role": "assistant", "content": "Hi! I'm your IT assistant. Ask me anything."})

    for message in st.session_state.messages:
        role, content = message["role"], message["content"]
        bubble_class = "bot-message" if role == "assistant" else "user-message"
        st.markdown(f"""
        <div class="message-container {bubble_class}">
            <div class="bubble">{content}</div>
        </div>
        """, unsafe_allow_html=True)

    if st.session_state.get('feedback_request'):
        col1, col2, _ = st.columns([1, 1, 5])
        with col1:
            if st.button("👍"):
                st.session_state.messages.append({"role": "assistant", "content": "Glad I could help! Ask me anything else, or type 'bye' to end."})
                st.session_state.feedback_request = False
                st.rerun()
        with col2:
            if st.button("👎"):
                st.session_state.messages.append({"role": "assistant", "content": "Sorry! Try rephrasing your question."})
                st.session_state.feedback_request = False
                st.rerun()

    if prompt := st.chat_input("Ask me an IT question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        bot_response = get_bot_response(prompt, st.session_state.df, st.session_state.nn_model, model)
        st.session_state.messages.append({"role": "assistant", "content": bot_response})
        st.session_state.feedback_request = True
        st.rerun()
