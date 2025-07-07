import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import time
from streamlit.components.v1 import html

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="HCIL IT Helpdesk Chat-Bot",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="auto",
)

# -------------------------------
# Custom Styling
# -------------------------------
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Background animation (optional)
html("""
<style>
body {
  background: linear-gradient(135deg, #f3f4f6, #dbeafe);
}
</style>
""", height=0)

# -------------------------------
# Model Loading (Cached)
# -------------------------------
@st.cache_resource
def load_sentence_transformer():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_sentence_transformer()

# -------------------------------
# Helper Functions
# -------------------------------
def get_bot_response(user_query, df, nn_model, model):
    with st.spinner("Thinking... 🤔"):
        query_embed = model.encode([user_query])
        distances, indices = nn_model.kneighbors(query_embed)
        best_idx = indices[0][0]
        response = df.iloc[best_idx]['answers']
        time.sleep(1)
    return response

# -------------------------------
# Sidebar Configuration
# -------------------------------
with st.sidebar:
    st.image("https://i.imgur.com/OyQdV3P.png", width=100)
    st.header("⚙️ Bot Configuration")
    with st.expander("📂 Upload Knowledge Base", expanded=False):
        uploaded_file = st.file_uploader(
            "Upload Excel File",
            type=["xlsx"],
            help="Upload an Excel file with 'questions', 'answers', 'categories', and 'tags' columns."
        )
    st.caption("Say 'bye', 'quit', or 'end' to close the chat.")

# -------------------------------
# App Title
# -------------------------------
st.markdown("""
<div class="header-text">
    🤖 <b>Welcome to HCIL's Smart IT Helpdesk</b><br>
    <small>Ask me anything about internal IT issues!</small>
</div>
""", unsafe_allow_html=True)

# -------------------------------
# Session State Initialization
# -------------------------------
if 'knowledge_base_loaded' not in st.session_state:
    st.session_state['knowledge_base_loaded'] = False
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'chat_ended' not in st.session_state:
    st.session_state.chat_ended = False

# -------------------------------
# Load Knowledge Base
# -------------------------------
if uploaded_file is not None and not st.session_state.knowledge_base_loaded:
    with st.spinner("🚀 Initializing bot..."):
        try:
            df = pd.read_excel(uploaded_file)
            required_columns = {'questions', 'answers', 'categories', 'tags'}
            if not required_columns.issubset(df.columns):
                st.error("❌ Missing columns in Excel: 'questions', 'answers', 'categories', 'tags'.")
            else:
                st.session_state.df = df
                embeddings = model.encode(df['questions'].tolist())
                nn_model = NearestNeighbors(n_neighbors=1, metric='cosine')
                nn_model.fit(np.array(embeddings))
                st.session_state.nn_model = nn_model
                st.session_state.knowledge_base_loaded = True
                st.session_state.messages = []
                st.session_state.chat_ended = False
                st.success("✅ Bot is ready to chat!")
                time.sleep(1)
                st.rerun()
        except Exception as e:
            st.error(f"❌ An error occurred: {e}")

# -------------------------------
# Chat Interface
# -------------------------------
if st.session_state.knowledge_base_loaded:
    if not st.session_state.messages:
        st.session_state.messages.append({"role": "assistant", "content": "📢 Hi! I'm your IT Helpdesk Assistant. How can I assist you today?"})

    for message in st.session_state.messages:
        role = message["role"]
        bubble_class = "user-bubble" if role == "user" else "bot-bubble"
        with st.container():
            st.markdown(f"<div class='{bubble_class}'>{message['content']}</div>", unsafe_allow_html=True)

    if prompt := st.chat_input("Type your question here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        if prompt.lower().strip() in ["bye", "end", "quit"]:
            farewell = "Thank you for chatting. **Mata Ne!** 👋"
            st.session_state.messages.append({"role": "assistant", "content": farewell})
            time.sleep(2)
            st.session_state.messages = []
            st.session_state.feedback_request = False
            st.rerun()
        else:
            bot_response = get_bot_response(prompt, st.session_state.df, st.session_state.nn_model, model)
            st.session_state.messages.append({"role": "assistant", "content": bot_response})
            st.session_state.feedback_request = True
            st.rerun()

    if st.session_state.get('feedback_request'):
        col1, col2 = st.columns(2)
        with col1:
            if st.button("👍 Helpful"):
                msg = "Awesome! Ask me anything else."
                st.session_state.messages.append({"role": "assistant", "content": msg})
                st.session_state.feedback_request = False
                st.rerun()
        with col2:
            if st.button("👎 Not Helpful"):
                msg = "Sorry about that! Can you rephrase your query?"
                st.session_state.messages.append({"role": "assistant", "content": msg})
                st.session_state.feedback_request = False
                st.rerun()
else:
    st.info("⬆️ Upload a knowledge base file to get started.")
