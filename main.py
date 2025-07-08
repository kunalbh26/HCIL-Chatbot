import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import time

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="HCIL IT Assistant Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------------
# Custom CSS Loader
# -------------------------------
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

# -------------------------------
# Model Loader (Cached)
# -------------------------------
@st.cache_resource
def load_sentence_transformer():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_sentence_transformer()

# -------------------------------
# Helper Functions
# -------------------------------
def get_bot_response(user_query, df, nn_model, model):
    greetings = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"]
    how_are_you = ["how are you", "how are you doing", "how's it going"]
    farewells = ["bye", "end", "quit", "exit"]

    query_lower = user_query.lower().strip()
    # Greeting logic
    if any(greet in query_lower for greet in greetings):
        return "Hello! 👋 How can I assist you with IT today?"
    if any(phrase in query_lower for phrase in how_are_you):
        return "I'm just a bot, but I'm always ready to help you! 😊"

    # Farewell logic handled in chat input block

    # Knowledge base logic
    query_embed = model.encode([user_query])
    distances, indices = nn_model.kneighbors(query_embed)
    best_idx = indices[0][0]
    best_distance = distances[0][0]

    # Threshold for unknown answers (tune as needed)
    if best_distance > 0.35:
        return "I'm not sure about that. Could you please rephrase your question?"

    return df.iloc[best_idx]['answers']

def reset_chat():
    st.session_state.messages = []
    st.session_state.feedback_request = False

# -------------------------------
# Session State Initialization
# -------------------------------
if 'knowledge_base_loaded' not in st.session_state:
    st.session_state['knowledge_base_loaded'] = False
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'feedback_request' not in st.session_state:
    st.session_state.feedback_request = False
if 'sidebar_collapsed' not in st.session_state:
    st.session_state.sidebar_collapsed = False

# -------------------------------
# Collapsible Sidebar
# -------------------------------
def sidebar_toggle():
    st.session_state.sidebar_collapsed = not st.session_state.sidebar_collapsed

with st.sidebar:
    # Sidebar toggle button
    toggle_label = "⏪" if not st.session_state.sidebar_collapsed else "⏩"
    st.button(toggle_label, key="sidebar_toggle", on_click=sidebar_toggle)
    # HCIL branding
    st.markdown(
        '<div class="sidebar-title">HCIL</div>',
        unsafe_allow_html=True
    )
    if not st.session_state.sidebar_collapsed:
        with st.expander("📂 Upload Knowledge Base", expanded=True):
            uploaded_file = st.file_uploader(
                "Upload an Excel File",
                type=["xlsx"],
                help="Upload an Excel file with 'questions' and 'answers' columns."
            )
        st.caption("Built with ❤️ for HCIL")
    else:
        st.write("")  # Minimal sidebar when collapsed

# -------------------------------
# Load Knowledge Base
# -------------------------------
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None

if 'uploaded_file' in st.session_state and st.session_state.uploaded_file is not None:
    uploaded_file = st.session_state.uploaded_file
else:
    uploaded_file = st.session_state.get('uploaded_file', None)

if not st.session_state.sidebar_collapsed:
    if uploaded_file is None:
        uploaded_file = st.sidebar.file_uploader(
            "Upload an Excel File",
            type=["xlsx"],
            key="main_file_uploader",
            help="Upload an Excel file with 'questions' and 'answers' columns."
        )
        if uploaded_file:
            st.session_state.uploaded_file = uploaded_file

if uploaded_file is not None and not st.session_state.knowledge_base_loaded:
    with st.spinner("🚀 Initializing bot..."):
        try:
            df = pd.read_excel(uploaded_file)
            required_columns = {'questions', 'answers'}
            if not required_columns.issubset(df.columns):
                st.error("❌ Error: Missing required columns in Excel file. Please ensure 'questions' and 'answers' columns exist.")
            else:
                st.session_state.df = df
                embeddings = model.encode(df['questions'].tolist())
                nn_model = NearestNeighbors(n_neighbors=1, metric='cosine', algorithm='brute')
                nn_model.fit(np.array(embeddings))
                st.session_state.nn_model = nn_model
                st.session_state.knowledge_base_loaded = True
                reset_chat()
                st.success("✅ Knowledge base loaded! The bot is ready.")
                time.sleep(1)
                st.rerun()
        except Exception as e:
            st.error(f"❌ An error occurred while processing the file: {e}")

# -------------------------------
# Main Chat Interface
# -------------------------------
st.markdown(
    '''
    <div class="main-title">HCIL IT Assistant Chatbot</div>
    ''',
    unsafe_allow_html=True
)

if not st.session_state.knowledge_base_loaded:
    st.info("👋 Please upload a knowledge base file in the sidebar to start the chat.")
else:
    chat_container = st.container()
    with chat_container:
        if not st.session_state.messages:
            st.session_state.messages.append({
                "role": "assistant",
                "content": "Hi there! I'm your creative IT Assistant. How can I help you today?"
            })

        for message in st.session_state.messages:
            role = message["role"]
            content = message["content"]
            if role == "assistant":
                st.markdown(
                    f'''
                    <div class="message-container bot-message">
                        <div class="bubble bot-bubble">{content}</div>
                    </div>
                    ''',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'''
                    <div class="message-container user-message">
                        <div class="bubble user-bubble">{content}</div>
                    </div>
                    ''',
                    unsafe_allow_html=True
                )

        # Feedback Buttons
        if st.session_state.get('feedback_request'):
            col1, col2, _ = st.columns([1, 1, 5])
            with col1:
                if st.button("👍", key="thumbs_up", help="This was helpful!"):
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "Thank you! Please let me know if you have any other query or type 'bye' to end the conversation."
                    })
                    st.session_state.feedback_request = False
                    st.rerun()
            with col2:
                if st.button("👎", key="thumbs_down", help="This was not helpful."):
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "I'm sorry! Please try rephrasing your question or ask something else."
                    })
                    st.session_state.feedback_request = False
                    st.rerun()

        # Chat Input
        prompt = st.chat_input("Ask me an IT question...")
        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            prompt_lower = prompt.lower().strip()
            if prompt_lower in ["bye", "end", "quit", "exit"]:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "Thank you for chatting, <b>Mata ne!</b> (see you later) 👋"
                })
                st.session_state.feedback_request = False
                time.sleep(1)
                reset_chat()
                st.rerun()
            else:
                bot_response = get_bot_response(prompt, st.session_state.df, st.session_state.nn_model, model)
                st.session_state.messages.append({"role": "assistant", "content": bot_response})
                st.session_state.feedback_request = True
                st.rerun()
