import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import time
from datetime import datetime

# -------------------------------
# Custom CSS for Modern UI
# -------------------------------
st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) fixed;
    }
    .main {
        background: rgba(255,255,255,0.8) !important;
        border-radius: 18px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        padding: 2rem !important;
        max-width: 540px;
        margin: 2.5rem auto;
    }
    .chat-bubble {
        padding: 1rem 1.5rem;
        border-radius: 20px;
        margin-bottom: 8px;
        max-width: 75%;
        animation: fadeInUp 0.3s;
        position: relative;
        word-break: break-word;
        font-size: 1.08rem;
    }
    .user-bubble {
        background: linear-gradient(90deg, #43e97b 0%, #38f9d7 100%);
        color: #222;
        align-self: flex-end;
        margin-left: auto;
        margin-right: 0;
    }
    .bot-bubble {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: #fff;
        align-self: flex-start;
        margin-right: auto;
        margin-left: 0;
    }
    .avatar {
        width: 38px; height: 38px; border-radius: 50%; margin-right: 8px;
        display: inline-block; vertical-align: middle;
        background: #fff;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
        font-size: 1.7rem;
        text-align: center;
        line-height: 38px;
    }
    .timestamp {
        font-size: 0.78rem;
        color: #888;
        margin-top: 2px;
        margin-bottom: 8px;
        margin-left: 46px;
    }
    .input-bar {
        background: rgba(255,255,255,0.7);
        border-radius: 20px;
        padding: 0.7rem 1.2rem;
        box-shadow: 0 2px 8px rgba(31,38,135,0.07);
        margin-top: 1.5rem;
    }
    .quick-reply {
        display: inline-block;
        background: #e0e7ff;
        color: #4f46e5;
        border-radius: 18px;
        padding: 0.5rem 1.1rem;
        margin: 0.15rem;
        cursor: pointer;
        font-size: 0.98rem;
        transition: background 0.2s;
    }
    .quick-reply:hover {
        background: #c7d2fe;
    }
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px);}
        to { opacity: 1; transform: translateY(0);}
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="HCIL IT Helpdesk Chat-Bot",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="auto"
)

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
    query_embed = model.encode([user_query])
    distances, indices = nn_model.kneighbors(query_embed)
    best_idx = indices[0][0]
    response = df.iloc[best_idx]['answers']
    return response

def show_typing():
    st.markdown("""
        <div style="display:flex; align-items:center;">
            <div class="avatar">🤖</div>
            <span class="bot-bubble">
                <span class="typing">
                    <span>.</span><span>.</span><span>.</span>
                </span>
            </span>
        </div>
        <style>
        .typing span {
            animation: blink 1s infinite;
        }
        .typing span:nth-child(2) { animation-delay: 0.2s; }
        .typing span:nth-child(3) { animation-delay: 0.4s; }
        @keyframes blink {
            0%, 80%, 100% { opacity: 0; }
            40% { opacity: 1; }
        }
        </style>
    """, unsafe_allow_html=True)

def render_chat(messages):
    for message in messages:
        avatar = "🧑" if message["role"] == "user" else "🤖"
        bubble_class = "user-bubble" if message["role"] == "user" else "bot-bubble"
        align = "flex-end" if message["role"] == "user" else "flex-start"
        st.markdown(
            f"""
            <div style="display:flex; align-items:flex-end; justify-content:{align}; margin-bottom:0.2rem;">
                <div class="avatar">{avatar}</div>
                <div class="chat-bubble {bubble_class}">{message['content']}</div>
            </div>
            <div class="timestamp">{message['timestamp']}</div>
            """, unsafe_allow_html=True
        )

def get_time():
    return datetime.now().strftime("%I:%M %p")

# -------------------------------
# Sidebar Configuration
# -------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    with st.expander("📂 Knowledge Base Setup", expanded=False):
        uploaded_file = st.file_uploader(
            "Upload knowledge base!",
            type=["xlsx"],
            help="Upload an Excel file with 'questions', 'answers', 'categories', and 'tags' columns."
        )
    st.info("Say 'bye', 'quit', or 'end' to close our chat.")
    theme = st.radio("Theme", ["🌞 Light", "🌙 Dark"], horizontal=True)
    if theme == "🌙 Dark":
        st.markdown("""
            <style>
            body { background: linear-gradient(135deg, #232526 0%, #414345 100%) fixed; }
            .main { background: rgba(30,30,30,0.95) !important; color: #fff; }
            .user-bubble { background: linear-gradient(90deg, #f7971e 0%, #ffd200 100%); color: #222; }
            .bot-bubble { background: linear-gradient(90deg, #232526 0%, #414345 100%); color: #fff; }
            .quick-reply { background: #232526; color: #ffd200; }
            .quick-reply:hover { background: #414345; }
            </style>
        """, unsafe_allow_html=True)

# -------------------------------
# Main Application Logic
# -------------------------------
st.markdown('<div class="main">', unsafe_allow_html=True)
st.title("🤖 HCIL IT Helpdesk Chatbot")

# Initialize session state variables
if 'knowledge_base_loaded' not in st.session_state:
    st.session_state['knowledge_base_loaded'] = False
if 'messages' not in st.session_state:
    st.session_state['messages'] = []
if 'chat_ended' not in st.session_state:
    st.session_state['chat_ended'] = False
if 'feedback_request' not in st.session_state:
    st.session_state['feedback_request'] = False
if 'quick_replies' not in st.session_state:
    st.session_state['quick_replies'] = ["Reset password", "VPN issues", "Software install"]

# Handle knowledge base upload
if uploaded_file is not None and not st.session_state.knowledge_base_loaded:
    with st.spinner("🚀 Launching the bot... Please wait."):
        try:
            df = pd.read_excel(uploaded_file)
            required_columns = {'questions', 'answers', 'categories', 'tags'}
            if not required_columns.issubset(df.columns):
                st.error("❌ **Error:** The file is missing required columns: `questions`, `answers`, `categories`, `tags`.")
            else:
                st.session_state.df = df
                embeddings = model.encode(df['questions'].tolist())
                nn_model = NearestNeighbors(n_neighbors=1, metric='cosine')
                nn_model.fit(np.array(embeddings))
                st.session_state.nn_model = nn_model
                st.session_state.knowledge_base_loaded = True
                st.session_state.messages = []
                st.session_state.chat_ended = False
                st.success("✅ Knowledge base loaded! The bot is ready.")
                time.sleep(1)
                st.rerun()
        except Exception as e:
            st.error(f"❌ An error occurred: {e}")

# Display initial greeting if chat hasn't started
if not st.session_state.messages:
    st.session_state.messages.append({
        "role": "bot",
        "content": "👋 **Konichiwa!** How can I help you today?",
        "timestamp": get_time()
    })

# Main chat interface logic
if st.session_state.knowledge_base_loaded:
    render_chat(st.session_state.messages)

    # Quick Reply Chips
    st.markdown('<div style="margin-bottom:1rem;">', unsafe_allow_html=True)
    for reply in st.session_state.quick_replies:
        if st.button(reply, key=f"quick_{reply}"):
            st.session_state.messages.append({
                "role": "user",
                "content": reply,
                "timestamp": get_time()
            })
            show_typing()
            bot_response = get_bot_response(reply, st.session_state.df, st.session_state.nn_model, model)
            st.session_state.messages.append({
                "role": "bot",
                "content": bot_response,
                "timestamp": get_time()
            })
            st.session_state.feedback_request = True
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # Chat Input Bar
    with st.form("chat_input_form", clear_on_submit=True):
        user_input = st.text_input("", placeholder="Type your IT question...", key="input_bar")
        submit = st.form_submit_button("Send", use_container_width=True)
        if submit and user_input.strip():
            st.session_state.messages.append({
                "role": "user",
                "content": user_input,
                "timestamp": get_time()
            })
            if user_input.lower() in ["bye", "end", "quit"]:
                st.session_state.messages.append({
                    "role": "bot",
                    "content": "Thank you for chatting, **Mata Ne!** (see you later) 👋",
                    "timestamp": get_time()
                })
                st.session_state.chat_ended = True
                st.session_state.feedback_request = False
                time.sleep(1)
                st.session_state.messages = []
                st.rerun()
            else:
                show_typing()
                bot_response = get_bot_response(user_input, st.session_state.df, st.session_state.nn_model, model)
                st.session_state.messages.append({
                    "role": "bot",
                    "content": bot_response,
                    "timestamp": get_time()
                })
                st.session_state.feedback_request = True
                st.rerun()

    # Feedback Reactions
    if st.session_state.feedback_request:
        st.markdown("#### Was this helpful?")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button("👍", use_container_width=True):
                st.session_state.messages.append({
                    "role": "bot",
                    "content": "Great! Let me know if there is something else that I can help you with.",
                    "timestamp": get_time()
                })
                st.session_state.feedback_request = False
                st.rerun()
        with col2:
            if st.button("👎", use_container_width=True):
                st.session_state.messages.append({
                    "role": "bot",
                    "content": "I apologize. Could you please rephrase your question?",
                    "timestamp": get_time()
                })
                st.session_state.feedback_request = False
                st.rerun()
        with col3:
            if st.button("🤔", use_container_width=True):
                st.session_state.messages.append({
                    "role": "bot",
                    "content": "I'm here to help! Feel free to ask another question.",
                    "timestamp": get_time()
                })
                st.session_state.feedback_request = False
                st.rerun()
        with col4:
            if st.button("❤️", use_container_width=True):
                st.session_state.messages.append({
                    "role": "bot",
                    "content": "Thank you for your feedback! 😊",
                    "timestamp": get_time()
                })
                st.session_state.feedback_request = False
                st.rerun()
else:
    st.info("⬆️ Please upload a knowledge base file in the sidebar to begin the chat.")

st.markdown('</div>', unsafe_allow_html=True)
