import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import time
import re
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

# -------------------------------
# Custom CSS for Red-Black-White Theme
# -------------------------------
st.markdown("""
    <style>
    body {
        background: #111;
    }
    .main {
        background: #181818 !important;
        border-radius: 18px;
        box-shadow: 0 8px 32px 0 rgba(255,0,0,0.07);
        padding: 2rem !important;
        max-width: 540px;
        margin: 2.5rem auto;
    }
    .chat-bubble {
        padding: 1rem 1.5rem;
        border-radius: 20px;
        margin-bottom: 14px;
        max-width: 75%;
        animation: fadeInUp 0.3s;
        position: relative;
        word-break: break-word;
        font-size: 1.08rem;
        display: flex;
        align-items: center;
    }
    .user-bubble {
        background: #fff;
        color: #111;
        align-self: flex-end;
        margin-left: auto;
        margin-right: 0;
        border: 1.5px solid #e53935;
    }
    .bot-bubble {
        background: linear-gradient(90deg, #e53935 0%, #b71c1c 100%);
        color: #fff;
        align-self: flex-start;
        margin-right: auto;
        margin-left: 0;
        border: 1.5px solid #fff;
    }
    .avatar {
        width: 38px; height: 38px; border-radius: 75%; margin: 0 10px;
        background: #3d3d3d;
        box-shadow: 0 2px 8px rgba(229,57,53,0.12);
        font-size: 1.7rem;
        text-align: center;
        line-height: 38px;
        border: 2px solid #ff0000;
        display: flex; align-items: center; justify-content: center;
    }
    .user-row {
        display: flex; flex-direction: row; align-items: flex-end; justify-content: flex-end;
    }
    .bot-row {
        display: flex; flex-direction: row; align-items: flex-end; justify-content: flex-start;
    }
    .input-bar {
        background: #222;
        border-radius: 20px;
        box-shadow: 0 2px 8px rgba(229,57,53,0.12);
        margin-top: 1.5rem;
        display: flex;
        align-items: center;
        padding: 0.3rem 0.8rem;
    }
    .input-bar input {
        background: transparent;
        border: 2px solid #ff0000;
        color: #fff;
        width: 75%;
        padding: 0.7rem 0.8rem;
        outline: none;
        font-size: 1rem;
    }
    .send-btn {
        background: linear-gradient(90deg, #e53935 0%, #b71c1c 100%);
        color: #fff;
        border: none;
        border-radius: 50%;
        width: 38px;
        height: 38px;
        font-size: 1.2rem;
        cursor: pointer;
        margin-left: 8px;
        transition: background 0.2s;
        display: flex; align-items: center; justify-content: center;
    }
    .send-btn:hover {
        background: #fff;
        color: #e53935;
        border: 1.5px solid #ff00;
    }
    .quick-reply {
        display: inline-block;
        background: #fff;
        color: #e53935;
        border-radius: 18px;
        padding: 0.5rem 1.1rem;
        margin: 0.15rem;
        cursor: pointer;
        font-size: 0.98rem;
        border: 1.5px solid #e53935;
        font-weight: 500;
        transition: background 0.2s, color 0.2s;
    }
    .quick-reply:hover {
        background: #e53935;
        color: #fff;
    }
    .sidebar-title {
        font-size: 3.5rem;
        color: #ff0000;
        font-weight: 900;
        text-align: center;
        margin: 0.5rem 0 1.5rem 0;
        letter-spacing: 0.05em;
        width: 100%;
        line-height: 1.1;
    }
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px);}
        to { opacity: 1; transform: translateY(0);}
    }
    .typing-indicator {
        display: flex; align-items: center; margin-bottom: 1.1rem;
    }
    .typing-dots span {
        height: 10px; width: 10px; margin: 0 2px;
        background: #e53935; border-radius: 25%; display: inline-block;
        animation: blink 1.2s infinite both;
    }
    .typing-dots span:nth-child(2) { animation-delay: 0.2s; }
    .typing-dots span:nth-child(3) { animation-delay: 0.4s; }
    @keyframes blink {
        0%, 80%, 100% { opacity: 0.2; }
        40% { opacity: 1; }
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
# Sidebar Configuration
# -------------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-title">HCIL</div>', unsafe_allow_html=True)
    with st.expander("📂 Knowledge Base Setup", expanded=False):
        uploaded_file = st.file_uploader(
            "Upload knowledge base!",
            type=["xlsx"],
            help="Upload an Excel file with 'questions', 'answers', 'categories', and 'tags' columns."
        )
    st.info("Say 'bye', 'quit', or 'end' to close our chat.")

# -------------------------------
# Main Container
# -------------------------------
st.markdown('<div class="main">', unsafe_allow_html=True)
st.markdown("<h1 style='color:white; text-align:center; margin-top: -10px;'>🤖 HCIL IT Helpdesk Chatbot</h1>", unsafe_allow_html=True)

# Your session state, logic, chat handling and UI rendering goes here (unchanged)

# -------------------------------
# Chat Input Bar (Fix applied)
# -------------------------------
with st.form("chat_input_form", clear_on_submit=True):
    col1, col2 = st.columns([8, 1])
    with col1:
        user_input = st.text_input("", placeholder="Type here...", key="input_bar")
    with col2:
        send_clicked = st.form_submit_button("Send", use_container_width=True)

    if send_clicked and user_input.strip():
        user_input_clean = user_input.lower().strip()

        if user_input_clean in ["bye", "end", "quit"]:
            st.session_state.messages.append({
                "role": "bot",
                "content": "Thank you for chatting, <b><span style='font-size:1.2em;color:#ffff;'>Mata Ne!</span></b> (see you later) 👋"
            })
            st.session_state.chat_ended = True
            st.session_state.feedback_request = False
            st.session_state.show_typing = False
            st.session_state.chat_reset_time = time.time()
            st.rerun()
        else:
            st.session_state.messages.append({
                "role": "user",
                "content": user_input
            })
            st.session_state.show_typing = True
            st.rerun()

st.markdown('</div>', unsafe_allow_html=True)
