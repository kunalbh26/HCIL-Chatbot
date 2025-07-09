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
        background: #ffffff;
    }
    .main {
    background: transparent;
    border-radius: 0px;
    padding: 3rem !important;
    max-width: 640px;
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
        margin-top: 0.5rem;
        display: flex;
        align-items: center;
        padding: 0.3rem 0.8rem;
    }
    .input-bar input {
        background: transparent;
        border: 2px solid #ff0000;
        color: #fff;
        width: 100%;
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
        background: #4c4c4c;
        color: #fff;
    }
    .sidebar-title {
        font-size: 4.5rem;
        color: #ff0000;
        font-weight: 900;
        text-align: center;
        margin: 0.5rem 0 1.5rem 0;
        letter-spacing: 0.05em;
        width: 100%;
        line-height: 1.2;
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
# Helper Functions
# -------------------------------
def is_gibberish(text):
    # Simple gibberish detection: very short, mostly non-words, or non-ASCII, or repeated chars
    text = text.strip()
    if len(text) < 2:
        return True
    if re.fullmatch(r'[^\w\s]+', text):
        return True
    if len(set(text)) < 3:
        return True
    # If more than 50% of words are not alphabetic, likely gibberish
    words = text.split()
    if len(words) > 0 and sum(1 for w in words if not w.isalpha())/len(words) > 0.5:
        return True
    return False

def is_greeting(text):
    greetings = [
        "hello", "hi", "hey", "greetings", "good morning", "good afternoon", "good evening",
        "how are you", "what's up", "sup", "thank you", "thanks", "bye", "goodbye"
    ]
    text = text.lower()
    for greet in greetings:
        if fuzz.partial_ratio(greet, text) > 80:
            return greet
    return None

def get_greeting_response(greet):
    responses = {
        "hello": "Hello! 👋 How can I help you today?",
        "hi": "Hi there! How can I assist you?",
        "hey": "Hey! How can I help you?",
        "greetings": "Greetings! How can I help you?",
        "good morning": "Good morning! ☀️ How can I help?",
        "good afternoon": "Good afternoon! How can I help?",
        "good evening": "Good evening! How can I help?",
        "how are you": "I'm just a bot, but I'm here to help you! 😊",
        "what's up": "I'm here to help with your IT queries!",
        "sup": "All good! How can I assist you?",
        "thank you": "You're welcome! Let me know if you have more questions.",
        "thanks": "You're welcome!",
        "bye": "Thank you for chatting, **Mata Ne!** (see you later) 👋",
        "goodbye": "Thank you for chatting, **Mata Ne!** (see you later) 👋"
    }
    return responses.get(greet, "Hello! How can I help you?")

def get_bot_response(user_query, df, nn_model, model):
    # If gibberish, ask to rephrase
    if is_gibberish(user_query):
        return "I'm sorry, I couldn't understand that. Could you please rephrase your question?"
    # If greeting, respond accordingly
    greet = is_greeting(user_query)
    if greet:
        return get_greeting_response(greet)
    # Fuzzy match to questions for spelling/grammar errors
    questions = df['questions'].tolist()
    best_match, score = process.extractOne(user_query, questions, scorer=fuzz.token_sort_ratio)
    if score > 70:
        idx = questions.index(best_match)
        return df.iloc[idx]['answers']
    # Otherwise, use embedding similarity
    query_embed = model.encode([user_query])
    distances, indices = nn_model.kneighbors(query_embed)
    best_idx = indices[0][0]
    # If similarity is poor, ask to rephrase
    if distances[0][0] > 0.45:
        return "I'm sorry, I couldn't understand that. Could you please rephrase your question?"
    response = df.iloc[best_idx]['answers']
    return response

def render_chat(messages):
    for message in messages:
        if message["role"] == "user":
            st.markdown(
                f"""
                <div class="user-row">
                    <div class="chat-bubble user-bubble">{message['content']}</div>
                    <div class="avatar" style="margin-left:8px;">🧑‍💻</div>
                </div>
                """, unsafe_allow_html=True
            )
        else: # bot
            st.markdown(
                f"""
                <div class="bot-row">
                    <div class="avatar" style="margin-right:8px;">🤖</div>
                    <div class="chat-bubble bot-bubble">{message['content']}</div>
                </div>
                """, unsafe_allow_html=True
            )

def show_typing():
    st.markdown("""
        <div class="typing-indicator">
            <div class="avatar" style="margin-right:8px;">🤖</div>
            <div class="typing-dots">
                <span></span><span></span><span></span>
            </div>
        </div>
    """, unsafe_allow_html=True)

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
# Main Application Logic
# -------------------------------

# Initialize session state variables at the very top, before any usage
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
if 'chat_reset_time' not in st.session_state:
    st.session_state['chat_reset_time'] = None


# Place this at the very start of the main application logic
if st.session_state.chat_reset_time:
    if time.time() - st.session_state.chat_reset_time > 2:
        st.session_state.messages = []
        st.session_state.chat_ended = False
        st.session_state.feedback_request = False
        st.session_state.show_typing = False
        st.session_state.chat_reset_time = None
        st.experimental_rerun()


st.markdown("<h1 style='color:white; text-align:center; margin-top: -10px;'>🤖 HCIL IT Helpdesk Chatbot</h1>", unsafe_allow_html=True)
st.markdown('<div class="main">', unsafe_allow_html=True)

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
        "content": "👋 <b><span style='font-size:1.2em;color:#ffff;'>Konichiwa! </span></b> How can I help you today?"
    })

# Main chat interface logic
if st.session_state.knowledge_base_loaded:
    render_chat(st.session_state.messages)

    # Typing Animation (shows only if last message is from user and waiting for bot)
    if st.session_state.get("show_typing", False):
        show_typing()

    # Quick Reply Chips
    st.markdown('<div style="margin-bottom:1rem;">', unsafe_allow_html=True)
    for reply in st.session_state.quick_replies:
        if st.button(reply, key=f"quick_{reply}"):
            st.session_state.messages.append({
                "role": "user",
                "content": reply
            })
            st.session_state.show_typing = True
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # Feedback Reactions (above input bar)
    if st.session_state.feedback_request:
        st.markdown("#### Was this helpful?")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button("👍", use_container_width=True):
                st.session_state.messages.append({
                    "role": "bot",
                    "content": "Great! Let me know if there is something else that I can help you with."
                })
                st.session_state.feedback_request = False
                st.rerun()
        with col2:
            if st.button("👎", use_container_width=True):
                st.session_state.messages.append({
                    "role": "bot",
                    "content": "I apologize. Could you please rephrase your question?"
                })
                st.session_state.feedback_request = False
                st.rerun()
        with col3:
            if st.button("🤔", use_container_width=True):
                st.session_state.messages.append({
                    "role": "bot",
                    "content": "I'm here to help! Feel free to ask another question."
                })
                st.session_state.feedback_request = False
                st.rerun()
        with col4:
            if st.button("❤️", use_container_width=True):
                st.session_state.messages.append({
                    "role": "bot",
                    "content": "Thank you for your feedback! 😊"
                })
                st.session_state.feedback_request = False
                st.rerun()

# Input Bar (smaller send button to right)
    with st.form("chat_input_form", clear_on_submit=True):
        col1, col2 = st.columns([6, 1])
        with col1:
            user_input = st.text_input("", placeholder="Type here...", key="input_bar")
        with col2:
            send_clicked = st.form_submit_button("Send", use_container_width=True)

if send_clicked and user_input.strip():
    user_input_clean = user_input.lower().strip()

    if user_input_clean in ["bye", "end", "quit"]:
        # Show user's message first
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })
        # Then bot's reply
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
            
            # --- End of Corrected Logic ---

    # Bot response logic (after user or quick reply)
    if st.session_state.get("show_typing", False):
        time.sleep(1.2)
        user_message = None
        for msg in reversed(st.session_state.messages):
            if msg["role"] == "user":
                user_message = msg["content"]
                break
        if user_message:
            bot_response = get_bot_response(user_message, st.session_state.df, st.session_state.nn_model, model)
            st.session_state.messages.append({
                "role": "bot",
                "content": bot_response
            })
            st.session_state.feedback_request = True
            st.session_state.show_typing = False
            st.rerun()

else:
    st.info("⬆️ Please upload a knowledge base file in the sidebar to begin the chat.")


st.markdown('</div>', unsafe_allow_html=True)
