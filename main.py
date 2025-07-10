import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import time
import re
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

# --- Configuration for Pre-loaded Knowledge Base ---
KNOWLEDGE_BASE_PATH = 'dataset.xlsx'

# -------------------------------
# Custom CSS for Red-Black-White Theme (Fixed Input Bar and Button Positioning)
# -------------------------------
st.markdown(f"""
<style>
/* Strict Background Color for the entire app */
html, body, .stApp {{
    background-color: #1F1F1F !important;
    color: white !important;
}}

/* Main Chat Area Background - Added more bottom padding */
.main .block-container {{
    padding-bottom: 10rem !important; /* Prevents content from being hidden by the fixed input bar */
}}


/* Sidebar styling */
.stSidebar > div:first-child {{
    background-color: #323232 !important;
    border-right: 2px solid white;
}}

/* FIXED: Start Chat Button - Proper Centering */
.start-chat-container {{
    display: flex;
    justify-content: center;
    align-items: center;
    width: 100%;
    margin: 3rem 0;
}}

.stButton > button {{
    background: linear-gradient(90deg, #e53935 0%, #b71c1c 100%) !important;
    color: #fff !important;
    border-radius: 25px !important;
    padding: 1.5rem 3rem !important;
    font-size: 1.4rem !important;
    border: 3px solid #fff !important;
    font-weight: bold !important;
    transition: transform 0.2s !important;
    min-width: 200px !important;
    margin: 0 auto !important;
    display: block !important;
}}

.stButton > button:hover {{
    transform: scale(1.08) !important;
    color: #fff !important;
    border: 3px solid #fff !important;
}}

/* NEW: Wrapper for the fixed input bar */
.fixed-input-wrapper {{
    position: fixed;
    bottom: 0;
    left: 0;
    width: 100%;
    background: #1F1F1F; /* Match app background */
    padding: 1rem;
    z-index: 100;
    border-top: 1px solid #444;
}}

/* Adjustments for the form inside the fixed wrapper */
.fixed-input-wrapper .stForm {{
    max-width: 640px; /* Aligns with the main content width */
    margin: 0 auto;
}}

.input-bar input {{
    background: #323232;
    border: 2px solid #ff0000;
    color: #fff;
    width: 100%;
    padding: 0.7rem 0.8rem;
    outline: none;
    font-size: 1rem;
    border-radius: 5px;
}}

/* Style the submit button to look like a send icon */
.fixed-input-wrapper .stButton[kind="form_submit"] button {{
    background: linear-gradient(90deg, #e53935 0%, #b71c1c 100%);
    color: #fff;
    border: none;
    border-radius: 5px;
    width: 100%;
    height: 46px; /* Match input field height */
    font-size: 1.5rem;
    cursor: pointer;
    transition: background 0.2s;
    display: flex;
    align-items: center;
    justify-content: center;
}}
.fixed-input-wrapper .stButton[kind="form_submit"] button:hover {{
    background: #fff;
    color: #e53935;
    border: 1.5px solid #e53935;
    transform: scale(1.05) !important;
}}


/* Chat bubbles and avatars */
.chat-bubble {{
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
}}

.user-bubble {{
    background: #fff;
    color: #111;
    align-self: flex-end;
    margin-left: auto;
    margin-right: 0;
    border: 1.5px solid #e53935;
}}

.bot-bubble {{
    background: linear-gradient(90deg, #e53935 0%, #b71c1c 100%);
    color: #fff;
    align-self: flex-start;
    margin-right: auto;
    margin-left: 0;
    border: 1.5px solid #fff;
}}

.avatar {{
    width: 38px;
    height: 38px;
    border-radius: 75%;
    margin: 0 10px;
    background: #3d3d3d;
    box-shadow: 0 2px 8px rgba(229,57,53,0.12);
    font-size: 1.7rem;
    text-align: center;
    line-height: 38px;
    border: 2px solid #ff0000;
    display: flex;
    align-items: center;
    justify-content: center;
}}

.user-row {{
    display: flex;
    flex-direction: row;
    align-items: flex-end;
    justify-content: flex-end;
}}

.bot-row {{
    display: flex;
    flex-direction: row;
    align-items: flex-end;
    justify-content: flex-start;
}}

/* Quick reply buttons */
.quick-reply {{
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
}}

.quick-reply:hover {{
    background: #e53935;
    color: #fff;
}}

/* Enhanced Sidebar Title */
.sidebar-title {{
    font-size: 5.5rem;
    color: #EE4B2B;
    font-weight: 900;
    text-align: center;
    margin: 0.5rem 0 1.5rem 0;
    letter-spacing: 0.05em;
    width: 100%;
    line-height: 1.2;
    animation: rotate3D 5s infinite linear;
    transform-style: preserve-3d;
    perspective: 800px;
    text-shadow:
        0 0 5px rgba(238, 75, 43, 0.5),
        0 0 10px rgba(238, 75, 43, 0.4),
        0 0 15px rgba(238, 75, 43, 0.3),
        1px 1px 2px rgba(0,0,0,0.8);
}}

@keyframes rotate3D {{
    0% {{ transform: rotateY(0deg) scale(1); }}
    25% {{ transform: rotateY(90deg) scale(1.05); }}
    50% {{ transform: rotateY(180deg) scale(1); }}
    75% {{ transform: rotateY(270deg) scale(1.05); }}
    100% {{ transform: rotateY(360deg) scale(1); }}
}}

/* Main Chatbot Title Enhancement */
.elegant-heading {{
    font-size: 5.0rem !important;
    font-weight: 800;
    text-align: center;
    margin-top: -5px;
    color: #ffffff;
    animation: fadeInUp 2.0s ease-out;
}}

@keyframes fadeInUp {{
    0% {{ opacity: 0; transform: translateY(20px); }}
    100% {{ opacity: 1; transform: translateY(0); }}
}}

/* Typing indicator */
.typing-indicator {{
    display: flex;
    align-items: center;
    margin-bottom: 1.1rem;
}}

.typing-dots span {{
    height: 10px;
    width: 10px;
    margin: 0 2px;
    background: #e53935;
    border-radius: 25%;
    display: inline-block;
    animation: blink 1.2s infinite both;
}}

.typing-dots span:nth-child(2) {{ animation-delay: 0.2s; }}
.typing-dots span:nth-child(3) {{ animation-delay: 0.4s; }}

@keyframes blink {{
    0%, 80%, 100% {{ opacity: 0.2; }}
    40% {{ opacity: 1; }}
}}
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

# --- Pre-load Knowledge Base ---
@st.cache_resource
def load_knowledge_base(path):
    try:
        df = pd.read_excel(path)
        required_columns = {'questions', 'answers', 'categories', 'tags'}
        if not required_columns.issubset(df.columns):
            st.error(f"❌ **Error:** The knowledge base file '{path}' is missing required columns: `questions`, `answers`, `categories`, `tags`.")
            st.stop()
        embeddings = model.encode(df['questions'].tolist())
        nn_model = NearestNeighbors(n_neighbors=1, metric='cosine')
        nn_model.fit(np.array(embeddings))
        return df, nn_model
    except FileNotFoundError:
        st.error(f"❌ **Error:** Knowledge base file not found at '{path}'. Please ensure it's in the correct directory.")
        st.stop()
    except Exception as e:
        st.error(f"❌ An error occurred while loading the knowledge base: {e}")
        st.stop()

# Load the knowledge base globally when the app starts
if 'df' not in st.session_state or 'nn_model' not in st.session_state:
    st.session_state.df, st.session_state.nn_model = load_knowledge_base(KNOWLEDGE_BASE_PATH)
    st.session_state.knowledge_base_loaded = True

# -------------------------------
# Helper Functions
# -------------------------------
def is_gibberish(text):
    text = text.strip()
    if len(text) < 2:
        return True
    if re.fullmatch(r'[^\w\s]+', text):
        return True
    if len(set(text)) < 3:
        return True
    words = text.split()
    if len(words) > 0 and sum(1 for w in words if not w.isalpha()) / len(words) > 0.5:
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
    if is_gibberish(user_query):
        return "I'm sorry, I couldn't understand that. Could you please rephrase your question?"

    greet = is_greeting(user_query)
    if greet:
        return get_greeting_response(greet)

    questions = df['questions'].tolist()
    best_match, score = process.extractOne(user_query, questions, scorer=fuzz.token_sort_ratio)

    if score > 70:
        idx = questions.index(best_match)
        return df.iloc[idx]['answers']

    query_embed = model.encode([user_query])
    distances, indices = nn_model.kneighbors(query_embed)
    best_idx = indices[0][0]

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
        else:
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
    st.info("Say 'bye', 'quit', or 'end' to close our chat.")

# -------------------------------
# Main Application Logic
# -------------------------------
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
if 'show_typing' not in st.session_state:
    st.session_state['show_typing'] = False
if 'chat_started' not in st.session_state:
    st.session_state['chat_started'] = False
if 'show_quick_replies' not in st.session_state:
    st.session_state['show_quick_replies'] = False

st.markdown("""
<h1 class='elegant-heading'>🤖 HCIL IT Helpdesk Chatbot</h1>
""", unsafe_allow_html=True)

# Start Chat Button Logic
if not st.session_state.chat_started:
    st.markdown('<div class="start-chat-container">', unsafe_allow_html=True)
    if st.button("Start Chat", key="start_chat"):
        st.session_state.chat_started = True
        st.session_state.show_quick_replies = True
        st.session_state.messages.append({
            "role": "bot",
            "content": "👋 <b><span style='font-size:1.0em;color:#ffff;'>Konnichiwa!</span></b> How can I help you today?"
        })
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

else:
    # Main chat interface
    if st.session_state.get('knowledge_base_loaded', False):
        render_chat(st.session_state.messages)

        # Chat reset logic
        if st.session_state.chat_ended:
            time.sleep(2)
            st.session_state.messages = []
            st.session_state.chat_ended = False
            st.session_state.feedback_request = False
            st.session_state.show_typing = False
            st.session_state.chat_started = False
            st.session_state.show_quick_replies = False
            st.rerun()

        if st.session_state.get("show_typing", False):
            show_typing()

        # Quick replies - show only at the beginning
        if st.session_state.show_quick_replies:
           st.markdown("**Quick help topics:**")
           for reply in st.session_state.quick_replies:
              if st.button(reply, key=f"quick_reply_{reply}"):
                  st.session_state.messages.append({"role": "user", "content": reply})
                  st.session_state.show_typing = True
                  st.session_state.show_quick_replies = False
                  st.rerun()


        # Bot response logic
        if st.session_state.get("show_typing", False):
            time.sleep(1.2)
            last_user_message = next((msg["content"] for msg in reversed(st.session_state.messages) if msg["role"] == "user"), None)

            if last_user_message:
                bot_response = get_bot_response(last_user_message, st.session_state.df, st.session_state.nn_model, model)
                st.session_state.messages.append({"role": "bot", "content": bot_response})
                st.session_state.feedback_request = True
                st.session_state.show_typing = False
                st.rerun()

    else:
        st.info("Attempting to load knowledge base... If this message persists, check the 'KNOWLEDGE_BASE_PATH' in the code and ensure the file exists.")

# FIXED: Floating Input Bar and Feedback (Gemini-style, Clean Layout)

if st.session_state.chat_started and not st.session_state.chat_ended:

    # Spacing for chat area to avoid being hidden by fixed bar
    st.markdown("<br><br><br>", unsafe_allow_html=True)

    # Minimal Feedback Section (ABOVE input bar)
    if st.session_state.feedback_request:
        st.markdown("---")
        st.write("Was this helpful?")
        col1, col2, col3, col4 = st.columns(4)
        if col1.button("👍", key="fb_like"):
            st.session_state.messages.append({"role": "bot", "content": "Great! Let me know if there's anything else I can help with."})
            st.session_state.feedback_request = False
            st.rerun()
        if col2.button("👎", key="fb_dislike"):
            st.session_state.messages.append({"role": "bot", "content": "Sorry about that. Could you rephrase your question?"})
            st.session_state.feedback_request = False
            st.rerun()
        if col3.button("🤔", key="fb_confused"):
            st.session_state.messages.append({"role": "bot", "content": "I'm here to help! Try asking in a different way."})
            st.session_state.feedback_request = False
            st.rerun()
        if col4.button("❤️", key="fb_love"):
            st.session_state.messages.append({"role": "bot", "content": "Thanks for the love! 😊"})
            st.session_state.feedback_request = False
            st.rerun()

    # Gemini-style fixed center input bar
    st.markdown("""
    <style>
    .gemini-input-wrapper {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background: #1F1F1F;
        padding: 1rem 0.5rem;
        border-top: 1px solid #444;
        z-index: 999;
    }
    .gemini-form {
        max-width: 720px;
        margin: 0 auto;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="gemini-input-wrapper"><div class="gemini-form">', unsafe_allow_html=True)

    with st.form("chat_input_form", clear_on_submit=True):
        col1, col2 = st.columns([6, 1])
        with col1:
            user_input = st.text_input(
                "Your message...",
                key="input_bar",
                label_visibility="collapsed"
            )
        with col2:
            send_clicked = st.form_submit_button("▶")

        if send_clicked and user_input.strip():
            user_input_clean = user_input.lower().strip()
            st.session_state.messages.append({"role": "user", "content": user_input})

            if user_input_clean in ["bye", "end", "quit"]:
                st.session_state.messages.append({"role": "bot", "content": "Thank you for chatting, <b><span style='font-size:1.2em;color:#ffff;'>Mata Ne!</span></b> 👋"})
                st.session_state.chat_ended = True
                st.session_state.feedback_request = False
                st.session_state.show_typing = False
                st.rerun()
            else:
                st.session_state.show_typing = True
                st.session_state.show_quick_replies = False
                st.rerun()

    st.markdown('</div></div>', unsafe_allow_html=True)
