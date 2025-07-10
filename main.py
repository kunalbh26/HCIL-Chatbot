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
            st.markdown(f"<div style='background:#fff;color:#111;border:1px solid red;padding:10px;border-radius:10px;margin:8px 0;text-align:right;'>{message['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='background:#e53935;color:#fff;border:1px solid white;padding:10px;border-radius:10px;margin:8px 0;text-align:left;'>{message['content']}</div>", unsafe_allow_html=True)

# -------------------------------
# Main Application Logic
# -------------------------------
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'chat_started' not in st.session_state:
    st.session_state.chat_started = False

st.title("🤖 HCIL IT Helpdesk Chatbot")

if not st.session_state.chat_started:
    if st.button("Start Chat"):
        st.session_state.chat_started = True
        st.session_state.messages.append({"role": "bot", "content": "👋 Hello! How can I help you today?"})
        st.rerun()
else:
    render_chat(st.session_state.messages)

    with st.form("chat_input_form", clear_on_submit=True):
        user_input = st.text_input("Your message...", key="input_text")
        submit = st.form_submit_button("Send")

        if submit and user_input.strip():
            st.session_state.messages.append({"role": "user", "content": user_input})
            bot_response = get_bot_response(user_input, st.session_state.df, st.session_state.nn_model, model)
            st.session_state.messages.append({"role": "bot", "content": bot_response})
            st.rerun()
