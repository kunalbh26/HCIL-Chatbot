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
# Custom CSS for Red-Black-White Theme
# -------------------------------
st.markdown(f"""
<style>
/* Strict Background Color for the entire app */
html, body, .stApp {{
    background-color: #1F1F1F !important; /* Dark Gray Background */
    color: white !important;
}}

/* Main Chat Area Background */
.main {{
    background: #1F1F1F; /* Match the strict background */
    border-radius: 0px;
    padding: 3.5rem !important;
    max-width: 640px;
    margin: 2.0rem auto;
}}

/* Centered Start Chat Button */
.start-chat-button-container {{
    display: flex;
    justify-content: center;
    align-items: center;
    margin-top: 2rem;
}}

.start-chat-button {{
    background: linear-gradient(90deg, #e53935 0%, #b71c1c 100%);
    color: #fff;
    border-radius: 20px;
    padding: 1.5rem 2rem; /* Increased padding for larger button */
    cursor: pointer;
    font-size: 1.5rem; /* Increased font size */
    border: 2px solid #fff;
    font-weight: bold;
    transition: transform 0.2s;
}}

.start-chat-button:hover {{
    transform: scale(1.05);
}}

/* Chat bubbles and avatars - unchanged */
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
    width: 38px; height: 38px; border-radius: 75%; margin: 0 10px;
    background: #3d3d3d;
    box-shadow: 0 2px 8px rgba(229,57,53,0.12);
    font-size: 1.7rem;
    text-align: center;
    line-height: 38px;
    border: 2px solid #ff0000;
    display: flex; align-items: center; justify-content: center;
}}

.user-row {{
    display: flex; flex-direction: row; align-items: flex-end; justify-content: flex-end;
}}

.bot-row {{
    display: flex; flex-direction: row; align-items: flex-end; justify-content: flex-start;
}}

.input-bar {{
    background: #222;
    border-radius: 20px;
    box-shadow: 0 2px 8px rgba(229,57,53,0.12);
    position: fixed; /* Fixed position for input bar */
    bottom: 80px; /* Adjusted to position above the reaction buttons */
    left: 50%;
    transform: translateX(-50%); /* Centered horizontally */
    width: calc(100% - 40px); /* Full width minus some padding */
    max-width: 640px; /* Match the main chat area */
    display: flex;
    align-items: center;
    padding: 0.3rem 0.8rem;
    z-index: 10; /* Ensure it stays above other elements */
}}

.input-bar input {{
    background: #323232;
    border: 2px solid #ff0000;
    color: #fff;
    width: 100%;
    padding: 0.7rem 0.8rem;
    outline: none;
    font-size: 1rem;
}}

.send-btn {{
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
}}

.send-btn:hover {{
    background: #fff;
    color: #e53935;
    border: 1.5px solid #ff00;
}}

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

.sidebar-title {{
    font-size: 5.5rem; /* Bigger font size */
    color: #EE4B2B; /* Keep original color or change as desired */
    font-weight: 900;
    text-align: center;
    margin: 0.5rem 0 1.5rem 0;
    letter-spacing: 0.05em;
    width: 100%;
    line-height: 1.2;
    animation: rotate3D 5s infinite linear; /* Slower, more impactful rotation */
    transform-style: preserve-3d;
    perspective: 800px; /* Adds perspective for 3D effect */
    text-shadow:
        0 0 5px rgba(238, 75, 43, 0.5), /* Red glow */
        0 0 10px rgba(238, 75, 43, 0.4),
        0 0 15px rgba(238, 75, 43, 0.3),
        1px 1px 2px rgba(0,0,0,0.8); /* Subtle shadow for depth */
}}

@keyframes rotate3D {{
    0% {{
        transform: rotateY(0deg) scale(1);
    }}
    25% {{
        transform: rotateY(90deg) scale(1.05); /* Slight scale for emphasis */
    }}
    50% {{
        transform: rotateY(180deg) scale(1);
    }}
    75% {{
        transform: rotateY(270deg) scale(1.05);
    }}
    100% {{
        transform: rotateY(360deg) scale(1);
    }}
}}

.elegant-heading {{
    font-size: 5.0rem !important; /* Bigger font size */
    font-weight: 800;
    text-align: center;
    margin-top: -5px;
    color: #ffffff; /* White text color */
    animation: fadeInUp 2.0s ease-out;
}}

@keyframes fadeInUp {{
    0% {{ opacity: 0; transform: translateY(20px); }}
    100% {{ opacity: 1; transform: translateY(0); }}
}}

.typing-indicator {{
    display: flex; align-items: center; margin-bottom: 1.1rem;
}}

.typing-dots span {{
    height: 10px; width: 10px; margin: 0 2px;
    background: #e53935; border-radius: 25%; display: inline-block;
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
            st.stop() # Stop the app if columns are missing
        embeddings = model.encode(df['questions'].tolist())
        nn_model = NearestNeighbors(n_neighbors=1, metric='cosine')
        nn_model.fit(np.array(embeddings))
        return df, nn_model
    except FileNotFoundError:
        st.error(f"❌ **Error:** Knowledge base file not found at '{path}'. Please ensure it's in the correct directory.")
        st.stop() # Stop the app if file is not found
    except Exception as e:
        st.error(f"❌ An error occurred while loading the knowledge base: {e}")
        st.stop()

# Load the knowledge base globally when the app starts
if 'df' not in st.session_state or 'nn_model' not in st.session_state:
    st.session_state.df, st.session_state.nn_model = load_knowledge_base(KNOWLEDGE_BASE_PATH)
    st.session_state.knowledge_base_loaded = True # Set to True once loaded

# -------------------------------
# Helper Functions (Unchanged)
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
        else:  # bot
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
# Sidebar Configuration (Modified)
# -------------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-title">HCIL</div>', unsafe_allow_html=True)
    st.info("Say 'bye', 'quit', or 'end' to close our chat.")

# -------------------------------
# Main Application Logic
# -------------------------------
if 'knowledge_base_loaded' not in st.session_state:
    st.session_state['knowledge_base_loaded'] = False # This will be set to True by load_knowledge_base
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

st.markdown("""
<h1 class='elegant-heading'>🤖 HCIL IT Helpdesk Chatbot</h1>
""", unsafe_allow_html=True)

st.markdown('<div class="main">', unsafe_allow_html=True)

if not st.session_state.chat_started:
    # Centered container for the button
    st.markdown(
        """
        <div class="start-chat-button-container">
            <button class="start-chat-button">Start Chat</button>
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.button("Start Chat", key="start_chat"):
        st.session_state.chat_started = True
        st.session_state.messages.append({
            "role": "bot",
            "content": "👋 <b><span style='font-size:1.0em;color:#ffff;'>Konnichiwa!</span></b> How can I help you today?"
        })
        st.rerun()
else:  # This block runs only after 'Start Chat' is clicked
    if st.session_state.get('knowledge_base_loaded', False):  # Only proceed if KB is loaded
        render_chat(st.session_state.messages)

        # Show quick replies only at the beginning of the chat
        if len(st.session_state.messages) == 1:  # After the bot's greeting
            st.markdown('<div style="margin-bottom:3rem;">', unsafe_allow_html=True)
            for reply in st.session_state.quick_replies:
                if st.button(reply, key=f"quick_reply_{reply}"):
                    st.session_state.messages.append({"role": "user", "content": reply})
                    st.session_state.show_typing = True
                    st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        # Feedback Reactions
        if st.session_state.feedback_request:
            st.markdown("#### Was this helpful?")
            col1, col2, col3, col4 = st.columns(4)
            if col1.button("👍", use_container_width=True, key="feedback_like"):
                st.session_state.messages.append({"role": "bot", "content": "Great! Let me know if there is something else that I can help you with."})
                st.session_state.feedback_request = False
                st.rerun()
            if col2.button("👎", use_container_width=True, key="feedback_dislike"):
                st.session_state.messages.append({"role": "bot", "content": "I apologize. Could you please rephrase your question?"})
                st.session_state.feedback_request = False
                st.rerun()
            if col3.button("🤔", use_container_width=True, key="feedback_confused"):
                st.session_state.messages.append({"role": "bot", "content": "I'm here to help! Feel free to ask another question."})
                st.session_state.feedback_request = False
                st.rerun()
            if col4.button("❤️", use_container_width=True, key="feedback_love"):
               
