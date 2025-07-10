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
st.markdown("""
<style>
/* Ensure the very root HTML and body are black */
html, body {
    background-color: #000000 !important; /* Strict background color */
    color: white !important;
}

/* Target Streamlit's main content area */
.stApp {
    background-color: #000000 !important; /* Strict background color for the app container */
}

/* Your existing .main style, updated to the desired background */
.main {
    background: #000000; /* Use your desired dark background here */
    border-radius: 0px;
    padding: 3.5rem !important;
    max-width: 640px;
    margin: 2.5rem auto;
}

/* FIXED: Start Chat Button - Proper Centering */
.start-chat-container {
    display: flex;
    justify-content: center;
    align-items: center;
    width: 100%;
    margin: 3rem 0;
}

/* General Button Style for Start Chat */
.stButton > button {
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
}

.stButton > button:hover {
    transform: scale(1.08) !important;
    color: #fff !important;
    border: 3px solid #fff !important;
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
display: inline-block !important;
background: #ffffff;
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
    font-size: 4.5rem;
    color: #EE4B2B;
    font-weight: 900;
    text-align: center;
    margin: 0.5rem 0 1.5rem 0;
    letter-spacing: 0.05em;
    width: 100%;
    line-height: 1.2;
    /* Add animation here */
    animation: rotateY 3s infinite linear; /* Apply the animation */
    transform-style: preserve-3d; /* For 3D transformations */
}

/* Keyframes for vertical rotation */
@keyframes rotateY {
    0% {
        transform: rotateY(0deg);
    }
    50% {
        transform: rotateY(180deg);
    }
    100% {
        transform: rotateY(360deg);
    }
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

# --- Pre-load Knowledge Base ---
@st.cache_resource
def load_knowledge_base(path):
    try:
        df = pd.read_excel(path)
        required_columns = {'questions', 'answers', 'categories', 'tags'}
        if not required_columns.issubset(df.columns):
            st.error(f"❌ **Error:** Missing required columns: {required_columns}")
            st.stop()
        embeddings = model.encode(df['questions'].tolist())
        nn_model = NearestNeighbors(n_neighbors=1, metric='cosine')
        nn_model.fit(np.array(embeddings))
        return df, nn_model
    except FileNotFoundError:
        st.error(f"❌ File not found at '{path}'")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading KB: {e}")
        st.stop()

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
# Sidebar Configuration
# -------------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-title">HCIL</div>', unsafe_allow_html=True)
    st.info("Say 'bye', 'quit', or 'end' to close our chat.")

# -------------------------------
# Session State Initialization
# -------------------------------
defaults = {
    'knowledge_base_loaded': False,
    'messages': [],
    'chat_ended': False,
    'feedback_request': False,
    'quick_replies': ["Reset password", "VPN issues", "Software install"],
    'show_typing': False,
    'chat_started': False,
    'show_quick_replies': False
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# -------------------------------
# Main Application Logic
# -------------------------------

# Load knowledge base automatically
if 'df' not in st.session_state or 'nn_model' not in st.session_state:
    with st.spinner("🚀 Launching the bot... Please wait."):
        st.session_state.df, st.session_state.nn_model = load_knowledge_base(KNOWLEDGE_BASE_PATH)
        st.session_state.knowledge_base_loaded = True

st.markdown("""
<style>
.elegant-heading {
    font-size: 5.0rem;
    font-weight: 800;
    text-align: center;
    margin-top: -5px;
    color: #ffffff;
    animation: fadeInUp 1.6s ease-out;
}

@keyframes fadeInUp {
    0% { opacity: 0; transform: translateY(20px); }
    100% { opacity: 1; transform: translateY(0); }
}
</style>

<h1 class='elegant-heading'>🤖 HCIL IT Helpdesk Chatbot</h1>
""", unsafe_allow_html=True)

st.markdown('<div class="main">', unsafe_allow_html=True)


# -------------------------------
# Chat App Flow
# -------------------------------
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
    if st.session_state.knowledge_base_loaded:
        render_chat(st.session_state.messages)

        if st.session_state.chat_ended:
            time.sleep(2)
            # Reset the chat state to show the "Start Chat" button again
            for key in ['messages', 'feedback_request', 'show_typing', 'chat_started', 'show_quick_replies', 'chat_ended']:
                st.session_state[key] = defaults[key]
            st.rerun()

        if st.session_state.get("show_typing", False):
            show_typing()
        
        # Quick Replies
        st.markdown('<div style="margin-bottom:3rem;">', unsafe_allow_html=True)
      /*   if st.session_state.show_quick_replies:  */
            for reply in st.session_state.quick_replies:
                if st.button(reply, key=f"quick_{reply}"):
                    st.session_state.messages.append({"role": "user", "content": reply})
                    st.session_state.show_typing = True
              /*      st.session_state.show_quick_replies = False   */
                    st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        # Feedback Reactions
        if st.session_state.feedback_request:
            st.markdown("#### Was this helpful?")
            col1, col2, col3, col4 = st.columns(4)
            if col1.button("👍", use_container_width=True):
                st.session_state.messages.append({"role": "bot", "content": "Great! Let me know if there is something else that I can help you with."})
                st.session_state.feedback_request = False
                st.rerun()
            if col2.button("👎", use_container_width=True):
                st.session_state.messages.append({"role": "bot", "content": "I apologize. Could you please rephrase your question?"})
                st.session_state.feedback_request = False
                st.rerun()
            if col3.button("🤔", use_container_width=True):
                st.session_state.messages.append({"role": "bot", "content": "I'm here to help! Feel free to ask another question."})
                st.session_state.feedback_request = False
                st.rerun()
            if col4.button("❤️", use_container_width=True):
                st.session_state.messages.append({"role": "bot", "content": "Thank you for your feedback! 😊"})
                st.session_state.feedback_request = False
                st.rerun()

        # Input Bar
        if not st.session_state.chat_ended:
            with st.form("chat_input_form", clear_on_submit=True):
                col1, col2 = st.columns([6, 1])
                with col1:
                    user_input = st.text_input("user_input", placeholder="Type here...", key="input_bar", label_visibility="collapsed")
                with col2:
                    send_clicked = st.form_submit_button("Send")

            if send_clicked and user_input.strip():
                user_input_clean = user_input.lower().strip()
                st.session_state.messages.append({"role": "user", "content": user_input})

                if user_input_clean in ["bye", "end", "quit"]:
                    st.session_state.messages.append({"role": "bot", "content": "Thank you for chatting, <b><span style='font-size:1.2em;color:#ffff;'>Mata Ne!</span></b> (see you later) 👋"})
                    st.session_state.chat_ended = True
                    st.session_state.feedback_request = False
                    st.session_state.show_typing = False
                    st.rerun()
                else:
                    st.session_state.show_typing = True
                    st.session_state.show_quick_replies = False # Hide quick replies on new input
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

st.markdown('</div>', unsafe_allow_html=True)
