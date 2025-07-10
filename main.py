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
# Custom CSS (Revised and Fixed)
# -------------------------------
st.markdown(f"""
<style>
/* Strict Background Color for the entire app */
html, body, .stApp {{
    background-color: #1F1F1F !important;
    color: white !important;
}}

/* Main Chat Area: Increased bottom padding to prevent overlap with the new pinned bar */
.main > .block-container {{
    padding: 2rem 1rem 12rem 1rem !important;
}}

/* Sidebar Styling */
.stSidebar > div:first-child {{
    background-color: #323232 !important;
    border-right: 2px solid white;
}}

/* ### FIX 1: Correctly Centered 'Start Chat' Button ### */
.start-chat-container {{
    text-align: center;
    margin-top: 2rem;
}}
.start-chat-container .stButton button {{
    background: linear-gradient(90deg, #e53935 0%, #b71c1c 100%);
    color: #fff;
    border-radius: 20px;
    padding: 1rem 2rem; /* Adjusted padding */
    font-size: 1.3rem !important; /* Adjusted font size */
    border: 2px solid #fff;
    font-weight: bold;
    transition: transform 0.2s, box-shadow 0.2s;
}}
.start-chat-container .stButton button:hover {{
    transform: scale(1.05);
    box-shadow: 0 0 15px rgba(255, 255, 255, 0.3);
    color: #fff !important;
    border: 2px solid #fff !important;
}}

/* Chat bubbles and avatars - Unchanged */
.chat-bubble {{
    padding: 1rem 1.5rem; border-radius: 20px; margin-bottom: 14px; max-width: 85%;
    animation: fadeInUp 0.3s; position: relative; word-break: break-word; font-size: 1.08rem;
    display: flex; align-items: center;
}}
.user-bubble {{
    background: #fff; color: #111; align-self: flex-end; margin-left: auto; margin-right: 0;
    border: 1.5px solid #e53935;
}}
.bot-bubble {{
    background: linear-gradient(90deg, #e53935 0%, #b71c1c 100%); color: #fff;
    align-self: flex-start; margin-right: auto; margin-left: 0; border: 1.5px solid #fff;
}}
.avatar {{
    width: 38px; height: 38px; border-radius: 75%; margin: 0 10px; background: #3d3d3d;
    box-shadow: 0 2px 8px rgba(229,57,53,0.12); font-size: 1.7rem; text-align: center;
    line-height: 38px; border: 2px solid #ff0000; display: flex; align-items: center; justify-content: center;
}}
.user-row {{ display: flex; flex-direction: row; align-items: flex-end; justify-content: flex-end; }}
.bot-row {{ display: flex; flex-direction: row; align-items: flex-end; justify-content: flex-start; }}

/* ### FIX 3: Vertical Quick Replies ### */
.quick-reply-container {{
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.5rem; /* Space between vertical buttons */
    margin-bottom: 1.5rem;
}}
.quick-reply-container .stButton button {{
    background: #fff; color: #e53935; border-radius: 18px; padding: 0.5rem 1.1rem;
    cursor: pointer; font-size: 0.98rem; border: 1.5px solid #e53935; font-weight: 500;
    transition: background 0.2s, color 0.2s; width: 200px; /* Give them a consistent width */
}}
.quick-reply-container .stButton button:hover {{ background: #e53935; color: #fff; }}

/* Sidebar & Main Title - Unchanged */
.sidebar-title {{
    font-size: 5.5rem; color: #EE4B2B; font-weight: 900; text-align: center;
    margin: 0.5rem 0 1.5rem 0; letter-spacing: 0.05em; width: 100%; line-height: 1.2;
    animation: rotate3D 5s infinite linear; transform-style: preserve-3d; perspective: 800px;
    text-shadow: 0 0 5px rgba(238, 75, 43, 0.5), 0 0 10px rgba(238, 75, 43, 0.4),
                 0 0 15px rgba(238, 75, 43, 0.3), 1px 1px 2px rgba(0,0,0,0.8);
}}
@keyframes rotate3D {{
    0% {{ transform: rotateY(0deg) scale(1); }} 50% {{ transform: rotateY(180deg) scale(1); }}
    100% {{ transform: rotateY(360deg) scale(1); }}
}}
.elegant-heading {{
    font-size: 4.5rem !important; font-weight: 800; text-align: center;
    margin-top: -5px; color: #ffffff; animation: fadeInUp 2.0s ease-out;
}}
@keyframes fadeInUp {{
    0% {{ opacity: 0; transform: translateY(20px); }} 100% {{ opacity: 1; transform: translateY(0); }}
}}

/* Typing Indicator Styling - Unchanged */
.typing-indicator {{ display: flex; align-items: center; margin-bottom: 1.1rem; }}
.typing-dots span {{
    height: 10px; width: 10px; margin: 0 2px; background: #e53935; border-radius: 25%;
    display: inline-block; animation: blink 1.2s infinite both;
}}
.typing-dots span:nth-child(2) {{ animation-delay: 0.2s; }}
.typing-dots span:nth-child(3) {{ animation-delay: 0.4s; }}
@keyframes blink {{ 0%, 80%, 100% {{ opacity: 0.2; }} 40% {{ opacity: 1; }} }}

/* ### FIX 2: Correctly Pinned Input Bar (ChatGPT/Gemini Style) ### */
.pinned-input-container {{
    position: fixed;
    bottom: 0;
    left: 0;
    width: 100%;
    background-color: #1F1F1F; /* Opaque background */
    z-index: 999;
    /* A subtle gradient to separate from chat */
    background-image: linear-gradient(to top, rgba(31,31,31,1) 80%, rgba(31,31,31,0) 100%);
    padding: 1rem 0;
}}
.pinned-input-inner {{
    max-width: 640px; /* Match the width of the main chat content */
    margin: 0 auto;
    padding: 0 1rem;
}}
.feedback-container {{
    display: flex;
    justify-content: center;
    gap: 1rem;
    margin-bottom: 0.75rem; /* Reduced space between feedback and input bar */
}}
.feedback-container .stButton button {{
    background-color: #3d3d3d; font-size: 1.2rem; border-radius: 50%;
    width: 40px; height: 40px; border: 1px solid #fff;
}}
.feedback-container .stButton button:hover {{ transform: scale(1.1); border-color: #e53935; }}

/* Input Form Styling */
.pinned-input-inner .stTextInput > div > div > input {{
    background: #323232; border: 2px solid #ff0000; color: #fff;
    width: 100%; padding: 0.8rem 1rem; outline: none; font-size: 1.1rem;
    border-radius: 20px;
}}
.pinned-input-inner .stButton[kind="form_submit"] button {{
    background: linear-gradient(90deg, #e53935 0%, #b71c1c 100%);
    color: #fff; border: none; border-radius: 50%; width: 42px; height: 42px;
    font-size: 1.2rem; cursor: pointer; transition: background 0.2s;
    display: flex; align-items: center; justify-content: center;
}}
.pinned-input-inner .stButton[kind="form_submit"] button:hover {{ background: #fff; color: #e53935; }}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(page_title="HCIL IT Helpdesk Chat-Bot", page_icon="🤖", layout="centered", initial_sidebar_state="auto")

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
            st.error(f"❌ **Error:** Knowledge base file '{path}' missing required columns.")
            st.stop()
        embeddings = model.encode(df['questions'].tolist())
        nn_model = NearestNeighbors(n_neighbors=1, metric='cosine')
        nn_model.fit(np.array(embeddings))
        return df, nn_model
    except FileNotFoundError:
        st.error(f"❌ **Error:** Knowledge base file not found at '{path}'.")
        st.stop()
    except Exception as e:
        st.error(f"❌ An error occurred while loading the knowledge base: {e}")
        st.stop()

if 'df' not in st.session_state:
    st.session_state.df, st.session_state.nn_model = load_knowledge_base(KNOWLEDGE_BASE_PATH)

# -------------------------------
# Helper Functions
# -------------------------------
def is_gibberish(text):
    text = text.strip()
    if len(text) < 2: return True
    if re.fullmatch(r'[^\w\s]+', text): return True
    if len(set(text)) < 3: return True
    return False

def is_greeting(text):
    greetings = ["hello", "hi", "hey", "greetings", "good morning", "good afternoon", "good evening", "how are you", "what's up", "sup", "thank you", "thanks", "bye", "goodbye"]
    text = text.lower()
    for greet in greetings:
        if fuzz.partial_ratio(greet, text) > 80:
            return greet
    return None

def get_greeting_response(greet):
    responses = {
        "hello": "Hello! 👋 How can I help you today?", "hi": "Hi there! How can I assist you?", "hey": "Hey! How can I help you?",
        "greetings": "Greetings! How can I help you?", "good morning": "Good morning! ☀️ How can I help?", "good afternoon": "Good afternoon! How can I help?",
        "good evening": "Good evening! How can I help?", "how are you": "I'm just a bot, but I'm here to help you! 😊", "what's up": "I'm here to help with your IT queries!",
        "sup": "All good! How can I assist you?", "thank you": "You're welcome! Let me know if you have more questions.", "thanks": "You're welcome!",
        "bye": "Thank you for chatting, **Mata Ne!** (see you later) 👋", "goodbye": "Thank you for chatting, **Mata Ne!** (see you later) 👋"
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
    return df.iloc[best_idx]['answers']

def render_chat(messages):
    for msg in messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="user-row"><div class="chat-bubble user-bubble">{msg["content"]}</div><div class="avatar" style="margin-left:8px;">🧑‍💻</div></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bot-row"><div class="avatar" style="margin-right:8px;">🤖</div><div class="chat-bubble bot-bubble">{msg["content"]}</div></div>', unsafe_allow_html=True)

def show_typing():
    st.markdown('<div class="typing-indicator"><div class="avatar" style="margin-right:8px;">🤖</div><div class="typing-dots"><span></span><span></span><span></span></div></div>', unsafe_allow_html=True)

# -------------------------------
# Sidebar
# -------------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-title">HCIL</div>', unsafe_allow_html=True)
    st.info("Say 'bye', 'quit', or 'end' to close our chat.")

# -------------------------------
# Main Application Logic
# -------------------------------
# Initialize session state
if 'messages' not in st.session_state: st.session_state.messages = []
if 'chat_started' not in st.session_state: st.session_state.chat_started = False
if 'chat_ended' not in st.session_state: st.session_state.chat_ended = False
if 'feedback_request' not in st.session_state: st.session_state.feedback_request = False
if 'show_typing' not in st.session_state: st.session_state.show_typing = False
if 'quick_replies' not in st.session_state: st.session_state.quick_replies = ["Reset password", "VPN issues", "Software install"]

# Render main page title
st.markdown("<h1 class='elegant-heading'>🤖 HCIL IT Helpdesk Chatbot</h1>", unsafe_allow_html=True)

# Central container for the chat history and start button
if not st.session_state.chat_started:
    with st.container():
        st.markdown('<div class="start-chat-container">', unsafe_allow_html=True)
        if st.button("Start Chat", key="start_chat_main"):
            st.session_state.chat_started = True
            st.session_state.messages.append({"role": "bot", "content": "👋 <b><span style='font-size:1.0em;color:#ffff;'>Konnichiwa!</span></b> How can I help you today?"})
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
else:
    # Render the chat history
    render_chat(st.session_state.messages)

    # If chat has ended, reset everything after a delay
    if st.session_state.chat_ended:
        time.sleep(2)
        st.session_state.messages = []
        st.session_state.chat_started = False
        st.session_state.chat_ended = False
        st.session_state.feedback_request = False
        st.session_state.show_typing = False
        st.rerun()

    # Show typing indicator when waiting for bot response
    if st.session_state.show_typing:
        show_typing()

    # ### FIX 3: Show VERTICAL quick replies only at the start of the chat ###
    if len(st.session_state.messages) == 1:
        st.markdown('<div class="quick-reply-container">', unsafe_allow_html=True)
        for reply in st.session_state.quick_replies:
            if st.button(reply, key=f"quick_reply_{reply}"):
                st.session_state.messages.append({"role": "user", "content": reply})
                st.session_state.show_typing = True
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# ### FIX 2: Pinned input bar at the bottom ###
if st.session_state.chat_started and not st.session_state.chat_ended:
    st.markdown('<div class="pinned-input-container">', unsafe_allow_html=True)
    st.markdown('<div class="pinned-input-inner">', unsafe_allow_html=True)

    if st.session_state.feedback_request:
        st.markdown('<div class="feedback-container">', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        # Unique keys for feedback buttons
        if col1.button("👍", key="feedback_like", use_container_width=True):
            st.session_state.messages.append({"role": "bot", "content": "Great! Let me know if I can help with anything else."})
            st.session_state.feedback_request = False
            st.rerun()
        if col2.button("👎", key="feedback_dislike", use_container_width=True):
            st.session_state.messages.append({"role": "bot", "content": "I apologize. Could you please rephrase your question?"})
            st.session_state.feedback_request = False
            st.rerun()
        if col3.button("🤔", key="feedback_confused", use_container_width=True):
            st.session_state.messages.append({"role": "bot", "content": "I'm here to help! Feel free to ask another question."})
            st.session_state.feedback_request = False
            st.rerun()
        if col4.button("❤️", key="feedback_love", use_container_width=True):
            st.session_state.messages.append({"role": "bot", "content": "Thank you for your feedback! 😊"})
            st.session_state.feedback_request = False
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    with st.form("chat_input_form", clear_on_submit=True):
        col1, col2 = st.columns([6, 1])
        with col1:
            user_input = st.text_input("user_input", placeholder="Type here...", key="input_bar", label_visibility="collapsed")
        with col2:
            send_clicked = st.form_submit_button("➤")

        if send_clicked and user_input.strip():
            st.session_state.messages.append({"role": "user", "content": user_input})
            if user_input.lower().strip() in ["bye", "end", "quit"]:
                st.session_state.show_typing = True
            else:
                st.session_state.show_typing = True
                st.session_state.feedback_request = False # Hide feedback while processing
            st.rerun()

    st.markdown('</div></div>', unsafe_allow_html=True)

# Bot response logic (processed after user input)
if st.session_state.show_typing:
    last_message = st.session_state.messages[-1]
    if last_message["role"] == "user":
        ### FIX 4: FASTER BOT RESPONSE TIME ###
        time.sleep(0.6)
        user_input_clean = last_message["content"].lower().strip()

        if user_input_clean in ["bye", "end", "quit"]:
            bot_response = "Thank you for chatting, <b><span style='font-size:1.2em;color:#ffff;'>Mata Ne!</span></b> (see you later) 👋"
            st.session_state.chat_ended = True
            st.session_state.feedback_request = False
        else:
            bot_response = get_bot_response(last_message["content"], st.session_state.df, st.session_state.nn_model, model)
            st.session_state.feedback_request = True

        st.session_state.messages.append({"role": "bot", "content": bot_response})
        st.session_state.show_typing = False
        st.rerun()
