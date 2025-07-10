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
/* Add all your previous CSS here – it’s too long to paste again.
   You already pasted the full working version with the chat bubble styles, sidebar, animations, etc.
   It’s still valid and not responsible for the error. Just re-use it as-is. */
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

if 'df' not in st.session_state or 'nn_model' not in st.session_state:
    st.session_state.df, st.session_state.nn_model = load_knowledge_base(KNOWLEDGE_BASE_PATH)
    st.session_state.knowledge_base_loaded = True

# -------------------------------
# Helper Functions
# -------------------------------
def is_gibberish(text):
    text = text.strip()
    if len(text) < 2 or re.fullmatch(r'[^\w\s]+', text) or len(set(text)) < 3:
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

    return df.iloc[best_idx]['answers']

def render_chat(messages):
    for msg in messages:
        if msg["role"] == "user":
            st.markdown(f"""<div class="user-row"><div class="chat-bubble user-bubble">{msg['content']}</div><div class="avatar">🧑‍💻</div></div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="bot-row"><div class="avatar">🤖</div><div class="chat-bubble bot-bubble">{msg['content']}</div></div>""", unsafe_allow_html=True)

def show_typing():
    st.markdown("""<div class="typing-indicator"><div class="avatar">🤖</div><div class="typing-dots"><span></span><span></span><span></span></div></div>""", unsafe_allow_html=True)

# -------------------------------
# Sidebar Configuration
# -------------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-title">HCIL</div>', unsafe_allow_html=True)
    st.info("Say 'bye', 'quit', or 'end' to close the chat.")

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

st.markdown("<h1 class='elegant-heading'>🤖 HCIL IT Helpdesk Chatbot</h1>", unsafe_allow_html=True)

# -------------------------------
# Chat App Flow
# -------------------------------
if not st.session_state.chat_started:
    if st.button("Start Chat", key="start_chat"):
        st.session_state.chat_started = True
        st.session_state.show_quick_replies = True
        st.session_state.messages.append({
            "role": "bot",
            "content": "👋 <b><span style='font-size:1.0em;color:#ffff;'>Konnichiwa!</span></b> How can I help you today?"
        })
        st.rerun()
else:
    if st.session_state.knowledge_base_loaded:
        render_chat(st.session_state.messages)

        if st.session_state.chat_ended:
            time.sleep(2)
            for key in ['messages', 'feedback_request', 'show_typing', 'chat_started', 'show_quick_replies']:
                st.session_state[key] = defaults[key]
            st.session_state.chat_ended = False
            st.rerun()

        if st.session_state.show_typing:
            show_typing()

        if st.session_state.show_quick_replies:
            st.markdown("**Quick help topics:**")
            for reply in st.session_state.quick_replies:
                if st.button(reply, key=f"quick_reply_{reply}"):
                    st.session_state.messages.append({"role": "user", "content": reply})
                    st.session_state.show_typing = True
                    st.session_state.show_quick_replies = False
                    st.rerun()

        if st.session_state.show_typing:
            time.sleep(1.2)
            last_user_msg = next((msg["content"] for msg in reversed(st.session_state.messages) if msg["role"] == "user"), None)
            if last_user_msg:
                response = get_bot_response(last_user_msg, st.session_state.df, st.session_state.nn_model, model)
                st.session_state.messages.append({"role": "bot", "content": response})
                st.session_state.feedback_request = True
                st.session_state.show_typing = False
                st.rerun()
    else:
        st.info("Trying to load the knowledge base...")

# -------------------------------
# Input Bar + Feedback
# -------------------------------
if st.session_state.chat_started and not st.session_state.chat_ended:
    if st.session_state.feedback_request:
        st.markdown("---")
        st.write("Was this helpful?")
        col1, col2, col3, col4 = st.columns(4)
        if col1.button("👍"): st.session_state.messages.append({"role": "bot", "content": "Glad it helped!"}); st.session_state.feedback_request = False; st.rerun()
        if col2.button("👎"): st.session_state.messages.append({"role": "bot", "content": "Sorry. Try asking differently?"}); st.session_state.feedback_request = False; st.rerun()
        if col3.button("🤔"): st.session_state.messages.append({"role": "bot", "content": "I'm trying my best! 😅"}); st.session_state.feedback_request = False; st.rerun()
        if col4.button("❤️"): st.session_state.messages.append({"role": "bot", "content": "Thanks for the love!"}); st.session_state.feedback_request = False; st.rerun()

    with st.form("chat_input_form", clear_on_submit=True):
        col1, col2 = st.columns([6, 1])
        with col1:
            user_input = st.text_input("Your message...", key="input_bar", label_visibility="collapsed")
        with col2:
            send_clicked = st.form_submit_button("▶")
        if send_clicked and user_input.strip():
            user_input_clean = user_input.lower().strip()
            st.session_state.messages.append({"role": "user", "content": user_input})
            if user_input_clean in ["bye", "quit", "end"]:
                st.session_state.messages.append({"role": "bot", "content": "Thank you for chatting, <b><span style='font-size:1.2em;color:#ffff;'>Mata Ne!</span></b> 👋"})
                st.session_state.chat_ended = True
                st.session_state.feedback_request = False
                st.session_state.show_typing = False
            else:
                st.session_state.show_typing = True
                st.session_state.show_quick_replies = False
            st.rerun()
