# ==========================================
# HCIL IT Helpdesk Chat-Bot
# ==========================================
import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import time
import re
from fuzzywuzzy import fuzz, process

# ------------------------------------------------------------------
#  CONFIGURATION
# ------------------------------------------------------------------
KNOWLEDGE_BASE_PATH = "dataset.xlsx"

st.set_page_config(
    page_title="HCIL IT Helpdesk Chat-Bot",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="auto",
)

# ------------------------------------------------------------------
#  GLOBAL STYLES  (existing theme + new Gemini-style input bar)
# ------------------------------------------------------------------
st.markdown(
    """
<style>
/* ——— CORE DARK THEME ——— */
html, body, .stApp {
    background-color: #1F1F1F !important;
    color: #fff !important;
}
.main .block-container { padding-bottom: 12rem !important; }

/* Sidebar */
.stSidebar > div:first-child {
    background-color: #323232 !important;
    border-right: 2px solid #fff;
}

/* Chat Title */
.elegant-heading {
    font-size: 5rem !important;
    font-weight: 800;
    text-align: center;
    margin-top: -5px;
    color: #ffffff;
    animation: fadeInUp 2s ease-out;
}
@keyframes fadeInUp {
    0% {opacity:0; transform:translateY(20px);}
    100% {opacity:1; transform:translateY(0);}
}

/* Chat bubbles */
.chat-bubble {
    padding: 1rem 1.5rem;
    border-radius: 20px;
    margin-bottom: 14px;
    max-width: 75%;
    font-size: 1.08rem;
    display:flex; align-items:center;
    word-break: break-word;
}
.user-bubble {background:#fff; color:#111; margin-left:auto; border:1.5px solid #e53935;}
.bot-bubble  {background:linear-gradient(90deg,#e53935 0%,#b71c1c 100%); color:#fff; margin-right:auto; border:1.5px solid #fff;}
.avatar {
    width:38px;height:38px;border-radius:50%;background:#3d3d3d;
    display:flex;align-items:center;justify-content:center;
    font-size:1.7rem;border:2px solid #ff0000;box-shadow:0 2px 6px rgba(229,57,53,0.15);
}
.user-row {display:flex;flex-direction:row;justify-content:flex-end;}
.bot-row  {display:flex;flex-direction:row;justify-content:flex-start;}

/* Typing indicator */
.typing-indicator {display:flex;align-items:center;margin-bottom:1.1rem;}
.typing-dots span {
    height:10px;width:10px;margin:0 2px;background:#e53935;border-radius:50%;
    display:inline-block;animation:blink 1.2s infinite both;
}
.typing-dots span:nth-child(2){animation-delay:0.2s;}
.typing-dots span:nth-child(3){animation-delay:0.4s;}
@keyframes blink {0%,80%,100%{opacity:0.2;}40%{opacity:1;}}

/* ——— NEW GEMINI-STYLE INPUT BAR ——— */
.gemini-input-bar{
    position:fixed;bottom:1.3rem;left:50%;transform:translateX(-50%);
    width:min(680px,92%);background:#2A2A2A;border:1.5px solid #444;
    border-radius:28px;padding:.55rem .9rem;display:flex;gap:.5rem;
    box-shadow:0 4px 14px rgba(0,0,0,.45);z-index:999;
}
.gemini-input-bar input{
    flex:1;background:transparent;border:none;outline:none;color:#fff;font-size:1.05rem;
}
.gemini-input-bar button{
    background:#e53935;border:none;width:46px;height:46px;border-radius:50%;
    color:#fff;font-size:1.35rem;cursor:pointer;transition:.15s;
}
.gemini-input-bar button:hover{background:#fff;color:#e53935;}

/* Quick reply buttons – stacked */
.quick-btn {
    width:100%; background:#fff; color:#e53935; border:1.5px solid #e53935;
    border-radius:6px; padding:.55rem 0; font-size:1rem; margin-bottom:.4rem;
}
.quick-btn:hover {background:#e53935; color:#fff;}

/* Sidebar rotating logo */
.sidebar-title{
    font-size:5.2rem;color:#EE4B2B;font-weight:900;text-align:center;margin:0.5rem 0 1.5rem;
    animation:spin 5s linear infinite; text-shadow:0 0 5px rgba(238,75,43,.5),0 0 10px rgba(238,75,43,.4);
}
@keyframes spin{0%{transform:rotateY(0);}100%{transform:rotateY(360deg);}}

/* FadeIn animation for bubbles */
@keyframes fadeInUp {0%{opacity:0;transform:translateY(20px);}100%{opacity:1;transform:translateY(0);}}
</style>
""",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
#  MODEL & KNOWLEDGE BASE
# ------------------------------------------------------------------
@st.cache_resource
def load_sentence_transformer():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_knowledge_base(path: str):
    try:
        df = pd.read_excel(path)
        required = {"questions", "answers", "categories", "tags"}
        if not required.issubset(df.columns):
            st.error("❌ Required columns missing in knowledge base.")
            st.stop()
        embeddings = model.encode(df["questions"].tolist())
        nn = NearestNeighbors(n_neighbors=1, metric="cosine")
        nn.fit(np.array(embeddings))
        return df, nn
    except FileNotFoundError:
        st.error(f"❌ Knowledge base not found at '{path}'.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading knowledge base: {e}")
        st.stop()

model = load_sentence_transformer()
if "df" not in st.session_state or "nn" not in st.session_state:
    st.session_state.df, st.session_state.nn = load_knowledge_base(KNOWLEDGE_BASE_PATH)

# ------------------------------------------------------------------
#  HELPER FUNCTIONS
# ------------------------------------------------------------------
def is_gibberish(text: str) -> bool:
    text = text.strip()
    if len(text) < 2 or re.fullmatch(r"[^\w\s]+", text) or len(set(text)) < 3:
        return True
    words = text.split()
    if words and sum(1 for w in words if not w.isalpha()) / len(words) > 0.5:
        return True
    return False

def is_greeting(text: str):
    greetings = [
        "hello","hi","hey","greetings","good morning","good afternoon","good evening",
        "how are you","what's up","sup","thank you","thanks","bye","goodbye",
    ]
    for g in greetings:
        if fuzz.partial_ratio(g, text.lower()) > 80:
            return g
    return None

def greeting_response(key: str) -> str:
    mapping = {
        "hello":"Hello! 👋 How can I help you today?",
        "hi":"Hi there! How can I assist you?",
        "hey":"Hey! How can I help you?",
        "greetings":"Greetings! How can I help you?",
        "good morning":"Good morning! ☀️ How can I help?",
        "good afternoon":"Good afternoon! How can I help?",
        "good evening":"Good evening! How can I help?",
        "how are you":"I'm just a bot, but I'm here to help you! 😊",
        "what's up":"I'm here to help with your IT queries!",
        "sup":"All good! How can I assist you?",
        "thank you":"You're welcome! Let me know if you have more questions.",
        "thanks":"You're welcome!",
        "bye":"Thank you for chatting, **Mata Ne!** 👋",
        "goodbye":"Thank you for chatting, **Mata Ne!** 👋",
    }
    return mapping.get(key, "Hello! How can I help you?")

def get_bot_response(query: str) -> str:
    df, nn = st.session_state.df, st.session_state.nn
    if is_gibberish(query):
        return "I'm sorry, I couldn't understand that. Could you please rephrase?"

    greet = is_greeting(query)
    if greet:
        return greeting_response(greet)

    # Fuzzy direct match
    questions = df["questions"].tolist()
    best_q, score = process.extractOne(query, questions, scorer=fuzz.token_sort_ratio)
    if score > 70:
        return df.loc[questions.index(best_q), "answers"]

    # Embedding search
    dist, idx = nn.kneighbors(model.encode([query]))
    if dist[0][0] > 0.45:
        return "I'm sorry, I couldn't understand that. Could you please rephrase?"
    return df.iloc[idx[0][0]]["answers"]

def render_chat(msgs):
    for m in msgs:
        if m["role"] == "user":
            st.markdown(
                f"""
<div class="user-row">
    <div class="chat-bubble user-bubble">{m['content']}</div>
    <div class="avatar" style="margin-left:8px;">🧑‍💻</div>
</div>""",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
<div class="bot-row">
    <div class="avatar" style="margin-right:8px;">🤖</div>
    <div class="chat-bubble bot-bubble">{m['content']}</div>
</div>""",
                unsafe_allow_html=True,
            )

def show_typing_indicator():
    st.markdown(
        """
<div class="typing-indicator">
    <div class="avatar" style="margin-right:8px;">🤖</div>
    <div class="typing-dots"><span></span><span></span><span></span></div>
</div>""",
        unsafe_allow_html=True,
    )

# ------------------------------------------------------------------
#  SIDEBAR
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-title">HCIL</div>', unsafe_allow_html=True)
    st.info("Say **bye**, **quit**, or **end** to close the chat.")

# ------------------------------------------------------------------
#  SESSION STATE INIT
# ------------------------------------------------------------------
for k, v in {
    "messages": [],
    "chat_started": False,
    "show_quick_replies": False,
    "show_typing": False,
    "chat_ended": False,
    "feedback_request": False,
    "quick_replies": ["Reset password", "VPN issues", "Software install"],
}.items():
    st.session_state.setdefault(k, v)

# ------------------------------------------------------------------
#  PAGE TITLE
# ------------------------------------------------------------------
st.markdown("<h1 class='elegant-heading'>🤖 HCIL IT Helpdesk Chat-Bot</h1>", unsafe_allow_html=True)

# ------------------------------------------------------------------
#  START CHAT BUTTON
# ------------------------------------------------------------------
if not st.session_state.chat_started:
    if st.button("Start Chat", key="start_chat", help="Begin"):
        st.session_state.chat_started = True
        st.session_state.show_quick_replies = True
        st.session_state.messages.append(
            {
                "role": "bot",
                "content": "👋 <b>Konnichiwa!</b> How can I help you today?",
            }
        )
        st.experimental_rerun()
    st.stop()

# ------------------------------------------------------------------
#  MAIN CHAT AREA
# ------------------------------------------------------------------
render_chat(st.session_state.messages)

# Reset logic after chat ends
if st.session_state.chat_ended:
    time.sleep(1.5)
    st.session_state.clear()
    st.experimental_rerun()

# Typing indicator
if st.session_state.show_typing:
    show_typing_indicator()

# ------------------------------------------------------------------
#  QUICK REPLIES (stacked buttons)
# ------------------------------------------------------------------
if st.session_state.show_quick_replies:
    for reply in st.session_state.quick_replies:
        if st.button(reply, key=f"qr_{reply}", help=reply, type="secondary"):
            st.session_state.messages.append({"role": "user", "content": reply})
            st.session_state.show_typing = True
            st.session_state.show_quick_replies = False
            st.experimental_rerun()

# ------------------------------------------------------------------
#  FEEDBACK (simple buttons)  ———  placed just above the input bar
# ------------------------------------------------------------------
def record_feedback(kind: str):
    mapping = {
        "like": "Great! Let me know if there’s anything else.",
        "dislike": "I’m sorry. Could you please rephrase?",
        "confused": "I’m here to help! Feel free to ask another question.",
        "love": "Thank you for your feedback! 😊",
    }
    st.session_state.messages.append({"role": "bot", "content": mapping[kind]})
    st.session_state.feedback_request = False
    st.experimental_rerun()

if st.session_state.feedback_request:
    fb1, fb2, fb3, fb4 = st.columns(4)
    if fb1.button("👍"): record_feedback("like")
    if fb2.button("👎"): record_feedback("dislike")
    if fb3.button("🤔"): record_feedback("confused")
    if fb4.button("❤️"): record_feedback("love")

# ------------------------------------------------------------------
#  GEMINI-STYLE INPUT BAR (fixed at bottom-center)
# ------------------------------------------------------------------
def process_user_message(text: str):
    user_clean = text.lower().strip()
    st.session_state.messages.append({"role": "user", "content": text})

    # End chat keywords
    if user_clean in {"bye", "quit", "end"}:
        st.session_state.messages.append(
            {
                "role": "bot",
                "content": "Thank you for chatting, <b>Mata Ne!</b> 👋",
            }
        )
        st.session_state.chat_ended = True
        st.session_state.feedback_request = False
        st.session_state.show_typing = False
        st.experimental_rerun()
    else:
        st.session_state.show_typing = True
        st.session_state.show_quick_replies = False
        st.experimental_rerun()

# The fixed bar itself
with st.form("gemini_input", clear_on_submit=True):
    st.markdown('<div class="gemini-input-bar">', unsafe_allow_html=True)
    user_txt = st.text_input("", placeholder="Ask HCIL bot…", label_visibility="collapsed")
    send = st.form_submit_button("▶")
    st.markdown("</div>", unsafe_allow_html=True)

# Handle send
if send and user_txt.strip():
    process_user_message(user_txt)

# ------------------------------------------------------------------
#  BOT RESPONSE GENERATOR (runs after rerun flag)
# ------------------------------------------------------------------
if st.session_state.show_typing:
    time.sleep(1)  # Simulated thinking
    last = next((m["content"] for m in reversed(st.session_state.messages) if m["role"] == "user"), "")
    bot_reply = get_bot_response(last)
    st.session_state.messages.append({"role": "bot", "content": bot_reply})
    st.session_state.show_typing = False
    st.session_state.feedback_request = True
    st.experimental_rerun()
