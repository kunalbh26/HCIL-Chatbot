import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time
import re
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import random
from datetime import datetime

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="HCIL IT-Helpdesk ChatBot",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="auto"
)


# --- Configuration for Pre-loaded Knowledge Base ---
KNOWLEDGE_BASE_PATH = 'dataset.xlsx'

# Advanced CSS with Elite-Level UI Features
st.markdown("""
<style>
/* Import Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;600&display=swap');

/* CSS Variables for Theme */
:root {
    /* Primary gradient: bright → deep red */
    --primary-gradient: linear-gradient(135deg, #e53935 0%, #b71c1c 100%);
    
    /* Secondary gradient: soft coral → crimson */
    --secondary-gradient: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
    
    /* Subtle red-tinted glassmorphism */
    --glass-bg: rgba(229, 57, 53, 0.06);     /* faint red overlay */
    --glass-border: rgba(229, 57, 53, 0.25); /* stronger red border tint */
    
    /* Red neon glow */
    --neon-glow: 0 0 20px rgba(229, 57, 53, 0.5);
    
    /* Text colors on black */
    --text-primary: #ffffff;                /* white for main text */
    --text-secondary: rgba(255, 255, 255, 0.8); /* softer gray-white */
}

/* --- UNIFIED BACKGROUND AND GLOBAL STYLES --- */
body {
    font-family: 'Inter', sans-serif !important;
    color: var(--text-primary) !important;
}

/* Simplified background for better performance */
[data-testid="stAppViewContainer"] {
    background-color: #000000;
}

.stApp {
    background: #000000 !important;
}


/* --- MAIN CONTAINER AND OTHER STYLES --- */
.main {
    background: linear-gradient(135deg, rgba(255,255,255,0.03) 0%, rgba(255,255,255,0.01) 100%) !important;
    backdrop-filter: blur(20px) saturate(200%);
    -webkit-backdrop-filter: blur(20px) saturate(200%);
    border: 1px solid var(--glass-border);
    border-radius: 30px !important;
    padding: 2.5rem !important;
    max-width: 800px !important;
    margin: 2rem auto;
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3),
        inset 0 1px 0 rgba(255, 255, 255, 0.1),
        0 0 100px rgba(229, 57, 53, 0.12);
    position: relative;
    z-index: 1;
    animation: mainFadeIn 1s ease-out;
}

@keyframes mainFadeIn {
    from { opacity: 0; transform: translateY(30px) scale(0.95); }
    to   { opacity: 1; transform: translateY(0)    scale(1);    }
}

/* Sidebar Styling – Black with Red Accent */
.stSidebar > div:first-child {
    background: linear-gradient(180deg, rgba(20, 20, 20, 0.95) 0%, rgba(10, 10, 10, 0.95) 100%) !important;
    backdrop-filter: blur(10px);
    
    /* changed to red-accented border */
    border-right: 1px solid rgba(229, 57, 53, 0.6);
    
    /* subtle red glow instead of plain black shadow */
    box-shadow: 
        5px 0 20px rgba(0, 0, 0, 0.5),
        0 0 15px rgba(229, 57, 53, 0.25);
}


/* Elite Title with Neon Effect – Red + Fixed Top */
.sidebar-title {
    position: sticky;             /* stick to its container while scrolling */
    top: 0 !important;                       /* pin to top of sidebar */
    z-index: 5;                  /* stay above sidebar content */
    margin-bottom: 1rem; 
    padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    font-size: 4rem;
    font-weight: 900;
    text-align: center;
    margin: 0.7rem 0;
    
    /* red animated gradient */
    background: linear-gradient(45deg, #b71c1c, #e53935, #ef4444, #f87171);
    background-size: 300% 300%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;

    animation: gradientShift 3s ease infinite;
    filter: drop-shadow(0 0 30px rgba(229, 57, 53, 0.5));
    letter-spacing: 0.1em;
    text-transform: uppercase;

    padding: 0.5rem 0;            /* small buffer so it doesn’t overlap */
    background-color: rgba(0, 0, 0, 0.85); /* subtle black bg so text stays readable */
}

/* Main Title Enhancement – Red Gradient + Glow */
.elegant-heading {
    font-size: 3.5rem !important;
    font-weight: 800;
    text-align: center;
    margin: 2rem 0 3rem 0 !important;

    /* red gradient text */
    background: linear-gradient(135deg, #b71c1c 0%, #e53935 50%, #f87171 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;

    position: relative;
    animation: titlePulse 2s ease-in-out infinite;
    letter-spacing: -0.02em;
}

@keyframes titlePulse {
    0%, 100% { 
        filter: brightness(1) drop-shadow(0 0 20px rgba(229, 57, 53, 0.5));  /* soft red glow */
    }
    50% { 
        filter: brightness(1.2) drop-shadow(0 0 40px rgba(229, 57, 53, 0.8)); /* stronger red glow */
    }
}


/* Start Chat Button - Premium Design (Red Theme) */
.start-chat-btn {
    background: var(--primary-gradient) !important;  /* uses red gradient from :root */
    color: white !important;
    border-radius: 60px !important;
    padding: 1.2rem 3.5rem !important;
    font-size: 1.2rem !important;
    font-weight: 600 !important;
    border: none !important;
    position: relative;
    overflow: hidden;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;

    /* swapped purple shadow → red glow */
    box-shadow: 
        0 10px 30px rgba(229, 57, 53, 0.4),
        inset 0 1px 0 rgba(255, 255, 255, 0.2);
}

.start-chat-btn::before {
    content: '';
    position: absolute;
    top: 50%;
    left: 50%;
    width: 0;
    height: 0;
    border-radius: 50%;
    background: rgba(255, 255, 255, 0.3);
    transform: translate(-50%, -50%);
    transition: width 0.6s, height 0.6s;
}

.start-chat-btn:hover {
    transform: translateY(-3px) scale(1.05) !important;

    /* hover glow → deeper red */
    box-shadow: 
        0 20px 40px rgba(229, 57, 53, 0.6),
        inset 0 1px 0 rgba(255, 255, 255, 0.3);
}

.start-chat-btn:hover::before {
    width: 300px;
    height: 300px;
}


/* Chat Bubbles - Glass Design */
.chat-bubble {
    padding: 1.2rem 1.8rem;
    border-radius: 24px;
    margin-bottom: 1.2rem;
    max-width: 75%;
    position: relative;
    font-size: 1.05rem;
    line-height: 1.6;
    animation: messageSlide 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    transition: transform 0.2s ease;
}

@keyframes messageSlide {
    from {
        opacity: 0;
        transform: translateY(20px) scale(0.95);
    }
    to {
        opacity: 1;
        transform: translateY(0) scale(1);
    }
}

/* USER messages – Red gradient */
.user-bubble {
    background: linear-gradient(135deg, rgba(229, 57, 53, 0.9) 0%, rgba(183, 28, 28, 0.9) 100%);
    color: white;
    margin-left: auto;

    /* red glow */
    box-shadow: 
        0 8px 24px rgba(229, 57, 53, 0.35),
        inset 0 1px 0 rgba(255, 255, 255, 0.2);

    border: 1px solid rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(10px);
}

/* BOT messages – frosted glass */
.bot-bubble {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.08) 0%, rgba(255, 255, 255, 0.05) 100%);
    color: var(--text-primary);
    border: 1px solid rgba(255, 255, 255, 0.15);
    backdrop-filter: blur(20px);

    /* subtle shadow to stand out from black bg */
    box-shadow: 
        0 8px 24px rgba(0, 0, 0, 0.25),
        inset 0 1px 0 rgba(255, 255, 255, 0.1);
}

/* Hover effect – slight nudge */
.chat-bubble:hover {
    transform: translateX(2px);
}

/* Avatar Design */
.avatar {
    width: 45px;
    height: 45px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.5rem;
    margin: 0 12px;
    position: relative;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
}

/* User avatar – matches primary red gradient */
.user-avatar {
    background: var(--primary-gradient);
    border: 2px solid rgba(255, 255, 255, 0.2);
    animation: avatarPulse 2s ease-in-out infinite;
}

/* Bot avatar – deep red gradient */
.bot-avatar {
    background: linear-gradient(135deg, #e53935 0%, #b71c1c 100%);
    border: 2px solid rgba(255, 255, 255, 0.2);
    animation: avatarRotate 3s linear infinite;
}

/* User avatar pulse – red glow instead of blue */
@keyframes avatarPulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(229, 57, 53, 0.4); }
    50%      { box-shadow: 0 0 0 10px rgba(229, 57, 53, 0); }
}

/* Bot avatar spin */
@keyframes avatarRotate {
    from { transform: rotate(0deg); }
    to   { transform: rotate(360deg); }
}

/* Quick Reply & Feedback Buttons */
.quick-reply-buttons .stButton > button,
.feedback-buttons .stButton > button {
    background: rgba(255, 255, 255, 0.05) !important;   /* frosted glass */
    color: var(--text-primary) !important;              /* white text */
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    border-radius: 50px !important;
    padding: 0.6rem 1.5rem !important;
    margin: 0.3rem !important;
    font-weight: 500 !important;
    backdrop-filter: blur(10px);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    position: relative;
    overflow: hidden;
}

/* Ripple glow effect from red gradient */
.quick-reply-buttons .stButton > button::before,
.feedback-buttons .stButton > button::before {
    content: '';
    position: absolute;
    top: 50%; left: 50%;
    width: 0; height: 0;
    background: var(--primary-gradient);    /* red gradient ripple */
    border-radius: 50%;
    transform: translate(-50%, -50%);
    transition: width 0.5s, height 0.5s;
    z-index: -1;
}

/* Hover: red glow + stronger red border */
.quick-reply-buttons .stButton > button:hover,
.feedback-buttons .stButton > button:hover {
    transform: translateY(-2px) scale(1.05) !important;
    box-shadow: 0 8px 20px rgba(229, 57, 53, 0.4) !important;   /* red aura */
    border-color: rgba(229, 57, 53, 0.5) !important;           /* red border */
}

.quick-reply-buttons .stButton > button:hover::before,
.feedback-buttons .stButton > button:hover::before {
    width: 200px;
    height: 200px;
}


/* Input Field - Modern Design */
.stTextInput > div > div > input {
    background: rgba(255, 255, 255, 0.03) !important;        /* frosted glass input */
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 15px !important;
    color: var(--text-primary) !important;                   /* white text */
    padding: 0.8rem 1.2rem !important;
    font-size: 1rem !important;
    backdrop-filter: blur(10px);
    transition: all 0.3s ease !important;
}

/* Focus state → red border + red glow */
.stTextInput > div > div > input:focus {
    border-color: rgba(229, 57, 53, 0.6) !important;         /* red border highlight */
    box-shadow: 0 0 20px rgba(229, 57, 53, 0.3) !important;  /* red glow */
    background: rgba(255, 255, 255, 0.05) !important;
}

/* Send Button */
.stButton > button[kind="secondary"] {
    background: var(--primary-gradient) !important;          /* red gradient */
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.8rem 1.2rem !important;
    font-weight: 600 !important;
    transition: all 0.3s cubic-bezier(0.4,0,0.2,1) !important;
}

/* Hover → red glow */
.stButton > button[kind="secondary"]:hover {
    transform: scale(1.05) !important;
    box-shadow: 0 10px 25px rgba(229, 57, 53, 0.5) !important;
}

/* Typing Indicator - Premium Animation */
.typing-indicator {
    display: flex;
    align-items: center;
    padding: 1rem 0;
}

.typing-dots {
    display: flex;
    gap: 4px;
    padding: 0 1rem;
}

.typing-dots span {
    width: 12px;
    height: 12px;
    background: var(--primary-gradient);          /* red gradient */
    border-radius: 50%;
    animation: typingPulse 1.4s infinite ease-in-out;

    /* red glow instead of purple */
    box-shadow: 0 0 10px rgba(229, 57, 53, 0.5);
}

.typing-dots span:nth-child(1) { animation-delay: 0s; }
.typing-dots span:nth-child(2) { animation-delay: 0.2s; }
.typing-dots span:nth-child(3) { animation-delay: 0.4s; }

@keyframes typingPulse {
    0%, 80%, 100% { 
        transform: scale(0.8);
        opacity: 0.5;
    }
    40% { 
        transform: scale(1.2);
        opacity: 1;
    }
}

/* Info Messages */
.stAlert {
    background: rgba(255, 255, 255, 0.03) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 15px !important;
    backdrop-filter: blur(10px);
    color: var(--text-primary) !important;
}

/* Scrollbar Styling */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: rgba(255, 255, 255, 0.02);
    border-radius: 10px;
}

/* Red scrollbar thumb */
::-webkit-scrollbar-thumb {
    background: linear-gradient(135deg, rgba(229, 57, 53, 0.5), rgba(191, 18, 18, 0.5));
    border-radius: 10px;
}

::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(135deg, rgba(229, 57, 53, 0.7), rgba(191, 18, 18, 0.7));
}

/* Loading Animation */
.loading-wave {
    display: inline-block;
    animation: wave 1.5s ease-in-out infinite;
}

@keyframes wave {
    0%, 100% { transform: translateY(0); }
    50%      { transform: translateY(-10px); }
}

/* Spacer Divs */
.transparent-spacer1 { height: 100px; background: transparent; }
.transparent-spacer2 { height: 50px; background: transparent; }

/* Hide Streamlit Elements */
footer { visibility: hidden; }
.stDeployButton { display: none; }

/* Custom Animations */
@keyframes float {
    0%, 100% { transform: translateY(0px); }
    50%      { transform: translateY(-10px); }
}

/* Glow animation → red glow */
@keyframes glow {
    0%, 100% { box-shadow: 0 0 20px rgba(229, 57, 53, 0.5); }
    50%      { box-shadow: 0 0 40px rgba(229, 57, 53, 0.8); }
}

/* Responsive Design */
@media (max-width: 768px) {
    .elegant-heading { font-size: 2.5rem !important; }
    .sidebar-title   { font-size: 3rem; }
    .chat-bubble     { max-width: 85%; }
    .main            { padding: 1.5rem !important; margin: 1rem !important; }
}

/* Performance Optimizations */
* {
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}

/* Status Indicator (system online ping) */
.status-indicator {
    display: inline-block;
    width: 8px;
    height: 8px;
    background: #4ade80;  /* green means online */
    border-radius: 50%;
    animation: statusPulse 2s ease-in-out infinite;
    margin-left: 8px;
}

@keyframes statusPulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(74, 222, 128, 0.4); }
    50%      { box-shadow: 0 0 0 8px rgba(74, 222, 128, 0); }
}
</style>
""", unsafe_allow_html=True)


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
            st.error(f"⚠️ **Error:** Missing required columns: {required_columns}")
            st.stop()
        embeddings = model.encode(df['questions'].tolist())
        nn_model = NearestNeighbors(n_neighbors=1, metric='cosine')
        nn_model.fit(np.array(embeddings))
        return df, nn_model
    except FileNotFoundError:
        st.error(f"⚠️ File not found at '{path}'")
        st.stop()
    except Exception as e:
        st.error(f"⚠️ Error loading KB: {e}")
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
        "hello": "Hello! 👋 Welcome to HCIL IT Support. How may I assist you today?",
        "hi": "Hi there! 🌟 Ready to help with your IT needs!",
        "hey": "Hey! 💫 What can I help you with today?",
        "greetings": "Greetings! 🎯 I'm here to assist with any IT issues.",
        "good morning": "Good morning! ☀️ How can I brighten your day with IT solutions?",
        "good afternoon": "Good afternoon! 🌤️ Ready to tackle any IT challenges!",
        "good evening": "Good evening! 🌙 How may I assist you?",
        "how are you": "I'm functioning optimally and ready to help! 🤖✨ What brings you here?",
        "what's up": "Ready to solve IT problems! 💪 What's on your mind?",
        "sup": "All systems operational! 🚀 How can I help?",
        "thank you": "You're very welcome! 🙏 Happy to help anytime!",
        "thanks": "My pleasure! ✨",
        "bye": "Thank you for using HCIL IT Support! **Sayonara!** 👋✨",
        "goodbye": "Until next time! **Mata ne!** 🌟 Have a great day!"
    }
    return responses.get(greet, "Hello! How can I help you?")

def get_bot_response(user_query, df, nn_model, model, chat_history=[]):
    if is_gibberish(user_query):
        return "I'm sorry, I couldn't understand your request. Could you please rephrase it?"

    greet = is_greeting(user_query)
    if greet:
        return get_greeting_response(greet)

    # Enhanced contextual understanding
    contextual_query = user_query
    if chat_history:
        # Give more weight to the most recent user message
        recent_user_msgs = [msg['content'] for msg in chat_history if msg['role'] == 'user']
        if recent_user_msgs:
            contextual_query = f"{recent_user_msgs[-1]} {user_query}"

    questions = df['questions'].tolist()
    
    # Hybrid Search: TF-IDF + Semantic
    # 1. TF-IDF for keyword matching
    vectorizer = TfidfVectorizer().fit(questions)
    query_tfidf = vectorizer.transform([contextual_query])
    question_tfidf = vectorizer.transform(questions)
    tfidf_scores = cosine_similarity(query_tfidf, question_tfidf).flatten()
    
    # 2. Semantic Search for contextual meaning
    query_embed = model.encode([contextual_query])
    distances, indices = nn_model.kneighbors(query_embed)
    semantic_scores = 1 - distances.flatten() # Convert distance to similarity

    # Combine scores with weighting
    hybrid_scores = (0.4 * tfidf_scores) + (0.6 * semantic_scores[indices.flatten()])
    
    best_idx = np.argmax(hybrid_scores)
    best_score = hybrid_scores[best_idx]

    # Dynamic threshold based on query length
    threshold = 0.65 if len(user_query.split()) > 3 else 0.75

    if best_score > threshold:
        return df.iloc[best_idx]['answers']
    else:
        # Fallback to fuzzy matching for short queries or typos
        best_match, score = process.extractOne(contextual_query, questions, scorer=fuzz.token_sort_ratio)
        if score > 80:
            idx = questions.index(best_match)
            return df.iloc[idx]['answers']
        
        return "I couldn't find an exact match in my knowledge base. Could you please provide more details or rephrase your question?"

def render_chat(messages):
    for msg in messages:
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="user-row" style="display: flex; justify-content: flex-end; align-items: flex-end;">
                <div class="chat-bubble user-bubble">{msg['content']}</div>
                <div class="avatar user-avatar">👨‍💻</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="bot-row" style="display: flex; justify-content: flex-start; align-items: flex-end;">
                <div class="avatar bot-avatar">🤖</div>
                <div class="chat-bubble bot-bubble">{msg['content']}</div>
            </div>
            """, unsafe_allow_html=True)

def show_typing():
    st.markdown("""
    <div class="typing-indicator">
        <div class="avatar bot-avatar">🤖</div>
        <div class="typing-dots">
            <span></span>
            <span></span>
            <span></span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# -------------------------------
# Sidebar Configuration
# -------------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-title">HCIL</span></div>', unsafe_allow_html=True)
    
    # System Status
    st.markdown("""
    <div style="background: rgba(255, 255, 255, 0.03); border-radius: 15px; padding: 1rem; margin: 1rem 0; border: 1px solid rgba(255, 255, 255, 0.1);">
        <h4 style="color: #4ade80; margin: 0;">🟢 System Online</h4>
        <p style="color: rgba(255, 255, 255, 0.7); font-size: 0.9rem; margin: 0.5rem 0 0 0;">This is an AI-powered IT Helpdesk chatbot for Honda Cars India that instantly answers repetitive IT queries, saving time and boosting support efficiency.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick Stats
    if 'messages' in st.session_state and len(st.session_state.messages) > 0:
        msg_count = len([m for m in st.session_state.messages if m['role'] == 'user'])
        st.markdown(f"""
        <div style="background: rgba(102, 126, 234, 0.1); border-radius: 12px; padding: 0.8rem; margin: 1rem 0; border: 1px solid rgba(102, 126, 234, 0.3);">
            <p style="margin: 0; font-size: 0.9rem;">Messages: {msg_count}</p>
            <p style="margin: 0; font-size: 0.9rem;">Response Time: ~1.2s</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.info("💡**Pro Tip:** Type 'bye' to end the conversation")

# -------------------------------
# Session State Initialization
# -------------------------------
defaults = {
    'knowledge_base_loaded': False,
    'messages': [],
    'chat_ended': False,
    'feedback_request': False,
    'quick_replies': ["Reset Password", "VPN Issues", "Software Install", "Hardware Problems"],
    'show_typing': False,
    'chat_started': False,
    'show_quick_replies': False
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# Main Title with Animation
st.markdown(""" <div style="display: flex; justify-content: center; align-items: center; gap: 1rem;"> <span class="loading-wave" style=" font-size: 5.5rem; background: linear-gradient(135deg, #b71c1c 0%, #e53935 50%, #f87171 100%); -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent; color: transparent; ">🤖</span> <h1 class='elegant-heading' style="margin-right: -5.2rem;">HCIL IT Helpdesk AI Assistant</h1> </div> """, unsafe_allow_html=True)


st.markdown('<div class="transparent-spacer1"></div>', unsafe_allow_html=True)

# -------------------------------
# Chat App Flow
# -------------------------------
if not st.session_state.chat_started:
    # Welcome Message
    st.markdown("""
    <div style="text-align: center; margin: 4rem 0;">
        <p style="color: rgba(255, 255, 255, 0.8); font-size: 1.1rem; margin-bottom: 2rem;">
            Experience next-generation IT support powered by AI
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Start Chat Button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Start Chat", key="start_chat_button", use_container_width=True):
            st.session_state.chat_started = True
            st.session_state.show_quick_replies = True
            welcome_messages = [
                "<b>Konnichiwa!</b> Welcome to HCIL's AI Support System. How may I assist you today?",
                "<b>Welcome!</b> I'm your AI assistant, ready to solve any IT challenge!",
                "<b>Hello!</b> Let's get your IT issues resolved quickly and efficiently!"
            ]
            st.session_state.messages = [{
                "role": "bot",
                "content": random.choice(welcome_messages)
            }]
            st.rerun()

else:
    st.markdown('<div class="main">', unsafe_allow_html=True)  # Apply the main container class

    if st.session_state.knowledge_base_loaded:
        # Chat History
        render_chat(st.session_state.messages)

        if st.session_state.chat_ended:
            time.sleep(2)
            for key in ['messages', 'feedback_request', 'show_typing', 'chat_started', 'show_quick_replies']:
                st.session_state[key] = defaults[key]
            st.session_state.chat_ended = False
            st.rerun()

        if st.session_state.show_typing:
            show_typing()

        # Quick Replies with Enhanced Design
        if st.session_state.show_quick_replies:
            st.markdown("""
            <div style="margin: 1.5rem 0;">
                <p style="color: rgba(255, 255, 255, 0.6); font-size: 0.9rem; margin-bottom: 0.8rem;">
                    Quick Actions:
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown('<div class="quick-reply-buttons">', unsafe_allow_html=True)
            cols = st.columns(len(st.session_state.quick_replies))
            for idx, (col, reply) in enumerate(zip(cols, st.session_state.quick_replies)):
                with col:
                    if st.button(reply, key=f"quick_{idx}", use_container_width=True):
                        clean_reply = reply.split(' ', 1)[1] if ' ' in reply else reply
                        st.session_state.messages.append({"role": "user", "content": clean_reply})
                        st.session_state.show_typing = True
                        st.session_state.show_quick_replies = False
                        st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        # Process typing animation
        if st.session_state.show_typing:
            time.sleep(1.2)
            last_user_msg = next((msg["content"] for msg in reversed(st.session_state.messages) if msg["role"] == "user"), None)
            if last_user_msg:
                response = get_bot_response(last_user_msg, st.session_state.df, st.session_state.nn_model, model, chat_history=st.session_state.messages)
                st.session_state.messages.append({"role": "bot", "content": response})
                st.session_state.feedback_request = True
                st.session_state.show_typing = False
                st.rerun()
    else:
        st.info("🔄 Loading AI Knowledge Base...")

# -------------------------------
# Input Bar + Feedback Section
# -------------------------------
if st.session_state.chat_started and not st.session_state.chat_ended:

    # Spacer when feedback request is shown
    if st.session_state.feedback_request:
        st.markdown('<div class="transparent-spacer2"></div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1)); 
                    border-radius: 20px; padding: 0.8rem; margin: 1.0rem 0; 
                    border: 1px solid rgba(229, 57, 53, 0.3);">
            <p style="color: rgba(255, 255, 255, 0.9); font-size: 1.5rem; margin-bottom: 0rem; text-align: center;">
                ✨ Was this response helpful?
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="feedback-buttons">', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        
        feedback_options = [
            ("😊 Perfect", "Excellent! I'm here if you need anything else! ✨"),
            ("🤔 Unclear", "Let me try to explain differently. Could you provide more details? 💭"),
            ("👎 Not Helpful", "I apologize. Let me connect you with better resources. 🔄"),
            ("💬 More Help", "Sure! What else would you like to know? 💡")
        ]
        
        for col, (btn_text, response) in zip([col1, col2, col3, col4], feedback_options):
            with col:
                if st.button(btn_text, use_container_width=True):
                    st.session_state.messages.append({"role": "bot", "content": response})
                    st.session_state.feedback_request = False
                    st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)


    
    # Modern Input Form
    st.markdown("""
    <div style="margin-top: 2rem; padding-top: 1rem; border-top: 1px solid rgba(255, 255, 255, 0.1); padding-bottom: 2rem;">
    </div>
    """, unsafe_allow_html=True)

    
    
    with st.form("chat_input_form", clear_on_submit=True):
        col1, col2 = st.columns([6, 1])
        with col1:
            user_input = st.text_input(
                "user_input", 
                placeholder="Type your message here......", 
                key="input_bar", 
                label_visibility="collapsed"
            )
        with col2:
            send_clicked = st.form_submit_button("Send ▶️", use_container_width=True)
        
        if send_clicked and user_input.strip():
            user_input_clean = user_input.lower().strip()
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            if user_input_clean in ["bye", "quit", "exit", "end"]:
                farewell_messages = [
                    "Thank you for using HCIL IT Support! <b>Mata ne!</b> 🌟 Have an amazing day!",
                    "It was great helping you! <b>Mata ne!</b> ✨ See you next time!",
                    "Thanks for choosing HCIL! <b>Goodbye!</b> 🚀 Stay awesome!"
                ]
                st.session_state.messages.append({
                    "role": "bot", 
                    "content": random.choice(farewell_messages)
                })
                st.session_state.chat_ended = True
                st.session_state.feedback_request = False
                st.session_state.show_typing = False
            else:
                st.session_state.show_typing = True
                st.session_state.show_quick_replies = False
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True) # Close the main container div

# Footer
st.markdown("""
<div style="margin-top: 3rem; padding-top: 2rem; border-top: 1px solid rgba(255, 255, 255, 0.1); text-align: center;">
    <p style="color: rgba(255, 255, 255, 0.8); font-size: 0.85rem;">
        Powered by Advanced AI | HCIL IT Support © 2025 ~ By Kunal Bhardwaj
    </p>
</div>
""", unsafe_allow_html=True)
