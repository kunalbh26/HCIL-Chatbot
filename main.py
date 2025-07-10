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
# Custom CSS (Final, Forceful Version)
# -------------------------------
st.markdown(f"""
<style>
/* --- BASE & UTILITIES --- */
html, body, .stApp {{
    background-color: #1F1F1F !important;
    color: white !important;
}}
/* Hide Streamlit's default "Made with Streamlit" footer */
.st-emotion-cache-1c7y2kd {{
    display: none;
}}

/* --- MAIN CHAT & SIDEBAR --- */
/* Main chat area: Add massive padding at the bottom to make space for the pinned input bar */
.main > .block-container {{
    padding: 2rem 1rem 18rem 1rem !important;
}}
.stSidebar > div:first-child {{
    background-color: #323232 !important;
    border-right: 2px solid white;
}}
.sidebar-title {{
    font-size: 5.5rem; color: #EE4B2B; font-weight: 900; text-align: center; margin: 0.5rem 0 1.5rem 0;
    animation: rotate3D 5s infinite linear; transform-style: preserve-3d; perspective: 800px;
    text-shadow: 0 0 5px rgba(238,75,43,0.5), 1px 1px 2px rgba(0,0,0,0.8);
}}

/* --- PRE-CHAT / START SCREEN (FIX 1) --- */
.start-screen-container {{
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    min-height: 60vh; /* Use viewport height to center vertically */
}}
.elegant-heading {{
    font-size: 4.5rem !important; font-weight: 800; color: #ffffff;
    animation: fadeInUp 1s ease-out;
}}
.start-chat-button .stButton button {{
    background: linear-gradient(90deg, #e53935 0%, #b71c1c 100%);
    color: #fff; border-radius: 20px; padding: 1rem 2.5rem;
    font-size: 1.5rem !important; border: 2px solid #fff; font-weight: bold;
    transition: transform 0.2s, box-shadow 0.2s;
    margin-top: 2rem; /* Space below title */
}}
.start-chat-button .stButton button:hover {{
    transform: scale(1.05); box-shadow: 0 0 20px rgba(255, 255, 255, 0.3);
    color: #fff !important; border: 2px solid #fff !important;
}}

/* --- CHAT BUBBLES --- */
.chat-bubble {{
    padding: 1rem 1.5rem; border-radius: 20px; margin-bottom: 1rem; max-width: 85%;
    word-break: break-word; font-size: 1.05rem; display: flex; align-items: center;
    animation: fadeInUp 0.5s;
}}
.user-bubble {{ background: #fff; color: #111; align-self: flex-end; margin-left: auto; border: 1.5px solid #e53935; }}
.bot-bubble {{ background: linear-gradient(90deg, #e53935 0%, #b71c1c 100%); color: #fff; align-self: flex-start; border: 1.5px solid #fff; }}
.avatar {{ width: 40px; height: 40px; border-radius: 50%; margin: 0 10px; background: #3d3d3d; border: 2px solid #ff0000; display: flex; align-items: center; justify-content: center; font-size: 1.5rem; }}
.user-row {{ display: flex; flex-direction: row; align-items: flex-end; justify-content: flex-end; }}
.bot-row {{ display: flex; flex-direction: row; align-items: flex-end; justify-content: flex-start; }}

/* --- PINNED INPUT BAR (FIX 2) --- */
.pinned-input-container {{
    position: fixed;
    bottom: 0;
    left: 0;
    width: 100%;
    z-index: 1000;
    background: linear-gradient(to top, rgba(31,31,31,1) 70%, transparent 100%); /* Gradient effect */
    pointer-events: none; /* Allow scrolling through the container */
}}
.pinned-input-inner {{
    max-width: 640px; /* Match main chat width */
    margin: 0 auto;
    padding: 1rem;
    pointer-events: auto; /* Re-enable interaction for the contents */
}}
/* Target Streamlit's form element directly to remove unwanted margins */
.pinned-input-inner .st-emotion-cache-1629p8f {{
    border: none;
    padding: 0;
}}
.feedback-container {{
    display: flex; justify-content: center; gap: 0.5rem;
    margin-bottom: 0.75rem; /* TIGHT spacing between feedback and input */
}}
.feedback-container .stButton button {{
    background-color: #3d3d3d; font-size: 1.2rem; border-radius: 50%;
    width: 40px; height: 40px; border: 1px solid #fff;
}}
.feedback-container .stButton button:hover {{ transform: scale(1.1); border-color: #e53935; }}
.pinned-input-inner .stTextInput > div > div > input {{
    background: #323232; border: 2px solid #ff0000; color: #fff;
    padding: 0.9rem 1rem; font-size: 1.1rem; border-radius: 18px;
}}
.pinned-input-inner .stButton[kind="form_submit"] button {{
    background: linear-gradient(90deg, #e53935 0%, #b71c1c 100%); color: #fff;
    border: none; border-radius: 50%; width: 45px; height: 45px; font-size: 1.3rem;
}}
.pinned-input-inner .stButton[kind="form_submit"] button:hover {{ background: #fff; color: #e53935; }}

/* --- VERTICAL QUICK REPLIES --- */
.quick-reply-container {{
    display: flex; flex-direction: column; align-items: center;
    gap: 0.5rem; margin-bottom: 1.5rem;
}}
.quick-reply-container .stButton button {{
    background: #fff; color: #e53935; border-radius: 18px; padding: 0.5rem 1.1rem;
    font-size: 0.98rem; border: 1.5px solid #e53935; font-weight: 500;
    width: 250px;
}}
.quick-reply-container .stButton button:hover {{ background: #e53935; color: #fff; }}

/* --- ANIMATIONS --- */
@keyframes fadeInUp {{
    from {{ opacity: 0; transform: translateY(20px); }}
    to {{ opacity: 1; transform: translateY(0); }}
}}
@keyframes rotate3D {{
    0% {{ transform: rotateY(-15deg); }} 50% {{ transform: rotateY(15deg); }}
    100% {{ transform: rotateY(-15deg); }}
}}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Page & Model Configuration
# -------------------------------
st.set_page_config(page_title="HCIL IT Helpdesk Chat-Bot", page_icon="🤖", layout="centered", initial_sidebar_state="auto")

@st.cache_resource
def load_models():
    """Load all models and data once."""
    sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
    try:
        df = pd.read_excel(KNOWLEDGE_BASE_PATH)
        required_columns = {'questions', 'answers', 'categories', 'tags'}
        if not required_columns.issubset(df.columns):
            st.error(f"❌ **Error:** Knowledge base file missing required columns.")
            st.stop()
        embeddings = sentence_model.encode(df['questions'].tolist())
        nn_model = NearestNeighbors(n_neighbors=1, metric='cosine')
        nn_model.fit(np.array(embeddings))
        return sentence_model, df, nn_model
    except FileNotFoundError:
        st.error(f"❌ **Error:** Knowledge base file not found at '{KNOWLEDGE_BASE_PATH}'.")
        st.stop()
    except Exception as e:
        st.error(f"❌ An error occurred while loading the knowledge base: {e}")
        st.stop()

model, df, nn_model = load_models()

# -------------------------------
# Helper Functions
# -------------------------------
def get_bot_response(user_query):
    """Generates a response from the knowledge base."""
    # Handle greetings
    greetings = {"hello": "Hello! 👋 How can I help you today?", "hi": "Hi there! How can I assist you?", "hey": "Hey! What can I do for you?", "thanks": "You're welcome!", "thank you": "You're welcome! Is there anything else?"}
    for greet, response in greetings.items():
        if fuzz.ratio(user_query.lower(), greet) > 85:
            return response

    # Handle gibberish
    if len(user_query.split()) < 2 and len(user_query) < 5:
        return "I'm sorry, I couldn't understand that. Could you please rephrase your question with more detail?"

    # Find best match using a combination of fuzzy matching and semantic search
    best_fuzzy_match, fuzzy_score = process.extractOne(user_query, df['questions'].tolist(), scorer=fuzz.token_sort_ratio)

    query_embed = model.encode([user_query])
    distances, indices = nn_model.kneighbors(query_embed)
    semantic_distance = distances[0][0]
    best_semantic_idx = indices[0][0]

    # If fuzzy score is very high, trust it. Otherwise, use semantic search result.
    if fuzzy_score > 85:
        best_idx = df[df['questions'] == best_fuzzy_match].index[0]
    else:
        best_idx = best_semantic_idx

    # If the semantic distance is too high, the query is likely out of scope.
    if semantic_distance > 0.5 and fuzzy_score < 80:
        return "I'm sorry, I don't have information on that topic. Could you please ask something related to IT issues?"

    return df.iloc[best_idx]['answers']

def render_chat_history():
    """Displays the chat messages."""
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="user-row"><div class="chat-bubble user-bubble">{msg["content"]}</div><div class="avatar">🧑‍💻</div></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bot-row"><div class="avatar">🤖</div><div class="chat-bubble bot-bubble">{msg["content"]}</div></div>', unsafe_allow_html=True)

# -------------------------------
# Session State Initialization
# -------------------------------
if 'messages' not in st.session_state: st.session_state.messages = []
if 'chat_started' not in st.session_state: st.session_state.chat_started = False
if 'chat_ended' not in st.session_state: st.session_state.chat_ended = False
if 'feedback_request' not in st.session_state: st.session_state.feedback_request = False
if 'quick_replies' not in st.session_state: st.session_state.quick_replies = ["Reset password", "VPN issues", "Software install"]

# -------------------------------
# Sidebar
# -------------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-title">HCIL</div>', unsafe_allow_html=True)
    st.info("Say 'bye', 'quit', or 'end' to close our chat.")

# -------------------------------
# Main Application UI
# -------------------------------
if not st.session_state.chat_started:
    # --- Show Start Screen ---
    st.markdown('<div class="start-screen-container">', unsafe_allow_html=True)
    st.markdown("<h1 class='elegant-heading'>🤖 HCIL IT Helpdesk Chatbot</h1>", unsafe_allow_html=True)
    st.markdown('<div class="start-chat-button">', unsafe_allow_html=True)
    if st.button("Start Chat", key="start_chat_main"):
        st.session_state.chat_started = True
        st.session_state.messages = [{"role": "bot", "content": "👋 **Konnichiwa!** How can I help you today?"}]
        st.rerun()
    st.markdown('</div></div>', unsafe_allow_html=True)

else:
    # --- Show Main Chat Interface ---
    render_chat_history()

    # If chat has ended, show final message and then reset
    if st.session_state.chat_ended:
        time.sleep(2)
        st.session_state.clear() # Clear the entire session state for a clean restart
        st.rerun()

    # Show VERTICAL quick replies only at the very start of the chat
    if len(st.session_state.messages) == 1:
        st.markdown('<div class="quick-reply-container">', unsafe_allow_html=True)
        for reply in st.session_state.quick_replies:
            if st.button(reply, key=f"quick_reply_{reply}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": reply})
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # --- Pinned Input Bar ---
    st.markdown('<div class="pinned-input-container">', unsafe_allow_html=True)
    st.markdown('<div class="pinned-input-inner">', unsafe_allow_html=True)

    # Display feedback buttons when requested
    if st.session_state.feedback_request:
        st.markdown('<div class="feedback-container">', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        if col1.button("👍", key="feedback_like", use_container_width=True):
            st.session_state.messages.append({"role": "bot", "content": "Great! Let me know if I can help with anything else."})
            st.session_state.feedback_request = False
            st.rerun()
        if col2.button("👎", key="feedback_dislike", use_container_width=True):
            st.session_state.messages.append({"role": "bot", "content": "I apologize. Could you please rephrase your question?"})
            st.session_state.feedback_request = False
            st.rerun()
        if col3.button("❤️", key="feedback_love", use_container_width=True):
            st.session_state.messages.append({"role": "bot", "content": "Thank you for your feedback! 😊"})
            st.session_state.feedback_request = False
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # The actual input form
    with st.form("chat_input_form", clear_on_submit=True):
        col1, col2 = st.columns([10, 2])
        with col1:
            user_input = st.text_input("user_input", placeholder="Ask about IT issues...", key="input_bar", label_visibility="collapsed")
        with col2:
            send_clicked = st.form_submit_button("➤")

        if send_clicked and user_input.strip():
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.session_state.feedback_request = False # Hide feedback while processing
            st.rerun()
            
    st.markdown('</div></div>', unsafe_allow_html=True)


    # --- Bot Response Logic ---
    last_message = st.session_state.messages[-1]
    if last_message["role"] == "user":
        # Check for end-of-chat keywords
        if last_message["content"].lower().strip() in ["bye", "end", "quit", "goodbye"]:
            bot_response = "Thank you for chatting, **Mata Ne!** (see you later) 👋"
            st.session_state.messages.append({"role": "bot", "content": bot_response})
            st.session_state.chat_ended = True
            st.rerun()
        else:
            with st.spinner("🤖 Typing..."):
                time.sleep(0.6) # Faster, but still perceptible, typing delay
                bot_response = get_bot_response(last_message["content"])
            st.session_state.messages.append({"role": "bot", "content": bot_response})
            st.session_state.feedback_request = True
            st.rerun()
