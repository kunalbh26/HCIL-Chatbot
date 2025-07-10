import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import time
import re
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

# Configuration
KNOWLEDGE_BASE_PATH = 'dataset.xlsx'

# Custom CSS
st.markdown("""
<style>
    /* Main app styling */
    html, body, .stApp {
        background-color: #1F1F1F !important;
        color: white !important;
    }
    
    /* Chat container */
    .chat-container {
        max-width: 640px;
        margin: 2rem auto;
        padding-bottom: 120px; /* Space for fixed input */
    }
    
    /* Start Chat button */
    .start-button-container {
        display: flex;
        justify-content: center;
        margin: 3rem 0;
    }
    .start-chat-button {
        background: linear-gradient(90deg, #e53935 0%, #b71c1c 100%);
        color: white;
        border: 2px solid white;
        border-radius: 20px;
        padding: 1.2rem 2.5rem;
        font-size: 1.3rem;
        font-weight: bold;
        cursor: pointer;
        transition: transform 0.2s;
    }
    .start-chat-button:hover {
        transform: scale(1.05);
    }
    
    /* Chat bubbles */
    .message-row {
        display: flex;
        margin-bottom: 1rem;
    }
    .bot-row {
        justify-content: flex-start;
    }
    .user-row {
        justify-content: flex-end;
    }
    .chat-bubble {
        padding: 1rem 1.5rem;
        border-radius: 20px;
        max-width: 75%;
        word-break: break-word;
    }
    .bot-bubble {
        background: linear-gradient(90deg, #e53935 0%, #b71c1c 100%);
        color: white;
        border: 1.5px solid white;
    }
    .user-bubble {
        background: white;
        color: #111;
        border: 1.5px solid #e53935;
    }
    
    /* Quick replies - shown only initially */
    .quick-replies {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin: 1rem 0 2rem 0;
        justify-content: center;
    }
    .quick-reply-btn {
        background: white;
        color: #e53935;
        border: 1.5px solid #e53935;
        border-radius: 18px;
        padding: 0.5rem 1rem;
        cursor: pointer;
        transition: all 0.2s;
    }
    .quick-reply-btn:hover {
        background: #e53935;
        color: white;
    }
    
    /* Fixed input bar */
    .input-container {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: #1F1F1F;
        padding: 1rem;
        z-index: 100;
    }
    .input-inner {
        max-width: 640px;
        margin: 0 auto;
        display: flex;
        gap: 0.5rem;
    }
    .chat-input {
        flex-grow: 1;
        padding: 0.8rem;
        border-radius: 20px;
        border: 2px solid #ff0000;
        background: #323232;
        color: white;
    }
    .send-btn {
        background: linear-gradient(90deg, #e53935 0%, #b71c1c 100%);
        color: white;
        border: none;
        border-radius: 50%;
        width: 48px;
        height: 48px;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
    }
</style>
""", unsafe_allow_html=True)

# Load models and data
@st.cache_resource
def load_resources():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    try:
        df = pd.read_excel(KNOWLEDGE_BASE_PATH)
        embeddings = model.encode(df['questions'].tolist())
        nn_model = NearestNeighbors(n_neighbors=1, metric='cosine').fit(embeddings)
        return model, df, nn_model
    except Exception as e:
        st.error(f"Error loading resources: {e}")
        st.stop()

model, df, nn_model = load_resources()

# Helper functions
def is_gibberish(text):
    return len(text.strip()) < 2 or not any(c.isalpha() for c in text)

def is_greeting(text):
    greetings = ["hello", "hi", "hey", "greetings"]
    text = text.lower()
    return any(greet in text for greet in greetings)

def get_response(query):
    if is_gibberish(query):
        return "Could you please rephrase that?"
    
    if is_greeting(query):
        return "Hello! How can I help you today?"
    
    # Fuzzy matching
    questions = df['questions'].tolist()
    best_match, score = process.extractOne(query, questions, scorer=fuzz.token_sort_ratio)
    
    if score > 70:
        return df.iloc[questions.index(best_match)]['answers']
    
    # Semantic matching
    query_embed = model.encode([query])
    distances, indices = nn_model.kneighbors(query_embed)
    
    if distances[0][0] <= 0.45:
        return df.iloc[indices[0][0]]['answers']
    
    return "I'm not sure I understand. Could you provide more details?"

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
    st.session_state.show_quick_replies = False

# Main app layout
st.markdown("<h1 style='text-align: center; margin-bottom: 0.5rem;'>🤖 HCIL IT Assistant Bot</h1>", unsafe_allow_html=True)

if not st.session_state.get('chat_started', False):
    # Start chat screen
    st.markdown("""
    <div class="start-button-container">
        <button class="start-chat-button" onclick="window.startChat()">Start Chat</button>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("Start Chat", key="start_chat_main"):
        st.session_state.chat_started = True
        st.session_state.messages = [{"role": "bot", "content": "👋 Hello! How can I help you today?"}]
        st.session_state.show_quick_replies = True
        st.rerun()
else:
    # Chat interface
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    for msg in st.session_state.messages:
        row_class = "bot-row" if msg["role"] == "bot" else "user-row"
        bubble_class = "bot-bubble" if msg["role"] == "bot" else "user-bubble"
        
        st.markdown(f"""
        <div class="message-row {row_class}">
            <div class="chat-bubble {bubble_class}">{msg['content']}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Show quick replies after first message if not shown yet
        if msg["role"] == "bot" and st.session_state.show_quick_replies:
            st.markdown('<div class="quick-replies">', unsafe_allow_html=True)
            st.write("Quick replies:")
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Reset Password", key="qr1"):
                    st.session_state.messages.append({"role": "user", "content": "Reset password"})
                    st.session_state.show_quick_replies = False
                    st.rerun()
            with col2:
                if st.button("VPN Issues", key="qr2"):
                    st.session_state.messages.append({"role": "user", "content": "VPN issues"})
                    st.session_state.show_quick_replies = False
                    st.rerun()
            with col3:
                if st.button("Software", key="qr3"):
                    st.session_state.messages.append({"role": "user", "content": "Software install"})
                    st.session_state.show_quick_replies = False
                    st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
            st.session_state.show_quick_replies = False
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Fixed input bar at bottom
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    st.markdown('<div class="input-inner">', unsafe_allow_html=True)
    
    query = st.text_input("Type your message...", key="user_input", label_visibility="collapsed", 
                         help="Press Enter to send")
    
    if st.button("➔", key="send_button", help="Send message"):
        if query.strip():
            st.session_state.messages.append({"role": "user", "content": query})
            response = get_response(query)
            st.session_state.messages.append({"role": "bot", "content": response})
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# JavaScript for button click handler
components.html("""
<script>
    window.startChat = function() {
        window.location.href = window.location.pathname + '?start_chat=true';
    }
</script>
""", height=0)
