import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import time
import re

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="HCIL IT Helpdesk",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="auto",
)

# -------------------------------
# Custom Styling and Assets
# -------------------------------
def load_css(file_name):
    """Function to load a local CSS file."""
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Inject custom CSS
load_css("style.css")

# -------------------------------
# Model Loading (Cached)
# -------------------------------
@st.cache_resource
def load_sentence_transformer():
    """Loads the sentence transformer model, cached for performance."""
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_sentence_transformer()

# -------------------------------
# Helper Functions
# -------------------------------
def get_bot_response(user_query, df, nn_model, model):
    """
    Finds the best response from the knowledge base for a given user query.
    """
    query_embed = model.encode([user_query])
    distances, indices = nn_model.kneighbors(query_embed)
    best_idx = indices[0][0]
    
    # A simple threshold to see if the result is relevant enough
    if distances[0][0] > 0.6: # This threshold may need tuning
        return "I'm not sure I have information on that. Could you try rephrasing your question?"
        
    return df.iloc[best_idx]['answers']

def get_farewell_message():
    return "Thank you for chatting. Feel free to start a new conversation anytime! 👋"

def handle_greetings(query):
    """Handles common greetings and returns a specific response."""
    greetings = {
        "hello": "Hello! How can I assist you with your IT questions today?",
        "hi": "Hi there! What can I help you with?",
        "hey": "Hey! I'm here to help with your IT issues.",
        "how are you": "I'm just a bot, but I'm running at full capacity! What can I do for you?"
    }
    # Use regex for flexible matching
    for greeting, response in greetings.items():
        if re.search(r'\b' + greeting + r'\b', query, re.IGNORECASE):
            return response
    return None

# -------------------------------
# Session State Initialization
# -------------------------------
if 'knowledge_base_loaded' not in st.session_state:
    st.session_state.knowledge_base_loaded = False
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'chat_ended' not in st.session_state:
    st.session_state.chat_ended = False

# -------------------------------
# Sidebar Configuration
# -------------------------------
with st.sidebar:
    st.image("https://i.imgur.com/vL5aF4i.png", width=80) # Using a white logo for dark bg
    st.header("⚙️ Bot Configuration")
    
    with st.expander("📂 Upload Knowledge Base", expanded=True):
        uploaded_file = st.file_uploader(
            "Upload an Excel File",
            type=["xlsx"],
            help="Upload an Excel file with 'questions' and 'answers' columns."
        )

    if st.button("🔄 Reset Chat Session"):
        for key in list(st.session_state.keys()):
            if key not in ['knowledge_base_loaded', 'df', 'nn_model']: # Keep KB loaded
                del st.session_state[key]
        st.rerun()

    st.caption("Built with ❤️ for HCIL")

# -------------------------------
# Load Knowledge Base
# -------------------------------
if uploaded_file is not None and not st.session_state.knowledge_base_loaded:
    with st.spinner("🚀 Initializing bot... This may take a moment."):
        try:
            df = pd.read_excel(uploaded_file)
            required_columns = {'questions', 'answers'}
            if not required_columns.issubset(df.columns):
                st.error("❌ Error: Missing required columns. Please ensure 'questions' and 'answers' columns exist.")
            else:
                st.session_state.df = df
                embeddings = model.encode(df['questions'].tolist())
                nn_model = NearestNeighbors(n_neighbors=1, metric='cosine', algorithm='brute')
                nn_model.fit(np.array(embeddings))
                
                st.session_state.nn_model = nn_model
                st.session_state.knowledge_base_loaded = True
                st.session_state.messages = [] # Clear messages on new KB
                st.session_state.chat_ended = False

                st.success("✅ Knowledge base loaded! The bot is ready.")
                time.sleep(1.5)
                st.rerun()
        except Exception as e:
            st.error(f"❌ An error occurred while processing the file: {e}")

# --- Message rendering functions ---
def render_bot_message(content):
    st.markdown(f"""
    <div class="message-container bot-message">
        <div class="avatar-bot">🤖</div>
        <div class="bubble">{content}</div>
    </div>
    """, unsafe_allow_html=True)

def render_user_message(content):
    st.markdown(f"""
    <div class="message-container user-message">
        <div class="bubble">{content}</div>
    </div>
    """, unsafe_allow_html=True)

def render_typing_indicator():
    st.markdown("""
    <div class="message-container bot-message">
        <div class="avatar-bot">🤖</div>
        <div class="bubble typing-indicator">
            <span></span><span></span><span></span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# -------------------------------
# Main Chat Interface
# -------------------------------
if not st.session_state.knowledge_base_loaded:
    st.info("👋 Welcome! Please upload a knowledge base file in the sidebar to start.")
else:
    # --- Chat Header ---
    st.markdown('<div class="chat-header">HCIL IT Helpdesk</div>', unsafe_allow_html=True)
    
    # --- Chat Messages ---
    message_container = st.container()
    with message_container:
        if not st.session_state.messages:
            st.session_state.messages.append({"role": "assistant", "content": "Hi there! I'm your friendly IT Helpdesk bot. How can I assist you today?"})

        for message in st.session_state.messages:
            if message["role"] == "assistant":
                render_bot_message(message["content"])
            else:
                render_user_message(message["content"])

    # --- Chat Input ---
    # Disable input if chat has ended
    if st.session_state.chat_ended:
        st.info("This chat has ended. Please press 'Reset Chat Session' in the sidebar to start a new one.")
    else:
        if prompt := st.chat_input("Ask me an IT question..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            render_user_message(prompt) # Immediately show user message

            farewells = ["bye", "end", "quit", "exit", "goodbye"]
            cleaned_prompt = prompt.lower().strip()

            # 1. Check for farewells
            if any(farewell in cleaned_prompt for farewell in farewells):
                farewell_message = get_farewell_message()
                st.session_state.messages.append({"role": "assistant", "content": farewell_message})
                st.session_state.chat_ended = True
                st.rerun()

            # 2. Check for greetings
            elif (greeting_response := handle_greetings(cleaned_prompt)):
                st.session_state.messages.append({"role": "assistant", "content": greeting_response})
                st.rerun()
            
            # 3. Process with AI model
            else:
                with st.spinner("Thinking..."):
                    render_typing_indicator()
                    time.sleep(1.5) # Simulate thinking and network latency
                    bot_response = get_bot_response(prompt, st.session_state.df, st.session_state.nn_model, model)
                    st.session_state.messages.append({"role": "assistant", "content": bot_response})
                    st.rerun()
