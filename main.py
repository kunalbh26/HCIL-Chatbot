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
# Session State Initialization
# -------------------------------
def init_session_state():
    """Initializes all necessary session state variables."""
    if 'knowledge_base_loaded' not in st.session_state:
        st.session_state.knowledge_base_loaded = False
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'sidebar_state' not in st.session_state:
        st.session_state.sidebar_state = 'expanded'
    if 'feedback_given' not in st.session_state:
        st.session_state.feedback_given = True # Start with true to not show buttons initially

init_session_state()

# -------------------------------
# Helper Functions
# -------------------------------
def get_bot_response(user_query, df, nn_model, model):
    """
    Finds the best response from the knowledge base for a given user query.
    """
    with st.spinner("Thinking... 🤔"):
        query_embed = model.encode([user_query])
        distances, indices = nn_model.kneighbors(query_embed)
        
        # Adding a threshold for relevance
        if distances[0][0] < 0.6: # Cosine distance threshold
            best_idx = indices[0][0]
            response = df.iloc[best_idx]['answers']
        else:
            response = "I'm sorry, I don't have an answer for that. Could you please try rephrasing your question?"
        
        time.sleep(1) # Simulate thinking
    return response

def handle_greetings_and_farewells(prompt):
    """Handles common greetings and farewells for a more natural conversation."""
    greetings = r"\b(hi|hello|hey|yo|how are you)\b"
    farewells = r"\b(bye|quit|exit|end|goodbye)\b"
    
    if re.search(greetings, prompt, re.IGNORECASE):
        return "Hello there! How can I assist you with your IT questions today?"
    
    if re.search(farewells, prompt, re.IGNORECASE):
        return "Thank you for chatting. Have a great day! 👋"
        
    return None

# -------------------------------
# Sidebar Configuration & Logic
# -------------------------------
# ** FIX: Initialize uploaded_file to None to prevent NameError **
uploaded_file = None 

with st.sidebar:
    # Toggle button for sidebar is removed as Streamlit handles it.
    # The sidebar can be closed by the user with the 'X' button.
    
    st.markdown("<div class='sidebar-title'>HCIL</div>", unsafe_allow_html=True)
    st.header("⚙️ Bot Configuration")
    
    with st.expander("📂 Upload Knowledge Base", expanded=True):
        uploaded_file = st.file_uploader(
            "Upload your Excel File",
            type=["xlsx"],
            help="Upload an Excel file with 'questions' and 'answers' columns."
        )
    st.caption("Built with ❤️ for HCIL")


# -------------------------------
# Load Knowledge Base
# -------------------------------
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None

if uploaded_file is not None and uploaded_file != st.session_state.uploaded_file:
    with st.spinner("🚀 Initializing bot... This might take a moment."):
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
                st.session_state.uploaded_file = uploaded_file
                st.session_state.messages = [] # Reset chat on new file
                
                st.success("✅ Knowledge base loaded! The bot is ready.")
                time.sleep(1.5)
                st.rerun()

        except Exception as e:
            st.error(f"❌ An error occurred while processing the file: {e}")

# -------------------------------
# Main Chat Interface
# -------------------------------
if not st.session_state.knowledge_base_loaded:
    st.markdown("<div class='welcome-message'><h1>HCIL IT Assistant Chatbot</h1><p>Please upload a knowledge base file in the sidebar to begin.</p></div>", unsafe_allow_html=True)
else:
    # --- Chat Messages Display ---
    if not st.session_state.messages:
        st.session_state.messages.append({"role": "assistant", "content": "Hi there! I'm your friendly IT Helpdesk bot. How can I assist you today?"})

    for i, message in enumerate(st.session_state.messages):
        role = message["role"]
        content = message["content"]
        
        # Apply different styling based on role
        if role == "user":
            st.markdown(f"""
            <div class="message-container user-message">
                <div class="bubble">{content}</div>
            </div>
            """, unsafe_allow_html=True)
        else: # Assistant
            st.markdown(f"""
            <div class="message-container bot-message">
                <div class="bubble">{content}</div>
            </div>
            """, unsafe_allow_html=True)

        # --- Feedback Buttons Logic ---
        # Show feedback buttons only for the last assistant message and if no feedback was given yet.
        is_last_message = (i == len(st.session_state.messages) - 1)
        if role == "assistant" and is_last_message and not st.session_state.feedback_given:
            feedback_cols = st.columns([1, 1, 5]) # Ratio for button sizes
            with feedback_cols[0]:
                if st.button("👍", help="This was helpful!", key=f"thumb_up_{i}"):
                    st.session_state.messages.append({"role": "assistant", "content": "Great! I'm glad I could help. Feel free to ask another question."})
                    st.session_state.feedback_given = True
                    st.rerun()
            with feedback_cols[1]:
                if st.button("👎", help="This was not helpful.", key=f"thumb_down_{i}"):
                    st.session_state.messages.append({"role": "assistant", "content": "I'm sorry to hear that. Could you try rephrasing your question?"})
                    st.session_state.feedback_given = True
                    st.rerun()
    
    # --- Chat Input ---
    if prompt := st.chat_input("Ask me an IT question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.feedback_given = True # Hide old feedback buttons
        
        # Check for greetings or farewells first
        special_response = handle_greetings_and_farewells(prompt)
        
        if special_response:
            st.session_state.messages.append({"role": "assistant", "content": special_response})
            # If it's a farewell, reset the chat after a delay
            if "Thank you" in special_response:
                time.sleep(2)
                # Reset the core session state to start fresh
                st.session_state.messages = []
                st.session_state.feedback_given = True
        else:
            # Get response from the AI model
            bot_response = get_bot_response(prompt, st.session_state.df, st.session_state.nn_model, model)
            st.session_state.messages.append({"role": "assistant", "content": bot_response})
            st.session_state.feedback_given = False # Request feedback for this new response
            
        st.rerun()
