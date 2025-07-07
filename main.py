import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import time
import base64

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
    with st.spinner("Thinking... 🤔"):
        query_embed = model.encode([user_query])
        distances, indices = nn_model.kneighbors(query_embed)
        best_idx = indices[0][0]
        response = df.iloc[best_idx]['answers']
        time.sleep(1) # Simulate thinking
    return response

# -------------------------------
# Session State Initialization
# -------------------------------
if 'knowledge_base_loaded' not in st.session_state:
    st.session_state['knowledge_base_loaded'] = False
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'feedback_request' not in st.session_state:
    st.session_state.feedback_request = False

# -------------------------------
# Sidebar Configuration
# -------------------------------
with st.sidebar:
    st.image("https://i.imgur.com/OyQdV3P.png", width=80)
    st.header("⚙️ Bot Configuration")
    with st.expander("📂 Upload Knowledge Base", expanded=True):
        uploaded_file = st.file_uploader(
            "Upload an Excel File",
            type=["xlsx"],
            help="Upload an Excel file with 'questions' and 'answers' columns."
        )
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
                st.error("❌ Error: Missing required columns in Excel file. Please ensure 'questions' and 'answers' columns exist.")
            else:
                st.session_state.df = df
                embeddings = model.encode(df['questions'].tolist())
                nn_model = NearestNeighbors(n_neighbors=1, metric='cosine', algorithm='brute')
                nn_model.fit(np.array(embeddings))
                
                st.session_state.nn_model = nn_model
                st.session_state.knowledge_base_loaded = True
                st.session_state.messages = []
                st.session_state.feedback_request = False
                
                st.success("✅ Knowledge base loaded! The bot is ready.")
                time.sleep(1.5)
                st.rerun()
        except Exception as e:
            st.error(f"❌ An error occurred while processing the file: {e}")

# -------------------------------
# Main Chat Interface
# -------------------------------
if not st.session_state.knowledge_base_loaded:
    st.info("👋 Welcome! Please upload a knowledge base file in the sidebar to start the chat.")
else:
    # --- Chat Header ---
    st.markdown("""
        <div class="chat-header">
            <div class="avatar-header">🤖</div>
            <div class="header-info">
                <span class="header-title">HCIL IT Helpdesk</span>
                <span class="header-status">Online</span>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # --- Chat Messages ---
    if not st.session_state.messages:
        st.session_state.messages.append({"role": "assistant", "content": "Hi there! I'm your friendly IT Helpdesk bot. How can I assist you today?"})

    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]
        if role == "assistant":
            st.markdown(f"""
            <div class="message-container bot-message">
                <div class="bubble">{content}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="message-container user-message">
                <div class="bubble">{content}</div>
            </div>
            """, unsafe_allow_html=True)

    # --- Feedback Buttons ---
    if st.session_state.get('feedback_request'):
        feedback_container = st.container()
        with feedback_container:
            col1, col2, _ = st.columns([1, 1, 5])
            with col1:
                if st.button("👍", help="This was helpful!"):
                    st.session_state.messages.append({"role": "assistant", "content": "Great! I'm glad I could help. Feel free to ask another question."})
                    st.session_state.feedback_request = False
                    st.rerun()
            with col2:
                if st.button("👎", help="This was not helpful."):
                    st.session_state.messages.append({"role": "assistant", "content": "I'm sorry to hear that. Could you try rephrasing your question, or ask something else?"})
                    st.session_state.feedback_request = False
                    st.rerun()
    
    # --- Chat Input ---
    if prompt := st.chat_input("Ask me an IT question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        if prompt.lower().strip() in ["bye", "end", "quit", "exit"]:
            farewell = "Thank you for chatting. Have a great day! 👋"
            st.session_state.messages.append({"role": "assistant", "content": farewell})
            st.session_state.feedback_request = False
        else:
            bot_response = get_bot_response(prompt, st.session_state.df, st.session_state.nn_model, model)
            st.session_state.messages.append({"role": "assistant", "content": bot_response})
            st.session_state.feedback_request = True
        
        st.rerun()
