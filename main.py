import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import time

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="🤖 HCIL IT Helpdesk Chat-Bot",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="auto"
)

# -------------------------------
# Model Loading (Cached)
# -------------------------------
@st.cache_resource
def load_sentence_transformer():
    """Loads the SentenceTransformer model."""
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
# Sidebar Configuration
# -------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    with st.expander("📂 Knowledge Base Setup", expanded=False):
        uploaded_file = st.file_uploader(
            "Upload knowledge base!",
            type=["xlsx"],
            help="Upload an Excel file with 'questions', 'answers', 'categories', and 'tags' columns."
        )

    st.info(" Say 'bye', 'quit', or 'end' to close our chat.")

# -------------------------------
# Main Application Logic
# -------------------------------
st.title("🤖 HCIL IT Helpdesk Chatbot")

# Initialize session state variables if they don't exist
if 'knowledge_base_loaded' not in st.session_state:
    st.session_state['knowledge_base_loaded'] = False
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'chat_ended' not in st.session_state:
    st.session_state.chat_ended = False


# Handle knowledge base upload
if uploaded_file is not None and not st.session_state.knowledge_base_loaded:
    with st.spinner("🚀 Launching the bot... Please wait."):
        try:
            df = pd.read_excel(uploaded_file)
            required_columns = {'questions', 'answers', 'categories', 'tags'}
            if not required_columns.issubset(df.columns):
                st.error("❌ **Error:** The file is missing required columns: `questions`, `answers`, `categories`, `tags`.")
            else:
                st.session_state.df = df
                embeddings = model.encode(df['questions'].tolist())
                nn_model = NearestNeighbors(n_neighbors=1, metric='cosine')
                nn_model.fit(np.array(embeddings))
                
                st.session_state.nn_model = nn_model
                st.session_state.knowledge_base_loaded = True
                st.session_state.messages = [] # Clear any previous messages
                st.session_state.chat_ended = False # Reset chat state
                st.success("✅ Knowledge base loaded! The bot is ready.")
                time.sleep(2)
                st.rerun()
        except Exception as e:
            st.error(f"❌ An error occurred: {e}")

# Display initial greeting if chat hasn't started
if not st.session_state.messages:
    st.markdown("👋 **Konichiwa!** How can I help you today?")


# Main chat interface logic
if st.session_state.knowledge_base_loaded:
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle user input
    if prompt := st.chat_input("Ask your IT question here..."):
        # Append and display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # *** NEW: LOGIC TO END AND RESET CHAT ***
        if prompt.lower() in ["bye", "end", "quit"]:
            response = "Thank you for chatting, **Mata Ne!**(see you later)👋"
            with st.chat_message("assistant"):
                st.markdown(response)
            
            # Reset the chat state after a short delay
            time.sleep(2)
            st.session_state.messages = []
            st.session_state.feedback_request = False
            st.rerun()

        else:
            # Generate and display bot response if not ending chat
            with st.chat_message("assistant"):
                bot_response = get_bot_response(prompt, st.session_state.df, st.session_state.nn_model, model)
                st.markdown(bot_response)
                st.session_state.messages.append({"role": "assistant", "content": bot_response})
            
            st.session_state.feedback_request = True
            st.rerun()

    # Handle feedback display and logic
    if st.session_state.get('feedback_request'):
        col1, col2 = st.columns(2)
        with col1:
            if st.button("👍 Helpful", use_container_width=True):
                feedback_response = "Great! Let me know if there is something else that I can help you with."
                st.session_state.messages.append({"role": "assistant", "content": feedback_response})
                st.session_state.feedback_request = False
                st.rerun()
        with col2:
            if st.button("👎 Not Helpful", use_container_width=True):
                feedback_response = "I apologize. Could you please rephrase your question?"
                st.session_state.messages.append({"role": "assistant", "content": feedback_response})
                st.session_state.feedback_request = False
                st.rerun()
else:
    st.info("⬆️ Please upload a knowledge base file in the sidebar to begin the chat.")
