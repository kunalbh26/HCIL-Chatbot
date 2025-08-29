import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import time
import re
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import random
from datetime import datetime
import json
import pickle
import os
import html
from typing import List, Dict, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
KNOWLEDGE_BASE_PATH = 'dataset.xlsx'
CACHE_DIR = 'cache'
EMBEDDINGS_CACHE_FILE = os.path.join(CACHE_DIR, 'embeddings.pkl')
MODEL_CACHE_FILE = os.path.join(CACHE_DIR, 'model.pkl')

# Create cache directory if it doesn't exist
os.makedirs(CACHE_DIR, exist_ok=True)

# Page Configuration
st.set_page_config(
    page_title="HCIL IT-Helpdesk ChatBot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS (Streamlit-Compatible)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

:root {
    --primary-color: #e53935;
    --secondary-color: #b71c1c;
    --accent-color: #f87171;
    --text-primary: #ffffff;
    --text-secondary: rgba(255, 255, 255, 0.8);
    --bg-dark: #000000;
    --glass-bg: rgba(255, 255, 255, 0.05);
    --glass-border: rgba(255, 255, 255, 0.1);
}

* {
    font-family: 'Inter', sans-serif !important;
}

.main {
    background: var(--glass-bg) !important;
    backdrop-filter: blur(10px);
    border: 1px solid var(--glass-border);
    border-radius: 20px !important;
    padding: 2rem !important;
    margin: 1rem auto;
    max-width: 900px !important;
}

.chat-bubble {
    padding: 1rem 1.5rem;
    border-radius: 18px;
    margin-bottom: 1rem;
    max-width: 80%;
    position: relative;
    font-size: 1rem;
    line-height: 1.5;
    transition: transform 0.2s ease;
}

.user-bubble {
    background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
    color: white;
    margin-left: auto;
    box-shadow: 0 4px 15px rgba(229, 57, 53, 0.3);
}

.bot-bubble {
    background: var(--glass-bg);
    color: var(--text-primary);
    border: 1px solid var(--glass-border);
    backdrop-filter: blur(10px);
}

.avatar {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.2rem;
    margin: 0 10px;
}

.user-avatar {
    background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
    border: 2px solid rgba(255, 255, 255, 0.2);
}

.bot-avatar {
    background: linear-gradient(135deg, var(--secondary-color), var(--primary-color));
    border: 2px solid rgba(255, 255, 255, 0.2);
}

.stButton > button {
    background: linear-gradient(135deg, var(--primary-color), var(--secondary-color)) !important;
    color: white !important;
    border: none !important;
    border-radius: 25px !important;
    padding: 0.8rem 1.5rem !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 20px rgba(229, 57, 53, 0.4) !important;
}

.stTextInput > div > div > input {
    background: var(--glass-bg) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 15px !important;
    color: var(--text-primary) !important;
    padding: 0.8rem 1.2rem !important;
    font-size: 1rem !important;
}

.stTextInput > div > div > input:focus {
    border-color: var(--primary-color) !important;
    box-shadow: 0 0 15px rgba(229, 57, 53, 0.3) !important;
}

.stSidebar > div:first-child {
    background: linear-gradient(180deg, rgba(20, 20, 20, 0.95), rgba(10, 10, 10, 0.95)) !important;
    border-right: 1px solid rgba(229, 57, 53, 0.3);
}

@media (max-width: 768px) {
    .main { padding: 1rem !important; margin: 0.5rem !important; }
    .chat-bubble { max-width: 90%; }
}

footer { visibility: hidden; }
.stDeployButton { display: none; }

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
    width: 8px;
    height: 8px;
    background: var(--primary-color);
    border-radius: 50%;
    animation: typingPulse 1.4s infinite ease-in-out;
}

.typing-dots span:nth-child(1) { animation-delay: 0s; }
.typing-dots span:nth-child(2) { animation-delay: 0.2s; }
.typing-dots span:nth-child(3) { animation-delay: 0.4s; }

@keyframes typingPulse {
    0%, 80%, 100% { transform: scale(0.8); opacity: 0.5; }
    40% { transform: scale(1.2); opacity: 1; }
}

.elegant-heading {
    font-size: 2.5rem !important;
    font-weight: 800;
    text-align: center;
    margin: 1rem 0 2rem 0 !important;
    background: linear-gradient(135deg, var(--secondary-color), var(--primary-color), var(--accent-color));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.status-indicator {
    display: inline-block;
    width: 8px;
    height: 8px;
    background: #4ade80;
    border-radius: 50%;
    margin-left: 8px;
    animation: statusPulse 2s ease-in-out infinite;
}

@keyframes statusPulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(74, 222, 128, 0.4); }
    50% { box-shadow: 0 0 0 8px rgba(74, 222, 128, 0); }
}
</style>
""", unsafe_allow_html=True)

# Enhanced Backend Classes
class ConversationManager:
    """Manages conversation context and history"""
    
    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self.conversation_history = []
        self.context_window = []
    
    def add_message(self, role: str, content: str, timestamp: datetime = None):
        """Add a message to conversation history"""
        if timestamp is None:
            timestamp = datetime.now()
        
        # Sanitize content for security
        sanitized_content = html.escape(content)
        
        message = {
            'role': role,
            'content': sanitized_content,
            'timestamp': timestamp
        }
        
        self.conversation_history.append(message)
        
        # Maintain context window
        if len(self.context_window) >= self.max_history:
            self.context_window.pop(0)
        self.context_window.append(message)
    
    def get_context(self) -> List[Dict]:
        """Get recent conversation context"""
        return self.context_window[-5:] if len(self.context_window) > 5 else self.context_window
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history.clear()
        self.context_window.clear()

class EnhancedKnowledgeBase:
    """Enhanced knowledge base with caching and better search"""
    
    def __init__(self, model: SentenceTransformer):
        self.model = model
        self.df = None
        self.embeddings = None
        self.nn_model = None
        self.question_embeddings = None
        self.answer_embeddings = None
        self.categories = set()
        self.tags = set()
    
    def load_data(self, file_path: str) -> bool:
        """Load and preprocess knowledge base data"""
        try:
            if not os.path.exists(file_path):
                logger.error(f"Knowledge base file not found: {file_path}")
                return False
            
            self.df = pd.read_excel(file_path)
            required_columns = {'questions', 'answers', 'categories', 'tags'}
            
            if not required_columns.issubset(self.df.columns):
                missing_cols = required_columns - set(self.df.columns)
                logger.error(f"Missing required columns: {missing_cols}")
                return False
            
            # Clean and preprocess data
            self.df = self.df.dropna(subset=['questions', 'answers'])
            self.df['questions_clean'] = self.df['questions'].astype(str).str.lower().str.strip()
            self.df['answers_clean'] = self.df['answers'].astype(str).str.strip()
            
            # Extract categories and tags
            self.categories = set(self.df['categories'].dropna().unique())
            self.tags = set(self.df['tags'].dropna().unique())
            
            # Generate embeddings
            self._generate_embeddings()
            
            logger.info(f"Knowledge base loaded successfully with {len(self.df)} entries")
            return True
            
        except Exception as e:
            logger.error(f"Error loading knowledge base: {e}")
            return False
    
    def _generate_embeddings(self):
        """Generate embeddings for questions and answers"""
        try:
            # Check cache first
            if os.path.exists(EMBEDDINGS_CACHE_FILE):
                try:
                    with open(EMBEDDINGS_CACHE_FILE, 'rb') as f:
                        cached_data = pickle.load(f)
                        current_hash = hash(str(self.df['questions'].tolist()))
                        if cached_data.get('df_hash') == current_hash:
                            self.question_embeddings = cached_data['question_embeddings']
                            self.answer_embeddings = cached_data['answer_embeddings']
                            logger.info("Loaded embeddings from cache")
                            return
                except Exception as e:
                    logger.warning(f"Cache loading failed: {e}")
            
            # Generate new embeddings
            logger.info("Generating embeddings...")
            self.question_embeddings = self.model.encode(self.df['questions_clean'].tolist())
            self.answer_embeddings = self.model.encode(self.df['answers_clean'].tolist())
            
            # Cache embeddings
            try:
                cache_data = {
                    'df_hash': hash(str(self.df['questions'].tolist())),
                    'question_embeddings': self.question_embeddings,
                    'answer_embeddings': self.answer_embeddings
                }
                with open(EMBEDDINGS_CACHE_FILE, 'wb') as f:
                    pickle.dump(cache_data, f)
                logger.info("Embeddings generated and cached")
            except Exception as e:
                logger.warning(f"Failed to cache embeddings: {e}")
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def search(self, query: str, method: str = 'hybrid', top_k: int = 3) -> List[Dict]:
        """Enhanced search with multiple methods - FIXED VERSION"""
        if not hasattr(self, 'df') or self.df is None:
            return []
        
        query_clean = query.lower().strip()
        if not query_clean:
            return []
        
        try:
            query_embedding = self.model.encode([query_clean])
        except Exception as e:
            logger.error(f"Error encoding query: {e}")
            return []
        
        results = []
        
        if method == 'semantic':
            results = self._semantic_search(query_embedding, top_k)
        elif method == 'fuzzy':
            results = self._fuzzy_search(query_clean, top_k)
        elif method == 'hybrid':
            # FIXED: No more infinite recursion - call internal methods directly
            semantic_results = self._semantic_search(query_embedding, top_k)
            fuzzy_results = self._fuzzy_search(query_clean, top_k)
            results = self._merge_results(semantic_results, fuzzy_results, top_k)
        
        return results
    
    def _semantic_search(self, query_embedding, top_k: int) -> List[Dict]:
        """Perform semantic search using embeddings"""
        try:
            if self.nn_model is None:
                self.nn_model = NearestNeighbors(n_neighbors=min(top_k, len(self.df)), metric='cosine')
                self.nn_model.fit(self.question_embeddings)
            
            distances, indices = self.nn_model.kneighbors(query_embedding)
            
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < len(self.df):
                    results.append({
                        'index': int(idx),
                        'question': str(self.df.iloc[idx]['questions']),
                        'answer': str(self.df.iloc[idx]['answers']),
                        'category': str(self.df.iloc[idx]['categories']) if pd.notna(self.df.iloc[idx]['categories']) else '',
                        'tags': str(self.df.iloc[idx]['tags']) if pd.notna(self.df.iloc[idx]['tags']) else '',
                        'score': float(1 - distance),
                        'method': 'semantic'
                    })
            
            return results
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return []
    
    def _fuzzy_search(self, query_clean: str, top_k: int) -> List[Dict]:
        """Perform fuzzy string matching"""
        try:
            fuzzy_results = process.extract(
                query_clean, 
                self.df['questions_clean'].tolist(), 
                scorer=fuzz.token_sort_ratio, 
                limit=top_k
            )
            
            results = []
            for question, score in fuzzy_results:
                try:
                    idx = self.df[self.df['questions_clean'] == question].index[0]
                    results.append({
                        'index': int(idx),
                        'question': str(self.df.iloc[idx]['questions']),
                        'answer': str(self.df.iloc[idx]['answers']),
                        'category': str(self.df.iloc[idx]['categories']) if pd.notna(self.df.iloc[idx]['categories']) else '',
                        'tags': str(self.df.iloc[idx]['tags']) if pd.notna(self.df.iloc[idx]['tags']) else '',
                        'score': float(score / 100),
                        'method': 'fuzzy'
                    })
                except (IndexError, KeyError):
                    continue
            
            return results
        except Exception as e:
            logger.error(f"Error in fuzzy search: {e}")
            return []
    
    def _merge_results(self, semantic_results: List[Dict], fuzzy_results: List[Dict], top_k: int) -> List[Dict]:
        """Merge and deduplicate search results"""
        try:
            all_results = semantic_results + fuzzy_results
            seen_indices = set()
            merged_results = []
            
            for result in all_results:
                if result['index'] not in seen_indices:
                    seen_indices.add(result['index'])
                    merged_results.append(result)
            
            # Sort by score and take top_k
            merged_results.sort(key=lambda x: x['score'], reverse=True)
            return merged_results[:top_k]
        except Exception as e:
            logger.error(f"Error merging results: {e}")
            return semantic_results[:top_k] if semantic_results else fuzzy_results[:top_k]
    
    def get_categories(self) -> List[str]:
        """Get all available categories"""
        return list(self.categories)
    
    def get_tags(self) -> List[str]:
        """Get all available tags"""
        return list(self.tags)
    
    def filter_by_category(self, category: str) -> pd.DataFrame:
        """Filter knowledge base by category"""
        return self.df[self.df['categories'] == category]

class ResponseGenerator:
    """Enhanced response generation with context awareness"""
    
    def __init__(self, knowledge_base: EnhancedKnowledgeBase):
        self.kb = knowledge_base
        self.greetings = {
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
    
    def is_greeting(self, text: str) -> Optional[str]:
        """Check if text is a greeting"""
        text_lower = text.lower().strip()
        for greet in self.greetings.keys():
            if fuzz.partial_ratio(greet, text_lower) > 80:
                return greet
        return None
    
    def is_gibberish(self, text: str) -> bool:
        """Check if text is gibberish"""
        text = text.strip()
        if len(text) < 2:
            return True
        if re.fullmatch(r'[^\w\s]+', text):
            return True
        if len(set(text)) < 3:
            return True
        words = text.split()
        if len(words) > 0 and sum(1 for w in words if not w.isalpha()) / len(words) > 0.5:
            return True
        return False
    
    def generate_response(self, query: str, context: List[Dict] = None) -> Dict:
        """Generate enhanced response with context awareness"""
        start_time = time.time()
        
        try:
            # Check for gibberish
            if self.is_gibberish(query):
                return {
                    'response': "🤔 I couldn't quite understand that. Could you please rephrase your question?",
                    'confidence': 0.0,
                    'method': 'gibberish_detection',
                    'processing_time': time.time() - start_time
                }
            
            # Check for greetings
            greeting = self.is_greeting(query)
            if greeting:
                return {
                    'response': self.greetings[greeting],
                    'confidence': 1.0,
                    'method': 'greeting',
                    'processing_time': time.time() - start_time
                }
            
            # Search knowledge base
            search_results = self.kb.search(query, method='hybrid', top_k=3)
            
            if not search_results:
                return {
                    'response': "🤔 I couldn't find a specific answer. Could you provide more details or try rephrasing?",
                    'confidence': 0.0,
                    'method': 'no_results',
                    'processing_time': time.time() - start_time
                }
            
            # Get best match
            best_match = search_results[0]
            
            # Enhance response based on context
            enhanced_response = self._enhance_response(best_match, context)
            
            return {
                'response': enhanced_response,
                'confidence': best_match['score'],
                'method': best_match['method'],
                'category': best_match['category'],
                'tags': best_match['tags'],
                'processing_time': time.time() - start_time,
                'alternatives': search_results[1:] if len(search_results) > 1 else []
            }
        
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                'response': "😔 I encountered an error while processing your request. Please try again.",
                'confidence': 0.0,
                'method': 'error',
                'processing_time': time.time() - start_time
            }
    
    def _enhance_response(self, match: Dict, context: List[Dict] = None) -> str:
        """Enhance response with additional context and formatting"""
        try:
            base_response = str(match['answer'])
            
            # Add category context if available
            if match.get('category') and match['category'].strip():
                category_info = f"\n\n**Category:** {match['category']}"
                base_response += category_info
            
            # Add related tags if available
            if match.get('tags') and match['tags'].strip():
                tags_info = f"\n\n**Related:** {match['tags']}"
                base_response += tags_info
            
            return base_response
        except Exception as e:
            logger.error(f"Error enhancing response: {e}")
            return str(match.get('answer', ''))

# Cached Model Loading
@st.cache_resource
def load_model():
    """Load and cache the sentence transformer model"""
    try:
        if os.path.exists(MODEL_CACHE_FILE):
            try:
                with open(MODEL_CACHE_FILE, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cached model: {e}")
        
        model = SentenceTransformer("all-MiniLM-L6-v2")
        
        try:
            with open(MODEL_CACHE_FILE, 'wb') as f:
                pickle.dump(model, f)
        except Exception as e:
            logger.warning(f"Failed to cache model: {e}")
        
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return SentenceTransformer("all-MiniLM-L6-v2")

# Initialize Components
if 'model' not in st.session_state:
    st.session_state.model = load_model()

if 'knowledge_base' not in st.session_state:
    st.session_state.knowledge_base = EnhancedKnowledgeBase(st.session_state.model)
    if not st.session_state.knowledge_base.load_data(KNOWLEDGE_BASE_PATH):
        st.error("Failed to load knowledge base. Please check the dataset file.")
        st.stop()

if 'conversation_manager' not in st.session_state:
    st.session_state.conversation_manager = ConversationManager()

if 'response_generator' not in st.session_state:
    st.session_state.response_generator = ResponseGenerator(st.session_state.knowledge_base)

# UI Components
def render_chat_message(role: str, content: str):
    """Render a single chat message with security"""
    try:
        # Sanitize content for display
        safe_content = html.escape(content)
        
        if role == "user":
            st.markdown(f"""
            <div style="display: flex; justify-content: flex-end; align-items: flex-end; margin-bottom: 1rem;">
                <div class="chat-bubble user-bubble">{safe_content}</div>
                <div class="avatar user-avatar">👨‍💻</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="display: flex; justify-content: flex-start; align-items: flex-end; margin-bottom: 1rem;">
                <div class="avatar bot-avatar">🤖</div>
                <div class="chat-bubble bot-bubble">{safe_content}</div>
            </div>
            """, unsafe_allow_html=True)
    except Exception as e:
        logger.error(f"Error rendering chat message: {e}")

def show_typing_indicator():
    """Show typing indicator"""
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

# Sidebar
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h2 style="color: var(--primary-color); margin: 0;">HCIL IT Support</h2>
        <p style="color: var(--text-secondary); font-size: 0.9rem; margin: 0.5rem 0;">
            AI-Powered Helpdesk
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # System Status
    st.markdown("""
    <div style="background: var(--glass-bg); border-radius: 15px; padding: 1rem; margin: 1rem 0; border: 1px solid var(--glass-border);">
        <h4 style="color: #4ade80; margin: 0;">🟢 System Online</h4>
        <p style="color: var(--text-secondary); font-size: 0.9rem; margin: 0.5rem 0 0 0;">
            AI-powered IT Helpdesk chatbot for Honda Cars India
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick Stats
    if st.session_state.conversation_manager.conversation_history:
        msg_count = len([m for m in st.session_state.conversation_manager.conversation_history if m['role'] == 'user'])
        st.markdown(f"""
        <div style="background: var(--glass-bg); border-radius: 12px; padding: 0.8rem; margin: 1rem 0; border: 1px solid var(--glass-border);">
            <p style="margin: 0; font-size: 0.9rem; color: var(--text-secondary);">Messages: {msg_count}</p>
            <p style="margin: 0; font-size: 0.9rem; color: var(--text-secondary);">Response Time: ~1.2s</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Categories
    if st.session_state.knowledge_base.categories:
        st.markdown("### 📂 Categories")
        for category in sorted(st.session_state.knowledge_base.categories):
            st.markdown(f"- {category}")
    
    st.markdown("---")
    
    # Clear Chat Button
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.conversation_manager.clear_history()
        st.rerun()
    
    st.info("💡 **Pro Tip:** Type 'bye' to end the conversation")

# Main Chat Interface
st.markdown('<h1 class="elegant-heading">HCIL IT Helpdesk AI Assistant</h1>', unsafe_allow_html=True)

# Chat History
if st.session_state.conversation_manager.conversation_history:
    for message in st.session_state.conversation_manager.conversation_history:
        render_chat_message(message['role'], message['content'])

# Quick Actions
if not st.session_state.conversation_manager.conversation_history:
    st.markdown("""
    <div style="text-align: center; margin: 3rem 0;">
        <p style="color: var(--text-secondary); font-size: 1.1rem; margin-bottom: 2rem;">
            Experience next-generation IT support powered by AI
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick Reply Buttons
    quick_replies = ["Reset Password", "VPN Issues", "Software Install", "Hardware Problems"]
    cols = st.columns(len(quick_replies))
    
    for col, reply in zip(cols, quick_replies):
        with col:
            if st.button(reply, use_container_width=True):
                # Process quick reply
                response_data = st.session_state.response_generator.generate_response(reply)
                
                # Add to conversation
                st.session_state.conversation_manager.add_message("user", reply)
                st.session_state.conversation_manager.add_message("bot", response_data['response'])
                
                st.rerun()

# Input Section
st.markdown("---")
with st.form("chat_input", clear_on_submit=True):
    col1, col2 = st.columns([6, 1])
    
    with col1:
        user_input = st.text_input(
            "Type your message...",
            key="user_input",
            label_visibility="collapsed",
            placeholder="Ask me anything about IT support..."
        )
    
    with col2:
        submit_button = st.form_submit_button("Send", use_container_width=True)
    
    if submit_button and user_input.strip():
        # Check for exit commands
        if user_input.lower().strip() in ['bye', 'quit', 'exit', 'end']:
            farewell_messages = [
                "Thank you for using HCIL IT Support! **Mata ne!** 🌟 Have an amazing day!",
                "It was great helping you! **Mata ne!** ✨ See you next time!",
                "Thanks for choosing HCIL! **Goodbye!** 🚀 Stay awesome!"
            ]
            
            st.session_state.conversation_manager.add_message("user", user_input)
            st.session_state.conversation_manager.add_message("bot", random.choice(farewell_messages))
            
        else:
            # Show typing indicator
            with st.spinner("🤖 Thinking..."):
                # Generate response
                response_data = st.session_state.response_generator.generate_response(
                    user_input, 
                    st.session_state.conversation_manager.get_context()
                )
                
                # Add messages to conversation
                st.session_state.conversation_manager.add_message("user", user_input)
                st.session_state.conversation_manager.add_message("bot", response_data['response'])
        
        st.rerun()

# Footer
st.markdown("""
<div style="margin-top: 3rem; padding-top: 2rem; border-top: 1px solid var(--glass-border); text-align: center;">
    <p style="color: var(--text-secondary); font-size: 0.85rem;">
        Powered by Advanced AI | HCIL IT Support © 2025 ~ By Kunal Bhardwaj
    </p>
</div>
""", unsafe_allow_html=True)
