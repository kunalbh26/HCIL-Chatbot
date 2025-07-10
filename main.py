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
# IMPORTANT: Place your 'knowledge_base.xlsx' file in the same directory as this script,
# or provide the correct path (e.g., 'data/knowledge_base.xlsx').
KNOWLEDGE_BASE_PATH = 'dataset.xlsx'

# -------------------------------
# Custom CSS for Red-Black-White Theme (Modified for strict background, animations, and new styles)
# -------------------------------
st.markdown(f"""
<style>
/* Strict Background Color for the entire app */
html, body, .stApp {{
    background-color: #1F1F1F !important; /* Dark Gray Background */
    color: white !important;
}}

/* Main Chat Area Background */
.main {{
    background: #1F1F1F; /* Match the strict background */
    border-radius: 0px;
    padding: 3.5rem !important;
    max-width: 640px;
    margin: 2.0rem auto;
}}

/* Sidebar Background */
.stSidebar > div:first-child {{
    background-color: #323232 !important; /* Darker Gray for Sidebar */
    border-right: 2px solid white; /* Added white border to the right */
}}

/* Chat bubbles and avatars - unchanged */
.chat-bubble {{
padding: 1rem 1.5rem;
border-radius: 20px;
margin-bottom: 14px;
max-width: 75%;
animation: fadeInUp 0.3s;
position: relative;
word-break: break-word;
font-size: 1.08rem;
display: flex;
align-items: center;
}}
.user-bubble {{
background: #fff;
color: #111;
align-self: flex-end;
margin-left: auto;
margin-right: 0;
border: 1.5px solid #e53935;
}}
.bot-bubble {{
background: linear-gradient(90deg, #e53935 0%, #b71c1c 100%);
color: #fff;
align-self: flex-start;
margin-right: auto;
margin-left: 0;
border: 1.5px solid #fff;
}}
.avatar {{
width: 38px; height: 38px; border-radius: 75%; margin: 0 10px;
background: #3d3d3d;
box-shadow: 0 2px 8px rgba(229,57,53,0.12);
font-size: 1.7rem;
text-align: center;
line-height: 38px;
border: 2px solid #ff0000;
display: flex; align-items: center; justify-content: center;
}}
.user-row {{
display: flex; flex-direction: row; align-items: flex-end; justify-content: flex-end;
}}
.bot-row {{
display: flex; flex-direction: row; align-items: flex-end; justify-content: flex-start;
}}
.input-bar {{
background: #222;
border-radius: 20px;
box-shadow: 0 2px 8px rgba(229,57,53,0.12);
margin-top: 0.5rem;
display: flex;
align-items: center;
padding: 0.3rem 0.8rem;
}}
.input-bar input {{
background: #323232;
border: 2px solid #ff0000;
color: #fff;
width: 100%;
padding: 0.7rem 0.8rem;
outline: none;
font-size: 1rem;
}}
.send-btn {{
background: linear-gradient(90deg, #e53935 0%, #b71c1c 100%);
color: #fff;
border: none;
border-radius: 50%;
width: 38px;
height: 38px;
font-size: 1.2rem;
cursor: pointer;
margin-left: 8px;
transition: background 0.2s;
display: flex; align-items: center; justify-content: center;
}}
.send-btn:hover {{
background: #fff;
color: #e53935;
border: 1.5px solid #ff00;
}}
.quick-reply {{
display: inline-block;
background: #fff;
color: #e53935;
border-radius: 18px;
padding: 0.5rem 1.1rem;
margin: 0.15rem;
cursor: pointer;
font-size: 0.98rem;
border: 1.5px solid #e53935;
font-weight: 500;
transition: background 0.2s, color 0.2s;
}}
.quick-reply:hover {{
background: #e53935;
color: #fff;
}}

/* Enhanced Sidebar Title */
.sidebar-title {{
font-size: 5.5rem; /* Bigger font size */
color: #EE4B2B; /* Keep original color or change as desired */
font-weight: 900;
text-align: center;
margin: 0.5rem 0 1.5rem 0;
letter-spacing: 0.05em;
width: 100%;
line-height: 1.2;
/* 3D rotation animation */
animation: rotate3D 5s infinite linear; /* Slower, more impactful rotation */
transform-style: preserve-3d;
perspective: 800px; /* Adds perspective for 3D effect */
/* Optional: text shadow for more depth, subtle glow */
text-shadow:
    0 0 5px rgba(238, 75, 43, 0.5), /* Red glow */
    0 0 10px rgba(238, 75, 43, 0.4),
    0 0 15px rgba(238, 75, 43, 0.3),
    1px 1px 2px rgba(0,0,0,0.8); /* Subtle shadow for depth */
}}

/* Keyframes for a more 3D rotation */
@keyframes rotate3D {{
    0% {{
        transform: rotateY(0deg) scale(1);
    }}
    25% {{
        transform: rotateY(90deg) scale(1.05); /* Slight scale for emphasis */
    }}
    50% {{
        transform: rotateY(180deg) scale(1);
    }}
    75% {{
        transform: rotateY(270deg) scale(1.05);
    }}
    100% {{
        transform: rotateY(360deg) scale(1);
    }}
}}


/* Main Chatbot Title Enhancement */
.elegant-heading {{
    font-size: 5.0rem !important; /* Bigger font size */
    font-weight: 800;
    text-align: center;
    margin-top: -5px;
    color: #ffffff; /* White text color */
    animation: fadeInUp 2.0s ease-out;
}}


@keyframes fadeInUp {{
    0% {{ opacity: 0; transform: translateY(20px); }}
    100% {{ opacity: 1; transform: translateY(0); }}
}}
.typing-indicator {{
display: flex; align-items: center; margin-bottom: 1.1rem;
}}
.typing-dots span {{
height: 10px; width: 10px; margin: 0 2px;
background: #e53935; border-radius: 25%; display: inline-block;
animation: blink 1.2s infinite both;
}}
.typing-dots span:nth-child(2) {{ animation-delay: 0.2s; }}
.typing-dots span:nth-child(3) {{ animation-delay: 0.4s; }}
@keyframes blink {{
0%, 80%, 100% {{ opacity: 0.2; }}
40% {{ opacity: 1; }}
}}

/* New style for the "Start Chat" button */
/* Apply styling to the Streamlit button itself and center its container */
div.stButton {{
    text-align: center; /* Center the button container */
    margin: 3rem auto; /* Center the button itself within the main div */
    width: fit-content; /* Ensure the div itself wraps the button */
}}

div.stButton > button {{
    background: linear-gradient(90deg, #e53935 0%, #b71c1c 100%);
    color: #fff;
    border: 2px solid #fff;
    border-radius: 20px; /* Rounded corners like a bubble */
    padding: 1rem 2rem;
    font-size: 1.5rem;
    font-weight: bold;
    cursor: pointer;
    transition: background 0.2s, color 0.2s, transform 0.2s;
    line-height: 1; /* Adjust line height for better vertical centering of text */
    display: inline-block; /* Allow it to be centered with text-align */
}}
div.stButton > button:hover {{
    background: #fff;
    color: #e53935;
    border: 2px solid #e53935;
    transform: scale(1.05);
}}
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
