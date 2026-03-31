import os
import torch
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Hardware Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8" # For faster-whisper

# Directory Setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_DIR = os.path.join(BASE_DIR, "temp")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# Ensure directories exist
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# API Keys
HF_TOKEN = os.getenv("HF_TOKEN")

# Model Names
WHISPER_MODEL_SIZE = "small"
SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
SUMMARIZATION_MODEL = "facebook/bart-large-cnn"

# Pipeline Thresholds
AUDIO_SAMPLE_RATE = 16000
MAX_WORDS_PER_CHUNK = 150 # Split long turns for BART token limits
BART_MAX_LENGTH = 130
BART_MIN_LENGTH = 30

# --- Groq Configuration ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"  