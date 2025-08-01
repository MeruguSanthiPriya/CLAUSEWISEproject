import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Hugging Face Configuration
HF_TOKEN = os.getenv("HF_TOKEN", "")  # Load from .env file

# Model Configurations - Using API-based models
MODELS = {
    "primary": {
        "name": "gpt2",  # Public model that doesn't require auth
        "type": "api",  # Use API instead of download
        "max_length": 2048,
        "temperature": 0.7,
        "description": "Primary model for general Q&A and document analysis"
    },
    "granite": {
        "name": "gpt2",  # Public model as fallback
        "type": "api",
        "max_length": 2048,
        "temperature": 0.6,
        "description": "IBM Granite model for specialized tasks"
    },
    "embedding": {
        "name": "sentence-transformers/all-MiniLM-L6-v2",
        "type": "api",
        "description": "Embedding model for document similarity"
    },
    "fallback": {
        "name": "gpt2",
        "type": "api",
        "max_length": 1024,
        "temperature": 0.8,
        "description": "Fallback model if others fail"
    }
}

# API Configuration
API_BASE_URL = "https://api-inference.huggingface.co"
API_TIMEOUT = 30  # seconds

# Loading strategies for API-based models
LOADING_STRATEGIES = {
    "device_map": "auto",
    "torch_dtype": "auto",
    "trust_remote_code": True
}

def get_model_config(model_type: str) -> dict:
    """Get configuration for a specific model"""
    return MODELS.get(model_type, MODELS["fallback"])

def has_hf_token() -> bool:
    """Check if HF token is configured"""
    return bool(HF_TOKEN and HF_TOKEN != "your_hf_token_here" and HF_TOKEN != "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")

def get_hf_token() -> str:
    """Get HF token - returns empty string if not configured"""
    if has_hf_token():
        return HF_TOKEN
    return ""  # Return empty string for public models

def get_api_url() -> str:
    """Get API base URL"""
    return API_BASE_URL 