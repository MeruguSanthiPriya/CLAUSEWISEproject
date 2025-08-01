#!/usr/bin/env python3
"""
Simple setup script for the project
"""

import os

def create_env_file():
    """Create .env file if it doesn't exist"""
    if not os.path.exists('.env'):
        env_content = """# Hugging Face Configuration
HF_TOKEN=your_hf_token_here

# API Configuration
API_URL=http://127.0.0.1:8001
"""
        with open('.env', 'w', encoding='utf-8') as f:
            f.write(env_content)
        print("✅ Created .env file")
        print("📝 Please edit .env and add your HF token")
    else:
        print("✅ .env file already exists")

def main():
    print("🚀 Quick Setup")
    print("=" * 30)
    
    create_env_file()
    
    print("\n📋 Next Steps:")
    print("1. Get your HF token from: https://huggingface.co/settings/tokens")
    print("2. Edit .env file and replace 'your_hf_token_here' with your token")
    print("3. Run: python main.py (Terminal 1)")
    print("4. Run: streamlit run app.py (Terminal 2)")
    print("\n🎉 Your project is ready!")

if __name__ == "__main__":
    main() 