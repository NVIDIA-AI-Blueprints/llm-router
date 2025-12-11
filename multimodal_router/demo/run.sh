#!/bin/bash

# Multimodal LLM Router Demo - Startup Script

echo "🚀 Starting Multimodal LLM Router Demo"
echo "======================================="
echo ""

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "⚠️  Virtual environment not found. Creating one..."
    uv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Install dependencies
echo "📦 Installing dependencies..."
pip install -q -r requirements.txt

# Check for .env file
if [ ! -f ".env" ]; then
    echo ""
    echo "⚠️  Warning: .env file not found!"
    echo "Please create a .env file with your API keys:"
    echo ""
    echo "ROUTER_ENDPOINT=http://localhost:8001/sfc_router/chat/completions"
    echo "OPENAI_API_KEY=your_openai_api_key"
    echo "NVIDIA_API_KEY=your_nvidia_api_key"
    echo ""
    echo "You can copy from .env.example:"
    echo "  cp .env.example .env"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Launch the app
echo ""
echo "🌐 Launching demo app..."
echo "Access the app at: http://localhost:7860"
echo ""
python app.py

