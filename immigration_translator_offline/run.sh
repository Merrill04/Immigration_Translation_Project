#!/bin/bash
# Linux/macOS shell script to run Immigration Translator Offline
# Make sure Python 3.10+ is installed

echo "========================================"
echo "Immigration Translator Offline"
echo "========================================"
echo

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed or not in PATH"
    echo "Please install Python 3.10+ from your package manager"
    exit 1
fi

# Check Python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
required_version="3.10"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "ERROR: Python $python_version is too old. Please install Python 3.10+"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to create virtual environment"
        exit 1
    fi
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install requirements if needed
if [ ! -d "venv/lib/python*/site-packages/fastapi" ]; then
    echo "Installing requirements..."
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to install requirements"
        exit 1
    fi
fi

# Check if models are downloaded
if [ ! -d "models" ]; then
    echo "Models not found. Downloading models..."
    python scripts/download_models.py
    if [ $? -ne 0 ]; then
        echo "WARNING: Model download failed. You may need to download models manually."
    fi
fi

# Set offline environment variables
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Start the application
echo
echo "Starting Immigration Translator Offline..."
echo "Open your browser and go to: http://localhost:8000"
echo "Press Ctrl+C to stop the server"
echo

python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

