#!/usr/bin/env python3
"""
Model download script for immigration_translator_offline
Downloads all required models locally for offline operation
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, Any
import logging

# IMPORTANT: Remove offline mode during download!
# Only set these when RUNNING the app, not when downloading
# os.environ["HF_HUB_OFFLINE"] = "1"
# os.environ["TRANSFORMERS_OFFLINE"] = "1"

try:
    from huggingface_hub import snapshot_download, hf_hub_download
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    import whisper
    import torch
except ImportError as e:
    print(f"Missing dependencies: {e}")
    print("Please install requirements.txt first: pip install -r requirements.txt")
    exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Model configurations
MODELS = {
    "whisper": {
        "model_name": "small",
        "local_path": "./models/whisper-small",
        "type": "whisper"
    },
    "nllb": {
        "repo_id": "facebook/nllb-200-distilled-600M",
        "local_path": "./models/nllb-200-distilled-600M",
        "type": "transformers"
    }
}

def create_directories():
    """Create necessary directories"""
    dirs = ["./models", "./voices", "./samples", "./config"]
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {dir_path}")

def download_whisper_model(model_name: str, local_path: str):
    """Download Whisper model"""
    try:
        logger.info(f"Downloading Whisper model: {model_name}")
        logger.info("This may take a few minutes...")
        
        # Whisper downloads to cache, then we can reference it
        model = whisper.load_model(model_name, download_root="./models")
        logger.info(f"✓ Successfully downloaded Whisper {model_name}")
        return True
        
    except Exception as e:
        logger.error(f"✗ Failed to download Whisper {model_name}: {e}")
        return False

def download_nllb_model(repo_id: str, local_path: str):
    """Download NLLB translation model"""
    try:
        if os.path.exists(local_path) and os.listdir(local_path):
            logger.info(f"NLLB model already exists at {local_path}, skipping download")
            return True
        
        logger.info(f"Downloading NLLB model: {repo_id}")
        logger.info("This is ~1.2GB and may take 5-10 minutes...")
        
        # Download tokenizer
        logger.info("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(repo_id)
        tokenizer.save_pretrained(local_path)
        
        # Download model
        logger.info("Downloading model weights (this is the large part)...")
        model = AutoModelForSeq2SeqLM.from_pretrained(repo_id)
        model.save_pretrained(local_path)
        
        logger.info(f"✓ Successfully downloaded NLLB model to {local_path}")
        return True
        
    except Exception as e:
        logger.error(f"✗ Failed to download NLLB model: {e}")
        logger.error("Please check your internet connection and try again")
        return False

def download_model(model_info: dict):
    """Download model based on type"""
    model_name = model_info.get("model_name", model_info.get("repo_id"))
    local_path = model_info["local_path"]
    model_type = model_info["type"]
    
    if model_type == "whisper":
        return download_whisper_model(model_name, local_path)
    elif model_type == "transformers":
        repo_id = model_info["repo_id"]
        return download_nllb_model(repo_id, local_path)
    
    return False

def create_sample_files():
    """Create sample audio files for testing"""
    samples_dir = Path("./samples")
    
    try:
        # Create a simple test audio file (silence)
        import numpy as np
        import soundfile as sf
        
        # Generate 2 seconds of silence at 16kHz
        sample_rate = 16000
        duration = 2.0
        silence = np.zeros(int(sample_rate * duration), dtype=np.float32)
        
        sample_file = samples_dir / "test_silence.wav"
        sf.write(sample_file, silence, sample_rate)
        logger.info(f"Created sample file: {sample_file}")
    except Exception as e:
        logger.warning(f"Could not create sample files: {e}")

def verify_models():
    """Verify that downloaded models can be loaded"""
    logger.info("\n" + "="*50)
    logger.info("Verifying model downloads...")
    logger.info("="*50)
    
    all_ok = True
    
    # Test Whisper model
    try:
        logger.info("Testing Whisper model...")
        model = whisper.load_model("small", download_root="./models")
        logger.info("✓ Whisper model loaded successfully")
    except Exception as e:
        logger.error(f"✗ Whisper model verification failed: {e}")
        all_ok = False
    
    # Test NLLB model
    try:
        logger.info("Testing NLLB model...")
        nllb_path = "./models/nllb-200-distilled-600M"
        
        if not os.path.exists(nllb_path):
            raise FileNotFoundError(f"NLLB model directory not found: {nllb_path}")
        
        # Check for required files
        required_files = ["config.json", "tokenizer_config.json"]
        missing_files = [f for f in required_files if not os.path.exists(os.path.join(nllb_path, f))]
        
        if missing_files:
            raise FileNotFoundError(f"Missing files in NLLB directory: {missing_files}")
        
        # Try loading
        tokenizer = AutoTokenizer.from_pretrained(nllb_path, local_files_only=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(nllb_path, local_files_only=True)
        
        logger.info("✓ NLLB model loaded successfully")
        
        # Quick translation test
        logger.info("Running quick translation test...")
        inputs = tokenizer("Hello, world!", return_tensors="pt", src_lang="eng_Latn")
        tokenizer.src_lang = "eng_Latn"
        translated = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id["hin_Deva"])
        result = tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
        logger.info(f"Test translation (en→hi): '{result}'")
        logger.info("✓ Translation test successful")
        
    except Exception as e:
        logger.error(f"✗ NLLB model verification failed: {e}")
        all_ok = False
    
    logger.info("="*50)
    if all_ok:
        logger.info("✓ All models verified successfully!")
        logger.info("\nYou can now run the application with: ./run.bat (or ./run.sh)")
    else:
        logger.error("✗ Some models failed verification")
        logger.error("Please check the errors above and try downloading again")
    logger.info("="*50)
    
    return all_ok

def main():
    """Main download function"""
    print("\n" + "="*60)
    print("  Immigration Translator - Model Download Script")
    print("="*60 + "\n")
    
    logger.info("Starting model download process...")
    logger.info("This will download ~1.5GB of models. Please be patient.\n")
    
    # Create directories
    create_directories()
    
    # Download models
    success_count = 0
    total_models = len(MODELS)
    
    for model_name, model_info in MODELS.items():
        logger.info(f"\n--- Downloading {model_name} ---")
        if download_model(model_info):
            success_count += 1
        logger.info("")
    
    # Create sample files
    logger.info("Creating sample files...")
    create_sample_files()
    
    # Verify models
    logger.info("\n")
    verify_success = verify_models()
    
    # Summary
    print("\n" + "="*60)
    print("  Download Summary")
    print("="*60)
    logger.info(f"Models downloaded: {success_count}/{total_models}")
    
    if success_count == total_models and verify_success:
        logger.info("\n✓✓✓ SUCCESS! All models downloaded and verified ✓✓✓")
        logger.info("\nNext steps:")
        logger.info("  1. Run: .\\run.bat (Windows) or ./run.sh (Linux)")
        logger.info("  2. Open: http://localhost:8000")
        logger.info("  3. Start translating!")
    else:
        logger.warning("\n⚠ Some models failed to download or verify")
        logger.warning("Please check the errors above and try again")
        logger.warning("\nTroubleshooting:")
        logger.warning("  - Check your internet connection")
        logger.warning("  - Try running the script again: python scripts/download_models.py")
        logger.warning("  - Check if you have enough disk space (~2GB needed)")
    
    print("="*60 + "\n")

if __name__ == "__main__":
    main()