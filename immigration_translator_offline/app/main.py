"""
FastAPI backend for immigration_translator_offline
"""

import os
import logging
from typing import Dict, Any, Optional
from pathlib import Path
import json
import io

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Set offline mode
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from app.config import config
from app.asr import asr_backend
from app.mt import mt_backend
from app.tts import tts_backend

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Immigration Translator Offline",
    description="Offline speech-to-speech translation for Hindi ↔ English",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Pydantic models
class TranslationRequest(BaseModel):
    text: str
    src_lang: str
    tgt_lang: str

class TTSRequest(BaseModel):
    text: str
    language: str

class ConfigRequest(BaseModel):
    key: str
    value: Any

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "mode": config.get("mode"),
        "asr_ready": asr_backend.is_ready(),
        "mt_ready": mt_backend.is_ready(),
        "tts_ready": tts_backend.is_ready()
    }

# Configuration endpoint
@app.get("/config")
async def get_config():
    """Get current configuration"""
    return {
        "mode": config.get("mode"),
        "languages": config.get("languages"),
        "src_lang_code": config.get("src_lang_code"),
        "tgt_lang_code": config.get("tgt_lang_code"),
        "tts_backend": config.get("tts_backend"),
        "device_config": config.get_device_config(),
        "supported_languages": {
            "asr": asr_backend.get_supported_languages(),
            "mt": mt_backend.get_supported_languages(),
            "tts": tts_backend.get_supported_languages()
        }
    }

@app.post("/config")
async def update_config(request: ConfigRequest):
    """Update configuration"""
    try:
        config.set(request.key, request.value)
        config.save_config()
        return {"message": f"Updated {request.key} to {request.value}"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ASR endpoints
@app.post("/asr")
async def transcribe_audio(
    file: UploadFile = File(...),
    language: str = Form("auto")
):
    """Transcribe audio file to text"""
    try:
        # Read audio data
        audio_data = await file.read()
        
        # Log file info for debugging
        logger.info(f"Received audio file: {file.filename}, size: {len(audio_data)} bytes, type: {file.content_type}")
        
        if len(audio_data) == 0:
            raise HTTPException(status_code=400, detail="Empty audio file")
        
        # Try direct transcription first, fallback to file-based
        result = asr_backend.transcribe_audio_direct(audio_data, language)
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return result
        
    except Exception as e:
        logger.error(f"ASR error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/asr/chunk")
async def transcribe_chunk(
    file: UploadFile = File(...),
    language: str = Form("auto")
):
    """Transcribe audio chunk for streaming"""
    try:
        # Read audio data
        audio_data = await file.read()
        
        # Convert to numpy array for chunk processing
        import soundfile as sf
        import numpy as np
        import io
        
        audio_array, sample_rate = sf.read(io.BytesIO(audio_data))
        
        # Transcribe chunk
        result = asr_backend.transcribe_chunk(audio_array, language)
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return result
        
    except Exception as e:
        logger.error(f"ASR chunk error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Translation endpoint
@app.post("/translate")
async def translate_text(request: TranslationRequest):
    """Translate text from source to target language"""
    try:
        result = mt_backend.translate(
            request.text,
            request.src_lang,
            request.tgt_lang
        )
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return result
        
    except Exception as e:
        logger.error(f"Translation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# TTS endpoint
@app.post("/tts")
async def synthesize_speech(request: TTSRequest):
    """Synthesize speech from text"""
    try:
        result = tts_backend.synthesize(request.text, request.language)
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        # Return audio as streaming response
        return StreamingResponse(
            io.BytesIO(result["audio_data"]),
            media_type="audio/wav",
            headers={
                "Content-Disposition": f"attachment; filename=speech_{request.language}.wav"
            }
        )
        
    except Exception as e:
        logger.error(f"TTS error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Frontend endpoint
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the frontend HTML"""
    try:
        frontend_path = Path("static/index.html")
        if frontend_path.exists():
            return FileResponse(frontend_path)
        else:
            return HTMLResponse("""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Immigration Translator Offline</title>
                <meta charset="utf-8">
                <meta name="viewport" content="width=device-width, initial-scale=1">
                <style>
                    body { font-family: Arial, sans-serif; margin: 40px; }
                    .container { max-width: 800px; margin: 0 auto; }
                    .error { color: red; }
                    .success { color: green; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>Immigration Translator Offline</h1>
                    <p class="error">Frontend not found. Please check static/index.html</p>
                    <p>API endpoints are available at:</p>
                    <ul>
                        <li>GET /health - Health check</li>
                        <li>GET /config - Get configuration</li>
                        <li>POST /asr - Transcribe audio</li>
                        <li>POST /translate - Translate text</li>
                        <li>POST /tts - Synthesize speech</li>
                    </ul>
                </div>
            </body>
            </html>
            """)
    except Exception as e:
        logger.error(f"Frontend error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Initialize backends on startup
@app.on_event("startup")
async def startup_event():
    """Initialize backends on startup"""
    logger.info("Starting Immigration Translator Offline...")
    
    # Initialize backends
    logger.info("Initializing ASR backend...")
    asr_backend.initialize()
    
    logger.info("Initializing MT backend...")
    mt_backend.initialize()
    
    logger.info("Initializing TTS backend...")
    tts_backend.initialize()
    
    logger.info("All backends initialized successfully!")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Immigration Translator Offline...")

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
