"""
TTS (Text-to-Speech) backend using pyttsx3 (Windows-friendly)
"""

import os
import logging
from typing import Optional, Dict, Any, List
import io
import tempfile
from pathlib import Path

# Set offline mode
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

try:
    import pyttsx3
    import soundfile as sf
    import numpy as np
except ImportError as e:
    print(f"Missing TTS dependencies: {e}")
    print("Please install: pip install pyttsx3 soundfile")

from app.config import config

logger = logging.getLogger(__name__)

class TTSBackend:
    """TTS backend using pyttsx3 (Windows native TTS)"""
    
    def __init__(self):
        self.engine: Optional[pyttsx3.Engine] = None
        self.device = config.get("tts_device", "cpu")
        self.rate = config.get("tts_rate", 150)
        self.volume = config.get("tts_volume", 1.0)
        self._initialized = False
        self._voices = {}
    
    def initialize(self) -> bool:
        """Initialize pyttsx3 TTS engine"""
        try:
            logger.info("Initializing pyttsx3 TTS engine")
            
            # Initialize pyttsx3 engine
            self.engine = pyttsx3.init()
            
            # Set properties
            self.engine.setProperty('rate', self.rate)
            self.engine.setProperty('volume', self.volume)
            
            # Get available voices
            voices = self.engine.getProperty('voices')
            logger.info(f"Found {len(voices)} available voices")
            
            # Map voices by language
            for voice in voices:
                voice_name = voice.name.lower()
                if 'hindi' in voice_name or 'indian' in voice_name:
                    self._voices['hi'] = voice.id
                elif 'english' in voice_name or 'us' in voice_name or 'uk' in voice_name:
                    self._voices['en'] = voice.id
            
            # Set default voices if not found
            if 'hi' not in self._voices and voices:
                self._voices['hi'] = voices[0].id
            if 'en' not in self._voices and voices:
                self._voices['en'] = voices[0].id
            
            self._initialized = True
            logger.info("pyttsx3 TTS engine initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize TTS engine: {e}")
            return False
    
    def synthesize(self, text: str, language: str, **kwargs) -> Dict[str, Any]:
        """Synthesize speech from text"""
        if not self._initialized:
            if not self.initialize():
                return {"error": "TTS engine not initialized"}
        
        try:
            if not text.strip():
                return {"audio_data": b"", "sample_rate": 22050}
            
            # Set voice for language
            if language in self._voices:
                self.engine.setProperty('voice', self._voices[language])
            
            # Create temporary file for audio output
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Synthesize to file
            self.engine.save_to_file(text, temp_path)
            self.engine.runAndWait()
            
            # Read the generated audio file
            with open(temp_path, 'rb') as f:
                audio_data = f.read()
            
            # Clean up temporary file
            os.unlink(temp_path)
            
            return {
                "audio_data": audio_data,
                "sample_rate": 22050,
                "language": language,
                "text": text
            }
            
        except Exception as e:
            logger.error(f"TTS synthesis failed: {e}")
            return {"error": str(e)}
    
    def synthesize_batch(self, texts: List[str], language: str, **kwargs) -> List[Dict[str, Any]]:
        """Synthesize multiple texts"""
        results = []
        for text in texts:
            result = self.synthesize(text, language, **kwargs)
            results.append(result)
        return results
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages"""
        return ["en", "hi"]
    
    def is_ready(self) -> bool:
        """Check if TTS backend is ready"""
        return self._initialized
    
    def set_rate(self, rate: int) -> None:
        """Set speech rate"""
        self.rate = rate
        if self.engine:
            self.engine.setProperty('rate', rate)
    
    def set_volume(self, volume: float) -> None:
        """Set speech volume (0.0 to 1.0)"""
        self.volume = max(0.0, min(1.0, volume))
        if self.engine:
            self.engine.setProperty('volume', self.volume)

# Global TTS instance
tts_backend = TTSBackend()
