"""
ASR (Automatic Speech Recognition) backend using openai-whisper
"""

import os
import io
import logging
from typing import Optional, List, Dict, Any
import numpy as np
import torch
from pathlib import Path
import subprocess

# Set offline mode
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

try:
    import whisper
    import soundfile as sf
except ImportError as e:
    print(f"Missing ASR dependencies: {e}")
    print("Please install: pip install openai-whisper soundfile")

from app.config import config

logger = logging.getLogger(__name__)

class ASRBackend:
    """ASR backend using openai-whisper"""
    
    def __init__(self):
        self.model: Optional[whisper.Whisper] = None
        self.model_path = config.get_model_path("asr")
        self.device = config.get("device", "auto")
        self.model_size = config.get("model_size", "small")
        self._initialized = False
    
    def _ensure_ffmpeg(self) -> None:
        """Ensure ffmpeg is available by augmenting PATH with local bin if present, and log version."""
        try:
            # If a local bin directory exists, prepend it to PATH
            local_bin = Path.cwd() / "bin"
            if local_bin.exists():
                os.environ["PATH"] = str(local_bin) + os.pathsep + os.environ.get("PATH", "")

            # Probe ffmpeg
            completed = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
            if completed.returncode == 0:
                logger.info("ffmpeg detected: %s", completed.stdout.splitlines()[0])
            else:
                logger.warning("ffmpeg not responding as expected (code %s)", completed.returncode)
        except FileNotFoundError:
            logger.warning("ffmpeg not found on PATH. Whisper may fail to process some formats.")
        except Exception as e:
            logger.warning("ffmpeg check failed: %s", e)

    def initialize(self) -> bool:
        """Initialize the ASR model"""
        try:
            # Ensure ffmpeg availability first
            self._ensure_ffmpeg()
            logger.info(f"Loading Whisper model: {self.model_size}")
            
            # Determine device
            if self.device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = self.device
            
            # Load model with download_root for offline use
            download_root = "./models" if os.path.exists("./models") else None
            self.model = whisper.load_model(
                self.model_size, 
                device=device,
                download_root=download_root
            )
            
            self._initialized = True
            logger.info(f"Whisper model loaded successfully on {device}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize ASR model: {e}")
            return False
    
    def transcribe_audio_direct(self, audio_data: bytes, language: str = "auto") -> Dict[str, Any]:
        """Transcribe audio data directly without temporary files"""
        if not self._initialized:
            if not self.initialize():
                return {"error": "ASR model not initialized"}
        
        try:
            # Try to read audio data directly with soundfile
            audio_array, sample_rate = sf.read(io.BytesIO(audio_data))
            
            # Ensure mono audio
            if len(audio_array.shape) > 1:
                audio_array = np.mean(audio_array, axis=1)
            
            # Resample to 16kHz if needed
            if sample_rate != 16000:
                audio_array = self._resample_audio(audio_array, sample_rate, 16000)
            
            # Transcribe with Whisper
            result = self.model.transcribe(
                audio_array,
                language=language if language != "auto" else None,
                fp16=False
            )
            
            return {
                "text": result["text"],
                "language": result.get("language", language),
                "duration": len(audio_array) / 16000
            }
            
        except Exception as e:
            logger.error(f"Direct transcription failed: {e}")
            # Fallback to file-based approach
            return self.transcribe_audio(audio_data, language)

    def transcribe_audio(self, audio_data: bytes, language: str = "auto") -> Dict[str, Any]:
        """Transcribe audio data to text"""
        if not self._initialized:
            if not self.initialize():
                return {"error": "ASR model not initialized"}
        
        temp_path = None
        try:
            # Save audio data to temporary file first
            import tempfile
            # Use .wav by default; ffmpeg can still detect actual container
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = temp_file.name
                temp_file.write(audio_data)
                temp_file.flush()
                os.fsync(temp_file.fileno())
            
            # Verify file was created and has content
            if not os.path.exists(temp_path):
                return {"error": "Failed to create temporary audio file"}
            
            file_size = os.path.getsize(temp_path)
            if file_size == 0:
                return {"error": "Audio file is empty"}
            
            logger.info(f"Transcribing audio file: {temp_path} ({file_size} bytes)")
            
            # Transcribe using Whisper (it handles various formats better)
            result = self.model.transcribe(
                temp_path,
                language=language if language != "auto" else None,
                fp16=False  # Disable FP16 to avoid CPU warnings
            )
            
            return {
                "text": result["text"],
                "language": result.get("language", language),
                "duration": result.get("duration", 0)
            }
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return {"error": str(e)}
        finally:
            # Clean up temporary file
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except Exception as cleanup_error:
                    logger.warning(f"Failed to clean up temp file {temp_path}: {cleanup_error}")
    
    def transcribe_file(self, file_path: str, language: str = "auto") -> Dict[str, Any]:
        """Transcribe audio file to text"""
        try:
            with open(file_path, 'rb') as f:
                audio_data = f.read()
            return self.transcribe_audio(audio_data, language)
        except Exception as e:
            logger.error(f"Failed to read audio file {file_path}: {e}")
            return {"error": str(e)}
    
    def transcribe_chunk(self, audio_chunk: np.ndarray, language: str = "auto") -> Dict[str, Any]:
        """Transcribe audio chunk for streaming"""
        if not self._initialized:
            if not self.initialize():
                return {"error": "ASR model not initialized"}
        
        try:
            # Ensure proper format
            if len(audio_chunk.shape) > 1:
                audio_chunk = np.mean(audio_chunk, axis=1)
            
            # Transcribe chunk
            segments, info = self.model.transcribe(
                audio_chunk,
                language=language if language != "auto" else None,
                beam_size=self.beam_size,
                condition_on_previous_text=False  # For streaming
            )
            
            # Get text from segments
            text = " ".join([segment.text.strip() for segment in segments])
            
            return {
                "text": text,
                "language": info.language,
                "language_probability": info.language_probability,
                "is_final": False
            }
            
        except Exception as e:
            logger.error(f"Chunk transcription failed: {e}")
            return {"error": str(e)}
    
    def _resample_audio(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio to target sample rate"""
        try:
            import librosa
            return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
        except ImportError:
            # Fallback using scipy
            from scipy import signal
            num_samples = int(len(audio) * target_sr / orig_sr)
            return signal.resample(audio, num_samples)
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages"""
        return ["en", "hi", "auto"]
    
    def is_ready(self) -> bool:
        """Check if ASR backend is ready"""
        return self._initialized and self.model is not None

# Global ASR instance
asr_backend = ASRBackend()
