"""
Configuration management for immigration_translator_offline
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class Config:
    """Configuration manager for the application"""
    
    def __init__(self, config_path: str = "./config/config.json"):
        self.config_path = Path(config_path)
        self._config: Dict[str, Any] = {}
        self.load_config()
    
    def load_config(self) -> None:
        """Load configuration from JSON file"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self._config = json.load(f)
                logger.info(f"Loaded configuration from {self.config_path}")
            else:
                logger.warning(f"Config file not found at {self.config_path}, using defaults")
                self._config = self._get_default_config()
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            self._config = self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "mode": "offline",
            "languages": ["en", "hi"],
            "src_lang_code": "eng_Latn",
            "tgt_lang_code": "hin_Deva",
            "tts_backend": "piper",
            "asr_model_path": "./models/whisper-small-ct2",
            "mt_model_path": "./models/nllb-200-distilled-600M",
            "piper_voice_hi": "./voices/hi-IN.onnx",
            "piper_voice_en": "./voices/en-US.onnx",
            "xtts_model_path": "./models/xtts-v2",
            "device": "auto",
            "compute_type": "int8",
            "beam_size": 1,
            "chunk_length": 30,
            "enable_vad": True,
            "mt_device": "auto",
            "tts_device": "cpu"
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self._config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value"""
        self._config[key] = value
    
    def save_config(self) -> None:
        """Save configuration to file"""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self._config, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved configuration to {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
    
    def get_model_path(self, model_type: str) -> str:
        """Get model path for specific type"""
        path_map = {
            "asr": self.get("asr_model_path"),
            "mt": self.get("mt_model_path"),
            "xtts": self.get("xtts_model_path")
        }
        return path_map.get(model_type, "")
    
    def get_voice_path(self, language: str) -> str:
        """Get voice file path for specific language"""
        voice_map = {
            "hi": self.get("piper_voice_hi"),
            "en": self.get("piper_voice_en")
        }
        return voice_map.get(language, "")
    
    def is_offline_mode(self) -> bool:
        """Check if running in offline mode"""
        return self.get("mode") == "offline"
    
    def get_device_config(self) -> Dict[str, str]:
        """Get device configuration"""
        return {
            "asr_device": self.get("device", "auto"),
            "mt_device": self.get("mt_device", "auto"),
            "tts_device": self.get("tts_device", "cpu")
        }

# Global config instance
config = Config()

