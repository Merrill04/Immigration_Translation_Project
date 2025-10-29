"""
Machine Translation backend using NLLB-200
"""

import os
import logging
from typing import Optional, Dict, Any, List
import torch
from pathlib import Path

# Set offline mode
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
except ImportError as e:
    print(f"Missing MT dependencies: {e}")
    print("Please install: pip install transformers torch")

from app.config import config

logger = logging.getLogger(__name__)

class MTBackend:
    """Machine Translation backend using NLLB-200"""
    
    def __init__(self):
        self.model: Optional[AutoModelForSeq2SeqLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model_path = config.get_model_path("mt")
        self.device = config.get("mt_device", "auto")
        self._initialized = False
        
        # Language code mappings
        self.lang_codes = {
            "en": "eng_Latn",
            "hi": "hin_Deva",
            "eng_Latn": "eng_Latn",
            "hin_Deva": "hin_Deva"
        }
    
    def initialize(self) -> bool:
        """Initialize the translation model"""
        try:
            if not os.path.exists(self.model_path):
                logger.error(f"MT model not found at {self.model_path}")
                return False
            
            logger.info(f"Loading MT model from {self.model_path}")
            
            # Determine device
            if self.device == "auto":
                if torch.cuda.is_available():
                    device = "cuda"
                    logger.info("Using CUDA for MT")
                else:
                    device = "cpu"
                    logger.info("Using CPU for MT")
            else:
                device = self.device
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                local_files_only=True
            )
            
            # Load model with device mapping
            try:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_path,
                    local_files_only=True,
                    device_map="auto" if device == "cuda" else None,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32
                )
                
                if device == "cpu":
                    self.model = self.model.to(device)
                    
            except Exception as e:
                logger.warning(f"Failed to load with device_map, falling back to CPU: {e}")
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_path,
                    local_files_only=True,
                    torch_dtype=torch.float32
                ).to("cpu")
                device = "cpu"
            
            self._initialized = True
            logger.info(f"MT model loaded successfully on {device}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize MT model: {e}")
            return False
    
    def translate(self, text: str, src_lang: str, tgt_lang: str) -> Dict[str, Any]:
        """Translate text from source to target language"""
        if not self._initialized:
            if not self.initialize():
                return {"error": "MT model not initialized"}
        
        try:
            # Normalize language codes
            src_code = self.lang_codes.get(src_lang, src_lang)
            tgt_code = self.lang_codes.get(tgt_lang, tgt_lang)
            
            if not text.strip():
                return {"translated_text": "", "src_lang": src_code, "tgt_lang": tgt_code}
            
            # Tokenize with source language
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            
            # Move to device
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate translation
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    forced_bos_token_id=self.tokenizer.lang_code_to_id[tgt_code],
                    max_length=512,
                    num_beams=2,
                    early_stopping=True,
                    do_sample=False
                )
            
            # Decode translation
            translated_text = self.tokenizer.decode(
                outputs[0], 
                skip_special_tokens=True
            )
            
            return {
                "translated_text": translated_text.strip(),
                "src_lang": src_code,
                "tgt_lang": tgt_code,
                "original_text": text
            }
            
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return {"error": str(e)}
    
    def translate_batch(self, texts: List[str], src_lang: str, tgt_lang: str) -> List[Dict[str, Any]]:
        """Translate multiple texts in batch"""
        if not self._initialized:
            if not self.initialize():
                return [{"error": "MT model not initialized"}] * len(texts)
        
        try:
            # Normalize language codes
            src_code = self.lang_codes.get(src_lang, src_lang)
            tgt_code = self.lang_codes.get(tgt_lang, tgt_lang)
            
            # Filter empty texts
            non_empty_texts = [text for text in texts if text.strip()]
            if not non_empty_texts:
                return [{"translated_text": "", "src_lang": src_code, "tgt_lang": tgt_code}] * len(texts)
            
            # Tokenize batch
            inputs = self.tokenizer(
                non_empty_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            
            # Move to device
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate translations
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    forced_bos_token_id=self.tokenizer.lang_code_to_id[tgt_code],
                    max_length=512,
                    num_beams=2,
                    early_stopping=True,
                    do_sample=False
                )
            
            # Decode translations
            translated_texts = self.tokenizer.batch_decode(
                outputs, 
                skip_special_tokens=True
            )
            
            # Create results
            results = []
            translated_idx = 0
            
            for text in texts:
                if text.strip():
                    results.append({
                        "translated_text": translated_texts[translated_idx].strip(),
                        "src_lang": src_code,
                        "tgt_lang": tgt_code,
                        "original_text": text
                    })
                    translated_idx += 1
                else:
                    results.append({
                        "translated_text": "",
                        "src_lang": src_code,
                        "tgt_lang": tgt_code,
                        "original_text": text
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Batch translation failed: {e}")
            return [{"error": str(e)}] * len(texts)
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages"""
        return ["en", "hi", "eng_Latn", "hin_Deva"]
    
    def is_ready(self) -> bool:
        """Check if MT backend is ready"""
        return self._initialized and self.model is not None and self.tokenizer is not None

# Global MT instance
mt_backend = MTBackend()

