# Immigration Translator Offline

A completely offline speech-to-speech translation system for Hindi ↔ English, built with Python 3.10+, FastAPI, and local AI models. No Docker, no cloud services, no network required at runtime.

## 🌟 Features

- **Speech-to-Speech Translation**: Hindi ↔ English bidirectional translation
- **Completely Offline**: All models run locally, no internet required
- **Modern Web Interface**: Clean, responsive HTML/JS frontend with MediaRecorder
- **Multiple TTS Backends**: Piper TTS and XTTS v2 support
- **Optimized Performance**: Uses int8 quantization and efficient models
- **Cross-Platform**: Windows batch script and Linux/macOS shell script

## 🏗️ Architecture

- **Backend**: FastAPI with REST endpoints
- **ASR**: faster-whisper with CTranslate2 (whisper-small-ct2)
- **MT**: Facebook NLLB-200-distilled-600M via Transformers
- **TTS**: Piper TTS (default) or XTTS v2 (configurable)
- **Frontend**: Minimal HTML/JS with MediaRecorder API

## 📋 Requirements

- **Python**: 3.10 or higher
- **RAM**: 8GB+ recommended (4GB minimum)
- **Storage**: 5GB+ for models
- **GPU**: Optional but recommended (4GB+ VRAM for GPU acceleration)

## 🚀 Quick Start

### Windows

1. **Clone/Download** the project folder
2. **Run the batch script**:
   ```cmd
   run.bat
   ```
3. **Open browser** to `http://localhost:8000`

### Linux/macOS

1. **Clone/Download** the project folder
2. **Make script executable**:
   ```bash
   chmod +x run.sh
   ```
3. **Run the shell script**:
   ```bash
   ./run.sh
   ```
4. **Open browser** to `http://localhost:8000`

The scripts will automatically:
- Create a virtual environment
- Install dependencies
- Download models (if not present)
- Start the server

## 📁 Project Structure

```
immigration_translator_offline/
├── app/                    # FastAPI backend
│   ├── main.py            # Main FastAPI app
│   ├── config.py          # Configuration management
│   ├── asr.py             # Speech recognition backend
│   ├── mt.py              # Machine translation backend
│   └── tts.py             # Text-to-speech backend
├── config/
│   └── config.json        # Configuration file
├── scripts/
│   └── download_models.py # Model download script
├── static/
│   └── index.html         # Web frontend
├── tests/
│   └── test_app.py        # Unit tests
├── models/                # AI models (created by download script)
├── voices/                # TTS voice files
├── samples/               # Test audio samples
├── requirements.txt       # Python dependencies
├── run.bat               # Windows startup script
├── run.sh                # Linux/macOS startup script
└── README.md             # This file
```

## 🔧 Manual Setup

If you prefer manual setup or the scripts don't work:

### 1. Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Models

```bash
python scripts/download_models.py
```

### 4. Set Offline Environment Variables

```bash
# Windows
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1

# Linux/macOS
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

### 5. Start the Server

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## 🎛️ Configuration

Edit `config/config.json` to customize:

```json
{
  "mode": "offline",
  "languages": ["en", "hi"],
  "src_lang_code": "eng_Latn",
  "tgt_lang_code": "hin_Deva",
  "tts_backend": "piper",
  "asr_model_path": "./models/whisper-small-ct2",
  "mt_model_path": "./models/nllb-200-distilled-600M",
  "piper_voice_hi": "./voices/hi-IN.onnx",
  "piper_voice_en": "./voices/en-US.onnx",
  "device": "auto",
  "compute_type": "int8",
  "beam_size": 1,
  "chunk_length": 30,
  "enable_vad": true,
  "mt_device": "auto",
  "tts_device": "cpu"
}
```

### Key Settings

- **`device`**: `"auto"`, `"cpu"`, or `"cuda"`
- **`compute_type`**: `"int8"`, `"int8_float16"`, or `"float16"`
- **`tts_backend`**: `"piper"` or `"xtts"`
- **`mt_device`**: Force MT to CPU if GPU memory is limited

## 🔌 API Endpoints

- **`GET /health`** - Health check and backend status
- **`GET /config`** - Get current configuration
- **`POST /config`** - Update configuration
- **`POST /asr`** - Transcribe audio file
- **`POST /asr/chunk`** - Transcribe audio chunk (streaming)
- **`POST /translate`** - Translate text
- **`POST /tts`** - Synthesize speech
- **`GET /`** - Serve web frontend

```


