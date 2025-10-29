#!/usr/bin/env python3
"""
Create sample audio files for testing
"""

import numpy as np
import soundfile as sf
from pathlib import Path

def create_sample_files():
    """Create sample audio files for testing"""
    samples_dir = Path("samples")
    samples_dir.mkdir(exist_ok=True)
    
    # Generate 2 seconds of silence at 16kHz
    sample_rate = 16000
    duration = 2.0
    silence = np.zeros(int(sample_rate * duration))
    
    sample_file = samples_dir / "test_silence.wav"
    sf.write(sample_file, silence, sample_rate)
    print(f"Created sample file: {sample_file}")
    
    # Generate a simple tone for testing
    t = np.linspace(0, duration, int(sample_rate * duration))
    tone = 0.1 * np.sin(2 * np.pi * 440 * t)  # 440 Hz tone
    tone_file = samples_dir / "test_tone.wav"
    sf.write(tone_file, tone, sample_rate)
    print(f"Created tone file: {tone_file}")

if __name__ == "__main__":
    create_sample_files()


