#!/usr/bin/env python3
"""
Test script for audio processing
"""

import os
import sys
import tempfile
import numpy as np
import soundfile as sf

def create_test_audio():
    """Create a test audio file"""
    # Generate 2 seconds of a simple tone
    sample_rate = 16000
    duration = 2.0
    frequency = 440  # A4 note
    
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = 0.3 * np.sin(2 * np.pi * frequency * t)
    
    # Save as WAV
    test_file = "test_audio.wav"
    sf.write(test_file, audio, sample_rate)
    print(f"Created test audio file: {test_file}")
    return test_file

def test_audio_processing():
    """Test audio processing pipeline"""
    print("Testing audio processing...")
    
    try:
        # Create test audio
        test_file = create_test_audio()
        
        # Read audio data
        with open(test_file, 'rb') as f:
            audio_data = f.read()
        
        print(f"Audio data size: {len(audio_data)} bytes")
        
        # Test temporary file creation
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_path = temp_file.name
            temp_file.write(audio_data)
        
        print(f"Temporary file created: {temp_path}")
        
        # Verify file exists and has content
        if os.path.exists(temp_path):
            file_size = os.path.getsize(temp_path)
            print(f"✓ File exists, size: {file_size} bytes")
            
            # Test reading with soundfile
            audio_array, sample_rate = sf.read(temp_path)
            print(f"✓ Audio loaded: {len(audio_array)} samples at {sample_rate} Hz")
            
            # Clean up
            os.unlink(temp_path)
            os.unlink(test_file)
            
            return True
        else:
            print("✗ Temporary file not created")
            return False
            
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_audio_processing()
    if success:
        print("\n🎉 Audio processing test passed!")
    else:
        print("\n❌ Audio processing test failed")
        sys.exit(1)


