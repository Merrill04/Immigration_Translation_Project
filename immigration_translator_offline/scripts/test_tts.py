#!/usr/bin/env python3
"""
Test script for pyttsx3 TTS functionality
"""

import pyttsx3
import tempfile
import os

def test_pyttsx3():
    """Test pyttsx3 TTS functionality"""
    print("Testing pyttsx3 TTS...")
    
    try:
        # Initialize engine
        engine = pyttsx3.init()
        
        # Get available voices
        voices = engine.getProperty('voices')
        print(f"Found {len(voices)} voices:")
        
        for i, voice in enumerate(voices):
            print(f"  {i}: {voice.name} ({voice.id})")
        
        # Test synthesis
        test_text = "Hello, this is a test of the TTS system."
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_path = temp_file.name
        
        # Synthesize to file
        engine.save_to_file(test_text, temp_path)
        engine.runAndWait()
        
        # Check if file was created
        if os.path.exists(temp_path):
            file_size = os.path.getsize(temp_path)
            print(f"✓ TTS synthesis successful! Generated {file_size} bytes")
            
            # Clean up
            os.unlink(temp_path)
            return True
        else:
            print("✗ TTS synthesis failed - no output file")
            return False
            
    except Exception as e:
        print(f"✗ TTS test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_pyttsx3()
    if success:
        print("\n🎉 pyttsx3 TTS is working correctly!")
    else:
        print("\n❌ pyttsx3 TTS test failed")


