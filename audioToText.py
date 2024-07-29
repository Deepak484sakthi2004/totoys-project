import os
import whisper
from pathlib import Path

def audio_to_transcript(audio_file,model):
    result = model.transcribe(audio_file)
    transcript = result["text"]
    print("THE AUDIO WAS!  : ",transcript)
    return transcript
    
