"""
Whisper-based transcription engine.
Uses OpenAI's Whisper for accurate speech-to-text.
"""
import sys
sys.path.insert(0, '.')

import whisper
import torch
from pathlib import Path
from typing import Optional, Dict
import time

from src.inference.postprocessor import TranscriptionCleaner


class WhisperTranscriber:
    """
    Production transcriber using OpenAI Whisper.
    
    Usage:
        transcriber = WhisperTranscriber()
        result = transcriber.transcribe("audio.mp3")
        print(result['text'])
    """
    
    def __init__(
        self,
        model_size: str = "base",
        device: Optional[str] = None
    ):
        """
        Initialize Whisper transcriber.
        
        Args:
            model_size: Whisper model size
                - tiny: fastest, least accurate (~39M params)
                - base: good balance (~74M params)
                - small: better accuracy (~244M params)
                - medium: high accuracy (~769M params)
                - large: best accuracy (~1.5B params)
            device: Device to use (auto-detects if None)
        """
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"  # MPS has issues with Whisper
        
        self.device = device
        self.model_size = model_size
        
        print(f"Loading Whisper '{model_size}' model...")
        self.model = whisper.load_model(model_size, device=device)
        print(f"✅ Whisper loaded on {device}")
        
        self.cleaner = TranscriptionCleaner()
    
    def transcribe(
        self,
        audio_path: str,
        language: str = "en",
        format_as_lyrics: bool = False,
        show_timing: bool = False
    ) -> Dict:
        """
        Transcribe audio file.
        
        Args:
            audio_path: Path to audio file
            language: Language code
            format_as_lyrics: Format output as lyrics
            show_timing: Include word-level timestamps
            
        Returns:
            Dict with transcription results
        """
        audio_path = Path(audio_path)
        
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        start_time = time.time()
        
        # Transcribe with Whisper
        result = self.model.transcribe(
            str(audio_path),
            language=language,
            word_timestamps=show_timing
        )
        
        inference_time = time.time() - start_time
        
        # Get raw text
        raw_text = result["text"].strip()
        
        # Apply rap-specific post-processing
        cleaned_text = self.cleaner.clean(raw_text, format_as_lyrics=format_as_lyrics)
        
        output = {
            "text": cleaned_text,
            "raw_text": raw_text,
            "language": result.get("language", language),
            "processing_time_seconds": round(inference_time, 3)
        }
        
        # Add segments
        if "segments" in result:
            output["segments"] = [
                {
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"].strip()
                }
                for seg in result["segments"]
            ]
        
        return output
    
    def transcribe_with_timestamps(self, audio_path: str) -> Dict:
        """Transcribe with word-level timestamps."""
        return self.transcribe(audio_path, show_timing=True)


def quick_transcribe(audio_path: str, model_size: str = "base") -> str:
    """
    Quick one-liner transcription.
    
    Usage:
        text = quick_transcribe("audio.mp3")
    """
    transcriber = WhisperTranscriber(model_size=model_size)
    result = transcriber.transcribe(audio_path)
    return result["text"]


if __name__ == "__main__":
    print("=" * 50)
    print("🎤 WHISPER TRANSCRIPTION TEST")
    print("=" * 50)
    print("")
    
    test_audio = Path("data/raw/test/sample.mp3")
    
    if not test_audio.exists():
        print(f"❌ Test audio not found: {test_audio}")
        sys.exit(1)
    
    print(f"Audio file: {test_audio}")
    print("")
    
    # Initialize
    transcriber = WhisperTranscriber(model_size="small")
    
    print("")
    print("Transcribing...")
    
    # Transcribe
    result = transcriber.transcribe(test_audio, format_as_lyrics=False)
    
    print("")
    print("─" * 50)
    print("TRANSCRIPTION:")
    print("─" * 50)
    print(result["text"])
    print("─" * 50)
    print("")
    print(f"⏱️  Processing time: {result['processing_time_seconds']:.2f}s")
    
    # Show segments
    if "segments" in result:
        print("")
        print("SEGMENTS:")
        for seg in result["segments"][:10]:
            print(f"  [{seg['start']:6.1f}s - {seg['end']:6.1f}s]: {seg['text']}")
        if len(result["segments"]) > 10:
            print(f"  ... and {len(result['segments']) - 10} more segments")
    
    print("")
    print("✅ Whisper integration working!")
