"""
Audio preprocessing for rap transcription.
Handles loading, normalization, resampling, and segmentation.
"""
import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Union


class AudioProcessor:
    """
    Preprocesses audio files for the transcription model.
    """
    
    def __init__(
        self,
        target_sample_rate: int = 16000,
        max_duration_sec: float = 30.0,
        normalize: bool = True
    ):
        self.target_sample_rate = target_sample_rate
        self.max_duration_sec = max_duration_sec
        self.normalize = normalize
        self.max_samples = int(max_duration_sec * target_sample_rate)
    
    def load_audio(self, audio_path: Union[str, Path]) -> Tuple[torch.Tensor, int]:
        """
        Load audio file and return waveform + sample rate.
        
        Args:
            audio_path: Path to audio file (wav, mp3, flac, etc.)
            
        Returns:
            waveform: Audio tensor of shape [channels, samples]
            sample_rate: Original sample rate
        """
        audio_path = Path(audio_path)
        
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        waveform, sample_rate = torchaudio.load(str(audio_path))
        return waveform, sample_rate
    
    def resample(
        self, 
        waveform: torch.Tensor, 
        orig_sample_rate: int
    ) -> torch.Tensor:
        """
        Resample audio to target sample rate.
        """
        if orig_sample_rate == self.target_sample_rate:
            return waveform
        
        resampler = torchaudio.transforms.Resample(
            orig_freq=orig_sample_rate,
            new_freq=self.target_sample_rate
        )
        return resampler(waveform)
    
    def to_mono(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Convert stereo audio to mono by averaging channels.
        """
        if waveform.shape[0] == 1:
            return waveform
        
        return waveform.mean(dim=0, keepdim=True)
    
    def normalize_audio(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Normalize audio to [-1, 1] range.
        """
        max_val = waveform.abs().max()
        if max_val > 0:
            waveform = waveform / max_val
        return waveform
    
    def trim_or_pad(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Trim audio to max duration or pad if shorter.
        """
        current_samples = waveform.shape[1]
        
        if current_samples > self.max_samples:
            waveform = waveform[:, :self.max_samples]
        elif current_samples < self.max_samples:
            padding = self.max_samples - current_samples
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        
        return waveform
    
    def process(
        self, 
        audio_path: Union[str, Path],
        trim_pad: bool = True
    ) -> torch.Tensor:
        """
        Full preprocessing pipeline.
        
        Args:
            audio_path: Path to audio file
            trim_pad: Whether to trim/pad to fixed length
            
        Returns:
            Processed audio tensor of shape [1, samples]
        """
        waveform, sample_rate = self.load_audio(audio_path)
        waveform = self.to_mono(waveform)
        waveform = self.resample(waveform, sample_rate)
        
        if self.normalize:
            waveform = self.normalize_audio(waveform)
        
        if trim_pad:
            waveform = self.trim_or_pad(waveform)
        
        return waveform
    
    def get_duration(self, audio_path: Union[str, Path]) -> float:
        """
        Get duration of audio file in seconds.
        """
        waveform, sample_rate = self.load_audio(audio_path)
        return waveform.shape[1] / sample_rate


if __name__ == "__main__":
    print("✅ AudioProcessor module loaded successfully!")
    print("")
    print("Usage example:")
    print("  from src.data.audio_processor import AudioProcessor")
    print("  processor = AudioProcessor()")
    print("  waveform = processor.process('path/to/audio.wav')")
