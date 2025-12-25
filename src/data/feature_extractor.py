"""
Feature extraction for rap transcription.
Converts audio waveforms to mel spectrograms.
"""
import torch
import torchaudio
from typing import Optional


class FeatureExtractor:
    """
    Extracts mel spectrogram features from audio waveforms.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 80,
        n_fft: int = 400,
        hop_length: int = 160,
        win_length: Optional[int] = None,
        f_min: float = 0.0,
        f_max: Optional[float] = 8000.0,
        normalize: bool = True
    ):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length or n_fft
        self.normalize = normalize
        
        # Create mel spectrogram transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=self.win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max
        )
    
    def extract(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extract mel spectrogram from waveform.
        
        Args:
            waveform: Audio tensor of shape [1, samples] or [samples]
            
        Returns:
            Mel spectrogram of shape [n_mels, time_frames]
        """
        # Ensure waveform is 2D
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        # Compute mel spectrogram
        mel_spec = self.mel_transform(waveform)
        
        # Convert to log scale (add small epsilon to avoid log(0))
        mel_spec = torch.log(mel_spec + 1e-9)
        
        # Remove channel dimension: [1, n_mels, time] -> [n_mels, time]
        mel_spec = mel_spec.squeeze(0)
        
        # Normalize if requested
        if self.normalize:
            mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-9)
        
        return mel_spec
    
    def get_output_length(self, input_samples: int) -> int:
        """
        Calculate output time frames for given input samples.
        """
        return (input_samples // self.hop_length) + 1
    
    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Allow using extractor as a callable.
        """
        return self.extract(waveform)


def extract_features(
    waveform: torch.Tensor,
    sample_rate: int = 16000,
    n_mels: int = 80
) -> torch.Tensor:
    """
    Convenience function for quick feature extraction.
    """
    extractor = FeatureExtractor(sample_rate=sample_rate, n_mels=n_mels)
    return extractor.extract(waveform)


if __name__ == "__main__":
    print("✅ FeatureExtractor module loaded successfully!")
    print("")
    
    # Quick test with dummy audio
    dummy_waveform = torch.randn(1, 16000 * 5)  # 5 seconds at 16kHz
    extractor = FeatureExtractor()
    features = extractor.extract(dummy_waveform)
    
    print(f"   Input shape:  {dummy_waveform.shape} (5 sec audio)")
    print(f"   Output shape: {features.shape} (mel spectrogram)")
    print(f"   Time frames:  {features.shape[1]}")
    print(f"   Mel bins:     {features.shape[0]}")
