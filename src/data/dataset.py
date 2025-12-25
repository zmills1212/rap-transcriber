"""
PyTorch Dataset for rap transcription.
"""
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Optional, Union
import json

from src.data.audio_processor import AudioProcessor
from src.data.feature_extractor import FeatureExtractor


class RapDataset(Dataset):
    """
    Dataset for loading rap audio and transcriptions.
    
    Expected data format (JSON manifest):
    [
        {
            "audio_path": "path/to/audio.wav",
            "text": "transcription text",
            "phonemes": "optional phoneme sequence",
            "duration": 5.2
        },
        ...
    ]
    """
    
    def __init__(
        self,
        manifest_path: Union[str, Path],
        audio_processor: Optional[AudioProcessor] = None,
        feature_extractor: Optional[FeatureExtractor] = None,
        max_duration_sec: float = 30.0,
        return_raw_audio: bool = False
    ):
        self.manifest_path = Path(manifest_path)
        self.max_duration_sec = max_duration_sec
        self.return_raw_audio = return_raw_audio
        
        # Initialize processors
        self.audio_processor = audio_processor or AudioProcessor(
            max_duration_sec=max_duration_sec
        )
        self.feature_extractor = feature_extractor or FeatureExtractor()
        
        # Load manifest
        self.samples = self._load_manifest()
    
    def _load_manifest(self) -> List[Dict]:
        """Load and validate manifest file."""
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")
        
        with open(self.manifest_path, 'r') as f:
            samples = json.load(f)
        
        # Filter by duration if specified
        if self.max_duration_sec:
            samples = [
                s for s in samples
                if s.get('duration', 0) <= self.max_duration_sec
            ]
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            Dictionary with:
                - features: mel spectrogram [n_mels, time]
                - text: transcription string
                - audio_path: path to audio file
                - (optional) waveform: raw audio if return_raw_audio=True
        """
        sample = self.samples[idx]
        audio_path = sample['audio_path']
        
        # Process audio
        waveform = self.audio_processor.process(audio_path, trim_pad=True)
        
        # Extract features
        features = self.feature_extractor.extract(waveform)
        
        result = {
            'features': features,
            'text': sample.get('text', ''),
            'audio_path': audio_path,
        }
        
        # Optional: include phonemes if available
        if 'phonemes' in sample:
            result['phonemes'] = sample['phonemes']
        
        # Optional: include raw audio
        if self.return_raw_audio:
            result['waveform'] = waveform
        
        return result


class SimpleAudioDataset(Dataset):
    """
    Simplified dataset for loading audio files from a directory.
    Useful for inference when you don't have a manifest.
    """
    
    def __init__(
        self,
        audio_dir: Union[str, Path],
        audio_extensions: List[str] = ['.wav', '.mp3', '.flac', '.m4a'],
        audio_processor: Optional[AudioProcessor] = None,
        feature_extractor: Optional[FeatureExtractor] = None
    ):
        self.audio_dir = Path(audio_dir)
        self.audio_processor = audio_processor or AudioProcessor()
        self.feature_extractor = feature_extractor or FeatureExtractor()
        
        # Find all audio files
        self.audio_files = []
        for ext in audio_extensions:
            self.audio_files.extend(self.audio_dir.glob(f'*{ext}'))
            self.audio_files.extend(self.audio_dir.glob(f'**/*{ext}'))
        
        self.audio_files = sorted(set(self.audio_files))
    
    def __len__(self) -> int:
        return len(self.audio_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        audio_path = self.audio_files[idx]
        
        waveform = self.audio_processor.process(audio_path, trim_pad=True)
        features = self.feature_extractor.extract(waveform)
        
        return {
            'features': features,
            'audio_path': str(audio_path),
            'filename': audio_path.name
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for batching samples.
    Handles variable length sequences.
    """
    # Stack features
    features = torch.stack([item['features'] for item in batch])
    
    # Collect texts
    texts = [item['text'] for item in batch]
    
    # Collect paths
    audio_paths = [item['audio_path'] for item in batch]
    
    result = {
        'features': features,
        'texts': texts,
        'audio_paths': audio_paths
    }
    
    # Optional: phonemes
    if 'phonemes' in batch[0]:
        result['phonemes'] = [item['phonemes'] for item in batch]
    
    return result


if __name__ == "__main__":
    print("✅ Dataset module loaded successfully!")
    print("")
    print("Classes available:")
    print("  - RapDataset: For training with manifest files")
    print("  - SimpleAudioDataset: For inference from a directory")
    print("  - collate_fn: For batching samples")
