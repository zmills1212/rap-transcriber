"""
Create a sample dataset for testing the training pipeline.
Uses your existing test audio file with sample transcriptions.

Usage:
    python scripts/create_sample_dataset.py
"""
import sys
sys.path.insert(0, '.')

import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import torch
import torchaudio
from pathlib import Path
import json
import shutil

from src.data.manifest import DataManifest, ManifestEntry
from src.data.audio_processor import AudioProcessor
from src.data.feature_extractor import FeatureExtractor


def create_audio_segments(
    source_audio: Path,
    output_dir: Path,
    segment_duration: float = 10.0,
    max_segments: int = 5
) -> list:
    """
    Split source audio into segments for training.
    
    Args:
        source_audio: Path to source audio file
        output_dir: Directory to save segments
        segment_duration: Duration of each segment in seconds
        max_segments: Maximum number of segments to create
        
    Returns:
        List of created segment paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load audio
    waveform, sample_rate = torchaudio.load(str(source_audio))
    
    # Convert to mono if needed
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    # Calculate segment samples
    segment_samples = int(segment_duration * sample_rate)
    total_samples = waveform.shape[1]
    
    segments = []
    
    for i in range(max_segments):
        start = i * segment_samples
        end = start + segment_samples
        
        if end > total_samples:
            break
        
        segment = waveform[:, start:end]
        
        # Save segment
        output_path = output_dir / f"segment_{i:03d}.wav"
        torchaudio.save(str(output_path), segment, sample_rate)
        
        segments.append({
            'path': output_path,
            'duration': segment_duration
        })
    
    return segments


def create_sample_transcriptions() -> list:
    """Create sample rap transcriptions for testing."""
    return [
        "i'm finna get this bread yeah every day i'm on my grind",
        "she bussin it down low yeah skrt skrt in the coupe",
        "no cap this fire got my homies with me tonight",
        "we getting money yeah stacking bands up to the ceiling",
        "ion know what they talking bout but we still winning",
        "ayy let's go yeah we on top now",
        "facts no cap i'm the realest in the game right now",
        "drip too hard yeah the ice is freezing cold",
    ]


def create_sample_dataset():
    """Create a complete sample dataset."""
    print("=" * 50)
    print("CREATING SAMPLE DATASET")
    print("=" * 50)
    
    # Paths
    source_audio = Path("data/raw/test/sample.mp3")
    segments_dir = Path("data/raw/segments")
    processed_dir = Path("data/processed")
    
    # Check for source audio
    if not source_audio.exists():
        print(f"\n❌ Source audio not found: {source_audio}")
        print("\nPlease ensure you have a test audio file.")
        return False
    
    print(f"\nSource audio: {source_audio}")
    
    # Create segments
    print("\nCreating audio segments...")
    segments = create_audio_segments(
        source_audio,
        segments_dir,
        segment_duration=10.0,
        max_segments=5
    )
    print(f"  Created {len(segments)} segments")
    
    # Get sample transcriptions
    transcriptions = create_sample_transcriptions()
    
    # Create manifest entries
    print("\nCreating manifest entries...")
    entries = []
    
    for i, segment in enumerate(segments):
        text = transcriptions[i % len(transcriptions)]
        
        entry = ManifestEntry(
            audio_path=str(segment['path']),
            text=text,
            duration=segment['duration'],
            artist="Sample Artist",
            song_title=f"Sample Track {i+1}"
        )
        entries.append(entry)
        
        # Also create text file
        text_path = segment['path'].with_suffix('.txt')
        with open(text_path, 'w') as f:
            f.write(text)
    
    print(f"  Created {len(entries)} entries")
    
    # Create train/val/test split
    print("\nCreating train/val/test splits...")
    
    if len(entries) >= 3:
        train_entries = entries[:-2]
        val_entries = [entries[-2]]
        test_entries = [entries[-1]]
    else:
        train_entries = entries
        val_entries = entries
        test_entries = entries
    
    train_manifest = DataManifest(train_entries)
    val_manifest = DataManifest(val_entries)
    test_manifest = DataManifest(test_entries)
    
    # Save manifests
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    train_manifest.save(processed_dir / 'train_manifest.json')
    val_manifest.save(processed_dir / 'val_manifest.json')
    test_manifest.save(processed_dir / 'test_manifest.json')
    
    # Pre-extract features for faster training
    print("\nPre-extracting features...")
    features_dir = processed_dir / 'features'
    features_dir.mkdir(exist_ok=True)
    
    processor = AudioProcessor(max_duration_sec=15.0)
    extractor = FeatureExtractor()
    
    for i, entry in enumerate(entries):
        waveform = processor.process(entry.audio_path)
        features = extractor.extract(waveform)
        
        feature_path = features_dir / f"segment_{i:03d}.pt"
        torch.save({
            'features': features,
            'text': entry.text,
            'audio_path': entry.audio_path
        }, feature_path)
    
    print(f"  Saved features to {features_dir}/")
    
    # Print summary
    print("\n" + "=" * 50)
    print("SAMPLE DATASET CREATED")
    print("=" * 50)
    print(f"\nAudio segments: {segments_dir}/")
    print(f"Manifests: {processed_dir}/")
    print(f"Features: {features_dir}/")
    print(f"\nDataset summary:")
    print(f"  Train: {len(train_entries)} samples")
    print(f"  Val:   {len(val_entries)} samples")
    print(f"  Test:  {len(test_entries)} samples")
    print(f"  Total duration: {sum(e.duration for e in entries):.1f}s")
    
    print("\n✅ Sample dataset ready for training!")
    print("\nTo train with this dataset:")
    print("  PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/train.py --debug --epochs 2")
    
    return True


def verify_dataset():
    """Verify the sample dataset is valid."""
    print("\nVerifying dataset...")
    
    processed_dir = Path("data/processed")
    
    # Check manifests exist
    manifests = ['train_manifest.json', 'val_manifest.json', 'test_manifest.json']
    
    for manifest_name in manifests:
        manifest_path = processed_dir / manifest_name
        if not manifest_path.exists():
            print(f"  ❌ Missing: {manifest_path}")
            return False
        
        manifest = DataManifest.load(manifest_path)
        print(f"  ✅ {manifest_name}: {len(manifest)} entries")
        
        # Verify audio files exist
        for entry in manifest.entries:
            if not Path(entry.audio_path).exists():
                print(f"    ⚠️  Audio not found: {entry.audio_path}")
    
    # Check features
    features_dir = processed_dir / 'features'
    if features_dir.exists():
        feature_files = list(features_dir.glob('*.pt'))
        print(f"  ✅ Features: {len(feature_files)} files")
    
    return True


if __name__ == "__main__":
    success = create_sample_dataset()
    
    if success:
        verify_dataset()
