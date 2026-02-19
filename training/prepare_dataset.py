"""
Phase D: Prepare Dataset (with Augmentation Support)
=====================================================
Converts original + augmented samples into HuggingFace Dataset format
for Whisper fine-tuning.

Usage:
    python -m training.prepare_dataset
    python -m training.prepare_dataset --no-augmented
"""

import json
from pathlib import Path
from typing import List, Dict

try:
    import librosa
    import numpy as np
    from datasets import Dataset, DatasetDict, Audio
except ImportError:
    print("ERROR: Missing dependencies. Run:")
    print("  pip install datasets librosa")
    exit(1)


# =============================================================================
# Config
# =============================================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "training_data"
MANIFEST_PATH = DATA_DIR / "manifest.json"
AUG_MANIFEST_PATH = DATA_DIR / "augmented_manifest.json"
OUTPUT_DIR = PROJECT_DIR / "training" / "hf_dataset" / "rap_transcription"

TARGET_SR = 16000
VAL_SPLIT = 0.15  # 15% validation


# =============================================================================
# Data Loading
# =============================================================================

def load_original_samples() -> List[Dict]:
    """Load original samples from manifest."""
    if not MANIFEST_PATH.exists():
        print(f"ERROR: Manifest not found at {MANIFEST_PATH}")
        return []
    
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
    
    return manifest.get("samples", [])


def load_augmented_samples() -> List[Dict]:
    """Load augmented samples from augmented manifest."""
    if not AUG_MANIFEST_PATH.exists():
        return []
    
    with open(AUG_MANIFEST_PATH) as f:
        aug_manifest = json.load(f)
    
    return aug_manifest.get("augmented_samples", [])


def load_lyrics(lyrics_path: str) -> str:
    """Load and clean lyrics from file."""
    path = Path(lyrics_path)
    if not path.is_absolute():
        path = DATA_DIR / path
    
    if not path.exists():
        return ""
    
    text = path.read_text(encoding="utf-8").strip()
    # Normalize whitespace
    text = " ".join(text.split())
    return text


def load_audio_array(audio_path: str) -> np.ndarray:
    """Load audio as numpy array at 16kHz."""
    path = Path(audio_path)
    if not path.is_absolute():
        path = DATA_DIR / path
    
    y, _ = librosa.load(str(path), sr=TARGET_SR, mono=True)
    return y


# =============================================================================
# Dataset Construction
# =============================================================================

def build_dataset(include_augmented: bool = True) -> DatasetDict:
    """
    Build HuggingFace DatasetDict from original + augmented samples.
    
    Validation set is ONLY original samples (never augmented).
    This ensures eval measures true generalization.
    """
    original = load_original_samples()
    augmented = load_augmented_samples() if include_augmented else []
    
    if not original:
        print("ERROR: No original samples found")
        return None
    
    print(f"\nLoading samples...")
    print(f"  Original:  {len(original)}")
    print(f"  Augmented: {len(augmented)}")
    
    # Split originals into train/val FIRST
    # Use deterministic split based on sample index
    np.random.seed(42)
    indices = np.random.permutation(len(original))
    n_val = max(1, int(len(original) * VAL_SPLIT))
    
    val_indices = set(indices[:n_val])
    
    train_samples = []
    val_samples = []
    
    # Track which original sample IDs go to validation
    val_source_ids = set()
    
    print("\nProcessing original samples...")
    skipped = 0
    for i, sample in enumerate(original):
        audio_path = sample.get("audio_file", "")
        lyrics_path = sample.get("transcript_file", "")
        
        # Resolve paths
        if not Path(audio_path).is_absolute():
            audio_path = str(DATA_DIR / audio_path)
        
        if not Path(audio_path).exists():
            skipped += 1
            continue
        
        lyrics = load_lyrics(lyrics_path)
        if not lyrics:
            skipped += 1
            continue
        
        entry = {
            "audio_path": audio_path,
            "text": lyrics,
            "artist": sample.get("artist", ""),
            "song": sample.get("song", ""),
            "augmented": False,
        }
        
        if i in val_indices:
            val_samples.append(entry)
            val_source_ids.add(sample.get("id", f"sample_{i}"))
        else:
            train_samples.append(entry)
    
    # Add augmented samples to TRAINING ONLY
    # Exclude augmented versions of validation samples
    print("Processing augmented samples...")
    aug_added = 0
    aug_excluded = 0
    
    for sample in augmented:
        source_id = sample.get("source_id", "")
        audio_path = sample.get("audio_file", "")
        lyrics_path = sample.get("transcript_file", "")
        
        # Don't add augmented versions of validation samples
        if source_id in val_source_ids:
            aug_excluded += 1
            continue
        
        if not Path(audio_path).exists():
            skipped += 1
            continue
        
        lyrics = load_lyrics(lyrics_path)
        if not lyrics:
            skipped += 1
            continue
        
        entry = {
            "audio_path": audio_path,
            "text": lyrics,
            "artist": sample.get("artist", ""),
            "song": sample.get("song", ""),
            "augmented": True,
        }
        train_samples.append(entry)
        aug_added += 1
    
    print(f"\n  Train samples:     {len(train_samples)} ({len(train_samples) - aug_added} original + {aug_added} augmented)")
    print(f"  Val samples:       {len(val_samples)} (original only)")
    print(f"  Aug excluded:      {aug_excluded} (from val sources)")
    print(f"  Skipped:           {skipped}")
    
    # Calculate total audio duration
    total_duration = 0
    
    print("\nLoading audio arrays...")
    
    def process_samples(samples: List[Dict], label: str) -> Dict:
        """Convert sample list to HuggingFace-compatible dict."""
        nonlocal total_duration
        
        audio_arrays = []
        texts = []
        processed = 0
        
        for s in samples:
            try:
                y = load_audio_array(s["audio_path"])
                audio_arrays.append({"array": y, "sampling_rate": TARGET_SR})
                texts.append(s["text"])
                total_duration += len(y) / TARGET_SR
                processed += 1
                
                if processed % 25 == 0:
                    print(f"  {label}: {processed}/{len(samples)}")
            except Exception as e:
                print(f"  ERROR loading {s.get('song', '?')}: {e}")
        
        print(f"  {label}: {processed}/{len(samples)} loaded")
        
        return {
            "audio": audio_arrays,
            "text": texts,
        }
    
    train_data = process_samples(train_samples, "Train")
    val_data = process_samples(val_samples, "Val")
    
    # Build DatasetDict
    dataset = DatasetDict({
        "train": Dataset.from_dict(train_data),
        "validation": Dataset.from_dict(val_data),
    })
    
    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(OUTPUT_DIR))
    
    total_minutes = total_duration / 60
    
    print(f"\n{'=' * 60}")
    print("DATASET READY")
    print(f"{'=' * 60}")
    print(f"  Training:    {len(train_data['text'])} samples")
    print(f"  Validation:  {len(val_data['text'])} samples")
    print(f"  Total audio: {total_minutes:.1f} min")
    print(f"  Saved to:    {OUTPUT_DIR}")
    print(f"{'=' * 60}\n")
    
    return dataset


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare dataset for Whisper fine-tuning")
    parser.add_argument(
        "--no-augmented", action="store_true",
        help="Exclude augmented samples (original only)"
    )
    
    args = parser.parse_args()
    build_dataset(include_augmented=not args.no_augmented)
