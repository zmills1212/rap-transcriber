"""
Phase D: Audio Data Augmentation Pipeline
==========================================
Takes existing rap transcription samples and creates augmented variants
with speed perturbation, pitch shifting, and noise injection.

Ground truth lyrics stay the same — only audio characteristics change.
This forces the model to learn acoustic-to-text mappings rather than
memorizing specific audio fingerprints.

Usage:
    python -m training.augment_audio
    python -m training.augment_audio --multiplier 5
    python -m training.augment_audio --dry-run
"""

import json
import shutil
import hashlib
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np

try:
    import librosa
    import soundfile as sf
except ImportError:
    print("ERROR: Missing dependencies. Run:")
    print("  pip install librosa soundfile")
    exit(1)


# =============================================================================
# Config
# =============================================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "training_data"
AUGMENTED_DIR = DATA_DIR / "augmented_audio"
MANIFEST_PATH = DATA_DIR / "manifest.json"
AUG_MANIFEST_PATH = DATA_DIR / "augmented_manifest.json"

# Target sample rate (Whisper expects 16kHz)
TARGET_SR = 16000

# Augmentation configs — each produces one variant per original sample
AUGMENTATIONS = [
    # Speed perturbations (same pitch, different tempo)
    {"name": "speed_slow",     "type": "speed",  "rate": 0.9},
    {"name": "speed_fast",     "type": "speed",  "rate": 1.1},
    
    # Pitch shifts (different pitch, same tempo)
    {"name": "pitch_down",     "type": "pitch",  "n_steps": -1.5},
    {"name": "pitch_up",       "type": "pitch",  "n_steps": 1.5},
    
    # Noise injection (light background noise)
    {"name": "noise_light",    "type": "noise",  "snr_db": 25},
    
    # Combined augmentations
    {"name": "slow_pitchdown", "type": "combo",  "rate": 0.93, "n_steps": -1.0},
    {"name": "fast_pitchup",   "type": "combo",  "rate": 1.07, "n_steps": 1.0},
]


# =============================================================================
# Augmentation Functions
# =============================================================================

def load_audio(audio_path: str) -> tuple:
    """Load audio file, return (waveform, sample_rate)."""
    y, sr = librosa.load(audio_path, sr=TARGET_SR, mono=True)
    return y, sr


def augment_speed(y: np.ndarray, sr: int, rate: float) -> np.ndarray:
    """
    Change playback speed without affecting pitch.
    rate < 1.0 = slower, rate > 1.0 = faster.
    Uses time-stretching so pitch stays constant.
    """
    return librosa.effects.time_stretch(y, rate=rate)


def augment_pitch(y: np.ndarray, sr: int, n_steps: float) -> np.ndarray:
    """
    Shift pitch by n_steps semitones without changing tempo.
    Positive = higher pitch, negative = lower pitch.
    """
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)


def augment_noise(y: np.ndarray, sr: int, snr_db: float) -> np.ndarray:
    """
    Add Gaussian noise at a given SNR (signal-to-noise ratio in dB).
    Higher SNR = less noise. 25dB is subtle background hiss.
    """
    # Calculate signal power
    signal_power = np.mean(y ** 2)
    
    # Calculate noise power from desired SNR
    noise_power = signal_power / (10 ** (snr_db / 10))
    
    # Generate noise
    noise = np.random.normal(0, np.sqrt(noise_power), len(y))
    
    return (y + noise).astype(np.float32)


def apply_augmentation(y: np.ndarray, sr: int, config: dict) -> np.ndarray:
    """Apply a single augmentation config to audio."""
    aug_type = config["type"]
    
    if aug_type == "speed":
        return augment_speed(y, sr, config["rate"])
    
    elif aug_type == "pitch":
        return augment_pitch(y, sr, config["n_steps"])
    
    elif aug_type == "noise":
        return augment_noise(y, sr, config["snr_db"])
    
    elif aug_type == "combo":
        result = y.copy()
        if "rate" in config:
            result = augment_speed(result, sr, config["rate"])
        if "n_steps" in config:
            result = augment_pitch(result, sr, config["n_steps"])
        if "snr_db" in config:
            result = augment_noise(result, sr, config["snr_db"])
        return result
    
    else:
        raise ValueError(f"Unknown augmentation type: {aug_type}")


# =============================================================================
# Pipeline
# =============================================================================

def generate_aug_id(original_id: str, aug_name: str) -> str:
    """Generate a unique ID for an augmented sample."""
    raw = f"{original_id}_{aug_name}"
    short_hash = hashlib.md5(raw.encode()).hexdigest()[:6]
    return f"aug_{short_hash}_{aug_name}"


def augment_dataset(
    multiplier: Optional[int] = None,
    dry_run: bool = False,
    verbose: bool = True
) -> List[Dict]:
    """
    Augment the entire training dataset.
    
    Args:
        multiplier: Max augmentations per sample (None = use all)
        dry_run: If True, just print what would happen
        verbose: Print progress
    
    Returns:
        List of augmented sample entries for manifest
    """
    # Load manifest
    if not MANIFEST_PATH.exists():
        print(f"ERROR: Manifest not found at {MANIFEST_PATH}")
        print("Run data collection first.")
        return []
    
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
    
    samples = manifest.get("samples", [])
    if not samples:
        print("ERROR: No samples in manifest")
        return []
    
    # Determine which augmentations to apply
    augs = AUGMENTATIONS[:multiplier] if multiplier else AUGMENTATIONS
    
    total_expected = len(samples) * len(augs)
    
    print("\n" + "=" * 60)
    print("AUDIO DATA AUGMENTATION")
    print("=" * 60)
    print(f"  Original samples:     {len(samples)}")
    print(f"  Augmentations/sample: {len(augs)}")
    print(f"  Expected output:      {total_expected} augmented samples")
    print(f"  Total after merge:    {len(samples) + total_expected} samples")
    print(f"  Output directory:     {AUGMENTED_DIR}")
    
    if dry_run:
        print("\n  [DRY RUN — no files will be created]")
        for aug in augs:
            print(f"    • {aug['name']}: {aug['type']} — {', '.join(f'{k}={v}' for k, v in aug.items() if k not in ('name', 'type'))}")
        print("=" * 60)
        return []
    
    print("=" * 60 + "\n")
    
    # Create output directory
    AUGMENTED_DIR.mkdir(parents=True, exist_ok=True)
    
    augmented_samples = []
    skipped = 0
    errors = 0
    
    for i, sample in enumerate(samples):
        audio_path = sample.get("audio_file", "")
        
        # Resolve relative paths
        if not Path(audio_path).is_absolute():
            audio_path = str(DATA_DIR / audio_path)
        
        if not Path(audio_path).exists():
            if verbose:
                print(f"  SKIP [{i+1}/{len(samples)}] {sample.get('song', '?')} — audio not found")
            skipped += 1
            continue
        
        if verbose:
            artist = sample.get("artist", "?")
            song = sample.get("song", "?")
            print(f"  [{i+1}/{len(samples)}] {artist} - {song}")
        
        # Load audio once, apply all augmentations
        try:
            y, sr = load_audio(audio_path)
        except Exception as e:
            print(f"    ERROR loading audio: {e}")
            errors += 1
            continue
        
        for aug_config in augs:
            aug_name = aug_config["name"]
            aug_id = generate_aug_id(sample.get("id", f"sample_{i}"), aug_name)
            
            # Output path
            out_filename = f"{aug_id}.wav"
            out_path = AUGMENTED_DIR / out_filename
            
            try:
                # Apply augmentation
                y_aug = apply_augmentation(y, sr, aug_config)
                
                # Normalize to prevent clipping
                peak = np.max(np.abs(y_aug))
                if peak > 0.99:
                    y_aug = y_aug * (0.99 / peak)
                
                # Save
                sf.write(str(out_path), y_aug, sr)
                
                # Create augmented sample entry
                aug_sample = {
                    "id": aug_id,
                    "audio_file": str(out_path),
                    "transcript_file": sample.get("transcript_file", ""),
                    "artist": sample.get("artist", ""),
                    "song": f"{sample.get('song', '')} [{aug_name}]",
                    "tags": sample.get("challenge_tags", []),
                    "augmentation": aug_name,
                    "source_id": sample.get("id", f"sample_{i}"),
                    "augmented": True,
                }
                augmented_samples.append(aug_sample)
                
                if verbose:
                    duration = len(y_aug) / sr
                    print(f"    ✓ {aug_name} ({duration:.1f}s)")
                
            except Exception as e:
                print(f"    ERROR [{aug_name}]: {e}")
                errors += 1
    
    # Save augmented manifest
    aug_manifest = {
        "augmented_samples": augmented_samples,
        "source_manifest": str(MANIFEST_PATH),
        "augmentation_configs": augs,
        "stats": {
            "original_samples": len(samples),
            "augmented_samples": len(augmented_samples),
            "skipped": skipped,
            "errors": errors,
        }
    }
    
    with open(AUG_MANIFEST_PATH, "w") as f:
        json.dump(aug_manifest, f, indent=2)
    
    print(f"\n{'=' * 60}")
    print("AUGMENTATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Generated:  {len(augmented_samples)} augmented samples")
    print(f"  Skipped:    {skipped}")
    print(f"  Errors:     {errors}")
    print(f"  Manifest:   {AUG_MANIFEST_PATH}")
    print(f"  Audio dir:  {AUGMENTED_DIR}")
    print(f"{'=' * 60}\n")
    
    return augmented_samples


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Augment rap transcription training data"
    )
    parser.add_argument(
        "--multiplier", type=int, default=None,
        help="Max augmentations per sample (default: all 7)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would happen without creating files"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Less verbose output"
    )
    
    args = parser.parse_args()
    
    augment_dataset(
        multiplier=args.multiplier,
        dry_run=args.dry_run,
        verbose=not args.quiet,
    )
