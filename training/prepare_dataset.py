"""
Phase D: Prepare Dataset (Memory-Efficient + Segments)
=======================================================
Converts original + augmented OR segmented samples into HuggingFace Dataset format.
Stores audio FILE PATHS instead of loading arrays into RAM.

Usage:
    python -m training.prepare_dataset
    python -m training.prepare_dataset --segments
    python -m training.prepare_dataset --no-augmented
"""

import json
from pathlib import Path
from typing import List, Dict

try:
    import numpy as np
    from datasets import Dataset, DatasetDict
except ImportError:
    print("ERROR: Missing dependencies. Run:")
    print("  pip install datasets")
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
VAL_SPLIT = 0.15


# =============================================================================
# Data Loading
# =============================================================================

def load_original_samples() -> List[Dict]:
    if not MANIFEST_PATH.exists():
        print(f"ERROR: Manifest not found at {MANIFEST_PATH}")
        return []
    with open(MANIFEST_PATH) as f:
        return json.load(f).get("samples", [])


def load_augmented_samples() -> List[Dict]:
    if not AUG_MANIFEST_PATH.exists():
        return []
    with open(AUG_MANIFEST_PATH) as f:
        return json.load(f).get("augmented_samples", [])


def load_lyrics(lyrics_path: str) -> str:
    path = Path(lyrics_path)
    if not path.is_absolute():
        path = DATA_DIR / path
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8").strip()
    return " ".join(text.split())


def resolve_audio_path(audio_path: str) -> str:
    path = Path(audio_path)
    if not path.is_absolute():
        path = DATA_DIR / path
    return str(path)


# =============================================================================
# Original + Augmented Dataset
# =============================================================================

def build_dataset(include_augmented: bool = True) -> DatasetDict:
    """Build dataset from original + augmented samples (full songs)."""
    original = load_original_samples()
    augmented = load_augmented_samples() if include_augmented else []

    if not original:
        print("ERROR: No original samples found")
        return None

    print(f"\nLoading samples...")
    print(f"  Original:  {len(original)}")
    print(f"  Augmented: {len(augmented)}")

    np.random.seed(42)
    indices = np.random.permutation(len(original))
    n_val = max(1, int(len(original) * VAL_SPLIT))
    val_indices = set(indices[:n_val])

    train_entries = []
    val_entries = []
    val_source_ids = set()
    skipped = 0

    print("\nProcessing original samples...")
    for i, sample in enumerate(original):
        audio_path = sample.get("audio_file", "")
        lyrics_path = sample.get("transcript_file", "")

        abs_audio = resolve_audio_path(audio_path)
        if not Path(abs_audio).exists():
            skipped += 1
            continue

        lyrics = load_lyrics(lyrics_path)
        if not lyrics:
            skipped += 1
            continue

        entry = {"audio": abs_audio, "text": lyrics}

        if i in val_indices:
            val_entries.append(entry)
            val_source_ids.add(sample.get("id", f"sample_{i}"))
        else:
            train_entries.append(entry)

    print("Processing augmented samples...")
    aug_added = 0
    aug_excluded = 0

    for sample in augmented:
        source_id = sample.get("source_id", "")
        audio_path = sample.get("audio_file", "")
        lyrics_path = sample.get("transcript_file", "")

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

        train_entries.append({"audio": audio_path, "text": lyrics})
        aug_added += 1

    print(f"\n  Train: {len(train_entries)} ({len(train_entries) - aug_added} original + {aug_added} augmented)")
    print(f"  Val:   {len(val_entries)} (original only)")
    print(f"  Skipped: {skipped}")

    print("\nBuilding dataset (paths only, no audio loading)...")

    dataset = DatasetDict({
        "train": Dataset.from_dict({
            "audio": [e["audio"] for e in train_entries],
            "text": [e["text"] for e in train_entries],
        }),
        "validation": Dataset.from_dict({
            "audio": [e["audio"] for e in val_entries],
            "text": [e["text"] for e in val_entries],
        }),
    })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(OUTPUT_DIR))

    print(f"\n{'=' * 60}")
    print("DATASET READY")
    print(f"{'=' * 60}")
    print(f"  Training:    {len(train_entries)} samples")
    print(f"  Validation:  {len(val_entries)} samples")
    print(f"  Saved to:    {OUTPUT_DIR}")
    print(f"{'=' * 60}\n")

    return dataset


# =============================================================================
# Segmented Dataset (30s chunks with aligned lyrics)
# =============================================================================

def build_segmented_dataset() -> DatasetDict:
    """Build dataset from 30s segmented audio with aligned lyrics."""
    seg_manifest = DATA_DIR / "segment_manifest.json"

    if not seg_manifest.exists():
        print("ERROR: No segment manifest. Run: python -m training.segment_audio")
        return None

    data = json.loads(seg_manifest.read_text())
    segments = data.get("segments", [])

    if not segments:
        print("ERROR: No segments found")
        return None

    print(f"\nLoading segmented data: {len(segments)} segments")

    # Group by source song to keep all chunks of a song in same split
    np.random.seed(42)
    by_source = {}
    for seg in segments:
        src = seg.get("original_sample_id", "unknown")
        by_source.setdefault(src, []).append(seg)

    source_ids = list(by_source.keys())
    np.random.shuffle(source_ids)
    n_val = max(1, int(len(source_ids) * VAL_SPLIT))
    val_sources = set(source_ids[:n_val])

    train_entries = []
    val_entries = []
    skipped = 0

    for seg in segments:
        audio_path = seg.get("audio_path", "")
        text = seg.get("text", "")

        if not Path(audio_path).exists() or not text or len(text.strip()) < 5:
            skipped += 1
            continue

        entry = {"audio": audio_path, "text": text}

        if seg.get("original_sample_id", "unknown") in val_sources:
            val_entries.append(entry)
        else:
            train_entries.append(entry)

    print(f"  Train: {len(train_entries)}")
    print(f"  Val:   {len(val_entries)}")
    print(f"  Skipped: {skipped}")

    print("\nBuilding dataset (paths only)...")

    dataset = DatasetDict({
        "train": Dataset.from_dict({
            "audio": [e["audio"] for e in train_entries],
            "text": [e["text"] for e in train_entries],
        }),
        "validation": Dataset.from_dict({
            "audio": [e["audio"] for e in val_entries],
            "text": [e["text"] for e in val_entries],
        }),
    })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(OUTPUT_DIR))

    print(f"\n{'=' * 60}")
    print("SEGMENTED DATASET READY")
    print(f"{'=' * 60}")
    print(f"  Training:    {len(train_entries)} segments")
    print(f"  Validation:  {len(val_entries)} segments")
    print(f"  Saved to:    {OUTPUT_DIR}")
    print(f"{'=' * 60}\n")

    return dataset


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare dataset for Whisper fine-tuning")
    parser.add_argument("--no-augmented", action="store_true",
                       help="Exclude augmented samples (original only)")
    parser.add_argument("--segments", action="store_true",
                       help="Use segmented 30s chunks with aligned lyrics")

    args = parser.parse_args()

    if args.segments:
        build_segmented_dataset()
    else:
        build_dataset(include_augmented=not args.no_augmented)
