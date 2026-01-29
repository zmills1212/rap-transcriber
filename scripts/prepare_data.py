"""
Data preparation script.
Creates train/val/test splits and prepares manifests for training.

Usage:
    python scripts/prepare_data.py --audio-dir data/raw --output-dir data/processed
"""
import sys
sys.path.insert(0, '.')

import argparse
from pathlib import Path
import json

from src.data.manifest import DataManifest, ManifestEntry, create_manifest_from_folder
from src.data.audio_validator import AudioValidator
from src.data.text_normalizer import TextNormalizer, LyricsPreprocessor


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Prepare data for training')
    
    parser.add_argument('--audio-dir', type=str, default='data/raw',
                        help='Directory containing audio files')
    parser.add_argument('--text-dir', type=str, default=None,
                        help='Directory containing text files (defaults to audio-dir)')
    parser.add_argument('--output-dir', type=str, default='data/processed',
                        help='Output directory for manifests')
    
    # Split ratios
    parser.add_argument('--train-ratio', type=float, default=0.8,
                        help='Training set ratio')
    parser.add_argument('--val-ratio', type=float, default=0.1,
                        help='Validation set ratio')
    parser.add_argument('--test-ratio', type=float, default=0.1,
                        help='Test set ratio')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    # Validation options
    parser.add_argument('--min-duration', type=float, default=1.0,
                        help='Minimum audio duration in seconds')
    parser.add_argument('--max-duration', type=float, default=30.0,
                        help='Maximum audio duration in seconds')
    parser.add_argument('--skip-validation', action='store_true',
                        help='Skip audio validation')
    
    # Text options
    parser.add_argument('--normalize-text', action='store_true', default=True,
                        help='Normalize text')
    
    return parser.parse_args()


def validate_audio_files(manifest: DataManifest, args) -> DataManifest:
    """Validate audio files and filter invalid ones."""
    print("\nValidating audio files...")
    
    validator = AudioValidator(
        min_duration=args.min_duration,
        max_duration=args.max_duration
    )
    
    valid_entries = []
    invalid_count = 0
    
    for i, entry in enumerate(manifest.entries):
        if (i + 1) % 50 == 0:
            print(f"  Validated {i + 1}/{len(manifest)} files...")
        
        result = validator.validate(entry.audio_path)
        
        if result.is_valid:
            # Update duration from validation
            entry.duration = result.duration
            valid_entries.append(entry)
        else:
            invalid_count += 1
            if invalid_count <= 5:  # Show first 5 invalid files
                print(f"  ⚠️  Invalid: {entry.audio_path}")
                print(f"      Issues: {result.issues}")
    
    print(f"\n  Valid: {len(valid_entries)}/{len(manifest)}")
    print(f"  Invalid: {invalid_count}")
    
    return DataManifest(valid_entries)


def normalize_texts(manifest: DataManifest, args) -> DataManifest:
    """Normalize text in manifest entries."""
    print("\nNormalizing texts...")
    
    preprocessor = LyricsPreprocessor()
    
    for entry in manifest.entries:
        entry.text = preprocessor.preprocess(entry.text)
    
    print(f"  Normalized {len(manifest)} entries")
    
    return manifest


def create_splits(manifest: DataManifest, args) -> tuple:
    """Create train/val/test splits."""
    print("\nCreating train/val/test splits...")
    
    train, val, test = manifest.split(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )
    
    print(f"  Train: {len(train)} entries ({args.train_ratio:.0%})")
    print(f"  Val:   {len(val)} entries ({args.val_ratio:.0%})")
    print(f"  Test:  {len(test)} entries ({args.test_ratio:.0%})")
    
    return train, val, test


def save_manifests(train, val, test, output_dir: Path):
    """Save manifest files."""
    print("\nSaving manifests...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train.save(output_dir / 'train_manifest.json')
    val.save(output_dir / 'val_manifest.json')
    test.save(output_dir / 'test_manifest.json')
    
    # Save combined stats
    stats = {
        'train': train.get_stats(),
        'val': val.get_stats(),
        'test': test.get_stats()
    }
    
    with open(output_dir / 'data_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"  Saved to {output_dir}/")


def print_summary(train, val, test):
    """Print data summary."""
    print("\n" + "=" * 50)
    print("DATA PREPARATION SUMMARY")
    print("=" * 50)
    
    total_entries = len(train) + len(val) + len(test)
    total_hours = train.total_duration_hours + val.total_duration_hours + test.total_duration_hours
    
    print(f"\nTotal entries: {total_entries}")
    print(f"Total duration: {total_hours:.2f} hours")
    
    print("\nSplit breakdown:")
    print(f"  Train: {len(train):5d} entries, {train.total_duration_hours:.2f} hours")
    print(f"  Val:   {len(val):5d} entries, {val.total_duration_hours:.2f} hours")
    print(f"  Test:  {len(test):5d} entries, {test.total_duration_hours:.2f} hours")
    
    # Sample entries
    if train.entries:
        print("\nSample training entry:")
        sample = train.entries[0]
        print(f"  Audio: {sample.audio_path}")
        print(f"  Text:  {sample.text[:50]}..." if len(sample.text) > 50 else f"  Text:  {sample.text}")
        print(f"  Duration: {sample.duration:.2f}s")
    
    print("\n" + "=" * 50)


def main():
    """Main data preparation function."""
    args = parse_args()
    
    audio_dir = Path(args.audio_dir)
    text_dir = Path(args.text_dir) if args.text_dir else audio_dir
    output_dir = Path(args.output_dir)
    
    print("=" * 50)
    print("DATA PREPARATION")
    print("=" * 50)
    print(f"\nAudio directory: {audio_dir}")
    print(f"Text directory: {text_dir}")
    print(f"Output directory: {output_dir}")
    
    # Check if audio directory exists
    if not audio_dir.exists():
        print(f"\n❌ Audio directory not found: {audio_dir}")
        print("\nTo use this script, you need:")
        print("  1. Audio files in data/raw/")
        print("  2. Matching text files (same name, .txt extension)")
        print("\nExample structure:")
        print("  data/raw/song1.mp3")
        print("  data/raw/song1.txt")
        print("\nFor now, creating a sample manifest with test data...")
        
        # Create sample manifest for testing
        create_sample_manifest(output_dir)
        return
    
    # Create manifest from folder
    print("\nScanning for audio files...")
    manifest = create_manifest_from_folder(
        audio_folder=audio_dir,
        text_folder=text_dir
    )
    
    if len(manifest) == 0:
        print("\n❌ No valid audio/text pairs found")
        print("\nMake sure you have:")
        print("  - Audio files (.mp3, .wav, .flac)")
        print("  - Matching text files (.txt)")
        print("\nCreating sample manifest for testing...")
        create_sample_manifest(output_dir)
        return
    
    print(f"  Found {len(manifest)} audio/text pairs")
    
    # Validate audio files
    if not args.skip_validation:
        manifest = validate_audio_files(manifest, args)
    
    # Normalize texts
    if args.normalize_text:
        manifest = normalize_texts(manifest, args)
    
    # Create splits
    train, val, test = create_splits(manifest, args)
    
    # Save manifests
    save_manifests(train, val, test, output_dir)
    
    # Print summary
    print_summary(train, val, test)
    
    print("\n✅ Data preparation complete!")


def create_sample_manifest(output_dir: Path):
    """Create a sample manifest for testing when no real data exists."""
    print("\nCreating sample manifest for testing...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create sample entries
    sample_entries = [
        ManifestEntry(
            audio_path="data/raw/sample1.mp3",
            text="i'm finna get this bread yeah every day i'm on my grind",
            duration=15.0,
            artist="Sample Artist",
            song_title="Sample Song 1"
        ),
        ManifestEntry(
            audio_path="data/raw/sample2.mp3",
            text="she bussin it down yeah skrt skrt in the coupe",
            duration=20.0,
            artist="Sample Artist",
            song_title="Sample Song 2"
        ),
        ManifestEntry(
            audio_path="data/raw/sample3.mp3",
            text="no cap this fire got my homies with me",
            duration=18.0,
            artist="Sample Artist 2",
            song_title="Sample Song 3"
        ),
    ]
    
    # Create manifests
    train_manifest = DataManifest([sample_entries[0], sample_entries[1]])
    val_manifest = DataManifest([sample_entries[2]])
    test_manifest = DataManifest([sample_entries[2]])
    
    train_manifest.save(output_dir / 'train_manifest.json')
    val_manifest.save(output_dir / 'val_manifest.json')
    test_manifest.save(output_dir / 'test_manifest.json')
    
    print(f"  ✅ Sample manifests created in {output_dir}/")
    print("\n  Note: These are placeholder manifests.")
    print("  Replace with real data for actual training.")


if __name__ == "__main__":
    main()
