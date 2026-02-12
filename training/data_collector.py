"""
Phase B1: Training Data Collection for Whisper Fine-Tuning

This module helps you collect and organize audio + lyrics pairs for fine-tuning.

Target: 50-100 clips (30-60 seconds each) = ~1-2 hours of rap audio with exact transcriptions.

Directory Structure:
    training_data/
    ├── manifest.json          # Index of all training samples
    ├── audio/                  # Audio clips (wav/mp3)
    │   ├── train_001.wav
    │   ├── train_002.wav
    │   └── ...
    └── transcripts/           # Exact lyrics for each clip
        ├── train_001.txt
        ├── train_002.txt
        └── ...

Usage:
    python -m training.data_collector add --audio clip.mp3 --lyrics lyrics.txt --artist "Artist" --song "Song"
    python -m training.data_collector status
    python -m training.data_collector export --format huggingface
"""

import json
import shutil
import hashlib
import subprocess
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional, List
import argparse


# =============================================================================
# Configuration
# =============================================================================

TRAINING_DATA_DIR = Path("training_data")
AUDIO_DIR = TRAINING_DATA_DIR / "audio"
TRANSCRIPT_DIR = TRAINING_DATA_DIR / "transcripts"
MANIFEST_FILE = TRAINING_DATA_DIR / "manifest.json"

TARGET_SAMPLE_RATE = 16000  # Whisper expects 16kHz
TARGET_CHANNELS = 1         # Mono


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class TrainingSample:
    """A single training sample."""
    id: str
    audio_file: str
    transcript_file: str
    artist: str
    song: str
    duration_seconds: float
    word_count: int
    challenge_tags: List[str]
    added_at: str
    
    def to_dict(self):
        return asdict(self)


@dataclass 
class TrainingManifest:
    """Collection of all training samples."""
    version: str = "1.0"
    created_at: str = ""
    samples: List[dict] = None
    
    def __post_init__(self):
        if self.samples is None:
            self.samples = []
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
    
    @property
    def total_duration(self) -> float:
        return sum(s.get("duration_seconds", 0) for s in self.samples)
    
    @property
    def total_words(self) -> int:
        return sum(s.get("word_count", 0) for s in self.samples)
    
    def to_dict(self):
        return {
            "version": self.version,
            "created_at": self.created_at,
            "sample_count": len(self.samples),
            "total_duration_seconds": self.total_duration,
            "total_duration_minutes": round(self.total_duration / 60, 1),
            "total_words": self.total_words,
            "samples": self.samples
        }


# =============================================================================
# Audio Processing
# =============================================================================

def get_audio_duration(audio_path: Path) -> float:
    """Get duration of audio file in seconds."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(audio_path)],
            capture_output=True, text=True, check=True
        )
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError):
        return 0.0


def convert_audio(input_path: Path, output_path: Path) -> bool:
    """Convert audio to Whisper-compatible format (16kHz mono WAV)."""
    try:
        subprocess.run([
            "ffmpeg", "-y", "-i", str(input_path),
            "-ar", str(TARGET_SAMPLE_RATE),
            "-ac", str(TARGET_CHANNELS),
            "-c:a", "pcm_s16le",
            str(output_path)
        ], capture_output=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error converting audio: {e}")
        return False


def extract_audio_from_video(video_path: Path, output_path: Path) -> bool:
    """Extract audio from video file."""
    try:
        subprocess.run([
            "ffmpeg", "-y", "-i", str(video_path),
            "-vn",  # No video
            "-ar", str(TARGET_SAMPLE_RATE),
            "-ac", str(TARGET_CHANNELS),
            "-c:a", "pcm_s16le",
            str(output_path)
        ], capture_output=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error extracting audio: {e}")
        return False


# =============================================================================
# Data Management
# =============================================================================

def init_training_data():
    """Initialize training data directory structure."""
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    
    if not MANIFEST_FILE.exists():
        manifest = TrainingManifest()
        save_manifest(manifest)
        print(f"✓ Initialized training data directory at {TRAINING_DATA_DIR}/")


def load_manifest() -> TrainingManifest:
    """Load training manifest from disk."""
    if not MANIFEST_FILE.exists():
        return TrainingManifest()
    
    data = json.loads(MANIFEST_FILE.read_text())
    return TrainingManifest(
        version=data.get("version", "1.0"),
        created_at=data.get("created_at", ""),
        samples=data.get("samples", [])
    )


def save_manifest(manifest: TrainingManifest):
    """Save training manifest to disk."""
    MANIFEST_FILE.write_text(json.dumps(manifest.to_dict(), indent=2))


def generate_sample_id(artist: str, song: str) -> str:
    """Generate unique sample ID."""
    base = f"{artist}_{song}".lower()
    base = "".join(c if c.isalnum() else "_" for c in base)
    hash_suffix = hashlib.md5(f"{base}_{datetime.now().isoformat()}".encode()).hexdigest()[:6]
    return f"train_{hash_suffix}"


def add_training_sample(
    audio_path: Path,
    lyrics_path: Path,
    artist: str,
    song: str,
    challenge_tags: List[str] = None
) -> Optional[TrainingSample]:
    """Add a new training sample."""
    init_training_data()
    
    # Generate ID
    sample_id = generate_sample_id(artist, song)
    
    # Process audio
    audio_ext = audio_path.suffix.lower()
    output_audio = AUDIO_DIR / f"{sample_id}.wav"
    
    if audio_ext in [".mp4", ".mov", ".avi", ".mkv"]:
        print(f"  Extracting audio from video...")
        if not extract_audio_from_video(audio_path, output_audio):
            return None
    elif audio_ext != ".wav" or True:  # Always convert to ensure format
        print(f"  Converting audio to 16kHz mono WAV...")
        if not convert_audio(audio_path, output_audio):
            return None
    else:
        shutil.copy(audio_path, output_audio)
    
    # Copy lyrics
    output_lyrics = TRANSCRIPT_DIR / f"{sample_id}.txt"
    lyrics_text = lyrics_path.read_text(encoding="utf-8").strip()
    output_lyrics.write_text(lyrics_text, encoding="utf-8")
    
    # Get metadata
    duration = get_audio_duration(output_audio)
    word_count = len(lyrics_text.split())
    
    # Create sample
    sample = TrainingSample(
        id=sample_id,
        audio_file=f"audio/{sample_id}.wav",
        transcript_file=f"transcripts/{sample_id}.txt",
        artist=artist,
        song=song,
        duration_seconds=round(duration, 2),
        word_count=word_count,
        challenge_tags=challenge_tags or [],
        added_at=datetime.now().isoformat()
    )
    
    # Update manifest
    manifest = load_manifest()
    manifest.samples.append(sample.to_dict())
    save_manifest(manifest)
    
    return sample


def get_status() -> dict:
    """Get current status of training data collection."""
    manifest = load_manifest()
    
    # Count by challenge tag
    tag_counts = {}
    artist_counts = {}
    
    for sample in manifest.samples:
        for tag in sample.get("challenge_tags", []):
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
        artist = sample.get("artist", "Unknown")
        artist_counts[artist] = artist_counts.get(artist, 0) + 1
    
    return {
        "total_samples": len(manifest.samples),
        "total_duration_minutes": round(manifest.total_duration / 60, 1),
        "total_words": manifest.total_words,
        "target_samples": 50,
        "target_duration_minutes": 60,
        "progress_pct": round(len(manifest.samples) / 50 * 100, 1),
        "by_artist": artist_counts,
        "by_challenge": tag_counts,
    }


def export_for_training(output_dir: Path, format: str = "huggingface"):
    """Export data in format ready for fine-tuning."""
    manifest = load_manifest()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if format == "huggingface":
        # Create HuggingFace datasets format
        data = []
        for sample in manifest.samples:
            audio_path = TRAINING_DATA_DIR / sample["audio_file"]
            transcript_path = TRAINING_DATA_DIR / sample["transcript_file"]
            
            if audio_path.exists() and transcript_path.exists():
                data.append({
                    "audio": str(audio_path.absolute()),
                    "text": transcript_path.read_text(encoding="utf-8").strip(),
                    "artist": sample["artist"],
                    "song": sample["song"],
                })
        
        output_file = output_dir / "train.json"
        output_file.write_text(json.dumps(data, indent=2))
        print(f"✓ Exported {len(data)} samples to {output_file}")
        
    elif format == "mlx":
        # Create MLX Whisper format (JSONL)
        output_file = output_dir / "train.jsonl"
        with open(output_file, "w") as f:
            for sample in manifest.samples:
                audio_path = TRAINING_DATA_DIR / sample["audio_file"]
                transcript_path = TRAINING_DATA_DIR / sample["transcript_file"]
                
                if audio_path.exists() and transcript_path.exists():
                    entry = {
                        "audio": str(audio_path.absolute()),
                        "text": transcript_path.read_text(encoding="utf-8").strip(),
                    }
                    f.write(json.dumps(entry) + "\n")
        
        print(f"✓ Exported to {output_file}")


# =============================================================================
# CLI
# =============================================================================

def print_status():
    """Print formatted status."""
    status = get_status()
    
    print("\n" + "=" * 60)
    print("TRAINING DATA COLLECTION STATUS")
    print("=" * 60)
    
    # Progress bar
    progress = min(status["progress_pct"], 100)
    bar_width = 40
    filled = int(bar_width * progress / 100)
    bar = "█" * filled + "░" * (bar_width - filled)
    print(f"\nProgress: [{bar}] {progress:.0f}%")
    
    print(f"\nSamples:  {status['total_samples']} / {status['target_samples']}")
    print(f"Duration: {status['total_duration_minutes']} / {status['target_duration_minutes']} minutes")
    print(f"Words:    {status['total_words']}")
    
    if status["by_artist"]:
        print("\nBy Artist:")
        for artist, count in sorted(status["by_artist"].items(), key=lambda x: -x[1]):
            print(f"  {artist}: {count}")
    
    if status["by_challenge"]:
        print("\nBy Challenge:")
        for tag, count in sorted(status["by_challenge"].items(), key=lambda x: -x[1]):
            print(f"  {tag}: {count}")
    
    print("\n" + "=" * 60)
    
    if status["total_samples"] < 10:
        print("\n💡 Tip: Aim for diverse artists and styles!")
        print("   Try to include: fast flow, melodic, mumble, regional dialects")
    elif status["total_samples"] < 50:
        print(f"\n💡 {50 - status['total_samples']} more samples to reach minimum for fine-tuning")
    else:
        print("\n✓ Ready for fine-tuning! Run: python -m training.fine_tune")


def main():
    parser = argparse.ArgumentParser(description="Manage training data for Whisper fine-tuning")
    subparsers = parser.add_subparsers(dest="command", help="Command")
    
    # init command
    init_parser = subparsers.add_parser("init", help="Initialize training data directory")
    
    # add command
    add_parser = subparsers.add_parser("add", help="Add a training sample")
    add_parser.add_argument("--audio", "-a", required=True, help="Path to audio file")
    add_parser.add_argument("--lyrics", "-l", required=True, help="Path to lyrics file")
    add_parser.add_argument("--artist", required=True, help="Artist name")
    add_parser.add_argument("--song", required=True, help="Song name")
    add_parser.add_argument("--tags", nargs="*", default=[], 
                          help="Challenge tags (fast_flow, mumble, heavy_slang, etc.)")
    
    # status command
    status_parser = subparsers.add_parser("status", help="Show collection status")
    
    # export command
    export_parser = subparsers.add_parser("export", help="Export for training")
    export_parser.add_argument("--output", "-o", default="training_export", help="Output directory")
    export_parser.add_argument("--format", "-f", choices=["huggingface", "mlx"], default="mlx",
                             help="Export format")
    
    args = parser.parse_args()
    
    if args.command == "init":
        init_training_data()
        print("✓ Training data directory initialized")
        
    elif args.command == "add":
        audio_path = Path(args.audio)
        lyrics_path = Path(args.lyrics)
        
        if not audio_path.exists():
            print(f"Error: Audio file not found: {audio_path}")
            return
        if not lyrics_path.exists():
            print(f"Error: Lyrics file not found: {lyrics_path}")
            return
        
        print(f"\nAdding training sample...")
        print(f"  Audio: {audio_path}")
        print(f"  Artist: {args.artist}")
        print(f"  Song: {args.song}")
        
        sample = add_training_sample(
            audio_path=audio_path,
            lyrics_path=lyrics_path,
            artist=args.artist,
            song=args.song,
            challenge_tags=args.tags
        )
        
        if sample:
            print(f"\n✓ Added sample: {sample.id}")
            print(f"  Duration: {sample.duration_seconds}s")
            print(f"  Words: {sample.word_count}")
            print_status()
        else:
            print("\n✗ Failed to add sample")
            
    elif args.command == "status":
        print_status()
        
    elif args.command == "export":
        export_for_training(Path(args.output), args.format)
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
