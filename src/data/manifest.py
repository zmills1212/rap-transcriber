"""
Data manifest management for rap transcription.
Handles creation, validation, and manipulation of training data manifests.
"""
import json
from pathlib import Path
from typing import List, Dict, Optional, Union
from dataclasses import dataclass, asdict
import random


@dataclass
class ManifestEntry:
    """A single entry in the data manifest."""
    audio_path: str
    text: str
    duration: float
    phonemes: Optional[str] = None
    artist: Optional[str] = None
    song_title: Optional[str] = None
    genre: Optional[str] = None
    has_adlibs: bool = False
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {k: v for k, v in asdict(self).items() if v is not None}
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ManifestEntry':
        """Create from dictionary."""
        return cls(
            audio_path=data['audio_path'],
            text=data['text'],
            duration=data['duration'],
            phonemes=data.get('phonemes'),
            artist=data.get('artist'),
            song_title=data.get('song_title'),
            genre=data.get('genre'),
            has_adlibs=data.get('has_adlibs', False)
        )


class DataManifest:
    """
    Manages a collection of training data entries.
    
    Manifest format (JSON):
    {
        "version": "1.0",
        "description": "Rap transcription training data",
        "total_duration_hours": 10.5,
        "num_entries": 1000,
        "entries": [
            {
                "audio_path": "data/raw/song1.mp3",
                "text": "lyrics here",
                "duration": 30.5,
                "artist": "Artist Name",
                ...
            },
            ...
        ]
    }
    """
    
    def __init__(self, entries: List[ManifestEntry] = None):
        self.entries = entries or []
        self.version = "1.0"
        self.description = "Rap transcription training data"
    
    def add_entry(self, entry: ManifestEntry):
        """Add a single entry."""
        self.entries.append(entry)
    
    def add_entries(self, entries: List[ManifestEntry]):
        """Add multiple entries."""
        self.entries.extend(entries)
    
    def remove_entry(self, index: int):
        """Remove entry by index."""
        if 0 <= index < len(self.entries):
            del self.entries[index]
    
    def filter_by_duration(
        self,
        min_duration: float = 0,
        max_duration: float = float('inf')
    ) -> 'DataManifest':
        """Filter entries by duration."""
        filtered = [
            e for e in self.entries
            if min_duration <= e.duration <= max_duration
        ]
        return DataManifest(filtered)
    
    def filter_by_artist(self, artist: str) -> 'DataManifest':
        """Filter entries by artist."""
        filtered = [
            e for e in self.entries
            if e.artist and artist.lower() in e.artist.lower()
        ]
        return DataManifest(filtered)
    
    def shuffle(self, seed: Optional[int] = None):
        """Shuffle entries in place."""
        if seed is not None:
            random.seed(seed)
        random.shuffle(self.entries)
    
    def split(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42
    ) -> tuple:
        """
        Split manifest into train/val/test sets.
        
        Args:
            train_ratio: Fraction for training
            val_ratio: Fraction for validation
            test_ratio: Fraction for testing
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (train_manifest, val_manifest, test_manifest)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001
        
        # Shuffle with seed
        entries = self.entries.copy()
        random.seed(seed)
        random.shuffle(entries)
        
        n = len(entries)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_entries = entries[:train_end]
        val_entries = entries[train_end:val_end]
        test_entries = entries[val_end:]
        
        return (
            DataManifest(train_entries),
            DataManifest(val_entries),
            DataManifest(test_entries)
        )
    
    @property
    def total_duration_hours(self) -> float:
        """Get total duration in hours."""
        total_seconds = sum(e.duration for e in self.entries)
        return total_seconds / 3600
    
    @property
    def num_entries(self) -> int:
        """Get number of entries."""
        return len(self.entries)
    
    def get_stats(self) -> Dict:
        """Get manifest statistics."""
        if not self.entries:
            return {
                'num_entries': 0,
                'total_duration_hours': 0,
                'avg_duration_seconds': 0,
                'min_duration_seconds': 0,
                'max_duration_seconds': 0,
                'unique_artists': 0
            }
        
        durations = [e.duration for e in self.entries]
        artists = set(e.artist for e in self.entries if e.artist)
        
        return {
            'num_entries': len(self.entries),
            'total_duration_hours': self.total_duration_hours,
            'avg_duration_seconds': sum(durations) / len(durations),
            'min_duration_seconds': min(durations),
            'max_duration_seconds': max(durations),
            'unique_artists': len(artists)
        }
    
    def save(self, path: Union[str, Path]):
        """Save manifest to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'version': self.version,
            'description': self.description,
            'total_duration_hours': self.total_duration_hours,
            'num_entries': self.num_entries,
            'entries': [e.to_dict() for e in self.entries]
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved manifest: {path} ({self.num_entries} entries)")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'DataManifest':
        """Load manifest from JSON file."""
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Manifest not found: {path}")
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        entries = [ManifestEntry.from_dict(e) for e in data['entries']]
        
        manifest = cls(entries)
        manifest.version = data.get('version', '1.0')
        manifest.description = data.get('description', '')
        
        return manifest
    
    def __len__(self) -> int:
        return len(self.entries)
    
    def __getitem__(self, index: int) -> ManifestEntry:
        return self.entries[index]
    
    def __iter__(self):
        return iter(self.entries)


def create_manifest_from_folder(
    audio_folder: Union[str, Path],
    text_folder: Optional[Union[str, Path]] = None,
    audio_extensions: List[str] = ['.mp3', '.wav', '.flac'],
    text_extension: str = '.txt'
) -> DataManifest:
    """
    Create manifest from folder of audio files.
    
    Expects matching text files with same name:
        audio_folder/song1.mp3 -> text_folder/song1.txt
    
    Args:
        audio_folder: Folder containing audio files
        text_folder: Folder containing text files (defaults to audio_folder)
        audio_extensions: Valid audio extensions
        text_extension: Extension for text files
        
    Returns:
        DataManifest with entries
    """
    import torchaudio
    
    audio_folder = Path(audio_folder)
    text_folder = Path(text_folder) if text_folder else audio_folder
    
    entries = []
    
    # Find all audio files
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(audio_folder.glob(f'*{ext}'))
        audio_files.extend(audio_folder.glob(f'*{ext.upper()}'))
    
    for audio_path in sorted(audio_files):
        # Find matching text file
        text_path = text_folder / f"{audio_path.stem}{text_extension}"
        
        if not text_path.exists():
            print(f"  Warning: No text file for {audio_path.name}, skipping")
            continue
        
        # Get audio duration
        try:
            info = torchaudio.info(str(audio_path))
            duration = info.num_frames / info.sample_rate
        except Exception as e:
            print(f"  Warning: Could not read {audio_path.name}: {e}")
            continue
        
        # Read text
        with open(text_path, 'r') as f:
            text = f.read().strip()
        
        # Create entry
        entry = ManifestEntry(
            audio_path=str(audio_path),
            text=text,
            duration=duration
        )
        entries.append(entry)
    
    return DataManifest(entries)


if __name__ == "__main__":
    print("✅ Manifest module loaded successfully!")
    print("")
    
    # Test creating entries
    print("Testing ManifestEntry...")
    entry = ManifestEntry(
        audio_path="data/raw/test.mp3",
        text="I'm finna get this bread",
        duration=30.5,
        artist="Test Artist"
    )
    print(f"   ✅ Entry created: {entry.audio_path}")
    print(f"   ✅ Duration: {entry.duration}s")
    print("")
    
    # Test manifest
    print("Testing DataManifest...")
    manifest = DataManifest()
    
    # Add sample entries
    for i in range(10):
        manifest.add_entry(ManifestEntry(
            audio_path=f"data/raw/song{i}.mp3",
            text=f"Lyrics for song {i}",
            duration=20.0 + i * 5,
            artist=f"Artist {i % 3}"
        ))
    
    stats = manifest.get_stats()
    print(f"   ✅ Entries: {stats['num_entries']}")
    print(f"   ✅ Total duration: {stats['total_duration_hours']:.2f} hours")
    print(f"   ✅ Avg duration: {stats['avg_duration_seconds']:.1f}s")
    print(f"   ✅ Unique artists: {stats['unique_artists']}")
    print("")
    
    # Test split
    print("Testing train/val/test split...")
    train, val, test = manifest.split(0.8, 0.1, 0.1)
    print(f"   ✅ Train: {len(train)} entries")
    print(f"   ✅ Val:   {len(val)} entries")
    print(f"   ✅ Test:  {len(test)} entries")
    print("")
    
    # Test save/load
    print("Testing save/load...")
    manifest.save("data/processed/test_manifest.json")
    loaded = DataManifest.load("data/processed/test_manifest.json")
    print(f"   ✅ Saved and loaded {len(loaded)} entries")
    
    # Cleanup
    Path("data/processed/test_manifest.json").unlink()
