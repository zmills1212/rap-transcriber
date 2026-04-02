"""
Auto-Collector: Rapid Training Data Scaling
=============================================
Fetches lyrics from Genius + downloads audio from SoundCloud (or uses local file),
then adds the pair to your training data in one command.

Usage:
    # With auto-download from SoundCloud
    python -m training.auto_collect --song "Magnolia" --artist "Playboi Carti" --tags mumble ad_libs

    # With local audio file
    python -m training.auto_collect --song "Plant Life" --artist "6 Dogs" --audio ~/Downloads/song.MP4

    # Batch from a text file
    python -m training.auto_collect --batch songs.txt

    # Dry run
    python -m training.auto_collect --song "Magnolia" --artist "Playboi Carti" --dry-run
"""

import os
import sys
import json
import shutil
import hashlib
import subprocess
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Optional, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.data_collector import (
    TRAINING_DATA_DIR, AUDIO_DIR, TRANSCRIPT_DIR, MANIFEST_FILE,
    TARGET_SAMPLE_RATE, TARGET_CHANNELS,
)


# =============================================================================
# Audio Download (SoundCloud via yt-dlp)
# =============================================================================

def search_and_download(
    song: str,
    artist: str,
    output_path: str,
    max_duration: int = 300,
) -> Optional[str]:
    """
    Search SoundCloud for a song and download audio.
    Returns path to downloaded WAV, or None if failed.
    """
    query = f"{artist} {song}"
    print(f"  Searching SoundCloud: {query}")

    out_template = output_path + ".%(ext)s"

    cmd = [
        "yt-dlp",
        f"scsearch1:{query}",
        "--extract-audio",
        "--audio-format", "wav",
        "--no-playlist",
        "-o", out_template,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        # Find the downloaded wav file
        out_dir = Path(output_path).parent
        for candidate in [output_path + ".wav", output_path]:
            if Path(candidate).exists():
                return candidate

        # Search directory for newest wav
        for f in sorted(out_dir.glob("*.wav"), key=lambda x: x.stat().st_mtime, reverse=True):
            return str(f)

        if result.returncode != 0:
            err = result.stderr[:300] if result.stderr else "Unknown error"
            print(f"  Download error: {err}")
        return None

    except subprocess.TimeoutExpired:
        print(f"  Download timed out")
        return None
    except FileNotFoundError:
        print(f"  ERROR: yt-dlp not found. Install with: pip install yt-dlp")
        return None


def convert_to_wav(input_path: str, output_path: str) -> bool:
    """Convert any audio file to 16kHz mono WAV using ffmpeg."""
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-ar", str(TARGET_SAMPLE_RATE),
        "-ac", str(TARGET_CHANNELS),
        "-f", "wav",
        output_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return result.returncode == 0 and Path(output_path).exists()
    except Exception:
        return False


def get_duration(audio_path: str) -> Optional[float]:
    """Get audio duration in seconds using ffprobe."""
    try:
        cmd = [
            "ffprobe", "-v", "quiet",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            audio_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            return float(result.stdout.strip())
    except Exception:
        pass
    return None


# =============================================================================
# Auto-Collect Pipeline
# =============================================================================

def auto_collect(
    song: str,
    artist: str,
    audio_path: str = None,
    tags: List[str] = None,
    dry_run: bool = False,
    max_duration: int = 300,
) -> bool:
    """
    Auto-collect a training sample.
    If audio_path provided, uses that file. Otherwise downloads from SoundCloud.
    Returns True if successful.
    """
    tags = tags or ["heavy_slang"]

    print(f"\n{'=' * 60}")
    print(f"  Collecting: {artist} - {song}")
    print(f"  Tags: {', '.join(tags)}")
    print(f"{'=' * 60}")

    # Step 1: Check for duplicates
    if MANIFEST_FILE.exists():
        manifest = json.loads(MANIFEST_FILE.read_text())
        for sample in manifest.get("samples", []):
            meta = sample.get("metadata", {})
            if (meta.get("artist", "").lower() == artist.lower() and
                meta.get("song", "").lower() == song.lower()):
                print(f"  SKIP: Already have {artist} - {song}")
                return False

    # Step 2: Fetch lyrics from Genius
    print(f"\n  Step 1: Fetching lyrics from Genius...")
    try:
        from lyrics_matcher.genius import GeniusClient
        genius = GeniusClient()
        lyrics = genius.fetch_lyrics(song, artist=artist)
    except Exception as e:
        print(f"  ERROR fetching lyrics: {e}")
        return False

    if not lyrics or len(lyrics.strip()) < 20:
        print(f"  SKIP: No lyrics found on Genius")
        return False

    word_count = len(lyrics.split())
    print(f"  Found {word_count} words of lyrics")

    if dry_run:
        print(f"\n  [DRY RUN] Would add to training data")
        return True

    # Step 3: Get audio
    if audio_path:
        source_audio = Path(audio_path)
        if not source_audio.exists():
            print(f"  ERROR: Audio file not found: {audio_path}")
            return False
        print(f"\n  Step 2: Using local audio: {source_audio.name}")
        downloaded = str(source_audio)
    else:
        print(f"\n  Step 2: Downloading audio from SoundCloud...")
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_output = os.path.join(tmp_dir, "audio")
            downloaded = search_and_download(song, artist, tmp_output, max_duration)
            if not downloaded:
                print(f"  ERROR: Failed to download audio")
                print(f"  TIP: Provide audio manually with --audio <file>")
                return False
            # Copy out of temp dir before it gets cleaned up
            persistent_path = os.path.join(
                tempfile.gettempdir(),
                f"autocollect_{hashlib.md5(f'{artist}{song}'.encode()).hexdigest()[:8]}.wav"
            )
            shutil.copy2(downloaded, persistent_path)
            downloaded = persistent_path

    print(f"  Audio source: {Path(downloaded).name}")

    # Step 4: Get duration
    duration = get_duration(downloaded)
    if duration:
        print(f"  Duration: {duration:.1f}s")

    # Step 5: Generate sample ID and save files
    sample_id = hashlib.md5(
        f"{artist}-{song}-{datetime.now().isoformat()}".encode()
    ).hexdigest()[:6]
    sample_name = f"train_{sample_id}"

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)

    audio_dest = AUDIO_DIR / f"{sample_name}.wav"
    transcript_dest = TRANSCRIPT_DIR / f"{sample_name}.txt"

    # Convert to 16kHz mono WAV
    print(f"  Converting to 16kHz WAV...")
    if not convert_to_wav(downloaded, str(audio_dest)):
        if downloaded.endswith(".wav"):
            shutil.copy2(downloaded, audio_dest)
        else:
            print(f"  ERROR: Conversion failed")
            return False

    # Save lyrics
    transcript_dest.write_text(lyrics, encoding="utf-8")

    # Step 6: Update manifest
    print(f"\n  Step 3: Updating manifest...")

    if MANIFEST_FILE.exists():
        manifest = json.loads(MANIFEST_FILE.read_text())
    else:
        manifest = {"samples": []}

    sample_entry = {
        "id": sample_name,
        "audio_file": f"audio/{sample_name}.wav",
        "transcript_file": f"transcripts/{sample_name}.txt",
        "metadata": {
            "artist": artist,
            "song": song,
            "duration_seconds": round(duration, 1) if duration else None,
            "word_count": word_count,
            "source": "auto_collect" + ("_local" if audio_path else "_soundcloud"),
            "collected_at": datetime.now().isoformat(),
        },
        "challenge_tags": tags,
    }

    manifest["samples"].append(sample_entry)
    MANIFEST_FILE.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\n  ✓ Added: {sample_name}")
    print(f"    Audio:  {audio_dest}")
    print(f"    Lyrics: {transcript_dest}")
    print(f"    Total samples: {len(manifest['samples'])}")

    return True


# =============================================================================
# Batch Collection
# =============================================================================

def batch_collect(file_path: str, default_tags: List[str] = None, dry_run: bool = False):
    """
    Collect multiple songs from a text file.

    File format (one per line):
        artist - song | tag1 tag2
        # comments ignored

    Example:
        6 Dogs - Plant Life | melodic heavy_slang
        K Suave - Besties | melodic ad_libs
    """
    default_tags = default_tags or ["heavy_slang"]
    lines = Path(file_path).read_text().strip().splitlines()

    songs = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        if "|" in line:
            song_part, tag_part = line.split("|", 1)
            tags = tag_part.strip().split()
        else:
            song_part = line
            tags = default_tags

        if " - " in song_part:
            artist, song = song_part.split(" - ", 1)
            songs.append((artist.strip(), song.strip(), tags))
        else:
            print(f"  SKIP invalid line: {line}")

    print(f"\n{'=' * 60}")
    print(f"  BATCH COLLECTION: {len(songs)} songs")
    print(f"{'=' * 60}")

    success = 0
    failed = 0
    skipped = 0

    for artist, song, tags in songs:
        try:
            result = auto_collect(song=song, artist=artist, tags=tags, dry_run=dry_run)
            if result:
                success += 1
            else:
                skipped += 1
        except Exception as e:
            print(f"  ERROR on {artist} - {song}: {e}")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"  BATCH COMPLETE")
    print(f"  Collected: {success}")
    print(f"  Skipped:   {skipped}")
    print(f"  Failed:    {failed}")
    print(f"{'=' * 60}")


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Auto-collect training data (SoundCloud + Genius)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  %(prog)s --song "Magnolia" --artist "Playboi Carti" --tags mumble ad_libs
  %(prog)s --song "Plant Life" --artist "6 Dogs" --audio ~/Downloads/song.MP4
  %(prog)s --batch songs.txt
  %(prog)s --song "Besties" --artist "K Suave" --dry-run
        """,
    )

    parser.add_argument("--song", "-s", help="Song title")
    parser.add_argument("--artist", "-a", help="Artist name")
    parser.add_argument("--audio", help="Local audio file (skips SoundCloud download)")
    parser.add_argument("--tags", "-t", nargs="+", default=["heavy_slang"],
                       help="Challenge tags (default: heavy_slang)")
    parser.add_argument("--batch", "-b", help="Batch file (one 'artist - song' per line)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be collected")
    parser.add_argument("--max-duration", type=int, default=300,
                       help="Max audio duration in seconds (default: 300)")

    args = parser.parse_args()

    if args.batch:
        batch_collect(args.batch, default_tags=args.tags, dry_run=args.dry_run)
    elif args.song and args.artist:
        auto_collect(
            song=args.song,
            artist=args.artist,
            audio_path=args.audio,
            tags=args.tags,
            dry_run=args.dry_run,
            max_duration=args.max_duration,
        )
    else:
        parser.print_help()
        print("\nProvide --song + --artist, or --batch <file>")


if __name__ == "__main__":
    main()
