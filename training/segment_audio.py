"""
Audio Segmentation Pipeline (v2 — sequential greedy alignment)
==============================================================
Splits full-length songs into ~30-second chunks with aligned lyrics
for each chunk. This fixes the training mismatch where Whisper processes
30s of audio but labels contain full-song lyrics.

Approach (v2):
  1. Split audio into 30s chunks (with 5s overlap for continuity)
  2. Transcribe each chunk with baseline Whisper
  3. Sequentially align each chunk's transcription against the full lyrics,
     starting the search where the previous chunk left off (a forward-only
     cursor). This enforces monotonic ordering and is the key fix over v1's
     proportional positioning, which assigned wrong lyrics to wrong chunks.
  4. Score each alignment via combined word-overlap + bigram-overlap and
     store `alignment_score` per segment for downstream filtering.
  5. Save (chunk_audio, aligned_lyrics, alignment_score) tuples.

Usage:
    python -m training.segment_audio
    python -m training.segment_audio --chunk-duration 30 --overlap 5
"""

import os
import re
import sys
import json
import subprocess
import hashlib
import shutil
from pathlib import Path
from typing import List, Tuple, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# =============================================================================
# Config
# =============================================================================

DATA_DIR = PROJECT_ROOT / "training_data"
MANIFEST_PATH = DATA_DIR / "manifest.json"
SEGMENTS_DIR = DATA_DIR / "segments"
SEGMENT_MANIFEST = DATA_DIR / "segment_manifest.json"

TARGET_SR = 16000
CHUNK_DURATION = 30  # seconds
OVERLAP = 5  # seconds overlap between chunks

# Alignment search bounds (in words), relative to the running cursor.
# Backward slack is small to keep ordering ~monotonic; forward slack lets the
# search find the chunk when word density varies between sections.
BACK_SLACK = 15
MIN_WINDOW = 5


# =============================================================================
# Audio Splitting
# =============================================================================

def get_duration(audio_path: str) -> float:
    """Get audio duration in seconds."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "csv=p=0",
        audio_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    return float(result.stdout.strip())


def split_audio(
    audio_path: str,
    output_dir: str,
    chunk_duration: int = 30,
    overlap: int = 5,
) -> List[dict]:
    """
    Split audio into overlapping chunks.
    Returns list of {path, start, end, duration} dicts.
    """
    total_duration = get_duration(audio_path)
    stem = Path(audio_path).stem
    chunks = []

    start = 0.0
    chunk_idx = 0
    step = chunk_duration - overlap

    while start < total_duration:
        end = min(start + chunk_duration, total_duration)
        actual_duration = end - start

        # Skip very short final chunks (< 10 seconds)
        if actual_duration < 10 and chunk_idx > 0:
            break

        out_path = os.path.join(output_dir, f"{stem}_chunk{chunk_idx:03d}.wav")

        cmd = [
            "ffmpeg", "-y",
            "-i", audio_path,
            "-ss", str(start),
            "-t", str(actual_duration),
            "-ar", str(TARGET_SR),
            "-ac", "1",
            "-f", "wav",
            out_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        if result.returncode == 0 and Path(out_path).exists():
            chunks.append({
                "path": out_path,
                "start": round(start, 1),
                "end": round(end, 1),
                "duration": round(actual_duration, 1),
                "index": chunk_idx,
            })

        start += step
        chunk_idx += 1

    return chunks


# =============================================================================
# Lyrics Alignment (v2 — sequential greedy)
# =============================================================================

# Common words ignored when measuring content-word overlap (mirrors the set
# used by model_filter so the segmenter and the filter agree on what counts).
_STOPWORDS = {
    'i', 'im', 'a', 'the', 'and', 'in', 'on', 'my', 'me', 'to',
    'it', 'is', 'that', 'you', 'we', 'he', 'she', 'they', 'this',
    'got', 'get', 'like', 'just', 'up', 'no', 'so', 'but', 'all',
    'yeah', 'yuh', 'uh', 'oh', 'ay', 'ayy', 'ooh', 'what',
}


def normalize_for_comparison(text: str) -> str:
    """Lowercase and strip punctuation for fuzzy comparison."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _content_words(text: str) -> set:
    return set(normalize_for_comparison(text).split()) - _STOPWORDS


def _bigrams(text: str) -> set:
    words = normalize_for_comparison(text).split()
    return set(zip(words, words[1:])) if len(words) > 1 else set()


def alignment_score(prediction: str, reference: str) -> float:
    """
    Combined word-overlap (0.4) + bigram-overlap (0.6) Jaccard similarity.
    Mirrors model_filter's combined_score weighting so the segmenter and the
    downstream filter speak the same scale.
    """
    pa, pb = _content_words(prediction), _content_words(reference)
    word = len(pa & pb) / len(pa | pb) if (pa and pb) else 0.0

    ba, bb = _bigrams(prediction), _bigrams(reference)
    seq = len(ba & bb) / len(ba | bb) if (ba and bb) else 0.0

    return 0.4 * word + 0.6 * seq


def transcribe_chunk(audio_path: str) -> str:
    """Transcribe a single chunk with Whisper."""
    import whisper
    global _whisper_model
    try:
        _whisper_model
    except NameError:
        _whisper_model = None

    if _whisper_model is None:
        _whisper_model = whisper.load_model("small")

    result = _whisper_model.transcribe(audio_path)
    return result["text"].strip()


def align_chunk_sequential(
    trans_words: List[str],
    lyric_words: List[str],
    cursor: int,
    window: int,
    back_slack: int = BACK_SLACK,
) -> Tuple[str, float, int]:
    """
    Greedily find the lyrics window that best matches a chunk transcription,
    searching forward from the running `cursor`.

    Candidate window starts are scanned over
        [cursor - back_slack, cursor + forward_slack]
    (clamped to the lyrics), and each window is scored against the chunk
    transcription. The forward-bounded search plus a small backward slack keep
    the alignment monotonic, so a chorus repeated elsewhere in the song can't
    be matched out of order (the failure mode of v1's proportional search).

    Returns (aligned_text, score, best_start).
    """
    n = len(lyric_words)
    if n == 0 or not trans_words:
        return "", 0.0, cursor

    W = max(MIN_WINDOW, min(window, n))
    forward_slack = max(W, 25)

    search_start = max(0, cursor - back_slack)
    search_end = min(n - 1, cursor + forward_slack)

    trans_text = " ".join(trans_words)
    best_score = -1.0
    best_start = min(cursor, max(0, n - W))

    for i in range(search_start, search_end + 1):
        candidate = " ".join(lyric_words[i:i + W])
        s = alignment_score(trans_text, candidate)
        if s > best_score:
            best_score = s
            best_start = i

    aligned = " ".join(lyric_words[best_start:best_start + W])
    return aligned, round(best_score, 4), best_start


# =============================================================================
# Main Pipeline
# =============================================================================

def segment_all(
    chunk_duration: int = CHUNK_DURATION,
    overlap: int = OVERLAP,
):
    """Segment all training samples into aligned chunks (v2 sequential)."""

    if not MANIFEST_PATH.exists():
        print("ERROR: No manifest found")
        return

    manifest = json.loads(MANIFEST_PATH.read_text())
    samples = manifest.get("samples", [])

    print(f"\n{'=' * 60}")
    print(f"  AUDIO SEGMENTATION PIPELINE (v2 sequential greedy)")
    print(f"  Chunk duration: {chunk_duration}s (overlap: {overlap}s)")
    print(f"  Samples to process: {len(samples)}")
    print(f"{'=' * 60}\n")

    # Create output directory
    SEGMENTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load Whisper once
    print("  Loading Whisper for chunk transcription...")
    import whisper
    global _whisper_model
    _whisper_model = whisper.load_model("small")
    print("  done\n")

    step_ratio = (chunk_duration - overlap) / chunk_duration

    all_segments = []
    total_chunks = 0
    skipped = 0
    score_sum = 0.0

    for idx, sample in enumerate(samples):
        audio_file = sample.get("audio_file", "")
        transcript_file = sample.get("transcript_file", "")
        artist = sample.get("metadata", {}).get("artist", "?")
        song = sample.get("metadata", {}).get("song", "?")

        # Resolve paths
        audio_path = str(DATA_DIR / audio_file) if not Path(audio_file).is_absolute() else audio_file
        transcript_path = str(DATA_DIR / transcript_file) if not Path(transcript_file).is_absolute() else transcript_file

        if not Path(audio_path).exists() or not Path(transcript_path).exists():
            skipped += 1
            continue

        # Get duration
        try:
            duration = get_duration(audio_path)
        except Exception:
            skipped += 1
            continue

        # Load full lyrics
        lyrics = Path(transcript_path).read_text(encoding="utf-8").strip()
        lyrics = " ".join(lyrics.split())
        lyric_words = lyrics.split()

        # Skip samples shorter than one chunk (already fine): keep as-is but
        # still transcribe once so the segment carries a real alignment_score.
        if duration <= chunk_duration + 5:
            try:
                chunk_trans = transcribe_chunk(audio_path)
            except Exception:
                chunk_trans = ""
            score = alignment_score(chunk_trans, lyrics)
            all_segments.append({
                "audio_path": audio_path,
                "text": lyrics,
                "alignment_score": round(score, 4),
                "artist": artist,
                "song": song,
                "chunk_index": 0,
                "start": 0,
                "end": round(duration, 1),
                "original_sample_id": sample.get("id", ""),
            })
            total_chunks += 1
            score_sum += score
            print(f"  [{idx+1}/{len(samples)}] {artist} - {song}: short ({duration:.0f}s), kept as-is (score {score:.2f})")
            continue

        # Split audio into chunks
        sample_seg_dir = SEGMENTS_DIR / sample.get("id", f"sample_{idx}")
        sample_seg_dir.mkdir(parents=True, exist_ok=True)

        chunks = split_audio(audio_path, str(sample_seg_dir), chunk_duration, overlap)

        if not chunks:
            skipped += 1
            continue

        total_words = len(lyric_words)
        words_per_chunk_est = max(1, total_words / max(1, len(chunks)))

        print(f"  [{idx+1}/{len(samples)}] {artist} - {song}: {duration:.0f}s → {len(chunks)} chunks", end="")

        # Sequential greedy alignment: cursor moves forward only, so each
        # chunk's lyrics search begins where the previous chunk landed.
        cursor = 0
        song_score_sum = 0.0
        for chunk in chunks:
            try:
                chunk_trans = transcribe_chunk(chunk["path"])
            except Exception as e:
                print(f" [error on chunk {chunk['index']}: {e}]", end="")
                chunk_trans = ""

            trans_words = normalize_for_comparison(chunk_trans).split()

            if trans_words:
                window = max(MIN_WINDOW, len(trans_words))
                text, score, best_start = align_chunk_sequential(
                    trans_words, lyric_words, cursor, window,
                )
                # Advance cursor by the non-overlap fraction of the window so
                # the next chunk picks up roughly where this one's audio ended.
                advance = max(1, int(window * step_ratio))
                cursor = min(best_start + advance, total_words)
            else:
                # Silence / instrumental: no reliable label. Leave text empty
                # (the filter drops it) and advance by the per-chunk estimate
                # so we stay roughly in sync with the audio.
                text, score = "", 0.0
                cursor = min(cursor + max(1, int(words_per_chunk_est * step_ratio)), total_words)

            all_segments.append({
                "audio_path": chunk["path"],
                "text": text,
                "alignment_score": round(score, 4),
                "artist": artist,
                "song": song,
                "chunk_index": chunk["index"],
                "start": chunk["start"],
                "end": chunk["end"],
                "original_sample_id": sample.get("id", ""),
            })
            total_chunks += 1
            score_sum += score
            song_score_sum += score

        avg = song_score_sum / max(1, len(chunks))
        print(f"  (avg align {avg:.2f}) ✓")

    mean_score = score_sum / max(1, total_chunks)
    above_thresh = sum(1 for s in all_segments if s["alignment_score"] >= 0.12)

    # Save segment manifest
    segment_data = {
        "total_segments": len(all_segments),
        "chunk_duration": chunk_duration,
        "overlap": overlap,
        "alignment": "v2_sequential_greedy",
        "mean_alignment_score": round(mean_score, 4),
        "segments": all_segments,
    }
    SEGMENT_MANIFEST.write_text(json.dumps(segment_data, indent=2), encoding="utf-8")

    print(f"\n{'=' * 60}")
    print(f"  SEGMENTATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Original samples:      {len(samples)}")
    print(f"  Skipped:               {skipped}")
    print(f"  Total segments:        {total_chunks}")
    print(f"  Mean alignment score:  {mean_score:.3f}")
    print(f"  Segments >= 0.12:      {above_thresh}  (rough pre-filter survival)")
    print(f"  Manifest saved:        {SEGMENT_MANIFEST}")
    print(f"{'=' * 60}\n")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Segment audio into aligned chunks for training")
    parser.add_argument("--chunk-duration", type=int, default=30, help="Chunk duration in seconds")
    parser.add_argument("--overlap", type=int, default=5, help="Overlap between chunks in seconds")

    args = parser.parse_args()
    segment_all(chunk_duration=args.chunk_duration, overlap=args.overlap)
