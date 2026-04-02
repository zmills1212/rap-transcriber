#!/usr/bin/env python3
"""
Rap Transcriber — Unified CLI
===============================
One command to transcribe rap audio with maximum accuracy.

Usage:
    # Transcribe with reference lyrics (highest accuracy)
    rap-transcribe audio.mp3 --lyrics lyrics.txt

    # Transcribe without reference (Whisper + post-processing only)
    rap-transcribe audio.mp3

    # Transcribe and save output
    rap-transcribe audio.mp3 --lyrics lyrics.txt -o output.txt

    # Evaluate pipeline on test samples
    rap-transcribe --eval

    # Compare models
    rap-transcribe audio.mp3 --model medium
"""

import sys
import time
import json
import argparse
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Lazy imports (fast CLI startup)
# =============================================================================

_whisper_model = None
_whisper_processor = None
_using_finetuned = False


def get_whisper_model(model_size: str = "small"):
    """Load Whisper model (cached). Uses fine-tuned adapter if available."""
    global _whisper_model, _whisper_processor, _using_finetuned
    if _whisper_model is not None:
        return _whisper_model

    adapter_path = PROJECT_ROOT / "training" / "fine_tuned" / "lora_adapter"

    if adapter_path.exists():
        try:
            import torch
            from transformers import WhisperForConditionalGeneration, WhisperProcessor
            from peft import PeftModel

            model_name = f"openai/whisper-{model_size}.en"
            print(f"  Loading fine-tuned Whisper ({model_size})...", end=" ", flush=True)

            _whisper_processor = WhisperProcessor.from_pretrained(model_name)
            base = WhisperForConditionalGeneration.from_pretrained(
                model_name, torch_dtype=torch.float32
            )
            model = PeftModel.from_pretrained(base, str(adapter_path))
            _whisper_model = model.merge_and_unload()
            _whisper_model.eval()
            _using_finetuned = True
            print("done (fine-tuned)")
            return _whisper_model
        except Exception as e:
            print(f"\n  Warning: Could not load fine-tuned model: {e}")
            print(f"  Falling back to baseline Whisper...")

    # Fallback to baseline
    try:
        import whisper
        print(f"  Loading Whisper ({model_size})...", end=" ", flush=True)
        _whisper_model = whisper.load_model(model_size)
        _using_finetuned = False
        print("done (baseline)")
    except ImportError:
        print("ERROR: openai-whisper not installed")
        sys.exit(1)
    return _whisper_model


def transcribe(audio_path: str, model_size: str = "small") -> str:
    """Transcribe audio with Whisper (fine-tuned or baseline)."""
    model = get_whisper_model(model_size)

    if _using_finetuned:
        import torch
        import librosa
        y, _ = librosa.load(audio_path, sr=16000, mono=True)
        inputs = _whisper_processor.feature_extractor(
            y, sampling_rate=16000, return_tensors="pt"
        )
        with torch.no_grad():
            ids = model.generate(inputs.input_features, max_length=448)
        return _whisper_processor.tokenizer.batch_decode(
            ids, skip_special_tokens=True
        )[0].strip()
    else:
        result = model.transcribe(audio_path)
        return result["text"].strip()


def postprocess(text: str) -> str:
    """Apply rap-specific post-processing."""
    try:
        from postprocessor.rap_postprocessor import RapPostProcessor
        pp = RapPostProcessor()
        return pp.process(text)
    except (ImportError, Exception):
        return text


def match_lyrics(transcription: str, reference: str) -> dict:
    """Run lyrics matcher."""
    from lyrics_matcher.lyrics_matcher import LyricsMatcher
    matcher = LyricsMatcher()
    result = matcher.match(transcription, reference)
    return {
        "text": result.text,
        "confidence": result.confidence,
        "stats": result.stats,
        "original_wer": result.transcription_wer,
        "corrected_wer": result.corrected_wer,
    }


def compute_wer(hyp: str, ref: str) -> float:
    """Quick WER computation."""
    from lyrics_matcher.lyrics_matcher import tokenize
    h = tokenize(hyp)
    r = tokenize(ref)
    if not r:
        return 0.0 if not h else 1.0
    rows, cols = len(r) + 1, len(h) + 1
    d = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        d[i][0] = i
    for j in range(cols):
        d[0][j] = j
    for i in range(1, rows):
        for j in range(1, cols):
            cost = 0 if r[i-1] == h[j-1] else 1
            d[i][j] = min(d[i-1][j]+1, d[i][j-1]+1, d[i-1][j-1]+cost)
    return d[len(r)][len(h)] / len(r)


# =============================================================================
# Main Pipeline
# =============================================================================

def fetch_genius_lyrics(song_title: str, artist: str = None) -> str:
    """Fetch lyrics from Genius API."""
    try:
        from lyrics_matcher.genius import GeniusClient
        genius = GeniusClient()
        print(f"\n  🔍 Searching Genius for: {artist + ' - ' if artist else ''}{song_title}")
        lyrics = genius.fetch_lyrics(song_title, artist=artist)
        if lyrics:
            word_count = len(lyrics.split())
            print(f"  ✓ Fetched {word_count} words of lyrics")
            return lyrics
        else:
            print(f"  ✗ No lyrics found on Genius")
            return ""
    except ValueError as e:
        print(f"  ERROR: {e}")
        return ""
    except Exception as e:
        print(f"  ERROR fetching from Genius: {e}")
        return ""


def run_transcription(
    audio_path: str,
    lyrics_path: str = None,
    song_title: str = None,
    artist: str = None,
    model_size: str = "small",
    output_path: str = None,
    verbose: bool = False,
    json_output: bool = False,
):
    """Run the full transcription pipeline."""

    audio_file = Path(audio_path)
    if not audio_file.exists():
        print(f"ERROR: Audio file not found: {audio_path}")
        sys.exit(1)

    reference = ""
    if lyrics_path:
        lp = Path(lyrics_path)
        if not lp.exists():
            print(f"ERROR: Lyrics file not found: {lyrics_path}")
            sys.exit(1)
        reference = lp.read_text(encoding="utf-8").strip()
    elif song_title:
        # Auto-fetch from Genius
        reference = fetch_genius_lyrics(song_title, artist=artist)

    results = {"audio": str(audio_file.name), "stages": {}}

    # Stage 1: Whisper
    if not json_output:
        print(f"\n  🎤 Transcribing: {audio_file.name}")
    t0 = time.time()
    raw = transcribe(str(audio_file), model_size)
    whisper_time = time.time() - t0
    results["stages"]["whisper"] = {"text": raw, "time": whisper_time}

    if verbose and not json_output:
        print(f"     Whisper ({whisper_time:.1f}s): {raw[:80]}...")

    # Stage 2: Post-processor
    processed = postprocess(raw)
    results["stages"]["postprocessor"] = {"text": processed}

    if verbose and not json_output:
        if processed != raw:
            print(f"     PostProc: {processed[:80]}...")
        else:
            print(f"     PostProc: [no changes]")

    # Stage 3: Lyrics Matcher (if reference available)
    if reference:
        matched = match_lyrics(processed, reference)
        final_text = matched["text"]
        results["stages"]["matcher"] = matched
        results["confidence"] = matched["confidence"]

        if verbose and not json_output:
            print(f"     Matcher ({matched['confidence']:.0%} conf): {final_text[:80]}...")

        # WER comparison
        whisper_wer = compute_wer(raw, reference)
        pp_wer = compute_wer(processed, reference)
        final_wer = compute_wer(final_text, reference)
        results["wer"] = {
            "whisper": whisper_wer,
            "postprocessor": pp_wer,
            "final": final_wer,
        }
    else:
        final_text = processed
        results["confidence"] = None

    results["final_text"] = final_text

    # Output
    if json_output:
        print(json.dumps(results, indent=2, default=str))
    else:
        print(f"\n{'━' * 60}")
        print(f"  TRANSCRIPTION")
        print(f"{'━' * 60}")
        print(f"\n{final_text}\n")

        if reference:
            wer = results["wer"]
            print(f"{'━' * 60}")
            print(f"  ACCURACY")
            print(f"{'━' * 60}")
            print(f"  Whisper:      {wer['whisper']:.1%} WER")
            print(f"  + PostProc:   {wer['postprocessor']:.1%} WER")
            print(f"  + Matcher:    {wer['final']:.1%} WER")
            print(f"  Confidence:   {results['confidence']:.1%}")
            print(f"  Improvement:  {wer['whisper'] - wer['final']:.1%} WER reduction")

        if not reference:
            print(f"  💡 Tip: Add --song 'Title' -a 'Artist' to auto-fetch lyrics from Genius")

    # Save to file
    if output_path:
        Path(output_path).write_text(final_text, encoding="utf-8")
        if not json_output:
            print(f"\n  Saved to: {output_path}")

    return results


# =============================================================================
# Batch Evaluation
# =============================================================================

def run_evaluation(test_dir: str = None, model_size: str = "small"):
    """Run pipeline evaluation on test samples."""
    from lyrics_matcher.pipeline_eval import run_batch_eval
    run_batch_eval(test_dir=test_dir, model_size=model_size)


# =============================================================================
# Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        prog="rap-transcribe",
        description="Rap Transcriber — accurate lyrics from audio",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  %(prog)s song.mp3                                        # Basic transcription
  %(prog)s song.mp3 --lyrics reference.txt                 # With lyrics file
  %(prog)s song.mp3 --song "Niggas Be Lame" -a "Yung Bans" # Auto-fetch from Genius
  %(prog)s song.mp3 -o output.txt                          # Save to file
  %(prog)s --eval                                          # Evaluate pipeline
        """,
    )

    parser.add_argument("audio", nargs="?", help="Audio file to transcribe")
    parser.add_argument("--lyrics", "-l", help="Reference lyrics file for matching")
    parser.add_argument("--song", "-s", help="Song title (auto-fetch lyrics from Genius)")
    parser.add_argument("--artist", "-a", help="Artist name (improves Genius search)")
    parser.add_argument("--output", "-o", help="Save transcription to file")
    parser.add_argument("--model", "-m", default="small",
                       choices=["tiny", "base", "small", "medium"],
                       help="Whisper model size (default: small)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Show each pipeline stage")
    parser.add_argument("--json", action="store_true", dest="json_output",
                       help="Output as JSON")
    parser.add_argument("--eval", action="store_true",
                       help="Run batch evaluation on test samples")
    parser.add_argument("--eval-dir", help="Custom test directory for --eval")

    args = parser.parse_args()

    if args.eval:
        run_evaluation(test_dir=args.eval_dir, model_size=args.model)
        return

    if not args.audio:
        parser.print_help()
        return

    run_transcription(
        audio_path=args.audio,
        lyrics_path=args.lyrics,
        song_title=args.song,
        artist=args.artist,
        model_size=args.model,
        output_path=args.output,
        verbose=args.verbose,
        json_output=args.json_output,
    )


if __name__ == "__main__":
    main()
