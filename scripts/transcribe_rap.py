#!/usr/bin/env python3
"""
Rap Transcription CLI

Simple command-line interface for transcribing rap audio with post-processing.

Usage:
    python scripts/transcribe_rap.py audio.mp3
    python scripts/transcribe_rap.py audio.mp3 --model medium --no-postprocess
    python scripts/transcribe_rap.py audio.mp3 --output lyrics.txt
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe rap audio with intelligent post-processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s song.mp3
    %(prog)s song.mp3 --model small --output lyrics.txt
    %(prog)s song.wav --no-postprocess --verbose
        """
    )
    parser.add_argument("audio", help="Path to audio file (mp3, wav, m4a, etc.)")
    parser.add_argument("--model", "-m", default="medium",
                        choices=["tiny", "base", "small", "medium"],
                        help="Whisper model size (default: medium)")
    parser.add_argument("--output", "-o", help="Save transcription to file")
    parser.add_argument("--no-postprocess", action="store_true",
                        help="Skip post-processing (raw Whisper output)")
    parser.add_argument("--conservative", action="store_true",
                        help="Use conservative post-processing (no explicit corrections)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show detailed output including corrections")
    parser.add_argument("--compare", action="store_true",
                        help="Show before/after comparison")
    args = parser.parse_args()

    # Validate input
    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_path}")
        sys.exit(1)

    # Import here to delay model loading
    from src.inference.whisper_engine import WhisperTranscriber
    from postprocessor.rap_postprocessor import RapPostProcessor

    # Transcribe
    print(f"\n🎤 Transcribing: {audio_path.name}")
    print(f"   Model: whisper-{args.model}")
    
    transcriber = WhisperTranscriber(model_size=args.model)
    
    start = time.time()
    result = transcriber.transcribe(str(audio_path))
    raw_text = result.get("text", "") if isinstance(result, dict) else str(result)
    elapsed = time.time() - start
    
    print(f"   Time: {elapsed:.1f}s")

    # Post-process (unless disabled)
    if args.no_postprocess:
        final_text = raw_text
        corrections = []
        print("   Post-processing: disabled")
    else:
        processor = RapPostProcessor(aggressive=not args.conservative)
        correction_result = processor.process(raw_text, track_changes=True)
        final_text = correction_result.corrected
        corrections = correction_result.changes
        mode = "conservative" if args.conservative else "aggressive"
        print(f"   Post-processing: {mode} ({len(corrections)} corrections)")

    # Output
    print("\n" + "=" * 60)
    
    if args.compare and not args.no_postprocess:
        print("RAW WHISPER OUTPUT:")
        print("-" * 60)
        print(raw_text)
        print("\n" + "-" * 60)
        print("CORRECTED OUTPUT:")
        print("-" * 60)
    
    print(final_text)
    print("=" * 60)

    # Show corrections if verbose
    if args.verbose and corrections:
        print("\nCORRECTIONS APPLIED:")
        for c in corrections:
            print(f"  • {c}")

    # Save to file if requested
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(final_text)
        print(f"\n✓ Saved to: {output_path}")

    print()


if __name__ == "__main__":
    main()
