#!/usr/bin/env python3
"""
Evaluate post-processor effectiveness.

Compares WER before and after post-processing to measure improvement.

Usage:
    python evaluation/eval_postprocessor.py --audio sample.mp3 --truth lyrics.txt
    python evaluation/eval_postprocessor.py --audio sample.mp3 --truth lyrics.txt --aggressive
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from evaluation.evaluate import compute_wer, tokenize, normalize_text
from postprocessor.rap_postprocessor import RapPostProcessor


def compare_wer(reference: str, whisper_output: str, aggressive: bool = False) -> dict:
    """
    Compare WER before and after post-processing.
    
    Returns dict with before/after WER and improvement stats.
    """
    processor = RapPostProcessor(aggressive=aggressive)
    
    # Normalize reference
    ref_tokens = tokenize(reference)
    
    # Before post-processing
    hyp_before = tokenize(whisper_output)
    wer_before = compute_wer(ref_tokens, hyp_before)
    
    # After post-processing
    corrected = processor.process(whisper_output, track_changes=True)
    hyp_after = tokenize(corrected.corrected)
    wer_after = compute_wer(ref_tokens, hyp_after)
    
    # Calculate improvement
    wer_reduction = wer_before["wer"] - wer_after["wer"]
    error_reduction = wer_before["total_errors"] - wer_after["total_errors"]
    pct_improvement = (wer_reduction / wer_before["wer"] * 100) if wer_before["wer"] > 0 else 0
    
    return {
        "reference": reference,
        "whisper_raw": whisper_output,
        "whisper_corrected": corrected.corrected,
        "changes_made": corrected.changes,
        "wer_before": wer_before["wer"],
        "wer_after": wer_after["wer"],
        "wer_reduction": wer_reduction,
        "errors_before": wer_before["total_errors"],
        "errors_after": wer_after["total_errors"],
        "errors_fixed": error_reduction,
        "pct_improvement": pct_improvement,
        "total_words": wer_before["total_ref_words"],
    }


def print_comparison(result: dict):
    """Print a formatted comparison report."""
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    RESET = "\033[0m"
    
    print("\n" + "=" * 70)
    print(f"{BOLD}POST-PROCESSOR EVALUATION{RESET}")
    print("=" * 70)
    
    print(f"\n{CYAN}REFERENCE:{RESET}")
    print(f"  {result['reference'][:100]}..." if len(result['reference']) > 100 else f"  {result['reference']}")
    
    print(f"\n{YELLOW}WHISPER RAW:{RESET}")
    print(f"  {result['whisper_raw'][:100]}..." if len(result['whisper_raw']) > 100 else f"  {result['whisper_raw']}")
    
    print(f"\n{GREEN}AFTER CORRECTION:{RESET}")
    print(f"  {result['whisper_corrected'][:100]}..." if len(result['whisper_corrected']) > 100 else f"  {result['whisper_corrected']}")
    
    print(f"\n{BOLD}CHANGES APPLIED:{RESET}")
    if result['changes_made']:
        for change in result['changes_made']:
            print(f"  • {change}")
    else:
        print("  (none)")
    
    print("\n" + "-" * 70)
    print(f"{BOLD}RESULTS:{RESET}")
    print("-" * 70)
    
    # WER comparison
    wer_color = GREEN if result['wer_after'] < result['wer_before'] else RED
    print(f"  WER Before:     {result['wer_before']:.1%} ({result['errors_before']} errors)")
    print(f"  WER After:      {wer_color}{result['wer_after']:.1%}{RESET} ({result['errors_after']} errors)")
    
    if result['wer_reduction'] > 0:
        print(f"\n  {GREEN}✓ IMPROVEMENT: {result['wer_reduction']:.1%} WER reduction ({result['errors_fixed']} errors fixed){RESET}")
        print(f"  {GREEN}  That's a {result['pct_improvement']:.1f}% relative improvement!{RESET}")
    elif result['wer_reduction'] < 0:
        print(f"\n  {RED}✗ REGRESSION: WER increased by {-result['wer_reduction']:.1%}{RESET}")
    else:
        print(f"\n  {YELLOW}→ NO CHANGE in WER{RESET}")
    
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Evaluate post-processor effectiveness")
    parser.add_argument("--audio", type=str, help="Audio file to transcribe")
    parser.add_argument("--truth", type=str, help="Ground truth lyrics file")
    parser.add_argument("--whisper-text", type=str, help="Pre-transcribed Whisper output (instead of --audio)")
    parser.add_argument("--model", type=str, default="medium", help="Whisper model size")
    parser.add_argument("--aggressive", action="store_true", help="Use aggressive correction mode")
    args = parser.parse_args()
    
    # Load ground truth
    if not args.truth:
        parser.error("--truth is required")
    
    with open(args.truth, "r") as f:
        reference = f.read().strip()
    
    # Get Whisper output
    if args.whisper_text:
        with open(args.whisper_text, "r") as f:
            whisper_output = f.read().strip()
    elif args.audio:
        print(f"Transcribing with whisper-{args.model}...")
        from src.inference.whisper_engine import WhisperTranscriber
        transcriber = WhisperTranscriber(model_size=args.model)
        result = transcriber.transcribe(args.audio)
        whisper_output = result.get("text", "") if isinstance(result, dict) else str(result)
        print(f"Transcription complete.\n")
    else:
        parser.error("Either --audio or --whisper-text is required")
    
    # Run comparison
    result = compare_wer(reference, whisper_output, aggressive=args.aggressive)
    print_comparison(result)
    
    return result


if __name__ == "__main__":
    main()
