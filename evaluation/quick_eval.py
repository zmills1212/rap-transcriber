#!/usr/bin/env python3
"""
Quick evaluation runner for the rap transcriber.

QUICK START:
    1. Place audio in test_data/audio/
    2. Place matching ground truth lyrics in test_data/ground_truth/
    3. Run: python evaluation/quick_eval.py

Or test a single file:
    python evaluation/quick_eval.py --audio path/to/song.mp3 --truth path/to/lyrics.txt

Or use your existing sample.mp3:
    python evaluation/quick_eval.py --audio sample.mp3 --truth test_data/ground_truth/sample_001.txt
"""

import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.evaluate import evaluate_sample, generate_report


def print_alignment_diff(result):
    """Print a color-coded word-by-word alignment (works in terminal)."""
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    RESET = "\033[0m"

    print("\n" + "=" * 70)
    print("WORD-BY-WORD ALIGNMENT")
    print("=" * 70)
    print(f"{GREEN}green=correct{RESET}  {YELLOW}yellow=substitution{RESET}  "
          f"{RED}red=deletion{RESET}  {CYAN}cyan=insertion{RESET}\n")

    ref_line = "REF: "
    hyp_line = "HYP: "

    for op, ref_word, hyp_word in result.alignment:
        max_len = max(len(ref_word), len(hyp_word), 1) + 1

        if op == "correct":
            ref_line += f"{GREEN}{ref_word:<{max_len}}{RESET}"
            hyp_line += f"{GREEN}{hyp_word:<{max_len}}{RESET}"
        elif op == "substitution":
            ref_line += f"{YELLOW}{ref_word:<{max_len}}{RESET}"
            hyp_line += f"{YELLOW}{hyp_word:<{max_len}}{RESET}"
        elif op == "deletion":
            ref_line += f"{RED}{ref_word:<{max_len}}{RESET}"
            hyp_line += f"{RED}{'***':<{max_len}}{RESET}"
        elif op == "insertion":
            ref_line += f"{CYAN}{'***':<{max_len}}{RESET}"
            hyp_line += f"{CYAN}{hyp_word:<{max_len}}{RESET}"

        # Wrap at ~100 chars
        if len(ref_line) > 200:
            print(ref_line)
            print(hyp_line)
            print()
            ref_line = "REF: "
            hyp_line = "HYP: "

    if ref_line.strip() != "REF:":
        print(ref_line)
        print(hyp_line)
    print()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Quick rap transcription evaluation")
    parser.add_argument("--audio", type=str, help="Audio file to evaluate")
    parser.add_argument("--truth", type=str, help="Ground truth lyrics file")
    parser.add_argument("--model", type=str, default="small",
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model size (default: small)")
    parser.add_argument("--tags", type=str, nargs="*", default=[],
                        help="Challenge tags for this sample (e.g. fast_flow heavy_slang)")
    parser.add_argument("--no-color", action="store_true", help="Disable colored output")
    parser.add_argument("--compare-models", action="store_true",
                        help="Run evaluation across tiny/base/small/medium and compare")
    args = parser.parse_args()

    if not args.audio or not args.truth:
        print("Usage: python evaluation/quick_eval.py --audio <file> --truth <lyrics.txt>")
        print("\nExample:")
        print("  python evaluation/quick_eval.py --audio sample.mp3 --truth lyrics.txt --model small")
        print("  python evaluation/quick_eval.py --audio sample.mp3 --truth lyrics.txt --compare-models")
        sys.exit(1)

    if args.compare_models:
        print("\n" + "=" * 70)
        print("MODEL COMPARISON")
        print("=" * 70)

        results = []
        for model in ["tiny", "base", "small", "medium"]:
            print(f"\nTesting whisper-{model}...")
            result = evaluate_sample(
                audio_path=args.audio,
                ground_truth_path=args.truth,
                sample_id=f"test_{model}",
                model_size=model,
                challenge_tags=args.tags,
            )
            results.append(result)
            print(f"  WER: {result.wer:.1%} | Time: {result.processing_time}s | "
                  f"Errors: {result.total_errors}/{result.total_ref_words}")

        print("\n" + "-" * 70)
        print(f"{'Model':<12} {'WER':>8} {'Errors':>8} {'Subs':>6} {'Ins':>5} {'Del':>5} {'Time':>8}")
        print("-" * 70)
        for r in results:
            print(f"  {r.model_size:<10} {r.wer:>7.1%} {r.total_errors:>7} "
                  f"{r.substitutions:>6} {r.insertions:>5} {r.deletions:>5} {r.processing_time:>7.1f}s")

        # Show detailed alignment for best model
        best = min(results, key=lambda r: r.wer)
        print(f"\nBest model: whisper-{best.model_size} (WER={best.wer:.1%})")
        if not args.no_color:
            print_alignment_diff(best)
        print(generate_report([best]))

    else:
        print(f"\nEvaluating with whisper-{args.model}...")
        result = evaluate_sample(
            audio_path=args.audio,
            ground_truth_path=args.truth,
            model_size=args.model,
            challenge_tags=args.tags,
        )

        if not args.no_color:
            print_alignment_diff(result)
        print(generate_report([result]))


if __name__ == "__main__":
    main()
