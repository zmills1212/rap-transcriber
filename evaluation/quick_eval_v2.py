#!/usr/bin/env python3
"""
Quick evaluation with post-processor comparison.

Runs Whisper, shows raw WER, applies post-processing, shows improved WER.

Usage:
    python evaluation/quick_eval_v2.py --audio sample.mp3 --truth lyrics.txt
    python evaluation/quick_eval_v2.py --audio sample.mp3 --truth lyrics.txt --aggressive
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
from evaluation.evaluate import compute_wer, tokenize, EvalResult, categorize_errors
from postprocessor.rap_postprocessor import RapPostProcessor


def evaluate_with_postprocessor(
    audio_path: str,
    ground_truth_path: str,
    model_size: str = "medium",
    aggressive: bool = False,
) -> dict:
    """Run full evaluation with before/after post-processing comparison."""
    
    # Load ground truth
    with open(ground_truth_path, "r") as f:
        reference = f.read().strip()
    ref_tokens = tokenize(reference)
    
    # Transcribe
    from src.inference.whisper_engine import WhisperTranscriber
    print(f"Transcribing with whisper-{model_size}...")
    transcriber = WhisperTranscriber(model_size=model_size)
    
    start = time.time()
    result = transcriber.transcribe(audio_path)
    elapsed = time.time() - start
    
    whisper_raw = result.get("text", "") if isinstance(result, dict) else str(result)
    print(f"Transcription complete ({elapsed:.1f}s)\n")
    
    # WER before post-processing
    hyp_before = tokenize(whisper_raw)
    wer_before = compute_wer(ref_tokens, hyp_before)
    cats_before = categorize_errors(wer_before["alignment"])
    
    # Apply post-processing
    processor = RapPostProcessor(aggressive=aggressive)
    corrected = processor.process(whisper_raw, track_changes=True)
    
    # WER after post-processing
    hyp_after = tokenize(corrected.corrected)
    wer_after = compute_wer(ref_tokens, hyp_after)
    cats_after = categorize_errors(wer_after["alignment"])
    
    return {
        "reference": reference,
        "whisper_raw": whisper_raw,
        "whisper_corrected": corrected.corrected,
        "changes": corrected.changes,
        "wer_before": wer_before,
        "wer_after": wer_after,
        "cats_before": cats_before,
        "cats_after": cats_after,
        "processing_time": elapsed,
        "model_size": model_size,
    }


def print_alignment(alignment: list, label: str):
    """Print color-coded alignment."""
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    RESET = "\033[0m"
    
    print(f"\n{label}")
    print("-" * 70)
    
    ref_line = "REF: "
    hyp_line = "HYP: "
    
    for op, ref_word, hyp_word in alignment:
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
        
        if len(ref_line) > 200:
            print(ref_line)
            print(hyp_line)
            print()
            ref_line = "REF: "
            hyp_line = "HYP: "
    
    if ref_line.strip() != "REF:":
        print(ref_line)
        print(hyp_line)


def print_report(result: dict):
    """Print comprehensive comparison report."""
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    RESET = "\033[0m"
    
    wer_b = result["wer_before"]
    wer_a = result["wer_after"]
    
    print("=" * 70)
    print(f"{BOLD}RAP TRANSCRIPTION EVALUATION WITH POST-PROCESSING{RESET}")
    print(f"Model: whisper-{result['model_size']}")
    print("=" * 70)
    
    # Show alignments
    print_alignment(wer_b["alignment"], f"{YELLOW}BEFORE POST-PROCESSING:{RESET}")
    print_alignment(wer_a["alignment"], f"{GREEN}AFTER POST-PROCESSING:{RESET}")
    
    # Changes made
    print(f"\n{BOLD}CORRECTIONS APPLIED:{RESET}")
    print("-" * 70)
    if result["changes"]:
        for change in result["changes"]:
            print(f"  • {change}")
    else:
        print("  (none)")
    
    # WER comparison table
    print(f"\n{BOLD}RESULTS COMPARISON:{RESET}")
    print("-" * 70)
    print(f"{'Metric':<25} {'Before':>12} {'After':>12} {'Change':>12}")
    print("-" * 70)
    
    # WER
    wer_change = wer_b["wer"] - wer_a["wer"]
    change_color = GREEN if wer_change > 0 else (RED if wer_change < 0 else RESET)
    print(f"{'WER':<25} {wer_b['wer']:>11.1%} {wer_a['wer']:>11.1%} {change_color}{wer_change:>+11.1%}{RESET}")
    
    # Errors
    err_change = wer_b["total_errors"] - wer_a["total_errors"]
    change_color = GREEN if err_change > 0 else (RED if err_change < 0 else RESET)
    print(f"{'Total Errors':<25} {wer_b['total_errors']:>12} {wer_a['total_errors']:>12} {change_color}{-err_change:>+12}{RESET}")
    
    # Substitutions
    sub_change = wer_b["substitutions"] - wer_a["substitutions"]
    change_color = GREEN if sub_change > 0 else RESET
    print(f"{'Substitutions':<25} {wer_b['substitutions']:>12} {wer_a['substitutions']:>12} {change_color}{-sub_change:>+12}{RESET}")
    
    # Insertions
    ins_change = wer_b["insertions"] - wer_a["insertions"]
    print(f"{'Insertions':<25} {wer_b['insertions']:>12} {wer_a['insertions']:>12} {-ins_change:>+12}")
    
    # Deletions
    del_change = wer_b["deletions"] - wer_a["deletions"]
    print(f"{'Deletions':<25} {wer_b['deletions']:>12} {wer_a['deletions']:>12} {-del_change:>+12}")
    
    print("-" * 70)
    
    # Error category comparison
    print(f"\n{BOLD}ERROR CATEGORIES:{RESET}")
    print("-" * 70)
    print(f"{'Category':<20} {'Before':>10} {'After':>10} {'Fixed':>10}")
    print("-" * 70)
    
    all_cats = set(result["cats_before"].keys()) | set(result["cats_after"].keys())
    for cat in sorted(all_cats):
        before_count = len(result["cats_before"].get(cat, []))
        after_count = len(result["cats_after"].get(cat, []))
        fixed = before_count - after_count
        color = GREEN if fixed > 0 else (RED if fixed < 0 else RESET)
        print(f"{cat:<20} {before_count:>10} {after_count:>10} {color}{fixed:>+10}{RESET}")
    
    # Summary
    print("\n" + "=" * 70)
    if wer_change > 0:
        pct_improve = (wer_change / wer_b["wer"]) * 100 if wer_b["wer"] > 0 else 0
        print(f"{GREEN}{BOLD}✓ POST-PROCESSOR IMPROVED WER BY {wer_change:.1%} ({pct_improve:.0f}% relative){RESET}")
        print(f"{GREEN}  Fixed {err_change} errors out of {wer_b['total_errors']}{RESET}")
    elif wer_change < 0:
        print(f"{RED}{BOLD}✗ POST-PROCESSOR MADE THINGS WORSE BY {-wer_change:.1%}{RESET}")
        print(f"{YELLOW}  Consider tuning correction rules{RESET}")
    else:
        print(f"{YELLOW}{BOLD}→ NO NET CHANGE IN WER{RESET}")
        print(f"  Post-processor may have traded some errors for others")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Evaluate transcription with post-processing")
    parser.add_argument("--audio", type=str, required=True, help="Audio file")
    parser.add_argument("--truth", type=str, required=True, help="Ground truth lyrics")
    parser.add_argument("--model", type=str, default="medium", help="Whisper model")
    parser.add_argument("--aggressive", action="store_true", help="Aggressive correction mode")
    args = parser.parse_args()
    
    result = evaluate_with_postprocessor(
        audio_path=args.audio,
        ground_truth_path=args.truth,
        model_size=args.model,
        aggressive=args.aggressive,
    )
    
    print_report(result)


if __name__ == "__main__":
    main()
