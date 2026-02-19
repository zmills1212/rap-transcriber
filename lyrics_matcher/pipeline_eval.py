"""
Phase A: Full Pipeline Evaluation
===================================
Runs the complete transcription pipeline and measures improvement at each stage:

    Audio → Whisper → Post-processor → Lyrics Matcher → Final output

Shows WER at each stage so you can see exactly where gains come from.

Usage:
    python -m lyrics_matcher.pipeline_eval \
        --audio test_data/audio/sample_001.wav \
        --reference test_data/ground_truth/sample_001.txt

    # Batch eval on all test samples
    python -m lyrics_matcher.pipeline_eval --batch

    # Skip Whisper (provide transcription directly)
    python -m lyrics_matcher.pipeline_eval \
        --transcription "whisper output text" \
        --reference "correct lyrics"
"""

import json
import time
from pathlib import Path
from typing import Optional, Dict

# Project imports
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lyrics_matcher.lyrics_matcher import LyricsMatcher, normalize_text, tokenize


# =============================================================================
# Whisper Transcription
# =============================================================================

def transcribe_with_whisper(audio_path: str, model_size: str = "small") -> str:
    """Run Whisper on an audio file."""
    try:
        import whisper
    except ImportError:
        # Try mlx_whisper
        try:
            import mlx_whisper
            result = mlx_whisper.transcribe(audio_path, path_or_hf_repo=f"mlx-community/whisper-{model_size}")
            return result["text"].strip()
        except ImportError:
            raise ImportError("Neither whisper nor mlx_whisper installed")

    model = whisper.load_model(model_size)
    result = model.transcribe(audio_path)
    return result["text"].strip()


# =============================================================================
# Post-Processing
# =============================================================================

def apply_postprocessor(text: str) -> str:
    """Apply rap-specific post-processing rules."""
    try:
        from postprocessor.rap_postprocessor import RapPostProcessor
        pp = RapPostProcessor()
        return pp.process(text)
    except (ImportError, Exception) as e:
        print(f"  [postprocessor not available: {e}]")
        return text


# =============================================================================
# WER Computation
# =============================================================================

def compute_wer(hypothesis: str, reference: str) -> float:
    """Compute Word Error Rate between hypothesis and reference."""
    hyp_tokens = tokenize(hypothesis)
    ref_tokens = tokenize(reference)

    if not ref_tokens:
        return 0.0 if not hyp_tokens else 1.0

    r = len(ref_tokens)
    h = len(hyp_tokens)
    d = [[0] * (h + 1) for _ in range(r + 1)]

    for i in range(r + 1):
        d[i][0] = i
    for j in range(h + 1):
        d[0][j] = j

    for i in range(1, r + 1):
        for j in range(1, h + 1):
            cost = 0 if ref_tokens[i-1] == hyp_tokens[j-1] else 1
            d[i][j] = min(d[i-1][j] + 1, d[i][j-1] + 1, d[i-1][j-1] + cost)

    return d[r][h] / r


# =============================================================================
# Single Sample Pipeline
# =============================================================================

def run_pipeline(
    audio_path: Optional[str] = None,
    transcription: Optional[str] = None,
    reference: str = "",
    model_size: str = "small",
    verbose: bool = True,
) -> Dict:
    """
    Run the full pipeline on a single sample.

    Returns dict with WER at each stage and final text.
    """
    results = {
        "stages": {},
        "reference": reference,
    }

    # Stage 1: Whisper
    if transcription:
        whisper_text = transcription
    elif audio_path:
        if verbose:
            print(f"\n  Stage 1: Whisper ({model_size})...")
        t0 = time.time()
        whisper_text = transcribe_with_whisper(audio_path, model_size)
        elapsed = time.time() - t0
        if verbose:
            print(f"    Time: {elapsed:.1f}s")
    else:
        raise ValueError("Provide either audio_path or transcription")

    whisper_wer = compute_wer(whisper_text, reference)
    results["stages"]["whisper"] = {
        "text": whisper_text,
        "wer": whisper_wer,
    }
    if verbose:
        print(f"    WER:  {whisper_wer:.1%}")
        print(f"    Text: {whisper_text[:100]}...")

    # Stage 2: Post-processor
    if verbose:
        print(f"\n  Stage 2: Post-processor...")
    pp_text = apply_postprocessor(whisper_text)
    pp_wer = compute_wer(pp_text, reference)
    results["stages"]["postprocessor"] = {
        "text": pp_text,
        "wer": pp_wer,
    }
    if verbose:
        improvement = whisper_wer - pp_wer
        print(f"    WER:  {pp_wer:.1%} ({'+' if improvement >= 0 else ''}{improvement:.1%})")
        print(f"    Text: {pp_text[:100]}...")

    # Stage 3: Lyrics Matcher
    if reference:
        if verbose:
            print(f"\n  Stage 3: Lyrics Matcher...")
        matcher = LyricsMatcher()
        match_result = matcher.match(pp_text, reference)
        matcher_wer = compute_wer(match_result.text, reference)
        results["stages"]["lyrics_matcher"] = {
            "text": match_result.text,
            "wer": matcher_wer,
            "confidence": match_result.confidence,
            "stats": match_result.stats,
        }
        if verbose:
            improvement = pp_wer - matcher_wer
            print(f"    WER:  {matcher_wer:.1%} ({'+' if improvement >= 0 else ''}{improvement:.1%})")
            print(f"    Conf: {match_result.confidence:.1%}")
            print(f"    Text: {match_result.text[:100]}...")
        results["final_text"] = match_result.text
        results["final_wer"] = matcher_wer
    else:
        results["final_text"] = pp_text
        results["final_wer"] = pp_wer
        if verbose:
            print(f"\n  [No reference lyrics — skipping lyrics matcher]")

    return results


# =============================================================================
# Batch Evaluation
# =============================================================================

def run_batch_eval(test_dir: Optional[str] = None, model_size: str = "small"):
    """
    Run pipeline on all test samples and produce a summary report.
    Looks for audio/ground_truth pairs in test_data/.
    """
    if test_dir:
        base = Path(test_dir)
    else:
        base = PROJECT_ROOT / "test_data"

    audio_dir = base / "audio"
    truth_dir = base / "ground_truth"

    if not audio_dir.exists():
        print(f"ERROR: No audio directory at {audio_dir}")
        return

    # Find matching pairs
    audio_files = sorted(audio_dir.glob("*"))
    results_all = []

    print("=" * 60)
    print("BATCH PIPELINE EVALUATION")
    print("=" * 60)

    for audio_file in audio_files:
        stem = audio_file.stem
        truth_file = truth_dir / f"{stem}.txt"

        if not truth_file.exists():
            print(f"\n  SKIP {stem}: no ground truth file")
            continue

        reference = truth_file.read_text().strip()
        print(f"\n{'─' * 60}")
        print(f"  {stem}")
        print(f"{'─' * 60}")

        try:
            result = run_pipeline(
                audio_path=str(audio_file),
                reference=reference,
                model_size=model_size,
                verbose=True,
            )
            result["sample"] = stem
            results_all.append(result)
        except Exception as e:
            print(f"  ERROR: {e}")

    # Summary
    if results_all:
        print(f"\n{'=' * 60}")
        print("SUMMARY")
        print(f"{'=' * 60}")

        print(f"\n  {'Sample':<30} {'Whisper':>10} {'PostProc':>10} {'Matcher':>10}")
        print(f"  {'─' * 30} {'─' * 10} {'─' * 10} {'─' * 10}")

        whisper_wers = []
        pp_wers = []
        matcher_wers = []

        for r in results_all:
            w = r["stages"]["whisper"]["wer"]
            p = r["stages"]["postprocessor"]["wer"]
            m = r["stages"].get("lyrics_matcher", {}).get("wer", p)

            whisper_wers.append(w)
            pp_wers.append(p)
            matcher_wers.append(m)

            print(f"  {r['sample']:<30} {w:>9.1%} {p:>9.1%} {m:>9.1%}")

        avg_w = sum(whisper_wers) / len(whisper_wers)
        avg_p = sum(pp_wers) / len(pp_wers)
        avg_m = sum(matcher_wers) / len(matcher_wers)

        print(f"  {'─' * 30} {'─' * 10} {'─' * 10} {'─' * 10}")
        print(f"  {'AVERAGE':<30} {avg_w:>9.1%} {avg_p:>9.1%} {avg_m:>9.1%}")
        print(f"\n  Pipeline improvement: {avg_w - avg_m:.1%} WER reduction")
        print(f"  (Whisper {avg_w:.1%} → PostProc {avg_p:.1%} → Matcher {avg_m:.1%})")

        # Save report
        report_path = PROJECT_ROOT / "evaluation" / "reports" / "pipeline_eval.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump({
                "results": results_all,
                "averages": {
                    "whisper_wer": avg_w,
                    "postprocessor_wer": avg_p,
                    "matcher_wer": avg_m,
                },
            }, f, indent=2, default=str)
        print(f"\n  Report saved to: {report_path}")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run full transcription pipeline")
    parser.add_argument("--audio", "-a", type=str, help="Audio file path")
    parser.add_argument("--transcription", "-t", type=str, help="Pre-computed Whisper output")
    parser.add_argument("--reference", "-r", type=str, help="Reference lyrics (text or file path)")
    parser.add_argument("--model", type=str, default="small", help="Whisper model size")
    parser.add_argument("--batch", action="store_true", help="Batch eval on all test samples")
    parser.add_argument("--test-dir", type=str, help="Test data directory for batch mode")

    args = parser.parse_args()

    if args.batch:
        run_batch_eval(test_dir=args.test_dir, model_size=args.model)
    else:
        if not args.audio and not args.transcription:
            print("ERROR: Provide --audio or --transcription")
            exit(1)

        reference = ""
        if args.reference:
            if Path(args.reference).exists():
                reference = Path(args.reference).read_text().strip()
            else:
                reference = args.reference

        result = run_pipeline(
            audio_path=args.audio,
            transcription=args.transcription,
            reference=reference,
            model_size=args.model,
        )

        print(f"\n{'=' * 60}")
        print(f"FINAL OUTPUT")
        print(f"{'=' * 60}")
        print(f"\n{result['final_text']}")
