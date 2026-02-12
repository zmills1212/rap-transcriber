"""
Rap Transcription Evaluation Engine

Evaluates Whisper transcription accuracy against ground truth lyrics.
Categorizes errors by type (slang, speed, dialect, etc.) to guide optimization.

Usage:
    python -m evaluation.evaluate --manifest test_data/test_manifest.json --model small
    python -m evaluation.evaluate --audio path/to/file.mp3 --truth path/to/lyrics.txt
"""

import json
import re
import time
import argparse
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional
from collections import Counter


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """Normalize text for fair comparison."""
    text = text.lower().strip()
    # Remove common punctuation but keep apostrophes in contractions
    text = re.sub(r"[^\w\s']", "", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    """Split normalized text into word tokens."""
    return normalize_text(text).split()


# ---------------------------------------------------------------------------
# Word Error Rate (Levenshtein on word sequences)
# ---------------------------------------------------------------------------

def compute_wer(reference: list[str], hypothesis: list[str]) -> dict:
    """
    Compute Word Error Rate with edit operation breakdown.
    Returns dict with wer, substitutions, insertions, deletions, and alignment.
    """
    r, h = reference, hypothesis
    n = len(r)
    m = len(h)

    # DP table
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if r[i - 1] == h[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = 1 + min(
                    d[i - 1][j],      # deletion
                    d[i][j - 1],      # insertion
                    d[i - 1][j - 1],  # substitution
                )

    # Backtrace to get alignment
    alignment = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and r[i - 1] == h[j - 1]:
            alignment.append(("correct", r[i - 1], h[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + 1:
            alignment.append(("substitution", r[i - 1], h[j - 1]))
            i -= 1
            j -= 1
        elif j > 0 and d[i][j] == d[i][j - 1] + 1:
            alignment.append(("insertion", "", h[j - 1]))
            j -= 1
        else:
            alignment.append(("deletion", r[i - 1], ""))
            i -= 1

    alignment.reverse()

    subs = sum(1 for op, _, _ in alignment if op == "substitution")
    ins = sum(1 for op, _, _ in alignment if op == "insertion")
    dels = sum(1 for op, _, _ in alignment if op == "deletion")
    correct = sum(1 for op, _, _ in alignment if op == "correct")
    total_errors = subs + ins + dels
    wer = total_errors / max(n, 1)

    return {
        "wer": round(wer, 4),
        "total_ref_words": n,
        "total_hyp_words": m,
        "correct": correct,
        "substitutions": subs,
        "insertions": ins,
        "deletions": dels,
        "total_errors": total_errors,
        "alignment": alignment,
    }


# ---------------------------------------------------------------------------
# Error categorization
# ---------------------------------------------------------------------------

# Common rap slang / AAVE terms Whisper is likely to miss
RAP_SLANG = {
    "finna", "tryna", "bussin", "cap", "no cap", "bet", "bruh", "fam",
    "drip", "ice", "whip", "plug", "opp", "opps", "guap", "bands",
    "hunnid", "rack", "racks", "stack", "stacks", "peel", "slide",
    "glizzy", "choppa", "blicky", "stick", "pole", "heat", "strap",
    "thot", "simp", "lit", "fire", "mid", "sus", "vibes", "lowkey",
    "highkey", "deadass", "word", "facts", "aight", "ight", "nah",
    "ion", "ima", "boutta", "gotta", "wanna", "gonna", "coulda",
    "shoulda", "woulda", "aint", "ain't", "yuh", "yah", "skrt",
    "skrrt", "grrah", "brrr", "gang", "dawg", "shawty", "shorty",
    "lil", "big", "young", "yung", "diss", "flex", "finesse",
    "clout", "sauce", "draco", "bando", "trap", "juug", "lick",
    "whippin", "trappin", "sippin", "grippin", "drippin", "hittin",
}

COMMON_ADLIBS = {
    "skrt", "skrrt", "yeah", "yuh", "aye", "ay", "brr", "brrr",
    "grrah", "grr", "pow", "bang", "woo", "gang", "let's go",
    "sheesh", "ugh", "huh", "what", "ooh", "ah", "oh", "damn",
}


def categorize_errors(alignment: list[tuple], challenge_tags: list[str] = None) -> dict:
    """
    Categorize each error from the WER alignment.
    Returns error breakdown with examples.
    """
    categories = {
        "slang_miss": [],       # Known slang word missed
        "adlib_miss": [],       # Ad-lib missed or hallucinated
        "homophone": [],        # Sounds-alike substitution
        "truncation": [],       # Word partially captured
        "hallucination": [],    # Whisper inserted a word not in reference
        "dropped_word": [],     # Word in reference completely missing
        "other_sub": [],        # Other substitution errors
    }

    for op, ref_word, hyp_word in alignment:
        if op == "correct":
            continue

        ref_lower = ref_word.lower() if ref_word else ""
        hyp_lower = hyp_word.lower() if hyp_word else ""

        if op == "substitution":
            if ref_lower in RAP_SLANG:
                categories["slang_miss"].append((ref_word, hyp_word))
            elif _is_homophone(ref_lower, hyp_lower):
                categories["homophone"].append((ref_word, hyp_word))
            elif _is_truncation(ref_lower, hyp_lower):
                categories["truncation"].append((ref_word, hyp_word))
            else:
                categories["other_sub"].append((ref_word, hyp_word))

        elif op == "insertion":
            if hyp_lower in COMMON_ADLIBS:
                categories["adlib_miss"].append(("", hyp_word))
            else:
                categories["hallucination"].append(("", hyp_word))

        elif op == "deletion":
            if ref_lower in COMMON_ADLIBS:
                categories["adlib_miss"].append((ref_word, ""))
            elif ref_lower in RAP_SLANG:
                categories["slang_miss"].append((ref_word, ""))
            else:
                categories["dropped_word"].append((ref_word, ""))

    return categories


def _is_homophone(a: str, b: str) -> bool:
    """Rough heuristic: do these words sound similar?"""
    if not a or not b:
        return False
    # Same first 3 chars and similar length
    if len(a) >= 3 and len(b) >= 3 and a[:3] == b[:3] and abs(len(a) - len(b)) <= 2:
        return True
    # Levenshtein distance <= 2 for words of length 4+
    if len(a) >= 4 and len(b) >= 4 and _edit_distance(a, b) <= 2:
        return True
    return False


def _is_truncation(ref: str, hyp: str) -> bool:
    """Check if hypothesis is a truncated version of reference or vice versa."""
    if not ref or not hyp:
        return False
    return ref.startswith(hyp) or hyp.startswith(ref)


def _edit_distance(a: str, b: str) -> int:
    """Character-level edit distance."""
    n, m = len(a), len(b)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[m]


# ---------------------------------------------------------------------------
# Single-sample evaluation result
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    sample_id: str
    artist: str
    song: str
    model_size: str
    duration_seconds: Optional[float]
    processing_time: float
    wer: float
    total_ref_words: int
    total_errors: int
    substitutions: int
    insertions: int
    deletions: int
    error_categories: dict
    challenge_tags: list[str]
    reference_text: str
    hypothesis_text: str
    alignment: list[tuple] = field(default_factory=list, repr=False)

    def summary_line(self) -> str:
        cats = {k: len(v) for k, v in self.error_categories.items() if v}
        return (
            f"[{self.sample_id}] WER={self.wer:.1%} | "
            f"Errors: {self.total_errors}/{self.total_ref_words} | "
            f"Categories: {cats} | Tags: {self.challenge_tags}"
        )


# ---------------------------------------------------------------------------
# Run evaluation on a single audio file
# ---------------------------------------------------------------------------

def evaluate_sample(
    audio_path: str,
    ground_truth_path: str,
    sample_id: str = "unknown",
    model_size: str = "small",
    metadata: dict = None,
    challenge_tags: list[str] = None,
    whisper_transcriber=None,
) -> EvalResult:
    """
    Evaluate Whisper transcription of a single audio file against ground truth.
    
    Args:
        audio_path: Path to audio file
        ground_truth_path: Path to ground truth lyrics (plain text)
        sample_id: Identifier for this sample
        model_size: Whisper model size to use
        metadata: Optional dict with artist, song, etc.
        challenge_tags: List of challenge categories this sample tests
        whisper_transcriber: Optional pre-loaded WhisperTranscriber instance
    """
    metadata = metadata or {}
    challenge_tags = challenge_tags or []

    # Load ground truth
    gt_text = Path(ground_truth_path).read_text(encoding="utf-8").strip()
    ref_tokens = tokenize(gt_text)

    # Transcribe
    if whisper_transcriber is None:
        from src.inference.whisper_engine import WhisperTranscriber
        whisper_transcriber = WhisperTranscriber(model_size=model_size)

    start = time.time()
    result = whisper_transcriber.transcribe(audio_path)
    elapsed = time.time() - start

    hyp_text = result.get("text", "") if isinstance(result, dict) else str(result)
    hyp_tokens = tokenize(hyp_text)

    # Compute WER + alignment
    wer_result = compute_wer(ref_tokens, hyp_tokens)

    # Categorize errors
    error_cats = categorize_errors(wer_result["alignment"], challenge_tags)

    return EvalResult(
        sample_id=sample_id,
        artist=metadata.get("artist", ""),
        song=metadata.get("song", ""),
        model_size=model_size,
        duration_seconds=metadata.get("duration_seconds"),
        processing_time=round(elapsed, 2),
        wer=wer_result["wer"],
        total_ref_words=wer_result["total_ref_words"],
        total_errors=wer_result["total_errors"],
        substitutions=wer_result["substitutions"],
        insertions=wer_result["insertions"],
        deletions=wer_result["deletions"],
        error_categories={k: v for k, v in error_cats.items()},
        challenge_tags=challenge_tags,
        reference_text=gt_text,
        hypothesis_text=hyp_text,
        alignment=wer_result["alignment"],
    )


# ---------------------------------------------------------------------------
# Run full evaluation from manifest
# ---------------------------------------------------------------------------

def run_evaluation(manifest_path: str, model_size: str = "small") -> list[EvalResult]:
    """Run evaluation on all samples in a test manifest."""
    manifest = json.loads(Path(manifest_path).read_text())
    base_dir = Path(manifest_path).parent
    results = []

    # Lazy-load transcriber once
    from src.inference.whisper_engine import WhisperTranscriber
    transcriber = WhisperTranscriber(model_size=model_size)

    for sample in manifest["samples"]:
        audio_path = base_dir / sample["audio_file"]
        truth_path = base_dir / sample["ground_truth_file"]

        if not audio_path.exists():
            print(f"  SKIP {sample['id']}: audio not found at {audio_path}")
            continue
        if not truth_path.exists():
            print(f"  SKIP {sample['id']}: ground truth not found at {truth_path}")
            continue

        print(f"  Evaluating {sample['id']}...", end=" ", flush=True)
        result = evaluate_sample(
            audio_path=str(audio_path),
            ground_truth_path=str(truth_path),
            sample_id=sample["id"],
            model_size=model_size,
            metadata=sample.get("metadata", {}),
            challenge_tags=sample.get("challenge_tags", []),
            whisper_transcriber=transcriber,
        )
        print(f"WER={result.wer:.1%}")
        results.append(result)

    return results


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(results: list[EvalResult], output_path: str = None) -> str:
    """Generate a human-readable evaluation report."""
    lines = []
    lines.append("=" * 70)
    lines.append("RAP TRANSCRIPTION EVALUATION REPORT")
    lines.append("=" * 70)
    lines.append("")

    if not results:
        lines.append("No results to report.")
        report = "\n".join(lines)
        if output_path:
            Path(output_path).write_text(report)
        return report

    # Overall summary
    avg_wer = sum(r.wer for r in results) / len(results)
    total_errors = sum(r.total_errors for r in results)
    total_words = sum(r.total_ref_words for r in results)
    model = results[0].model_size

    lines.append(f"Model: whisper-{model}")
    lines.append(f"Samples evaluated: {len(results)}")
    lines.append(f"Average WER: {avg_wer:.1%}")
    lines.append(f"Total errors: {total_errors} / {total_words} words")
    lines.append("")

    # Per-sample breakdown
    lines.append("-" * 70)
    lines.append("PER-SAMPLE RESULTS")
    lines.append("-" * 70)
    for r in sorted(results, key=lambda x: x.wer, reverse=True):
        lines.append("")
        label = f"{r.artist} - {r.song}" if r.artist else r.sample_id
        lines.append(f"  {label}")
        lines.append(f"    WER: {r.wer:.1%} ({r.total_errors} errors / {r.total_ref_words} words)")
        lines.append(f"    Subs: {r.substitutions}  Ins: {r.insertions}  Del: {r.deletions}")
        lines.append(f"    Processing: {r.processing_time}s")
        if r.challenge_tags:
            lines.append(f"    Challenge tags: {', '.join(r.challenge_tags)}")

        # Error category breakdown
        for cat, examples in r.error_categories.items():
            if examples:
                ex_strs = [f"'{ref}'→'{hyp}'" for ref, hyp in examples[:5]]
                lines.append(f"    {cat} ({len(examples)}): {', '.join(ex_strs)}")

    # Aggregate error analysis
    lines.append("")
    lines.append("-" * 70)
    lines.append("AGGREGATE ERROR ANALYSIS")
    lines.append("-" * 70)

    all_cats = Counter()
    for r in results:
        for cat, examples in r.error_categories.items():
            all_cats[cat] += len(examples)

    lines.append("")
    for cat, count in all_cats.most_common():
        pct = count / max(total_errors, 1) * 100
        bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
        lines.append(f"  {cat:20s} {count:4d} ({pct:5.1f}%) {bar}")

    # Challenge tag analysis
    lines.append("")
    lines.append("-" * 70)
    lines.append("WER BY CHALLENGE TAG")
    lines.append("-" * 70)

    tag_wers = {}
    for r in results:
        for tag in r.challenge_tags:
            tag_wers.setdefault(tag, []).append(r.wer)

    lines.append("")
    for tag, wers in sorted(tag_wers.items(), key=lambda x: sum(x[1]) / len(x[1]), reverse=True):
        avg = sum(wers) / len(wers)
        lines.append(f"  {tag:25s}  avg WER={avg:.1%}  (n={len(wers)})")

    # Worst misses (most common substitution pairs)
    lines.append("")
    lines.append("-" * 70)
    lines.append("TOP SUBSTITUTION ERRORS (what Whisper gets wrong most)")
    lines.append("-" * 70)

    sub_pairs = Counter()
    for r in results:
        for cat in ["slang_miss", "homophone", "other_sub", "truncation"]:
            for ref, hyp in r.error_categories.get(cat, []):
                if ref and hyp:
                    sub_pairs[(ref.lower(), hyp.lower())] += 1

    lines.append("")
    for (ref, hyp), count in sub_pairs.most_common(20):
        lines.append(f"  '{ref}' → '{hyp}'  (x{count})")

    # Recommendations
    lines.append("")
    lines.append("-" * 70)
    lines.append("OPTIMIZATION RECOMMENDATIONS")
    lines.append("-" * 70)
    lines.append("")

    if all_cats.get("slang_miss", 0) > total_errors * 0.2:
        lines.append("  ⚠  HIGH SLANG MISS RATE - Consider building a slang post-processor")
        lines.append("     or fine-tuning Whisper with rap vocabulary.")
    if all_cats.get("hallucination", 0) > total_errors * 0.15:
        lines.append("  ⚠  HIGH HALLUCINATION RATE - Whisper is inserting words.")
        lines.append("     Try: higher temperature, or post-processing to filter low-confidence words.")
    if all_cats.get("dropped_word", 0) > total_errors * 0.2:
        lines.append("  ⚠  HIGH DROP RATE - Whisper is missing words.")
        lines.append("     Likely: fast delivery sections. Try: slower model or segment-level processing.")
    if all_cats.get("adlib_miss", 0) > total_errors * 0.1:
        lines.append("  ⚠  AD-LIB ISSUES - Consider separate ad-lib detection pass.")

    biggest = all_cats.most_common(1)
    if biggest:
        lines.append(f"\n  → Biggest problem area: {biggest[0][0]} ({biggest[0][1]} errors)")
        lines.append(f"    Focus optimization efforts here first.")

    report = "\n".join(lines)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(report, encoding="utf-8")
        print(f"\nReport saved to {output_path}")

    return report


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate Whisper rap transcription accuracy")
    parser.add_argument("--manifest", type=str, help="Path to test_manifest.json")
    parser.add_argument("--audio", type=str, help="Single audio file to evaluate")
    parser.add_argument("--truth", type=str, help="Ground truth lyrics file (with --audio)")
    parser.add_argument("--model", type=str, default="small", help="Whisper model size")
    parser.add_argument("--output", type=str, default=None, help="Save report to file")
    parser.add_argument("--json", type=str, default=None, help="Save raw results as JSON")
    args = parser.parse_args()

    if args.audio and args.truth:
        print(f"\nEvaluating single file: {args.audio}")
        print(f"Model: whisper-{args.model}\n")
        result = evaluate_sample(
            audio_path=args.audio,
            ground_truth_path=args.truth,
            model_size=args.model,
        )
        results = [result]
    elif args.manifest:
        print(f"\nRunning evaluation from manifest: {args.manifest}")
        print(f"Model: whisper-{args.model}\n")
        results = run_evaluation(args.manifest, model_size=args.model)
    else:
        parser.error("Provide either --manifest or both --audio and --truth")
        return

    report = generate_report(results, output_path=args.output)
    print("\n" + report)

    if args.json:
        json_results = []
        for r in results:
            d = asdict(r)
            d.pop("alignment", None)  # Too verbose for JSON
            json_results.append(d)
        Path(args.json).write_text(json.dumps(json_results, indent=2))
        print(f"\nJSON results saved to {args.json}")


if __name__ == "__main__":
    main()
