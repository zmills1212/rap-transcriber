"""
Model-Based Data Quality Filter for Rap-Transcriber
=====================================================
Uses the current fine-tuned model (or baseline Whisper) to transcribe
every training segment, then compares predictions to assigned lyrics.
Segments where the model's output has no resemblance to the reference
are flagged as misaligned and can be removed.

This is smarter than word-overlap heuristics because the model captures
phonetic and contextual similarity, not just exact word matches.

Run from project root:
    python -m training.model_filter                    # report only
    python -m training.model_filter --remove           # remove bad segments
    python -m training.model_filter --threshold 0.6    # stricter threshold
    python -m training.model_filter --use-baseline     # use base Whisper instead of fine-tuned
"""

import json
import argparse
import re
import sys
import torch
import numpy as np
import librosa
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).parent.parent
TRAINING_DATA = PROJECT_ROOT / "training_data"
MANIFEST_PATH = TRAINING_DATA / "segment_manifest.json"
ADAPTER_DIR = PROJECT_ROOT / "training" / "fine_tuned" / "lora_adapter"
PREDICTION_CACHE = TRAINING_DATA / "model_filter_predictions.json"


def normalize_for_comparison(text: str) -> str:
    """Normalize text for comparison: lowercase, strip punctuation."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def word_overlap_score(text_a: str, text_b: str) -> float:
    """Jaccard similarity between word sets."""
    words_a = set(normalize_for_comparison(text_a).split())
    words_b = set(normalize_for_comparison(text_b).split())
    
    # Remove very common words
    stopwords = {
        'i', 'im', 'a', 'the', 'and', 'in', 'on', 'my', 'me', 'to',
        'it', 'is', 'that', 'you', 'we', 'he', 'she', 'they', 'this',
        'got', 'get', 'like', 'just', 'up', 'no', 'so', 'but', 'all',
        'yeah', 'yuh', 'uh', 'oh', 'ay', 'ayy', 'ooh', 'what',
    }
    words_a -= stopwords
    words_b -= stopwords
    
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


def sequence_overlap_score(text_a: str, text_b: str) -> float:
    """
    Bigram overlap — catches sequential similarity that word sets miss.
    E.g. "I'm a stoner" vs "I'm a stoner" would score high even if
    individual words are common.
    """
    def bigrams(text):
        words = normalize_for_comparison(text).split()
        return set(zip(words, words[1:])) if len(words) > 1 else set()
    
    bg_a = bigrams(text_a)
    bg_b = bigrams(text_b)
    
    if not bg_a or not bg_b:
        return 0.0
    intersection = bg_a & bg_b
    union = bg_a | bg_b
    return len(intersection) / len(union)


def detect_repetition(text: str) -> bool:
    """Detect degenerate repetition loops (model collapse indicator)."""
    words = normalize_for_comparison(text).split()
    if len(words) < 10:
        return False
    
    # Check if any 2-4 word phrase repeats 5+ times
    for n in range(2, 5):
        ngrams = [' '.join(words[i:i+n]) for i in range(len(words) - n + 1)]
        counts = Counter(ngrams)
        if counts and counts.most_common(1)[0][1] >= 5:
            return True
    return False


def combined_score(prediction: str, reference: str) -> dict:
    """
    Compute multiple similarity metrics and a combined quality score.
    Returns dict with individual scores and overall quality assessment.
    """
    word_score = word_overlap_score(prediction, reference)
    seq_score = sequence_overlap_score(prediction, reference)
    is_repetitive = detect_repetition(prediction)
    
    # Combined score: weighted average favoring sequence overlap
    combined = 0.4 * word_score + 0.6 * seq_score
    
    # Penalty for repetitive output (strong signal of misalignment)
    if is_repetitive:
        combined *= 0.3
    
    return {
        'word_overlap': round(word_score, 4),
        'sequence_overlap': round(seq_score, 4),
        'combined': round(combined, 4),
        'is_repetitive': is_repetitive,
    }


def load_model(use_baseline: bool = False, model_size: str = "small"):
    """Load either fine-tuned or baseline Whisper model."""
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    
    model_name = f"openai/whisper-{model_size}.en"
    
    print(f"Loading base model: {model_name}")
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.float32
    )
    
    if not use_baseline and ADAPTER_DIR.exists():
        from peft import PeftModel
        print(f"Loading LoRA adapter from: {ADAPTER_DIR}")
        model = PeftModel.from_pretrained(model, str(ADAPTER_DIR))
        model = model.merge_and_unload()
        print("Adapter merged.")
    elif not use_baseline:
        print("WARNING: No adapter found, using baseline model.")
    else:
        print("Using baseline Whisper (no adapter).")
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    return model, processor, device


def transcribe_segment(model, processor, device, audio_path: str) -> str:
    """Transcribe a single audio segment."""
    try:
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    except Exception as e:
        return f"[ERROR: {e}]"
    
    input_features = processor.feature_extractor(
        audio, sampling_rate=16000, return_tensors="pt"
    ).input_features.to(device)
    
    with torch.no_grad():
        predicted_ids = model.generate(
            input_features,
            max_new_tokens=225,


        )
    
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription.strip()


def run_filter(
    manifest_path: Path,
    threshold: float = 0.08,
    use_baseline: bool = False,
    fresh: bool = False,
):
    """
    Run model-based quality filter on all segments.

    Args:
        manifest_path: Path to segment manifest
        threshold: Combined score below this = bad segment
        use_baseline: Use baseline Whisper instead of fine-tuned
        fresh: Ignore the prediction cache and re-transcribe everything
    """
    with open(manifest_path) as f:
        manifest = json.load(f)

    segments = manifest if isinstance(manifest, list) else manifest.get('segments', [])
    total = len(segments)
    print(f"  Total segments: {total}")

    # Resumable prediction cache: transcription is the expensive step, so cache
    # each segment's model output keyed by audio path. A re-run (after an
    # interruption, or to try a different threshold) reuses cached predictions,
    # and the model is loaded lazily so a fully-cached run skips loading it.
    # NOTE: the cache assumes the same adapter — pass fresh=True after retraining.
    cache = {}
    if not fresh and PREDICTION_CACHE.exists():
        try:
            cache = json.loads(PREDICTION_CACHE.read_text())
            print(f"  Loaded {len(cache)} cached predictions (resume)")
        except Exception:
            cache = {}

    _model = {}  # lazy holder so a fully-cached run never loads the model
    def _predict(audio_path):
        if not _model:
            _model['m'], _model['p'], _model['d'] = load_model(use_baseline=use_baseline)
        return transcribe_segment(_model['m'], _model['p'], _model['d'], audio_path)

    results = []
    good = 0
    bad = 0
    repetitive = 0
    cache_hits = 0
    new_preds = 0

    for i, seg in enumerate(segments):
        audio_path = seg.get('audio_path', seg.get('audio', ''))
        lyrics = seg.get('lyrics', seg.get('text', ''))

        if not Path(audio_path).exists():
            results.append({
                'index': i, 'status': 'missing',
                'audio': audio_path, 'scores': None,
            })
            bad += 1
            continue

        if not lyrics or len(lyrics.strip()) < 10:
            results.append({
                'index': i, 'status': 'empty_lyrics',
                'audio': audio_path, 'scores': None,
            })
            bad += 1
            continue

        # Transcribe (reuse cached prediction when available)
        cached = cache.get(audio_path)
        if cached is not None and not cached.startswith('[ERROR'):
            prediction = cached
            cache_hits += 1
        else:
            prediction = _predict(audio_path)
            if not prediction.startswith('[ERROR'):
                cache[audio_path] = prediction
                new_preds += 1
                if new_preds % 25 == 0:
                    PREDICTION_CACHE.write_text(json.dumps(cache))
        scores = combined_score(prediction, lyrics)
        
        status = 'good' if scores['combined'] >= threshold else 'bad'
        if scores['is_repetitive']:
            repetitive += 1
        
        if status == 'good':
            good += 1
        else:
            bad += 1
        
        results.append({
            'index': i,
            'status': status,
            'audio': Path(audio_path).name,
            'scores': scores,
            'prediction_preview': prediction[:80],
            'lyrics_preview': lyrics[:80],
        })
        
        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{total}] good={good} bad={bad} repetitive={repetitive}")

    # Persist prediction cache for resumability / threshold experiments.
    PREDICTION_CACHE.write_text(json.dumps(cache))
    print(f"  Predictions: {cache_hits} reused, {new_preds} new")

    # Final summary
    bad_indices = [r['index'] for r in results if r['status'] in ('bad', 'missing', 'empty_lyrics')]
    
    report = {
        'total': total,
        'good': good,
        'bad': bad,
        'repetitive_predictions': repetitive,
        'threshold': threshold,
        'bad_indices': bad_indices,
        'worst_examples': sorted(
            [r for r in results if r['status'] == 'bad' and r['scores']],
            key=lambda x: x['scores']['combined']
        )[:20],
        'best_examples': sorted(
            [r for r in results if r['status'] == 'good' and r['scores']],
            key=lambda x: -x['scores']['combined']
        )[:10],
    }
    
    return report


def remove_bad_segments(manifest_path: Path, bad_indices: list):
    """Remove bad segments from manifest."""
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    segments = manifest if isinstance(manifest, list) else manifest.get('segments', [])
    original_count = len(segments)
    
    bad_set = set(bad_indices)
    cleaned = [seg for i, seg in enumerate(segments) if i not in bad_set]
    
    # Backup
    backup_path = manifest_path.with_suffix('.json.bak2')
    with open(backup_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    # Save cleaned
    if isinstance(manifest, list):
        output = cleaned
    else:
        output = dict(manifest)
        output['segments'] = cleaned
    
    with open(manifest_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    return original_count, len(cleaned)


def main():
    parser = argparse.ArgumentParser(description='Model-based data quality filter')
    parser.add_argument('--threshold', type=float, default=0.08,
                        help='Combined score threshold (default: 0.08)')
    parser.add_argument('--remove', action='store_true',
                        help='Remove flagged segments from manifest')
    parser.add_argument('--use-baseline', action='store_true',
                        help='Use baseline Whisper instead of fine-tuned model')
    parser.add_argument('--manifest', type=str, default=None,
                        help='Path to segment_manifest.json')
    parser.add_argument('--fresh', action='store_true',
                        help='Ignore the prediction cache and re-transcribe all segments')
    args = parser.parse_args()
    
    manifest_path = Path(args.manifest) if args.manifest else MANIFEST_PATH
    
    if not manifest_path.exists():
        print(f"ERROR: Manifest not found: {manifest_path}")
        sys.exit(1)
    
    print(f"{'=' * 60}")
    print(f"  MODEL-BASED DATA QUALITY FILTER")
    print(f"{'=' * 60}")
    print(f"  Manifest:   {manifest_path}")
    print(f"  Threshold:  {args.threshold}")
    print(f"  Model:      {'baseline' if args.use_baseline else 'fine-tuned'}")
    print()
    
    report = run_filter(manifest_path, threshold=args.threshold, use_baseline=args.use_baseline, fresh=args.fresh)
    
    print(f"\n{'=' * 60}")
    print(f"  FILTER RESULTS")
    print(f"{'=' * 60}")
    print(f"  Total segments:       {report['total']}")
    print(f"  Good (score >= {args.threshold}):  {report['good']}")
    print(f"  Bad (score < {args.threshold}):   {report['bad']}")
    print(f"  Repetitive outputs:   {report['repetitive_predictions']}")
    print(f"  Survival rate:        {report['good']/report['total']*100:.1f}%")
    
    if report['best_examples']:
        print(f"\n  Best aligned segments:")
        for ex in report['best_examples'][:5]:
            print(f"    [{ex['index']}] {ex['audio']} (score: {ex['scores']['combined']})")
            print(f"      Model:  {ex['prediction_preview']}")
            print(f"      Lyrics: {ex['lyrics_preview']}")
    
    if report['worst_examples']:
        print(f"\n  Worst segments (will be removed):")
        for ex in report['worst_examples'][:10]:
            print(f"    [{ex['index']}] {ex['audio']} (score: {ex['scores']['combined']})")
            rep = " [REPETITIVE]" if ex['scores']['is_repetitive'] else ""
            print(f"      Model:  {ex['prediction_preview']}{rep}")
            print(f"      Lyrics: {ex['lyrics_preview']}")
    
    # Save report
    report_path = TRAINING_DATA / "model_filter_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Report saved: {report_path}")
    
    if args.remove and report['bad_indices']:
        print(f"\n  Removing {len(report['bad_indices'])} bad segments...")
        orig, cleaned = remove_bad_segments(manifest_path, report['bad_indices'])
        print(f"  Manifest: {orig} → {cleaned} segments")
        print(f"  Backup saved: {manifest_path.with_suffix('.json.bak2')}")
        print(f"\n  Next steps:")
        print(f"    python -m training.prepare_dataset --segments")
        print(f"    caffeinate -i python -m training.fine_tune")
    elif report['bad_indices']:
        print(f"\n  Run with --remove to clean the manifest.")
    
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
