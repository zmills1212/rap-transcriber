"""
Segment Alignment Auditor for Rap-Transcriber
===============================================
Spot-checks segment audio/lyrics alignment by running baseline Whisper
on a sample of segments and comparing to assigned lyrics.

Flags segments where Whisper's raw output has zero overlap with the
assigned lyrics — indicating misalignment.

Run from project root:
    python -m training.audit_segments                  # audit 20 random segments
    python -m training.audit_segments --n 50           # audit 50 segments
    python -m training.audit_segments --all            # audit everything (slow)
    python -m training.audit_segments --remove-bad     # audit + remove flagged segments
"""

import json
import random
import argparse
import re
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).parent.parent
TRAINING_DATA = PROJECT_ROOT / "training_data"
MANIFEST_PATH = TRAINING_DATA / "segment_manifest.json"


def normalize_text(text: str) -> set:
    """Extract word set from text for overlap comparison."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    words = set(text.split())
    # Remove very common words that don't indicate alignment
    stopwords = {
        'i', 'im', 'a', 'the', 'and', 'in', 'on', 'my', 'me', 'to',
        'it', 'is', 'that', 'you', 'we', 'he', 'she', 'they', 'this',
        'got', 'get', 'like', 'just', 'up', 'no', 'so', 'but', 'all',
        'yeah', 'yuh', 'uh', 'oh', 'ay', 'ayy', 'ooh',
    }
    return words - stopwords


def word_overlap_score(text_a: str, text_b: str) -> float:
    """Jaccard similarity between word sets (0.0 = no overlap, 1.0 = identical)."""
    words_a = normalize_text(text_a)
    words_b = normalize_text(text_b)
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


def check_metadata_in_lyrics(lyrics: str) -> list:
    """Check if lyrics contain obvious Genius metadata."""
    issues = []
    patterns = [
        (r'\d+\s*Contributors?', 'contributor_header'),
        (r'Translations?(?:Español|Русский|Português|Italiano|Français|Türkçe)', 'translation_tags'),
        (r'Read More', 'read_more'),
        (r'Lyrics$', 'lyrics_suffix'),
        (r'You might also like', 'genius_ad'),
        (r'\d*\s*Embed\s*$', 'embed_marker'),
    ]
    for pattern, label in patterns:
        if re.search(pattern, lyrics, re.MULTILINE | re.IGNORECASE):
            issues.append(label)
    return issues


def audit_segments(manifest_path: Path, n_samples: int = 20, 
                   audit_all: bool = False, use_whisper: bool = True) -> dict:
    """
    Audit segment alignment quality.
    
    Phase 1 (fast, no Whisper): Check lyrics for metadata contamination
    Phase 2 (slow, with Whisper): Transcribe audio and compare to assigned lyrics
    """
    
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    segments = manifest if isinstance(manifest, list) else manifest.get('segments', [])
    total = len(segments)
    
    print(f"  Total segments in manifest: {total}")
    
    # =========================================================================
    # Phase 1: Fast metadata scan (all segments)
    # =========================================================================
    print(f"\n  Phase 1: Scanning all {total} segments for metadata...")
    
    metadata_contaminated = []
    empty_lyrics = []
    
    for i, seg in enumerate(segments):
        lyrics = seg.get('lyrics', seg.get('text', ''))
        
        if not lyrics or len(lyrics.strip()) < 10:
            empty_lyrics.append(i)
            continue
        
        issues = check_metadata_in_lyrics(lyrics)
        if issues:
            metadata_contaminated.append({
                'index': i,
                'audio': seg.get('audio_path', seg.get('audio', 'unknown')),
                'issues': issues,
                'lyrics_preview': lyrics[:100],
            })
    
    print(f"    Metadata contaminated: {len(metadata_contaminated)}")
    print(f"    Empty/too-short lyrics: {len(empty_lyrics)}")
    
    # =========================================================================
    # Phase 2: Whisper alignment check (sampled segments)
    # =========================================================================
    alignment_results = []
    
    if use_whisper:
        try:
            import whisper
            import numpy as np
        except ImportError:
            print("\n  Phase 2: SKIPPED (whisper not importable)")
            print("    Install with: pip install openai-whisper")
            use_whisper = False
    
    if use_whisper:
        # Sample segments for Whisper check
        valid_indices = [i for i in range(total) if i not in empty_lyrics]
        if audit_all:
            sample_indices = valid_indices
        else:
            sample_indices = random.sample(valid_indices, min(n_samples, len(valid_indices)))
        
        print(f"\n  Phase 2: Whisper alignment check on {len(sample_indices)} segments...")
        
        model = whisper.load_model("tiny.en")  # fast model for checking
        
        for count, idx in enumerate(sample_indices):
            seg = segments[idx]
            audio_path = seg.get('audio_path', seg.get('audio', ''))
            lyrics = seg.get('lyrics', seg.get('text', ''))
            
            if not Path(audio_path).exists():
                alignment_results.append({
                    'index': idx,
                    'audio': audio_path,
                    'status': 'missing_audio',
                    'score': 0.0,
                })
                continue
            
            try:
                result = model.transcribe(str(audio_path), language='en')
                whisper_text = result['text']
                score = word_overlap_score(whisper_text, lyrics)
                
                alignment_results.append({
                    'index': idx,
                    'audio': str(Path(audio_path).name),
                    'score': round(score, 3),
                    'status': 'good' if score > 0.1 else 'bad' if score < 0.05 else 'marginal',
                    'whisper_preview': whisper_text[:80],
                    'lyrics_preview': lyrics[:80],
                })
                
                if (count + 1) % 10 == 0:
                    print(f"    Checked {count + 1}/{len(sample_indices)}...")
                    
            except Exception as e:
                alignment_results.append({
                    'index': idx,
                    'audio': audio_path,
                    'status': 'error',
                    'score': 0.0,
                    'error': str(e),
                })
    
    # =========================================================================
    # Build report
    # =========================================================================
    bad_segments = [r for r in alignment_results if r['status'] == 'bad']
    marginal_segments = [r for r in alignment_results if r['status'] == 'marginal']
    good_segments = [r for r in alignment_results if r['status'] == 'good']
    missing = [r for r in alignment_results if r['status'] == 'missing_audio']
    
    # Combine all bad segment indices
    all_bad_indices = set()
    all_bad_indices.update(empty_lyrics)
    all_bad_indices.update(seg['index'] for seg in metadata_contaminated)
    all_bad_indices.update(seg['index'] for seg in bad_segments)
    
    report = {
        'total_segments': total,
        'metadata_contaminated': len(metadata_contaminated),
        'metadata_details': metadata_contaminated[:20],
        'empty_lyrics': len(empty_lyrics),
        'whisper_checked': len(alignment_results),
        'alignment_good': len(good_segments),
        'alignment_marginal': len(marginal_segments),
        'alignment_bad': len(bad_segments),
        'missing_audio': len(missing),
        'bad_examples': bad_segments[:10],
        'all_bad_indices': sorted(all_bad_indices),
        'estimated_bad_total': len(all_bad_indices),
    }
    
    return report


def remove_bad_segments(manifest_path: Path, bad_indices: list):
    """Remove bad segments from manifest and save."""
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    segments = manifest if isinstance(manifest, list) else manifest.get('segments', [])
    original_count = len(segments)
    
    bad_set = set(bad_indices)
    cleaned = [seg for i, seg in enumerate(segments) if i not in bad_set]
    
    # Backup original
    backup_path = manifest_path.with_suffix('.json.bak')
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
    parser = argparse.ArgumentParser(description='Audit segment alignment quality')
    parser.add_argument('--n', type=int, default=20, help='Number of segments to spot-check')
    parser.add_argument('--all', action='store_true', help='Check all segments (slow)')
    parser.add_argument('--no-whisper', action='store_true', help='Skip Whisper check (fast metadata scan only)')
    parser.add_argument('--remove-bad', action='store_true', help='Remove flagged segments from manifest')
    parser.add_argument('--manifest', type=str, default=None, help='Path to segment_manifest.json')
    args = parser.parse_args()
    
    manifest_path = Path(args.manifest) if args.manifest else MANIFEST_PATH
    
    if not manifest_path.exists():
        print(f"ERROR: Manifest not found: {manifest_path}")
        return
    
    print(f"{'=' * 60}")
    print(f"  SEGMENT ALIGNMENT AUDITOR")
    print(f"{'=' * 60}")
    print(f"  Manifest: {manifest_path}\n")
    
    report = audit_segments(
        manifest_path, 
        n_samples=args.n,
        audit_all=args.all,
        use_whisper=not args.no_whisper,
    )
    
    # Print report
    print(f"\n{'=' * 60}")
    print(f"  AUDIT RESULTS")
    print(f"{'=' * 60}")
    print(f"  Total segments:          {report['total_segments']}")
    print(f"  Metadata contaminated:   {report['metadata_contaminated']}")
    print(f"  Empty/short lyrics:      {report['empty_lyrics']}")
    
    if report['whisper_checked'] > 0:
        print(f"\n  Whisper alignment check ({report['whisper_checked']} sampled):")
        print(f"    Good (>10% overlap):   {report['alignment_good']}")
        print(f"    Marginal (5-10%):      {report['alignment_marginal']}")
        print(f"    Bad (<5% overlap):     {report['alignment_bad']}")
        print(f"    Missing audio:         {report['missing_audio']}")
    
    print(f"\n  Estimated bad segments:  {report['estimated_bad_total']} / {report['total_segments']}")
    
    if report['bad_examples']:
        print(f"\n  Worst alignment examples:")
        for ex in report['bad_examples'][:5]:
            print(f"    [{ex['index']}] {ex['audio']} (score: {ex['score']})")
            print(f"      Whisper heard: {ex.get('whisper_preview', 'N/A')}")
            print(f"      Assigned lyrics: {ex.get('lyrics_preview', 'N/A')}")
    
    if report['metadata_details']:
        print(f"\n  Metadata contamination examples:")
        for ex in report['metadata_details'][:5]:
            print(f"    [{ex['index']}] {ex['issues']}")
            print(f"      {ex['lyrics_preview']}")
    
    # Remove bad segments if requested
    if args.remove_bad and report['all_bad_indices']:
        print(f"\n  Removing {len(report['all_bad_indices'])} bad segments...")
        orig, cleaned = remove_bad_segments(manifest_path, report['all_bad_indices'])
        print(f"  Manifest: {orig} → {cleaned} segments")
        print(f"  Backup saved: {manifest_path.with_suffix('.json.bak')}")
        print(f"\n  Now rebuild dataset and retrain:")
        print(f"    python -m training.prepare_dataset --segments")
        print(f"    python -m training.fine_tune")
    elif report['all_bad_indices']:
        print(f"\n  Run with --remove-bad to clean the manifest.")
    
    print(f"{'=' * 60}")
    
    # Save full report
    report_path = TRAINING_DATA / "audit_report.json"
    # Convert sets to lists for JSON
    report_json = {k: v for k, v in report.items()}
    with open(report_path, 'w') as f:
        json.dump(report_json, f, indent=2)
    print(f"  Full report saved: {report_path}")


if __name__ == '__main__':
    main()
