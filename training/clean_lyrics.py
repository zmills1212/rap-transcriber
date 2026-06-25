"""
Lyrics Cleaner for Rap-Transcriber
===================================
Scans all lyrics .txt files, detects and strips Genius metadata contamination.
Run from project root:
    python -m training.clean_lyrics           # dry run (report only)
    python -m training.clean_lyrics --fix      # apply fixes
    python -m training.clean_lyrics --fix --verbose  # apply + show diffs
"""

import re
import json
import argparse
from pathlib import Path
from difflib import unified_diff

PROJECT_ROOT = Path(__file__).parent.parent
TRAINING_DATA = PROJECT_ROOT / "training_data"


# =============================================================================
# Metadata patterns to strip
# =============================================================================

METADATA_PATTERNS = [
    # Contributor headers: "433 ContributorsTranslationsEspañol..."
    (r'\d+\s*Contributors?(?:Translations?[A-Za-zÀ-ÿА-яёЁ\s]*)?', 'contributor_header'),
    
    # Song title + "Lyrics" suffix: "No Role Modelz Lyrics", "Codeine Crazy Lyrics"
    # Match lines that end with "Lyrics" preceded by title-case words
    (r'(?:^|\n)[A-Z][A-Za-z0-9\s\'\-\(\)]*Lyrics(?:\n|$)', 'lyrics_header'),
    
    # Translation language tags: standalone language names
    (r'(?:Español|Русский\s*Russian|Português|Italiano|Français|Türkçe|azərbaycan|Polski|Deutsch|한국어|日本語|中文)', 'translation_tag'),
    
    # "Read More" markers
    (r'(?:…\s*)?Read More\.?', 'read_more'),
    
    # Genius song descriptions (editorial paragraphs about the song)
    # These typically contain phrases like "the song", "the track", "released", "single", "album"
    # and are written in third person
    (r'(?:^|\n)(?:["\u201c].*?["\u201d]\s+(?:talks about|is about|describes|explores|features|samples|references).*?)(?:\n|$)', 'description_quote'),
    
    # Lines with "released" + date patterns (editorial context)
    (r'(?:^|\n).*?(?:released|dropped)\s+(?:the\s+)?(?:song|track|single|album|video)\s+(?:on|in)\s+\w+\s+\d+.*?(?:\n|$)', 'release_info'),
    
    # Parenthetical production credits
    (r'\(Prod(?:uced)?\.?\s+(?:by\s+)?[^)]+\)', 'prod_credit'),
    
    # Embed markers
    (r'\d*\s*Embed\s*$', 'embed_marker'),
    
    # "You might also like" Genius insertions
    (r'You might also like', 'you_might_like'),
    
    # Verse/section markers from Genius that are malformed
    # (valid ones like [Verse 1] are fine, but broken ones aren't)
    (r'See\s+\w+\s+Live', 'see_live_ad'),
    
    # URLs
    (r'https?://\S+', 'url'),
    
    # "Get tickets as low as $XX" ads
    (r'Get tickets as low as \$\d+', 'ticket_ad'),
]

# Longer editorial detection: if a "sentence" has 3+ of these words,
# it's probably a Genius description, not lyrics
EDITORIAL_KEYWORDS = {
    'released', 'single', 'album', 'billboard', 'chart', 'certified',
    'platinum', 'gold', 'record', 'producer', 'featuring', 'sampled',
    'interpolat', 'remix', 'music video', 'directed', 'debuted',
    'peaked', 'nominated', 'grammy', 'award', 'viral', 'snippet',
    'instagram', 'twitter', 'tiktok', 'streaming', 'spotify',
    'apple music', 'soundcloud', 'youtube', 'contributor', 'annotation',
    'defined by merriam', 'widely regarded', 'physical altercation',
    'star-studded', 'following year', 'teasing a snippet',
}


def detect_editorial_lines(text: str) -> list:
    """Detect lines that read like editorial descriptions, not lyrics."""
    flagged = []
    for i, line in enumerate(text.split('\n')):
        line_lower = line.lower()
        hits = sum(1 for kw in EDITORIAL_KEYWORDS if kw in line_lower)
        # 2+ editorial keywords in one line = almost certainly metadata
        if hits >= 2 and len(line) > 40:
            flagged.append((i, line, f'editorial_line ({hits} keywords)'))
    return flagged


def clean_text(text: str, verbose: bool = False) -> tuple:
    """
    Clean a lyrics string. Returns (cleaned_text, list_of_changes).
    """
    changes = []
    cleaned = text
    
    # Apply regex patterns
    for pattern, label in METADATA_PATTERNS:
        matches = list(re.finditer(pattern, cleaned, re.MULTILINE | re.IGNORECASE))
        for match in reversed(matches):  # reverse to preserve indices
            matched_text = match.group().strip()
            if matched_text:  # don't log empty matches
                changes.append({
                    'type': label,
                    'removed': matched_text[:80] + ('...' if len(matched_text) > 80 else ''),
                    'position': match.start(),
                })
        cleaned = re.sub(pattern, '\n', cleaned, flags=re.MULTILINE | re.IGNORECASE)
    
    # Detect editorial lines
    editorial = detect_editorial_lines(cleaned)
    for line_num, line_text, reason in editorial:
        changes.append({
            'type': reason,
            'removed': line_text[:80] + ('...' if len(line_text) > 80 else ''),
            'position': line_num,
        })
    
    # Remove editorial lines
    if editorial:
        lines = cleaned.split('\n')
        editorial_indices = {item[0] for item in editorial}
        lines = [l for i, l in enumerate(lines) if i not in editorial_indices]
        cleaned = '\n'.join(lines)
    
    # Clean up excessive whitespace
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    cleaned = cleaned.strip()
    
    return cleaned, changes


def scan_lyrics_files(data_dir: Path) -> dict:
    """Scan all lyrics files and return contamination report."""
    report = {
        'total_files': 0,
        'contaminated_files': 0,
        'clean_files': 0,
        'files': {},
    }
    
    # Find all .txt files in training_data
    txt_files = sorted(data_dir.rglob('*.txt'))
    
    for txt_file in txt_files:
        report['total_files'] += 1
        original = txt_file.read_text(encoding='utf-8', errors='replace')
        cleaned, changes = clean_text(original)
        
        if changes:
            report['contaminated_files'] += 1
            report['files'][str(txt_file.relative_to(data_dir))] = {
                'changes': changes,
                'original_len': len(original),
                'cleaned_len': len(cleaned),
                'reduction_pct': round((1 - len(cleaned) / max(len(original), 1)) * 100, 1),
            }
        else:
            report['clean_files'] += 1
    
    return report


def fix_lyrics_files(data_dir: Path, verbose: bool = False) -> dict:
    """Apply fixes to all contaminated lyrics files."""
    report = scan_lyrics_files(data_dir)
    fixed = 0
    
    for rel_path, info in report['files'].items():
        filepath = data_dir / rel_path
        original = filepath.read_text(encoding='utf-8', errors='replace')
        cleaned, changes = clean_text(original)
        
        if verbose:
            diff = unified_diff(
                original.splitlines(keepends=True),
                cleaned.splitlines(keepends=True),
                fromfile=f'original/{rel_path}',
                tofile=f'cleaned/{rel_path}',
                n=1,
            )
            diff_text = ''.join(diff)
            if diff_text:
                print(f"\n{'=' * 60}")
                print(f"  {rel_path}")
                print(f"{'=' * 60}")
                print(diff_text[:500])
                if len(diff_text) > 500:
                    print(f"  ... ({len(diff_text) - 500} more chars)")
        
        # Write cleaned version
        filepath.write_text(cleaned, encoding='utf-8')
        fixed += 1
    
    report['fixed'] = fixed
    return report


def main():
    parser = argparse.ArgumentParser(description='Clean Genius metadata from lyrics files')
    parser.add_argument('--fix', action='store_true', help='Apply fixes (default: dry run)')
    parser.add_argument('--verbose', action='store_true', help='Show diffs')
    parser.add_argument('--data-dir', type=str, default=None, help='Path to training_data dir')
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir) if args.data_dir else TRAINING_DATA
    
    if not data_dir.exists():
        print(f"ERROR: Training data directory not found: {data_dir}")
        return
    
    print(f"{'=' * 60}")
    print(f"  LYRICS CLEANER {'(DRY RUN)' if not args.fix else '(APPLYING FIXES)'}")
    print(f"{'=' * 60}")
    print(f"  Scanning: {data_dir}\n")
    
    if args.fix:
        report = fix_lyrics_files(data_dir, verbose=args.verbose)
        print(f"\n{'=' * 60}")
        print(f"  CLEANUP COMPLETE")
        print(f"{'=' * 60}")
        print(f"  Total files:        {report['total_files']}")
        print(f"  Contaminated:       {report['contaminated_files']}")
        print(f"  Fixed:              {report['fixed']}")
        print(f"  Already clean:      {report['clean_files']}")
        print(f"{'=' * 60}")
    else:
        report = scan_lyrics_files(data_dir)
        print(f"  Total files:        {report['total_files']}")
        print(f"  Contaminated:       {report['contaminated_files']}")
        print(f"  Already clean:      {report['clean_files']}")
        
        if report['files']:
            print(f"\n  Contaminated files:")
            for rel_path, info in report['files'].items():
                print(f"\n    {rel_path}")
                print(f"      Size reduction: {info['reduction_pct']}%")
                for change in info['changes'][:5]:
                    print(f"      [{change['type']}] {change['removed']}")
                if len(info['changes']) > 5:
                    print(f"      ... and {len(info['changes']) - 5} more issues")
        
        print(f"\n  Run with --fix to apply changes.")
        print(f"  Run with --fix --verbose to see diffs.")


if __name__ == '__main__':
    main()
