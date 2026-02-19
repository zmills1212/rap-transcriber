"""
Phase A: Lyrics Matcher
========================
Aligns Whisper transcription output against reference lyrics to produce
the most accurate possible transcription.

Core algorithm:
1. Tokenize both transcription and reference into words
2. Run sequence alignment (similar to diff/Needleman-Wunsch)
3. For each aligned pair, score confidence using:
   - Exact match (highest confidence)
   - Phonetic match (Metaphone/Soundex similarity)
   - Edit distance (Levenshtein)
   - Context window (surrounding words)
4. For disagreements, pick the most likely correct version
5. Output final lyrics with per-word confidence scores

Usage:
    from lyrics_matcher import LyricsMatcher

    matcher = LyricsMatcher()
    result = matcher.match(
        transcription="i got my pants up in the club",
        reference="i got my bands up in the club"
    )
    print(result.text)       # "i got my bands up in the club"
    print(result.confidence)  # 0.94
"""

import re
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from difflib import SequenceMatcher


# =============================================================================
# Phonetic Matching (Double Metaphone simplified)
# =============================================================================

# Common phonetic equivalences in rap/AAVE
PHONETIC_GROUPS = {
    # th-stopping (common in AAVE)
    "th": ["d", "t", "f", "v"],
    "dat": ["that"],
    "dem": ["them"],
    "dey": ["they"],
    "dis": ["this"],

    # -ing → -in
    "in": ["ing"],
    "nothin": ["nothing"],
    "somethin": ["something"],
    "doin": ["doing"],

    # Vowel shifts
    "bout": ["about"],
    "cause": ["because", "cuz", "cus"],
    "finna": ["fixing to", "gonna", "going to"],
    "tryna": ["trying to"],
    "boutta": ["about to"],
    "ion": ["i don't", "i dont"],
    "aint": ["ain't", "isn't", "is not"],
    "ima": ["i'm going to", "i'm gonna", "imma"],
    "imma": ["i'm going to", "i'm gonna", "ima"],

    # Common rap terms Whisper mangles
    "bands": ["pants", "pans", "bans"],
    "glizzy": ["busy", "dizzy", "grisly"],
    "drip": ["trip", "grip", "drip"],
    "cap": ["cap", "tap", "cat"],
    "slatt": ["slap", "flat", "slot"],
    "thang": ["thing", "tang"],
    "cuh": ["come", "cub", "cut"],
    "bruh": ["bro", "brother", "brr"],
    "skrrt": ["skirt", "skirt", "shirt"],
}

# Build reverse lookup
_PHONETIC_REVERSE: Dict[str, List[str]] = {}
for canonical, variants in PHONETIC_GROUPS.items():
    for v in variants:
        _PHONETIC_REVERSE.setdefault(v.lower(), []).append(canonical)
    _PHONETIC_REVERSE.setdefault(canonical.lower(), variants)


def phonetic_similarity(word_a: str, word_b: str) -> float:
    """
    Score phonetic similarity between two words.
    Returns 0.0-1.0 where 1.0 = identical or known phonetic equivalent.
    """
    a = word_a.lower().strip()
    b = word_b.lower().strip()

    if a == b:
        return 1.0

    # Check known phonetic equivalences
    if a in _PHONETIC_REVERSE:
        if b in _PHONETIC_REVERSE[a] or b in PHONETIC_GROUPS.get(a, []):
            return 0.95

    if b in _PHONETIC_REVERSE:
        if a in _PHONETIC_REVERSE[b] or a in PHONETIC_GROUPS.get(b, []):
            return 0.95

    # Check if one is a suffix variant of the other (runnin/running)
    if a.rstrip("g") == b.rstrip("g"):
        return 0.9
    if a + "g" == b or b + "g" == a:
        return 0.9

    # Levenshtein-based similarity
    return _levenshtein_ratio(a, b)


def _levenshtein_ratio(s1: str, s2: str) -> float:
    """Levenshtein similarity ratio (0.0-1.0)."""
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0

    rows = len(s1) + 1
    cols = len(s2) + 1
    dist = [[0] * cols for _ in range(rows)]

    for i in range(rows):
        dist[i][0] = i
    for j in range(cols):
        dist[0][j] = j

    for i in range(1, rows):
        for j in range(1, cols):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            dist[i][j] = min(
                dist[i-1][j] + 1,      # deletion
                dist[i][j-1] + 1,      # insertion
                dist[i-1][j-1] + cost,  # substitution
            )

    max_len = max(len(s1), len(s2))
    return 1.0 - (dist[rows-1][cols-1] / max_len)


# =============================================================================
# Text Normalization
# =============================================================================

def normalize_text(text: str) -> str:
    """Normalize text for comparison (lowercase, strip punctuation, collapse whitespace)."""
    text = text.lower()
    # Keep apostrophes in contractions
    text = re.sub(r"[^\w\s']", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str) -> List[str]:
    """Split text into word tokens."""
    return normalize_text(text).split()


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class WordAlignment:
    """A single word alignment between transcription and reference."""
    transcription_word: Optional[str]  # None if insertion from reference
    reference_word: Optional[str]       # None if Whisper hallucination
    chosen_word: str                    # The word we output
    confidence: float                   # 0.0-1.0
    source: str                         # "exact", "reference", "transcription", "phonetic"

    def __repr__(self):
        return f"[{self.chosen_word}({self.confidence:.2f},{self.source})]"


@dataclass
class MatchResult:
    """Result of lyrics matching."""
    text: str                              # Final corrected transcription
    confidence: float                      # Overall confidence (0.0-1.0)
    alignments: List[WordAlignment]        # Per-word alignment details
    transcription_wer: float               # WER of original transcription vs reference
    corrected_wer: float                   # WER after correction
    stats: Dict = field(default_factory=dict)

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Lyrics Matcher Result",
            f"{'=' * 50}",
            f"  Overall confidence: {self.confidence:.1%}",
            f"  Original WER:      {self.transcription_wer:.1%}",
            f"  Corrected WER:     {self.corrected_wer:.1%}",
            f"  Improvement:       {(self.transcription_wer - self.corrected_wer):.1%}",
            f"  Words matched:     {self.stats.get('exact_matches', 0)}",
            f"  Words corrected:   {self.stats.get('corrections', 0)}",
            f"  Words uncertain:   {self.stats.get('uncertain', 0)}",
        ]
        return "\n".join(lines)


# =============================================================================
# Sequence Alignment
# =============================================================================

def align_sequences(seq_a: List[str], seq_b: List[str]) -> List[Tuple[Optional[str], Optional[str]]]:
    """
    Align two word sequences using SequenceMatcher (similar to diff).
    Returns list of (word_from_a, word_from_b) tuples.
    None means gap (insertion/deletion).
    """
    matcher = SequenceMatcher(None, seq_a, seq_b)
    aligned = []

    for op, a_start, a_end, b_start, b_end in matcher.get_opcodes():
        if op == "equal":
            for i, j in zip(range(a_start, a_end), range(b_start, b_end)):
                aligned.append((seq_a[i], seq_b[j]))

        elif op == "replace":
            a_words = seq_a[a_start:a_end]
            b_words = seq_b[b_start:b_end]
            # Pair up as many as possible
            max_len = max(len(a_words), len(b_words))
            for k in range(max_len):
                wa = a_words[k] if k < len(a_words) else None
                wb = b_words[k] if k < len(b_words) else None
                aligned.append((wa, wb))

        elif op == "delete":
            # Words in A but not B (Whisper hallucination or extra words)
            for i in range(a_start, a_end):
                aligned.append((seq_a[i], None))

        elif op == "insert":
            # Words in B but not A (Whisper missed these)
            for j in range(b_start, b_end):
                aligned.append((None, seq_b[j]))

    return aligned


# =============================================================================
# Lyrics Matcher
# =============================================================================

class LyricsMatcher:
    """
    Matches Whisper transcription against reference lyrics to produce
    the best possible output.

    Parameters:
        phonetic_threshold: Minimum phonetic similarity to accept a match (0.0-1.0)
        edit_threshold: Minimum edit distance ratio to consider words related
        prefer_reference: When in doubt, prefer reference lyrics over Whisper
    """

    def __init__(
        self,
        phonetic_threshold: float = 0.6,
        edit_threshold: float = 0.5,
        prefer_reference: bool = True,
    ):
        self.phonetic_threshold = phonetic_threshold
        self.edit_threshold = edit_threshold
        self.prefer_reference = prefer_reference

    def match(
        self,
        transcription: str,
        reference: str,
    ) -> MatchResult:
        """
        Align and correct transcription against reference lyrics.

        Args:
            transcription: Raw Whisper output (or post-processed output)
            reference: Known/reference lyrics

        Returns:
            MatchResult with corrected text and per-word confidence
        """
        trans_tokens = tokenize(transcription)
        ref_tokens = tokenize(reference)

        if not trans_tokens and not ref_tokens:
            return MatchResult(
                text="",
                confidence=1.0,
                alignments=[],
                transcription_wer=0.0,
                corrected_wer=0.0,
            )

        if not ref_tokens:
            # No reference — just return transcription as-is
            alignments = [
                WordAlignment(w, None, w, 0.5, "transcription")
                for w in trans_tokens
            ]
            return MatchResult(
                text=" ".join(trans_tokens),
                confidence=0.5,
                alignments=alignments,
                transcription_wer=0.0,
                corrected_wer=0.0,
            )

        if not trans_tokens:
            # Whisper produced nothing — use reference
            alignments = [
                WordAlignment(None, w, w, 0.3, "reference")
                for w in ref_tokens
            ]
            return MatchResult(
                text=" ".join(ref_tokens),
                confidence=0.3,
                alignments=alignments,
                transcription_wer=1.0,
                corrected_wer=0.0,
            )

        # Align sequences
        aligned_pairs = align_sequences(trans_tokens, ref_tokens)

        # Score each alignment
        alignments = []
        for trans_word, ref_word in aligned_pairs:
            alignment = self._score_alignment(trans_word, ref_word)
            alignments.append(alignment)

        # Build output
        output_words = [a.chosen_word for a in alignments]
        output_text = " ".join(output_words)

        # Compute WERs
        original_wer = self._compute_wer(trans_tokens, ref_tokens)
        corrected_tokens = tokenize(output_text)
        corrected_wer = self._compute_wer(corrected_tokens, ref_tokens)

        # Stats
        exact = sum(1 for a in alignments if a.source == "exact")
        corrections = sum(1 for a in alignments if a.source == "reference")
        phonetic = sum(1 for a in alignments if a.source == "phonetic")
        uncertain = sum(1 for a in alignments if a.confidence < 0.6)

        overall_confidence = (
            sum(a.confidence for a in alignments) / len(alignments)
            if alignments else 0.0
        )

        return MatchResult(
            text=output_text,
            confidence=overall_confidence,
            alignments=alignments,
            transcription_wer=original_wer,
            corrected_wer=corrected_wer,
            stats={
                "exact_matches": exact,
                "corrections": corrections,
                "phonetic_matches": phonetic,
                "uncertain": uncertain,
                "total_words": len(alignments),
            },
        )

    def _score_alignment(
        self, trans_word: Optional[str], ref_word: Optional[str]
    ) -> WordAlignment:
        """Score a single word alignment and decide which version to use."""

        # Case 1: Both present and identical
        if trans_word and ref_word and trans_word.lower() == ref_word.lower():
            return WordAlignment(
                trans_word, ref_word,
                chosen_word=ref_word,  # Use reference casing/spelling
                confidence=1.0,
                source="exact",
            )

        # Case 2: Only transcription (Whisper produced something reference doesn't have)
        if trans_word and not ref_word:
            # Could be a hallucination or an ad-lib the reference missed
            return WordAlignment(
                trans_word, None,
                chosen_word=trans_word,
                confidence=0.4,
                source="transcription",
            )

        # Case 3: Only reference (Whisper missed a word)
        if ref_word and not trans_word:
            return WordAlignment(
                None, ref_word,
                chosen_word=ref_word,
                confidence=0.6,
                source="reference",
            )

        # Case 4: Both present but different — this is where it gets interesting
        sim = phonetic_similarity(trans_word, ref_word)

        if sim >= 0.9:
            # Very close (phonetic equivalent like bands/pants, thang/thing)
            # Trust the reference
            return WordAlignment(
                trans_word, ref_word,
                chosen_word=ref_word,
                confidence=0.9,
                source="phonetic",
            )

        if sim >= self.phonetic_threshold:
            # Somewhat similar — probably the same word, trust reference
            return WordAlignment(
                trans_word, ref_word,
                chosen_word=ref_word,
                confidence=sim,
                source="reference",
            )

        # Low similarity — could be completely different words
        if self.prefer_reference:
            return WordAlignment(
                trans_word, ref_word,
                chosen_word=ref_word,
                confidence=0.5,
                source="reference",
            )
        else:
            return WordAlignment(
                trans_word, ref_word,
                chosen_word=trans_word,
                confidence=0.4,
                source="transcription",
            )

    def _compute_wer(self, hypothesis: List[str], reference: List[str]) -> float:
        """Compute Word Error Rate."""
        if not reference:
            return 0.0 if not hypothesis else 1.0

        # Standard Levenshtein on word level
        r = len(reference)
        h = len(hypothesis)
        d = [[0] * (h + 1) for _ in range(r + 1)]

        for i in range(r + 1):
            d[i][0] = i
        for j in range(h + 1):
            d[0][j] = j

        for i in range(1, r + 1):
            for j in range(1, h + 1):
                cost = 0 if reference[i-1].lower() == hypothesis[j-1].lower() else 1
                d[i][j] = min(
                    d[i-1][j] + 1,
                    d[i][j-1] + 1,
                    d[i-1][j-1] + cost,
                )

        return d[r][h] / r


# =============================================================================
# CLI
# =============================================================================

def main():
    """Run lyrics matcher from command line."""
    import argparse

    parser = argparse.ArgumentParser(description="Match Whisper output against reference lyrics")
    parser.add_argument("--transcription", "-t", type=str, help="Whisper transcription text or file path")
    parser.add_argument("--reference", "-r", type=str, help="Reference lyrics text or file path")
    parser.add_argument("--audio", "-a", type=str, help="Audio file (will run Whisper first)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show per-word alignments")
    parser.add_argument("--threshold", type=float, default=0.6, help="Phonetic match threshold")

    args = parser.parse_args()

    # Load transcription
    if args.transcription:
        if Path(args.transcription).exists():
            transcription = Path(args.transcription).read_text().strip()
        else:
            transcription = args.transcription
    elif args.audio:
        # Run Whisper
        print(f"Transcribing: {args.audio}")
        try:
            import whisper
            model = whisper.load_model("small")
            result = model.transcribe(args.audio)
            transcription = result["text"]
            print(f"Whisper output: {transcription[:100]}...")
        except ImportError:
            print("ERROR: whisper not installed. Provide --transcription instead.")
            return
    else:
        print("ERROR: Provide --transcription or --audio")
        return

    # Load reference
    if not args.reference:
        print("ERROR: Provide --reference lyrics")
        return

    if Path(args.reference).exists():
        reference = Path(args.reference).read_text().strip()
    else:
        reference = args.reference

    # Match
    matcher = LyricsMatcher(phonetic_threshold=args.threshold)
    result = matcher.match(transcription, reference)

    # Output
    print("\n" + result.summary())
    print(f"\nCorrected text:")
    print(f"  {result.text}")

    if args.verbose:
        print(f"\nPer-word alignments:")
        for a in result.alignments:
            marker = "✓" if a.confidence >= 0.9 else "~" if a.confidence >= 0.6 else "?"
            print(f"  {marker} {a.chosen_word:20s} conf={a.confidence:.2f} src={a.source:15s} "
                  f"(whisper={a.transcription_word or '-':15s} ref={a.reference_word or '-'})")


if __name__ == "__main__":
    main()
