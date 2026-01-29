"""
Text normalization for rap lyrics.
Cleans and standardizes text for training.
"""
import re
import unicodedata
from typing import List, Dict, Optional, Tuple
from pathlib import Path


class TextNormalizer:
    """
    Normalizes rap lyrics for training.
    
    Handles:
        - Unicode normalization
        - Removing/converting special characters
        - Standardizing whitespace
        - Handling annotations [Verse], [Chorus], etc.
        - Preserving important punctuation
    """
    
    def __init__(
        self,
        lowercase: bool = True,
        remove_annotations: bool = True,
        remove_punctuation: bool = False,
        preserve_apostrophes: bool = True,
        normalize_unicode: bool = True,
        expand_contractions: bool = False
    ):
        """
        Initialize normalizer.
        
        Args:
            lowercase: Convert to lowercase
            remove_annotations: Remove [Verse], [Chorus], etc.
            remove_punctuation: Remove all punctuation
            preserve_apostrophes: Keep apostrophes in contractions
            normalize_unicode: Normalize unicode characters
            expand_contractions: Expand contractions to full forms
        """
        self.lowercase = lowercase
        self.remove_annotations = remove_annotations
        self.remove_punctuation = remove_punctuation
        self.preserve_apostrophes = preserve_apostrophes
        self.normalize_unicode = normalize_unicode
        self.expand_contractions = expand_contractions
        
        # Common contractions
        self.contractions_map = {
            "i'm": "i am",
            "you're": "you are",
            "he's": "he is",
            "she's": "she is",
            "it's": "it is",
            "we're": "we are",
            "they're": "they are",
            "i've": "i have",
            "you've": "you have",
            "we've": "we have",
            "they've": "they have",
            "i'll": "i will",
            "you'll": "you will",
            "he'll": "he will",
            "she'll": "she will",
            "we'll": "we will",
            "they'll": "they will",
            "i'd": "i would",
            "you'd": "you would",
            "he'd": "he would",
            "she'd": "she would",
            "we'd": "we would",
            "they'd": "they would",
            "isn't": "is not",
            "aren't": "are not",
            "wasn't": "was not",
            "weren't": "were not",
            "haven't": "have not",
            "hasn't": "has not",
            "hadn't": "had not",
            "don't": "do not",
            "doesn't": "does not",
            "didn't": "did not",
            "won't": "will not",
            "wouldn't": "would not",
            "can't": "cannot",
            "couldn't": "could not",
            "shouldn't": "should not",
            "ain't": "is not",
            "y'all": "you all",
            "let's": "let us",
            "that's": "that is",
            "what's": "what is",
            "here's": "here is",
            "there's": "there is",
            "who's": "who is",
            "how's": "how is",
            "where's": "where is",
        }
        
        # Unicode replacements
        self.unicode_map = {
            ''': "'",
            ''': "'",
            '"': '"',
            '"': '"',
            '—': '-',
            '–': '-',
            '…': '...',
            '•': '',
            '·': '',
            '\u200b': '',  # Zero-width space
            '\ufeff': '',  # BOM
        }
        
        # Annotation pattern
        self.annotation_pattern = re.compile(r'\[.*?\]')
        
        # Multiple spaces
        self.multi_space_pattern = re.compile(r'\s+')
    
    def normalize(self, text: str) -> str:
        """
        Normalize text.
        
        Args:
            text: Raw text input
            
        Returns:
            Normalized text
        """
        if not text:
            return ""
        
        # Unicode normalization
        if self.normalize_unicode:
            text = self._normalize_unicode(text)
        
        # Remove annotations
        if self.remove_annotations:
            text = self._remove_annotations(text)
        
        # Lowercase
        if self.lowercase:
            text = text.lower()
        
        # Expand contractions
        if self.expand_contractions:
            text = self._expand_contractions(text)
        
        # Handle punctuation
        if self.remove_punctuation:
            text = self._remove_punctuation(text)
        else:
            text = self._clean_punctuation(text)
        
        # Normalize whitespace
        text = self._normalize_whitespace(text)
        
        return text
    
    def _normalize_unicode(self, text: str) -> str:
        """Normalize unicode characters."""
        # Apply manual replacements
        for old, new in self.unicode_map.items():
            text = text.replace(old, new)
        
        # Normalize to NFC form
        text = unicodedata.normalize('NFC', text)
        
        # Remove non-printable characters
        text = ''.join(c for c in text if c.isprintable() or c in '\n\t')
        
        return text
    
    def _remove_annotations(self, text: str) -> str:
        """Remove [Verse], [Chorus], etc."""
        return self.annotation_pattern.sub('', text)
    
    def _expand_contractions(self, text: str) -> str:
        """Expand contractions to full forms."""
        words = text.split()
        expanded = []
        
        for word in words:
            lower_word = word.lower()
            if lower_word in self.contractions_map:
                expanded.append(self.contractions_map[lower_word])
            else:
                expanded.append(word)
        
        return ' '.join(expanded)
    
    def _remove_punctuation(self, text: str) -> str:
        """Remove all punctuation."""
        if self.preserve_apostrophes:
            # Keep apostrophes
            text = re.sub(r"[^\w\s']", '', text)
        else:
            text = re.sub(r'[^\w\s]', '', text)
        return text
    
    def _clean_punctuation(self, text: str) -> str:
        """Clean up punctuation without removing it all."""
        # Standardize quotes
        text = re.sub(r'["""]', '"', text)
        text = re.sub(r"[''']", "'", text)
        
        # Remove excessive punctuation
        text = re.sub(r'([!?.]){2,}', r'\1', text)
        
        # Space around punctuation
        text = re.sub(r'\s*([,;:!?])\s*', r'\1 ', text)
        text = re.sub(r'\s*\.\s*', '. ', text)
        
        return text
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace."""
        # Replace newlines with spaces
        text = text.replace('\n', ' ').replace('\r', ' ')
        
        # Replace tabs with spaces
        text = text.replace('\t', ' ')
        
        # Collapse multiple spaces
        text = self.multi_space_pattern.sub(' ', text)
        
        # Strip
        text = text.strip()
        
        return text
    
    def normalize_batch(self, texts: List[str]) -> List[str]:
        """Normalize multiple texts."""
        return [self.normalize(text) for text in texts]


class LyricsPreprocessor:
    """
    Specialized preprocessor for rap lyrics.
    """
    
    def __init__(self):
        self.normalizer = TextNormalizer(
            lowercase=True,
            remove_annotations=True,
            remove_punctuation=False,
            preserve_apostrophes=True
        )
        
        # Patterns for lyrics-specific cleaning
        self.repeated_line_pattern = re.compile(r'^(.+?)\s*(?:x\s*\d+|\(\s*\d+\s*x?\s*\))$', re.IGNORECASE)
        self.ad_lib_pattern = re.compile(r'\(([^)]+)\)')
    
    def preprocess(self, lyrics: str) -> str:
        """
        Preprocess lyrics for training.
        
        Args:
            lyrics: Raw lyrics text
            
        Returns:
            Preprocessed lyrics
        """
        # Basic normalization
        text = self.normalizer.normalize(lyrics)
        
        # Handle repeated lines (e.g., "hook x4" or "chorus (2x)")
        text = self._expand_repeats(text)
        
        return text
    
    def _expand_repeats(self, text: str) -> str:
        """Expand repeated line indicators."""
        lines = text.split('\n')
        expanded = []
        
        for line in lines:
            match = self.repeated_line_pattern.match(line.strip())
            if match:
                # Just keep the line once (don't actually repeat)
                expanded.append(match.group(1).strip())
            else:
                expanded.append(line)
        
        return ' '.join(expanded)
    
    def extract_ad_libs(self, text: str) -> Tuple[str, List[str]]:
        """
        Extract ad-libs from text.
        
        Args:
            text: Text with ad-libs in parentheses
            
        Returns:
            Tuple of (text without ad-libs, list of ad-libs)
        """
        ad_libs = self.ad_lib_pattern.findall(text)
        text_without = self.ad_lib_pattern.sub('', text)
        text_without = self.normalizer._normalize_whitespace(text_without)
        
        return text_without, ad_libs


class VocabularyBuilder:
    """
    Builds vocabulary from normalized texts.
    """
    
    def __init__(self, min_frequency: int = 2):
        self.min_frequency = min_frequency
        self.word_counts: Dict[str, int] = {}
        self.char_counts: Dict[str, int] = {}
    
    def add_text(self, text: str):
        """Add text to vocabulary counts."""
        # Word counts
        words = text.split()
        for word in words:
            self.word_counts[word] = self.word_counts.get(word, 0) + 1
        
        # Character counts
        for char in text:
            self.char_counts[char] = self.char_counts.get(char, 0) + 1
    
    def add_texts(self, texts: List[str]):
        """Add multiple texts."""
        for text in texts:
            self.add_text(text)
    
    def get_vocabulary(self) -> List[str]:
        """Get vocabulary filtered by frequency."""
        return [
            word for word, count in self.word_counts.items()
            if count >= self.min_frequency
        ]
    
    def get_stats(self) -> Dict:
        """Get vocabulary statistics."""
        vocab = self.get_vocabulary()
        return {
            'total_words': sum(self.word_counts.values()),
            'unique_words': len(self.word_counts),
            'vocabulary_size': len(vocab),
            'unique_characters': len(self.char_counts),
            'most_common_words': sorted(
                self.word_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:20]
        }


if __name__ == "__main__":
    print("✅ TextNormalizer module loaded successfully!")
    print("")
    
    # Test normalizer
    print("Testing TextNormalizer...")
    normalizer = TextNormalizer()
    
    raw_text = """
    [Verse 1]
    I'm finna get this BREAD, yeah!!
    She don't know what's up...
    (ayy) Let's GO!!!
    """
    
    normalized = normalizer.normalize(raw_text)
    print(f"   Raw:        '{raw_text.strip()[:50]}...'")
    print(f"   Normalized: '{normalized}'")
    print("")
    
    # Test lyrics preprocessor
    print("Testing LyricsPreprocessor...")
    preprocessor = LyricsPreprocessor()
    
    lyrics = "[Hook] Yeah yeah yeah (x2)\nI'm on my grind (ayy)"
    processed = preprocessor.preprocess(lyrics)
    print(f"   Raw:       '{lyrics}'")
    print(f"   Processed: '{processed}'")
    print("")
    
    # Test ad-lib extraction
    print("Testing ad-lib extraction...")
    text = "I'm getting money (yeah) every day (skrt skrt)"
    clean, adlibs = preprocessor.extract_ad_libs(text)
    print(f"   Original: '{text}'")
    print(f"   Clean:    '{clean}'")
    print(f"   Ad-libs:  {adlibs}")
    print("")
    
    # Test vocabulary builder
    print("Testing VocabularyBuilder...")
    builder = VocabularyBuilder(min_frequency=1)
    
    sample_texts = [
        "i'm finna get this bread",
        "she bussin it down yeah",
        "i'm on my grind every day"
    ]
    
    for text in sample_texts:
        builder.add_text(text)
    
    stats = builder.get_stats()
    print(f"   Total words: {stats['total_words']}")
    print(f"   Unique words: {stats['unique_words']}")
    print(f"   Top words: {stats['most_common_words'][:5]}")
