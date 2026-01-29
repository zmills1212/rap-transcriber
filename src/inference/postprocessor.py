"""
Post-processing for rap transcription output.
Cleans up raw model output into readable text.
"""
import re
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import json


class TextPostProcessor:
    """
    Post-processes transcription output.
    
    Handles:
        - Capitalization
        - Punctuation restoration
        - Slang normalization
        - Profanity handling
        - Formatting
    """
    
    def __init__(
        self,
        capitalize_sentences: bool = True,
        restore_punctuation: bool = False,
        normalize_slang: bool = False,
        censor_profanity: bool = False
    ):
        self.capitalize_sentences = capitalize_sentences
        self.restore_punctuation = restore_punctuation
        self.normalize_slang = normalize_slang
        self.censor_profanity = censor_profanity
        
        # Common contractions and their expansions
        self.contractions = {
            "ima": "I'm gonna",
            "imma": "I'm gonna",
            "finna": "fixing to",
            "gonna": "going to",
            "wanna": "want to",
            "gotta": "got to",
            "tryna": "trying to",
            "kinda": "kind of",
            "sorta": "sort of",
            "lemme": "let me",
            "gimme": "give me",
            "dunno": "don't know",
            "ion": "I don't",
            "aint": "ain't",
        }
        
        # Words that should always be capitalized
        self.always_capitalize = {
            "i", "i'm", "i'll", "i've", "i'd",
        }
        
        # Common profanity patterns (for optional censoring)
        self.profanity_patterns = [
            (r'\bfuck\b', 'f***'),
            (r'\bshit\b', 's***'),
            (r'\bbitch\b', 'b****'),
            (r'\bass\b', 'a**'),
            (r'\bdamn\b', 'd***'),
            (r'\bhell\b', 'h***'),
            (r'\bnigga\b', 'n****'),
            (r'\bniggas\b', 'n****s'),
        ]
    
    def process(self, text: str) -> str:
        """
        Apply all post-processing steps.
        
        Args:
            text: Raw transcription text
            
        Returns:
            Processed text
        """
        if not text:
            return text
        
        # Clean whitespace
        text = self._clean_whitespace(text)
        
        # Normalize slang if requested
        if self.normalize_slang:
            text = self._normalize_slang(text)
        
        # Capitalize
        if self.capitalize_sentences:
            text = self._capitalize(text)
        
        # Restore punctuation if requested
        if self.restore_punctuation:
            text = self._restore_punctuation(text)
        
        # Censor profanity if requested
        if self.censor_profanity:
            text = self._censor_profanity(text)
        
        return text
    
    def _clean_whitespace(self, text: str) -> str:
        """Remove extra whitespace."""
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        # Trim
        text = text.strip()
        return text
    
    def _capitalize(self, text: str) -> str:
        """Capitalize sentences and special words."""
        if not text:
            return text
        
        # Split into words
        words = text.split()
        
        for i, word in enumerate(words):
            lower_word = word.lower()
            
            # Always capitalize certain words
            if lower_word in self.always_capitalize:
                if lower_word == "i":
                    words[i] = "I"
                elif lower_word.startswith("i'"):
                    words[i] = "I'" + word[2:]
            
            # Capitalize first word
            elif i == 0:
                words[i] = word.capitalize()
        
        return ' '.join(words)
    
    def _normalize_slang(self, text: str) -> str:
        """Expand slang to standard English."""
        words = text.split()
        
        for i, word in enumerate(words):
            lower_word = word.lower()
            if lower_word in self.contractions:
                words[i] = self.contractions[lower_word]
        
        return ' '.join(words)
    
    def _restore_punctuation(self, text: str) -> str:
        """
        Attempt to restore basic punctuation.
        Simple rule-based approach.
        """
        # Add period at end if missing
        if text and text[-1].isalnum():
            text += '.'
        
        # Capitalize after periods
        sentences = text.split('. ')
        sentences = [s.capitalize() for s in sentences]
        text = '. '.join(sentences)
        
        return text
    
    def _censor_profanity(self, text: str) -> str:
        """Replace profanity with censored versions."""
        for pattern, replacement in self.profanity_patterns:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text


class LyricsFormatter:
    """
    Formats transcription output as lyrics.
    
    Handles:
        - Line breaks
        - Verse/chorus detection
        - Ad-lib formatting
    """
    
    def __init__(
        self,
        max_line_length: int = 60,
        detect_structure: bool = True,
        format_adlibs: bool = True
    ):
        self.max_line_length = max_line_length
        self.detect_structure = detect_structure
        self.format_adlibs = format_adlibs
        
        # Common ad-libs
        self.adlibs = {
            'yeah', 'yuh', 'uh', 'ay', 'ayy', 'aye',
            'skrt', 'brr', 'woo', 'sheesh', 'gang',
            'what', 'huh', 'ok', 'okay', 'let\'s go',
            'facts', 'word', 'yo', 'hey'
        }
        
        # Patterns suggesting line breaks
        self.break_patterns = [
            r'(?<=[.!?])\s+',  # After punctuation
            r'(?<=yeah)\s+',   # After "yeah"
            r'(?<=ayy)\s+',    # After "ayy"
        ]
    
    def format(self, text: str) -> str:
        """
        Format text as lyrics.
        
        Args:
            text: Processed transcription text
            
        Returns:
            Formatted lyrics
        """
        if not text:
            return text
        
        # Format ad-libs
        if self.format_adlibs:
            text = self._format_adlibs(text)
        
        # Add line breaks
        lines = self._add_line_breaks(text)
        
        # Detect structure if requested
        if self.detect_structure:
            lines = self._detect_structure(lines)
        
        return '\n'.join(lines)
    
    def _format_adlibs(self, text: str) -> str:
        """Format ad-libs in parentheses."""
        words = text.split()
        result = []
        
        i = 0
        while i < len(words):
            word = words[i]
            lower_word = word.lower().strip('.,!?')
            
            # Check if it's an ad-lib
            if lower_word in self.adlibs:
                # Check if next word is also ad-lib (group them)
                adlib_group = [word]
                j = i + 1
                while j < len(words) and words[j].lower().strip('.,!?') in self.adlibs:
                    adlib_group.append(words[j])
                    j += 1
                
                if len(adlib_group) <= 3:  # Only format short ad-lib sequences
                    result.append(f"({' '.join(adlib_group)})")
                    i = j
                    continue
            
            result.append(word)
            i += 1
        
        return ' '.join(result)
    
    def _add_line_breaks(self, text: str) -> List[str]:
        """Split text into lines."""
        # First try natural break points
        for pattern in self.break_patterns:
            text = re.sub(pattern, '\n', text)
        
        lines = text.split('\n')
        
        # Break long lines
        result = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if len(line) <= self.max_line_length:
                result.append(line)
            else:
                # Break at word boundaries
                words = line.split()
                current_line = []
                current_length = 0
                
                for word in words:
                    if current_length + len(word) + 1 > self.max_line_length and current_line:
                        result.append(' '.join(current_line))
                        current_line = [word]
                        current_length = len(word)
                    else:
                        current_line.append(word)
                        current_length += len(word) + 1
                
                if current_line:
                    result.append(' '.join(current_line))
        
        return result
    
    def _detect_structure(self, lines: List[str]) -> List[str]:
        """Detect and mark verse/chorus structure."""
        # Simple heuristic: look for repeated lines (potential chorus)
        line_counts = {}
        for line in lines:
            normalized = line.lower().strip()
            line_counts[normalized] = line_counts.get(normalized, 0) + 1
        
        # Lines that appear multiple times might be chorus
        repeated_lines = {
            line for line, count in line_counts.items() 
            if count > 1
        }
        
        result = []
        in_chorus = False
        
        for i, line in enumerate(lines):
            normalized = line.lower().strip()
            
            if normalized in repeated_lines and not in_chorus:
                result.append('')
                result.append('[Chorus]')
                in_chorus = True
            elif normalized not in repeated_lines and in_chorus:
                result.append('')
                result.append('[Verse]')
                in_chorus = False
            
            result.append(line)
        
        return result


class TranscriptionCleaner:
    """
    Combines post-processing and formatting.
    """
    
    def __init__(
        self,
        postprocessor: Optional[TextPostProcessor] = None,
        formatter: Optional[LyricsFormatter] = None
    ):
        self.postprocessor = postprocessor or TextPostProcessor()
        self.formatter = formatter or LyricsFormatter()
    
    def clean(
        self,
        text: str,
        format_as_lyrics: bool = False
    ) -> str:
        """
        Clean and optionally format transcription.
        
        Args:
            text: Raw transcription
            format_as_lyrics: Whether to format as lyrics
            
        Returns:
            Cleaned (and optionally formatted) text
        """
        # Post-process
        text = self.postprocessor.process(text)
        
        # Format as lyrics if requested
        if format_as_lyrics:
            text = self.formatter.format(text)
        
        return text


if __name__ == "__main__":
    print("✅ PostProcessor module loaded successfully!")
    print("")
    
    # Test post-processor
    print("Testing TextPostProcessor...")
    processor = TextPostProcessor()
    
    raw_text = "i finna get this bread yeah ima do my thing"
    processed = processor.process(raw_text)
    
    print(f"   Raw:       '{raw_text}'")
    print(f"   Processed: '{processed}'")
    print("")
    
    # Test with slang normalization
    print("Testing slang normalization...")
    processor_slang = TextPostProcessor(normalize_slang=True)
    normalized = processor_slang.process(raw_text)
    print(f"   Normalized: '{normalized}'")
    print("")
    
    # Test lyrics formatter
    print("Testing LyricsFormatter...")
    formatter = LyricsFormatter()
    
    long_text = "Yeah I'm finna get this bread yeah yeah I'm on my grind every day yeah skrt skrt in the coupe yeah"
    formatted = formatter.format(long_text)
    
    print(f"   Input: '{long_text}'")
    print(f"   Formatted:")
    for line in formatted.split('\n'):
        print(f"      {line}")
    print("")
    
    # Test full cleaner
    print("Testing TranscriptionCleaner...")
    cleaner = TranscriptionCleaner()
    
    final = cleaner.clean(long_text, format_as_lyrics=True)
    print(f"   Final output:")
    for line in final.split('\n'):
        print(f"      {line}")
