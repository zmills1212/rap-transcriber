"""
Tokenizer for rap transcription.
Handles text-to-token and token-to-text conversion.
"""
import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import Counter


class SimpleTokenizer:
    """
    Simple character/word-level tokenizer.
    Good for initial testing before using BPE.
    """
    
    def __init__(
        self,
        vocab_size: int = 8192,
        min_frequency: int = 2,
        special_tokens: List[str] = None
    ):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        
        # Special tokens
        self.special_tokens = special_tokens or [
            '<blank>',  # CTC blank (index 0)
            '<unk>',    # Unknown token
            '<sos>',    # Start of sequence
            '<eos>',    # End of sequence
            '<pad>',    # Padding
        ]
        
        # Initialize vocabulary with special tokens
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        
        for i, token in enumerate(self.special_tokens):
            self.token_to_id[token] = i
            self.id_to_token[i] = token
        
        self.vocab_built = False
    
    def build_vocab(self, texts: List[str], level: str = 'word'):
        """
        Build vocabulary from texts.
        
        Args:
            texts: List of text strings
            level: 'word' or 'char'
        """
        # Count tokens
        counter = Counter()
        
        for text in texts:
            text = self._preprocess(text)
            if level == 'word':
                tokens = text.split()
            else:  # char
                tokens = list(text)
            counter.update(tokens)
        
        # Filter by frequency and limit vocab size
        filtered = [
            (token, count) for token, count in counter.most_common()
            if count >= self.min_frequency
        ]
        
        # Add to vocabulary (leave room for special tokens)
        max_tokens = self.vocab_size - len(self.special_tokens)
        
        for token, _ in filtered[:max_tokens]:
            if token not in self.token_to_id:
                idx = len(self.token_to_id)
                self.token_to_id[token] = idx
                self.id_to_token[idx] = token
        
        self.vocab_built = True
        print(f"Built vocabulary with {len(self.token_to_id)} tokens")
    
    def _preprocess(self, text: str) -> str:
        """Preprocess text for tokenization."""
        # Lowercase
        text = text.lower()
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    
    def encode(self, text: str, add_special: bool = True) -> List[int]:
        """
        Convert text to token IDs.
        
        Args:
            text: Input text
            add_special: Whether to add SOS/EOS tokens
            
        Returns:
            List of token IDs
        """
        text = self._preprocess(text)
        tokens = text.split()
        
        ids = []
        if add_special:
            ids.append(self.token_to_id['<sos>'])
        
        unk_id = self.token_to_id['<unk>']
        for token in tokens:
            ids.append(self.token_to_id.get(token, unk_id))
        
        if add_special:
            ids.append(self.token_to_id['<eos>'])
        
        return ids
    
    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """
        Convert token IDs back to text.
        
        Args:
            ids: List of token IDs
            skip_special: Whether to skip special tokens
            
        Returns:
            Decoded text string
        """
        special_ids = set(range(len(self.special_tokens))) if skip_special else set()
        
        tokens = []
        for idx in ids:
            if idx in special_ids:
                continue
            token = self.id_to_token.get(idx, '<unk>')
            tokens.append(token)
        
        return ' '.join(tokens)
    
    def save(self, path: str):
        """Save tokenizer to file."""
        data = {
            'vocab_size': self.vocab_size,
            'min_frequency': self.min_frequency,
            'special_tokens': self.special_tokens,
            'token_to_id': self.token_to_id
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'SimpleTokenizer':
        """Load tokenizer from file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        tokenizer = cls(
            vocab_size=data['vocab_size'],
            min_frequency=data['min_frequency'],
            special_tokens=data['special_tokens']
        )
        tokenizer.token_to_id = data['token_to_id']
        tokenizer.id_to_token = {int(k): v for k, v in 
                                  {v: k for k, v in data['token_to_id'].items()}.items()}
        # Rebuild id_to_token properly
        tokenizer.id_to_token = {v: k for k, v in tokenizer.token_to_id.items()}
        tokenizer.vocab_built = True
        
        return tokenizer
    
    @property
    def vocab_len(self) -> int:
        return len(self.token_to_id)
    
    @property
    def pad_id(self) -> int:
        return self.token_to_id['<pad>']
    
    @property
    def unk_id(self) -> int:
        return self.token_to_id['<unk>']
    
    @property
    def sos_id(self) -> int:
        return self.token_to_id['<sos>']
    
    @property
    def eos_id(self) -> int:
        return self.token_to_id['<eos>']
    
    @property
    def blank_id(self) -> int:
        return self.token_to_id['<blank>']


class RapTokenizer(SimpleTokenizer):
    """
    Tokenizer specialized for rap lyrics.
    Handles slang, contractions, and ad-libs.
    """
    
    def __init__(self, vocab_size: int = 8192, min_frequency: int = 2):
        super().__init__(vocab_size, min_frequency)
        
        # Common rap-specific tokens to always include
        self.rap_tokens = [
            # Ad-libs
            "yeah", "yuh", "yah", "uh", "ay", "ayy", "skrt", "brr",
            "woo", "sheesh", "gang", "aye",
            # Common slang
            "finna", "gonna", "wanna", "gotta", "tryna", "ima", "imma",
            "ion", "aint", "ain't", "bout", "'bout",
            "bussin", "fire", "lit", "dope", "sick",
            "cap", "facts", "bet", "lowkey", "highkey","deadhomies"
            # People
            "homie", "dawg", "bruh", "fam", "bro", "sis",
            "plug", "opps",
            # Money
            "bread", "bands", "guap", "racks", "stacks","chip",
            "cheddar","cheese",
            # Style
            "drip", "flex", "swag", "ice",
            # Common words
            "the", "a", "an", "is", "are", "was", "were",
            "i", "you", "he", "she", "it", "we", "they",
            "my", "your", "his", "her", "our", "their",
            "and", "but", "or", "so", "if", "when", "that",
            "in", "on", "at", "to", "for", "with", "from",
            "got", "get", "like", "know", "want", "need",
            "come", "go", "make", "take", "see", "look",
            "all", "no", "not", "just", "up", "out",
        ]
    
    def _preprocess(self, text: str) -> str:
        """Preprocess rap lyrics."""
        # Lowercase
        text = text.lower()
        
        # Normalize common variations
        replacements = {
            "'": "'",
            "'": "'",
            """: '"',
            """: '"',
            "…": "...",
            "n***a": "nigga",
            "n****": "nigga",
            "f**k": "fuck",
            "s**t": "shit",
            "b***h": "bitch",
            "a**": "ass",
            "p***y": "pussy",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Remove content in brackets [Verse 1], [Hook], etc.
        text = re.sub(r'\[.*?\]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def build_vocab(self, texts: List[str], level: str = 'word'):
        """Build vocab with rap-specific tokens."""
        # First add rap tokens
        for token in self.rap_tokens:
            if token not in self.token_to_id:
                idx = len(self.token_to_id)
                self.token_to_id[token] = idx
                self.id_to_token[idx] = token
        
        # Then build from texts
        super().build_vocab(texts, level)


class PhonemeTokenizer:
    """
    Tokenizer for ARPAbet phoneme sequences.
    """
    
    def __init__(self):
        # ARPAbet phonemes
        self.phonemes = [
            '<blank>', '<sos>', '<eos>', '<unk>', '<pad>',
            # Vowels
            'AA', 'AA0', 'AA1', 'AA2',
            'AE', 'AE0', 'AE1', 'AE2',
            'AH', 'AH0', 'AH1', 'AH2',
            'AO', 'AO0', 'AO1', 'AO2',
            'AW', 'AW0', 'AW1', 'AW2',
            'AY', 'AY0', 'AY1', 'AY2',
            'EH', 'EH0', 'EH1', 'EH2',
            'ER', 'ER0', 'ER1', 'ER2',
            'EY', 'EY0', 'EY1', 'EY2',
            'IH', 'IH0', 'IH1', 'IH2',
            'IY', 'IY0', 'IY1', 'IY2',
            'OW', 'OW0', 'OW1', 'OW2',
            'OY', 'OY0', 'OY1', 'OY2',
            'UH', 'UH0', 'UH1', 'UH2',
            'UW', 'UW0', 'UW1', 'UW2',
            # Consonants
            'B', 'CH', 'D', 'DH', 'F', 'G', 'HH', 'JH',
            'K', 'L', 'M', 'N', 'NG', 'P', 'R', 'S', 'SH',
            'T', 'TH', 'V', 'W', 'Y', 'Z', 'ZH',
            # Word boundary
            ' ',
        ]
        
        self.token_to_id = {p: i for i, p in enumerate(self.phonemes)}
        self.id_to_token = {i: p for i, p in enumerate(self.phonemes)}
    
    def encode(self, phonemes: List[str], add_special: bool = True) -> List[int]:
        """Convert phoneme list to IDs."""
        ids = []
        if add_special:
            ids.append(self.token_to_id['<sos>'])
        
        unk_id = self.token_to_id['<unk>']
        for p in phonemes:
            ids.append(self.token_to_id.get(p, unk_id))
        
        if add_special:
            ids.append(self.token_to_id['<eos>'])
        
        return ids
    
    def decode(self, ids: List[int], skip_special: bool = True) -> List[str]:
        """Convert IDs back to phonemes."""
        special_ids = {0, 1, 2, 3, 4} if skip_special else set()
        
        phonemes = []
        for idx in ids:
            if idx in special_ids:
                continue
            phonemes.append(self.id_to_token.get(idx, '<unk>'))
        
        return phonemes
    
    @property
    def vocab_size(self) -> int:
        return len(self.phonemes)
    
    @property
    def blank_id(self) -> int:
        return 0


if __name__ == "__main__":
    print("✅ Tokenizer module loaded successfully!")
    print("")
    
    # Test RapTokenizer
    print("Testing RapTokenizer...")
    tokenizer = RapTokenizer(vocab_size=1000)
    
    # Sample rap lyrics for vocab building
    sample_texts = [
        "I'm finna get this bread yeah",
        "She bussin it down ayy",
        "Got my homies with me we gonna make it",
        "Ion know what you mean bruh",
        "Skrt skrt in the coupe yeah",
        "Facts no cap this fire",
    ]
    
    tokenizer.build_vocab(sample_texts)
    
    # Test encoding/decoding
    test_text = "I'm finna get this bread"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    
    print(f"   Original: {test_text}")
    print(f"   Encoded:  {encoded}")
    print(f"   Decoded:  {decoded}")
    print(f"   Vocab size: {tokenizer.vocab_len}")
    print("")
    
    # Test PhonemeTokenizer
    print("Testing PhonemeTokenizer...")
    phoneme_tokenizer = PhonemeTokenizer()
    
    test_phonemes = ['F', 'IH1', 'N', 'AH0']  # "finna"
    encoded_p = phoneme_tokenizer.encode(test_phonemes)
    decoded_p = phoneme_tokenizer.decode(encoded_p)
    
    print(f"   Original: {test_phonemes}")
    print(f"   Encoded:  {encoded_p}")
    print(f"   Decoded:  {decoded_p}")
    print(f"   Vocab size: {phoneme_tokenizer.vocab_size}")
