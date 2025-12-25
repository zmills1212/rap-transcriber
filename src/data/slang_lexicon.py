"""
Slang lexicon for rap transcription.
Maps slang words to pronunciations and definitions.
"""
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json


class SlangLexicon:
    """
    Manages slang vocabulary with pronunciations and variants.
    """
    
    def __init__(self, lexicon_path: Optional[str] = None):
        self.lexicon: Dict[str, Dict] = {}
        
        # Load starter lexicon
        self._load_starter_lexicon()
        
        # Load custom lexicon if provided
        if lexicon_path:
            self.load_from_file(lexicon_path)
    
    def _load_starter_lexicon(self):
        """Load built-in starter slang vocabulary."""
        
        # Core rap slang with ARPAbet pronunciations
        starter = {
            # Common contractions/slang
            "finna": {
                "canonical": ["F", "IH1", "N", "AH0"],
                "variants": [["F", "IH1", "N"]],
                "definition": "going to, about to",
                "examples": ["I'm finna go to the store"]
            },
            "gonna": {
                "canonical": ["G", "AH1", "N", "AH0"],
                "variants": [["G", "AA1", "N", "AH0"]],
                "definition": "going to",
                "examples": ["I'm gonna make it"]
            },
            "wanna": {
                "canonical": ["W", "AA1", "N", "AH0"],
                "variants": [],
                "definition": "want to",
                "examples": ["I wanna be the best"]
            },
            "gotta": {
                "canonical": ["G", "AA1", "T", "AH0"],
                "variants": [],
                "definition": "got to, have to",
                "examples": ["I gotta go"]
            },
            "tryna": {
                "canonical": ["T", "R", "AY1", "N", "AH0"],
                "variants": [["T", "R", "AH0", "N", "AH0"]],
                "definition": "trying to",
                "examples": ["I'm tryna get paid"]
            },
            "ion": {
                "canonical": ["AY1", "OW0", "N"],
                "variants": [["AY1", "AH0", "N"]],
                "definition": "I don't",
                "examples": ["Ion know what you mean"]
            },
            "imma": {
                "canonical": ["IH1", "M", "AH0"],
                "variants": [["AY1", "M", "AH0"]],
                "definition": "I'm going to",
                "examples": ["Imma do it my way"]
            },
            "lemme": {
                "canonical": ["L", "EH1", "M", "IY0"],
                "variants": [],
                "definition": "let me",
                "examples": ["Lemme show you"]
            },
            "gimme": {
                "canonical": ["G", "IH1", "M", "IY0"],
                "variants": [],
                "definition": "give me",
                "examples": ["Gimme the mic"]
            },
            
            # Slang adjectives/expressions
            "bussin": {
                "canonical": ["B", "AH1", "S", "IH0", "N"],
                "variants": [["B", "UH1", "S", "N"]],
                "definition": "really good, excellent",
                "examples": ["This beat bussin"]
            },
            "fire": {
                "canonical": ["F", "AY1", "ER0"],
                "variants": [["F", "AY1", "R"]],
                "definition": "amazing, excellent",
                "examples": ["That track is fire"]
            },
            "lit": {
                "canonical": ["L", "IH1", "T"],
                "variants": [],
                "definition": "exciting, excellent",
                "examples": ["The party was lit"]
            },
            "dope": {
                "canonical": ["D", "OW1", "P"],
                "variants": [],
                "definition": "cool, excellent",
                "examples": ["That's dope"]
            },
            "sick": {
                "canonical": ["S", "IH1", "K"],
                "variants": [],
                "definition": "cool, impressive",
                "examples": ["That verse was sick"]
            },
            "lowkey": {
                "canonical": ["L", "OW1", "K", "IY1"],
                "variants": [],
                "definition": "kind of, secretly",
                "examples": ["I lowkey love this song"]
            },
            "highkey": {
                "canonical": ["H", "AY1", "K", "IY1"],
                "variants": [],
                "definition": "very much, openly",
                "examples": ["I highkey need this"]
            },
            "deadass": {
                "canonical": ["D", "EH1", "D", "AE2", "S"],
                "variants": [],
                "definition": "seriously, for real",
                "examples": ["I'm deadass tired"]
            },
            "cap": {
                "canonical": ["K", "AE1", "P"],
                "variants": [],
                "definition": "lie, false statement",
                "examples": ["No cap, that's the truth"]
            },
            "facts": {
                "canonical": ["F", "AE1", "K", "T", "S"],
                "variants": [["F", "AE1", "K", "S"]],
                "definition": "truth, agreement",
                "examples": ["Facts, you right"]
            },
            "bet": {
                "canonical": ["B", "EH1", "T"],
                "variants": [],
                "definition": "okay, agreement",
                "examples": ["Bet, let's do it"]
            },
            
            # People/roles
            "homie": {
                "canonical": ["H", "OW1", "M", "IY0"],
                "variants": [],
                "definition": "friend",
                "examples": ["That's my homie"]
            },
            "dawg": {
                "canonical": ["D", "AO1", "G"],
                "variants": [],
                "definition": "friend, buddy",
                "examples": ["What's up dawg"]
            },
            "bruh": {
                "canonical": ["B", "R", "AH1"],
                "variants": [["B", "R", "UH1"]],
                "definition": "brother, friend",
                "examples": ["Bruh, that's crazy"]
            },
            "fam": {
                "canonical": ["F", "AE1", "M"],
                "variants": [],
                "definition": "family, close friends",
                "examples": ["What's good fam"]
            },
            "plug": {
                "canonical": ["P", "L", "AH1", "G"],
                "variants": [],
                "definition": "connection, supplier",
                "examples": ["He's the plug"]
            },
            "opps": {
                "canonical": ["AA1", "P", "S"],
                "variants": [],
                "definition": "opponents, enemies",
                "examples": ["Watch out for the opps"]
            },
            
            # Money/success
            "bread": {
                "canonical": ["B", "R", "EH1", "D"],
                "variants": [],
                "definition": "money",
                "examples": ["I'm getting this bread"]
            },
            "bands": {
                "canonical": ["B", "AE1", "N", "D", "Z"],
                "variants": [],
                "definition": "money (thousands)",
                "examples": ["I got bands"]
            },
            "guap": {
                "canonical": ["G", "W", "AA1", "P"],
                "variants": [],
                "definition": "money",
                "examples": ["Counting guap"]
            },
            "racks": {
                "canonical": ["R", "AE1", "K", "S"],
                "variants": [],
                "definition": "money (thousands)",
                "examples": ["Racks on racks"]
            },
            "drip": {
                "canonical": ["D", "R", "IH1", "P"],
                "variants": [],
                "definition": "style, fashion",
                "examples": ["Check out my drip"]
            },
            "flex": {
                "canonical": ["F", "L", "EH1", "K", "S"],
                "variants": [],
                "definition": "show off",
                "examples": ["Don't flex on me"]
            },
            
            # Ad-libs
            "skrt": {
                "canonical": ["S", "K", "ER1", "T"],
                "variants": [["S", "K", "R", "T"]],
                "definition": "ad-lib (car sounds)",
                "examples": ["Skrt skrt"]
            },
            "ayy": {
                "canonical": ["EY1"],
                "variants": [["AY1"]],
                "definition": "ad-lib (excitement)",
                "examples": ["Ayy, let's go"]
            },
            "yuh": {
                "canonical": ["Y", "AH1"],
                "variants": [["Y", "UH1"]],
                "definition": "ad-lib (yeah)",
                "examples": ["Yuh yuh"]
            },
            "sheesh": {
                "canonical": ["SH", "IY1", "SH"],
                "variants": [],
                "definition": "ad-lib (impressed)",
                "examples": ["Sheesh!"]
            },
            "uh": {
                "canonical": ["AH1"],
                "variants": [],
                "definition": "ad-lib (filler)",
                "examples": ["Uh, yeah"]
            },
            "yeah": {
                "canonical": ["Y", "EH1"],
                "variants": [["Y", "AE1"]],
                "definition": "ad-lib (agreement)",
                "examples": ["Yeah yeah"]
            },
        }
        
        self.lexicon = starter
    
    def add_word(
        self,
        word: str,
        canonical: List[str],
        variants: List[List[str]] = None,
        definition: str = "",
        examples: List[str] = None
    ):
        """Add a word to the lexicon."""
        self.lexicon[word.lower()] = {
            "canonical": canonical,
            "variants": variants or [],
            "definition": definition,
            "examples": examples or []
        }
    
    def lookup(self, word: str) -> Optional[Dict]:
        """Look up a word in the lexicon."""
        return self.lexicon.get(word.lower())
    
    def get_pronunciations(self, word: str) -> List[List[str]]:
        """Get all pronunciations for a word."""
        entry = self.lookup(word)
        if not entry:
            return []
        
        pronunciations = [entry["canonical"]]
        pronunciations.extend(entry.get("variants", []))
        return pronunciations
    
    def is_slang(self, word: str) -> bool:
        """Check if a word is in the slang lexicon."""
        return word.lower() in self.lexicon
    
    def fuzzy_lookup(self, word: str, max_distance: int = 2) -> List[Tuple[str, int]]:
        """
        Find similar words in lexicon using edit distance.
        Returns list of (word, distance) tuples.
        """
        try:
            import editdistance
        except ImportError:
            return []
        
        matches = []
        word_lower = word.lower()
        
        for lexicon_word in self.lexicon.keys():
            distance = editdistance.eval(word_lower, lexicon_word)
            if distance <= max_distance:
                matches.append((lexicon_word, distance))
        
        return sorted(matches, key=lambda x: x[1])
    
    def save_to_file(self, path: str):
        """Save lexicon to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.lexicon, f, indent=2)
    
    def load_from_file(self, path: str):
        """Load additional words from JSON file."""
        with open(path, 'r') as f:
            custom_lexicon = json.load(f)
        self.lexicon.update(custom_lexicon)
    
    def __len__(self) -> int:
        return len(self.lexicon)
    
    def __contains__(self, word: str) -> bool:
        return self.is_slang(word)


if __name__ == "__main__":
    print("✅ SlangLexicon module loaded successfully!")
    print("")
    
    # Test the lexicon
    lexicon = SlangLexicon()
    
    print(f"   Total words: {len(lexicon)}")
    print("")
    
    # Test lookups
    test_words = ["finna", "bussin", "skrt", "hello"]
    print("   Sample lookups:")
    for word in test_words:
        if word in lexicon:
            entry = lexicon.lookup(word)
            print(f"   - {word}: {entry['definition']}")
        else:
            print(f"   - {word}: (not in lexicon)")
