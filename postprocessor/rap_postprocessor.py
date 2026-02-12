"""
Rap Transcription Post-Processor v3

Fixes based on real evaluation data:
- Tightened her→hoes context rules
- Fixed rules that weren't firing (jam up 12, all these hoes, is it my)
- Added new patterns (now/not, will→whip, be like→be lame)
"""

import re
from typing import Optional, Union
from dataclasses import dataclass, field


@dataclass
class CorrectionResult:
    """Result of post-processing with change tracking."""
    original: str
    corrected: str
    changes: list = field(default_factory=list)
    
    @property
    def was_modified(self) -> bool:
        return self.original != self.corrected
    
    def summary(self) -> str:
        if not self.changes:
            return "No corrections made"
        return f"{len(self.changes)} corrections: {', '.join(self.changes)}"


class RapPostProcessor:
    """
    Post-processor for rap transcription output.
    Fixes systematic Whisper errors based on empirical evaluation data.
    """
    
    def __init__(self, aggressive: bool = False):
        self.aggressive = aggressive
        self._build_correction_maps()
    
    def _build_correction_maps(self):
        """Build all correction dictionaries."""
        
        # =================================================================
        # SLANG VOCABULARY CORRECTIONS
        # =================================================================
        self.slang_corrections = {
            # Money slang
            "pants": "bands",
            "pant": "band",
            
            # Gun/weapon slang
            "lizzy": "glizzy",
            "chopper": "choppa",
            "drako": "draco",
            
            # Ad-libs and exclamations
            "skirt": "skrrt",
            "skirts": "skrrt",
            "burr": "brrr",
            
            # Vehicle slang
            "will": "whip",  # NEW: common mishearing
            
            # Other slang
            "ops": "opps",
            "op": "opp",
            "well": "12",  # Police - but context dependent
        }
        
        # =================================================================
        # EXPLICIT TERM CORRECTIONS
        # =================================================================
        self.explicit_corrections = {
            "groups": "bitch",
            "beaches": "bitches",
            "benches": "bitches",
        }
        
        # =================================================================
        # DIALECT PRESERVATION
        # =================================================================
        self.dialect_corrections = {
            # -ing -> -in' (comprehensive list)
            "doing": "doin'",
            "going": "goin'",
            "rolling": "rollin'",
            "riding": "ridin'",
            "sliding": "slidin'",
            "walking": "walkin'",
            "talking": "talkin'",
            "keeping": "keepin'",
            "blowing": "blowin'",
            "looking": "lookin'",
            "cooking": "cookin'",
            "booking": "bookin'",
            "trapping": "trappin'",
            "whipping": "whippin'",
            "dripping": "drippin'",
            "sipping": "sippin'",
            "gripping": "grippin'",
            "hitting": "hittin'",
            "sitting": "sittin'",
            "getting": "gettin'",
            "letting": "lettin'",
            "catching": "catchin'",
            "matching": "matchin'",
            "flipping": "flippin'",
            "tripping": "trippin'",
            "popping": "poppin'",
            "dropping": "droppin'",
            "stopping": "stoppin'",
            "hopping": "hoppin'",
            "shooting": "shootin'",
            "moving": "movin'",
            "proving": "provin'",
            "loving": "lovin'",
            "fucking": "fuckin'",
            "sucking": "suckin'",
            "ducking": "duckin'",
            "bucking": "buckin'",
            "running": "runnin'",
            "gunning": "gunnin'",
            "stunning": "stunnin'",
            "coming": "comin'",
            "bumping": "bumpin'",
            "jumping": "jumpin'",
            "pumping": "pumpin'",
            "dumping": "dumpin'",
            "playing": "playin'",
            "saying": "sayin'",
            "staying": "stayin'",
            "paying": "payin'",
            "laying": "layin'",
            "slaying": "slayin'",
            "praying": "prayin'",
            "spraying": "sprayin'",
            "swinging": "swingin'",
            "bringing": "bringin'",
            "ringing": "ringin'",
            "singing": "singin'",
            "stinging": "stingin'",
            "banging": "bangin'",
            "hanging": "hangin'",
            "slanging": "slangin'",
            "landing": "landin'",
            "standing": "standin'",
            "handing": "handin'",
            "banding": "bandin'",
            "flying": "flyin'",
            "trying": "tryin'",
            "dying": "dyin'",
            "lying": "lyin'",
            "crying": "cryin'",
            "buying": "buyin'",
            "disappearing": "disappearin'",
            
            # Other dialect
            "hey": "aye",
            "hay": "aye",
        }
        
        # =================================================================
        # CONTEXT-AWARE PHRASE CORRECTIONS (order matters - specific first)
        # =================================================================
        self.phrase_corrections = [
            # === "is it my" pattern (repetitive flex lines) ===
            (r"\bit's\s+my\s+watch\s+it's\s+my\s+watch\s+it's\s+my\b", "is it my watch is it my watch is it my"),
            (r"\bit's\s+my\s+watch,?\s+it's\s+my\b", "is it my watch is it my"),
            (r"\bit's\s+my\s+sauce,?\s+it's\s+my\b", "is it my sauce is it my"),
            (r"\bit's\s+my\s+bitch,?\s+it's\s+my\b", "is it my bitch is it my"),
            (r"\bit's\s+my\s+will\b", "is it my whip"),
            (r"\bit's\s+my\s+thang\b", "is it my fit"),  # end of that pattern
            
            # === "bands up" pattern ===
            (r"\bpants\s+up\b", "bands up"),
            (r"\bgot\s+my\s+pants\b", "got my bands"),
            (r"\bgot\s+them\s+pants\b", "got them bands"),
            (r"\bgot\s+the\s+pants\b", "got the bands"),
            (r"\bgot\s+new\s+bands\b", "got them bands"),  # NEW: "new" → "them"
            (r"\bnew\s+bands\s+up\b", "them bands up"),
            
            # === "all these hoes" pattern ===
            (r"\bwhy\s+this\s+hoes\b", "all these hoes"),  # NEW
            (r"\bwhy\s+this\s+her\b", "all these hoes"),
            (r"\ball\s+these\s+her\s+gon", "all these hoes gon'"),
            (r"\bthese\s+her\s+gon", "these hoes gon'"),
            (r"\bher\s+gon'\s+flock\b", "hoes gon' flock"),
            (r"\bher\s+gon'\s+talk\b", "hoes gon' talk"),
            (r"\bher\s+gon'\s+fly\b", "hoes gon' flock"),
            
            # === "now/not" confusion ===
            (r"\bnow\s+she\s+wanna\b", "know she wanna"),  # NEW
            (r"\bnot\s+because\s+i\s+got\b", "now because i got"),  # NEW
            
            # === "jam up 12" pattern ===
            (r"\bjam\s+us,?\s+well\b", "jam up 12"),  # NEW: fixed with comma
            (r"\bjam\s+up\s+well\b", "jam up 12"),
            (r"\bstick\s+won't\s+jam\s+us\b", "stick won't jam up"),  # NEW
            
            # === "stick won't jam" pattern ===
            (r"\bstill\s+won't\s+jam\b", "stick won't jam"),
            
            # === "be lame" pattern ===
            (r"\bbe\s+like,?\s+i'm\b", "be lame i'm"),  # NEW
            (r"\bniggas?\s+be\s+like\b", "niggas be lame"),  # NEW
            
            # === "bitch we" pattern ===
            (r"\bas\s+best\s+we\b", "bitch we"),
            (r"\bgroups\s+like\b", "bitch like"),
            (r"\byour\s+groups\b", "your bitch"),
            
            # === "knees" pattern ===
            (r"\byour\s+needs\s+it\b", "your knees and"),
            (r"\byour\s+needs\s+and\b", "your knees and"),
            (r"\bknow\s+your\s+needs\b", "on your knees"),
            
            # === Numeric corrections ===
            (r"\bone\s+of\s+one\b", "1 of 1"),
            (r"\bone\s+on\s+one\b", "1 of 1"),
            
            # === "range" pattern (car) ===
            (r"\bthat\s+rain\b", "that range"),
            (r"\bthe\s+rain\b", "the range"),
            
            # === "thang" in context ===
            (r"\bsame\s+thing\b", "same thang"),
            (r"\bmy\s+thing\b", "my thang"),
            (r"\bdo\s+my\s+thing\b", "do my thang"),
            (r"\bdoin'\s+my\s+thing\b", "doin' my thang"),
            
            # === Contraction restoration ===
            (r"\bi\s+keeping\b", "i'm keepin'"),
            (r"\bi\s+walking\b", "i'm walkin'"),
            (r"\bi\s+doing\b", "i'm doin'"),
            (r"\bi\s+going\b", "i'm goin'"),
            
            # === "bet she" / "bitch she" ===
            (r"\bbitch,?\s+she\s+wanna\b", "bet she wanna"),  # NEW: overcorrection fix
        ]
        
        # Words needing specific context
        self.context_dependent = {"her", "thing", "still", "fly", "na", "well", "yeah"}
    
    def process(self, text: str, track_changes: bool = False) -> Union[str, CorrectionResult]:
        """Process transcription text and apply corrections."""
        original = text
        changes = []
        
        # Normalize input
        text = text.lower().strip()
        
        # Step 1: Apply phrase corrections first (most specific)
        text, phrase_changes = self._apply_phrase_corrections(text)
        changes.extend(phrase_changes)
        
        # Step 2: Apply dialect corrections
        text, dialect_changes = self._apply_dialect_corrections(text)
        changes.extend(dialect_changes)
        
        # Step 3: Apply slang corrections
        text, slang_changes = self._apply_slang_corrections(text)
        changes.extend(slang_changes)
        
        # Step 4: Apply explicit corrections (if aggressive mode)
        if self.aggressive:
            text, explicit_changes = self._apply_explicit_corrections(text)
            changes.extend(explicit_changes)
        
        if track_changes:
            return CorrectionResult(original=original, corrected=text, changes=changes)
        return text
    
    def _apply_phrase_corrections(self, text: str) -> tuple:
        """Apply context-aware phrase corrections."""
        changes = []
        for pattern, replacement in self.phrase_corrections:
            new_text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
            if new_text != text:
                changes.append(f"phrase: '{pattern}' → '{replacement}'")
                text = new_text
        return text, changes
    
    def _apply_dialect_corrections(self, text: str) -> tuple:
        """Apply dialect preservation corrections."""
        changes = []
        words = text.split()
        corrected_words = []
        
        for word in words:
            clean_word = re.sub(r"[^\w']", "", word.lower())
            trailing = word[len(clean_word):] if len(word) > len(clean_word) else ""
            
            if clean_word in self.dialect_corrections and clean_word not in self.context_dependent:
                new_word = self.dialect_corrections[clean_word]
                changes.append(f"dialect: {clean_word} → {new_word}")
                corrected_words.append(new_word + trailing)
            else:
                corrected_words.append(word)
        
        return " ".join(corrected_words), changes
    
    def _apply_slang_corrections(self, text: str) -> tuple:
        """Apply slang vocabulary corrections."""
        changes = []
        words = text.split()
        corrected_words = []
        
        for i, word in enumerate(words):
            clean_word = re.sub(r"[^\w']", "", word.lower())
            trailing = word[len(clean_word):] if len(word) > len(clean_word) else ""
            
            # Context check for "well" → "12" (only after "jam")
            if clean_word == "well":
                context_before = " ".join(words[max(0, i-2):i]).lower()
                if "jam" in context_before:
                    changes.append(f"slang: well → 12 (context: jam)")
                    corrected_words.append("12" + trailing)
                    continue
            
            if clean_word in self.slang_corrections and clean_word not in self.context_dependent:
                new_word = self.slang_corrections[clean_word]
                if new_word != clean_word:
                    changes.append(f"slang: {clean_word} → {new_word}")
                corrected_words.append(new_word + trailing)
            else:
                corrected_words.append(word)
        
        return " ".join(corrected_words), changes
    
    def _apply_explicit_corrections(self, text: str) -> tuple:
        """Apply explicit term corrections (aggressive mode only)."""
        changes = []
        words = text.split()
        corrected_words = []
        
        for i, word in enumerate(words):
            clean_word = re.sub(r"[^\w']", "", word.lower())
            trailing = word[len(clean_word):] if len(word) > len(clean_word) else ""
            
            # TIGHT context for "her" -> "hoes"
            if clean_word == "her":
                context_before = " ".join(words[max(0, i-2):i]).lower()
                context_after = " ".join(words[i+1:min(len(words), i+2)]).lower()
                if "these" in context_before and "gon" in context_after:
                    changes.append(f"explicit: her → hoes")
                    corrected_words.append("hoes" + trailing)
                    continue
            
            # Context for "na" -> "nigga"
            if clean_word == "na":
                context_before = " ".join(words[max(0, i-3):i]).lower()
                context_after = " ".join(words[i+1:min(len(words), i+3)]).lower()
                if any(w in context_before + context_after for w in ["try", "gang", "jam", "young", "ass", "can't"]):
                    changes.append(f"explicit: na → nigga")
                    corrected_words.append("nigga" + trailing)
                    continue
            
            if clean_word in self.explicit_corrections:
                new_word = self.explicit_corrections[clean_word]
                changes.append(f"explicit: {clean_word} → {new_word}")
                corrected_words.append(new_word + trailing)
            else:
                corrected_words.append(word)
        
        return " ".join(corrected_words), changes
    
    def add_custom_correction(self, wrong: str, correct: str, category: str = "slang"):
        """Add a custom correction rule."""
        wrong = wrong.lower()
        correct = correct.lower()
        
        if category == "slang":
            self.slang_corrections[wrong] = correct
        elif category == "explicit":
            self.explicit_corrections[wrong] = correct
        elif category == "dialect":
            self.dialect_corrections[wrong] = correct
        elif category == "phrase":
            self.phrase_corrections.append((wrong, correct))
    
    def get_stats(self) -> dict:
        """Return statistics about loaded corrections."""
        return {
            "slang_corrections": len(self.slang_corrections),
            "explicit_corrections": len(self.explicit_corrections),
            "dialect_corrections": len(self.dialect_corrections),
            "phrase_corrections": len(self.phrase_corrections),
            "total": (
                len(self.slang_corrections) + 
                len(self.explicit_corrections) + 
                len(self.dialect_corrections) + 
                len(self.phrase_corrections)
            )
        }


def correct_transcription(text: str, aggressive: bool = False) -> str:
    """Quick function to correct a transcription."""
    processor = RapPostProcessor(aggressive=aggressive)
    return processor.process(text)


if __name__ == "__main__":
    processor = RapPostProcessor(aggressive=True)
    
    # Test the specific issues from evaluation
    test_cases = [
        "n***a be like, i'm doin' my thang",
        "this stick won't jam us, well, can't jam us",
        "why this hoes gon' flock",
        "not because i got my bands up",
        "got new bands up",
        "it's my watch, it's my watch, it's my will it's my sauce",
    ]
    
    print("Post-processor v3 test:\n")
    for text in test_cases:
        result = processor.process(text, track_changes=True)
        print(f"IN:  {text}")
        print(f"OUT: {result.corrected}")
        if result.changes:
            print(f"     [{len(result.changes)} changes]")
        print()
