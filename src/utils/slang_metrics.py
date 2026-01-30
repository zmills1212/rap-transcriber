"""
Slang-specific evaluation metrics for rap transcription.
Measures how well the model handles rap vocabulary.
"""
from typing import List, Dict, Set, Tuple
from pathlib import Path
import json
import re


class SlangAccuracyEvaluator:
    """
    Evaluates transcription accuracy specifically for slang terms.
    """
    
    def __init__(self, slang_lexicon_path: str = None):
        """
        Initialize evaluator.
        
        Args:
            slang_lexicon_path: Path to slang lexicon JSON
        """
        # Default slang terms if no lexicon provided
        self.slang_terms = {
            # Common rap slang
            "finna", "gonna", "wanna", "gotta", "tryna", "ima", "imma",
            "ion", "aint", "ain't", "bout", "ya", "yo", "yuh",
            # Ad-libs
            "ayy", "aye", "yeah", "yuh", "skrt", "brr", "woo", "sheesh",
            # Slang nouns
            "bussin", "cap", "fire", "lit", "dope", "sick", "bet",
            "facts", "lowkey", "highkey", "drip", "flex", "swag", "ice",
            "bread", "bands", "guap", "racks", "stacks",
            "homie", "dawg", "bruh", "fam", "bro", "plug", "opps",
            # Common variations
            "nigga", "niggas", "shit", "fuck", "bitch", "ass", "damn",
        }
        
        # Load from lexicon if provided
        if slang_lexicon_path:
            self._load_lexicon(slang_lexicon_path)
        
        # Common slang variations/misspellings
        self.slang_variants = {
            "finna": ["gonna", "fixing to", "about to"],
            "bussin": ["busting", "bustin"],
            "ima": ["i'm gonna", "i'm going to", "imma"],
            "ion": ["i don't", "i dont"],
            "tryna": ["trying to", "tryin to"],
            "aint": ["ain't", "isn't", "aren't"],
            "yuh": ["yeah", "yah", "ya"],
            "ayy": ["aye", "ay", "hey"],
        }
    
    def _load_lexicon(self, path: str):
        """Load slang terms from lexicon file."""
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            
            if 'words' in data:
                self.slang_terms.update(data['words'].keys())
        except Exception as e:
            print(f"Warning: Could not load lexicon: {e}")
    
    def extract_slang(self, text: str) -> List[str]:
        """
        Extract slang terms from text.
        
        Args:
            text: Input text
            
        Returns:
            List of slang terms found
        """
        text = text.lower()
        words = re.findall(r"\b[\w']+\b", text)
        
        found = []
        for word in words:
            if word in self.slang_terms:
                found.append(word)
        
        return found
    
    def compute_slang_accuracy(
        self,
        reference: str,
        hypothesis: str
    ) -> Dict:
        """
        Compute slang-specific accuracy.
        
        Args:
            reference: Ground truth text
            hypothesis: Predicted text
            
        Returns:
            Dictionary with slang metrics
        """
        ref_slang = self.extract_slang(reference)
        hyp_slang = self.extract_slang(hypothesis)
        
        ref_set = set(ref_slang)
        hyp_set = set(hyp_slang)
        
        # Correct slang (in both)
        correct = ref_set & hyp_set
        
        # Missed slang (in ref but not hyp)
        missed = ref_set - hyp_set
        
        # Extra slang (in hyp but not ref)
        extra = hyp_set - ref_set
        
        # Check for acceptable variants
        recovered_from_variants = set()
        for missed_term in list(missed):
            if missed_term in self.slang_variants:
                for variant in self.slang_variants[missed_term]:
                    if variant in hypothesis.lower():
                        recovered_from_variants.add(missed_term)
                        break
        
        # Recall: what fraction of reference slang was captured
        recall = len(correct) / len(ref_set) if ref_set else 1.0
        
        # Precision: what fraction of hypothesis slang was correct
        precision = len(correct) / len(hyp_set) if hyp_set else 1.0
        
        # F1 score
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'reference_slang': ref_slang,
            'hypothesis_slang': hyp_slang,
            'correct': list(correct),
            'missed': list(missed),
            'extra': list(extra),
            'recovered_variants': list(recovered_from_variants),
            'slang_recall': recall,
            'slang_precision': precision,
            'slang_f1': f1,
            'total_ref_slang': len(ref_slang),
            'total_hyp_slang': len(hyp_slang)
        }
    
    def compute_batch_slang_accuracy(
        self,
        references: List[str],
        hypotheses: List[str]
    ) -> Dict:
        """
        Compute slang accuracy over a batch.
        
        Args:
            references: List of ground truth texts
            hypotheses: List of predicted texts
            
        Returns:
            Aggregate slang metrics
        """
        total_ref_slang = 0
        total_correct = 0
        total_hyp_slang = 0
        
        all_missed = []
        all_extra = []
        
        for ref, hyp in zip(references, hypotheses):
            metrics = self.compute_slang_accuracy(ref, hyp)
            
            total_ref_slang += metrics['total_ref_slang']
            total_hyp_slang += metrics['total_hyp_slang']
            total_correct += len(metrics['correct'])
            
            all_missed.extend(metrics['missed'])
            all_extra.extend(metrics['extra'])
        
        # Aggregate metrics
        recall = total_correct / total_ref_slang if total_ref_slang > 0 else 1.0
        precision = total_correct / total_hyp_slang if total_hyp_slang > 0 else 1.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Most commonly missed slang
        missed_counts = {}
        for term in all_missed:
            missed_counts[term] = missed_counts.get(term, 0) + 1
        
        most_missed = sorted(missed_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            'slang_recall': recall,
            'slang_precision': precision,
            'slang_f1': f1,
            'total_reference_slang': total_ref_slang,
            'total_hypothesis_slang': total_hyp_slang,
            'total_correct': total_correct,
            'most_missed_slang': most_missed
        }


class AdLibEvaluator:
    """
    Evaluates transcription accuracy for ad-libs.
    """
    
    def __init__(self):
        self.adlibs = {
            "yeah", "yuh", "yah", "ya",
            "ayy", "aye", "ay", "hey",
            "uh", "um", "ah",
            "skrt", "skrrt",
            "brr", "brrr",
            "woo", "wooo",
            "sheesh",
            "gang",
            "what", "huh",
            "ok", "okay",
            "let's go", "lesgo",
            "facts", "word",
            "yo"
        }
    
    def extract_adlibs(self, text: str) -> List[str]:
        """Extract ad-libs from text."""
        text = text.lower()
        words = re.findall(r"\b[\w']+\b", text)
        
        return [w for w in words if w in self.adlibs]
    
    def compute_adlib_accuracy(
        self,
        reference: str,
        hypothesis: str
    ) -> Dict:
        """Compute ad-lib specific accuracy."""
        ref_adlibs = self.extract_adlibs(reference)
        hyp_adlibs = self.extract_adlibs(hypothesis)
        
        ref_set = set(ref_adlibs)
        hyp_set = set(hyp_adlibs)
        
        correct = ref_set & hyp_set
        
        recall = len(correct) / len(ref_set) if ref_set else 1.0
        precision = len(correct) / len(hyp_set) if hyp_set else 1.0
        
        return {
            'adlib_recall': recall,
            'adlib_precision': precision,
            'reference_adlibs': ref_adlibs,
            'hypothesis_adlibs': hyp_adlibs
        }


if __name__ == "__main__":
    print("✅ Slang metrics module loaded successfully!")
    print("")
    
    # Test slang evaluator
    print("Testing SlangAccuracyEvaluator...")
    evaluator = SlangAccuracyEvaluator()
    
    test_cases = [
        (
            "i'm finna get this bread yeah",
            "i'm finna get this bread yeah"
        ),
        (
            "she bussin it down skrt skrt",
            "she busting it down skirt skirt"
        ),
        (
            "no cap this fire bruh",
            "no cap this fire bro"
        ),
    ]
    
    for ref, hyp in test_cases:
        metrics = evaluator.compute_slang_accuracy(ref, hyp)
        print(f"\n   Ref: '{ref}'")
        print(f"   Hyp: '{hyp}'")
        print(f"   Slang Recall: {metrics['slang_recall']:.2%}")
        print(f"   Correct: {metrics['correct']}")
        print(f"   Missed: {metrics['missed']}")
    
    print("\n")
    
    # Test batch
    print("Testing batch slang accuracy...")
    refs = [tc[0] for tc in test_cases]
    hyps = [tc[1] for tc in test_cases]
    
    batch = evaluator.compute_batch_slang_accuracy(refs, hyps)
    print(f"   Overall Slang Recall: {batch['slang_recall']:.2%}")
    print(f"   Overall Slang F1: {batch['slang_f1']:.2%}")
    print(f"   Most missed: {batch['most_missed_slang']}")
