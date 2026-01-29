"""
Evaluation metrics for rap transcription.
Implements WER, CER, and other ASR metrics.
"""
import editdistance
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import re


@dataclass
class TranscriptionMetrics:
    """Container for transcription evaluation metrics."""
    wer: float  # Word Error Rate
    cer: float  # Character Error Rate
    word_accuracy: float
    char_accuracy: float
    insertions: int
    deletions: int
    substitutions: int
    reference_words: int
    hypothesis_words: int
    reference_chars: int
    hypothesis_chars: int


def normalize_text(text: str) -> str:
    """Normalize text for fair comparison."""
    # Lowercase
    text = text.lower()
    
    # Remove punctuation except apostrophes
    text = re.sub(r"[^\w\s']", '', text)
    
    # Normalize whitespace
    text = ' '.join(text.split())
    
    return text


def compute_wer(
    reference: str,
    hypothesis: str,
    normalize: bool = True
) -> Tuple[float, Dict]:
    """
    Compute Word Error Rate.
    
    WER = (S + D + I) / N
    
    Where:
        S = Substitutions
        D = Deletions
        I = Insertions
        N = Number of words in reference
    
    Args:
        reference: Ground truth transcription
        hypothesis: Predicted transcription
        normalize: Whether to normalize texts
        
    Returns:
        Tuple of (WER, details dict)
    """
    if normalize:
        reference = normalize_text(reference)
        hypothesis = normalize_text(hypothesis)
    
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    
    # Handle empty cases
    if len(ref_words) == 0:
        if len(hyp_words) == 0:
            return 0.0, {'insertions': 0, 'deletions': 0, 'substitutions': 0}
        else:
            return float(len(hyp_words)), {'insertions': len(hyp_words), 'deletions': 0, 'substitutions': 0}
    
    # Compute edit distance and operations
    distance = editdistance.eval(ref_words, hyp_words)
    
    # Get detailed operations using DP
    ops = _get_edit_operations(ref_words, hyp_words)
    
    wer = distance / len(ref_words)
    
    details = {
        'distance': distance,
        'reference_words': len(ref_words),
        'hypothesis_words': len(hyp_words),
        'insertions': ops['insertions'],
        'deletions': ops['deletions'],
        'substitutions': ops['substitutions']
    }
    
    return wer, details


def compute_cer(
    reference: str,
    hypothesis: str,
    normalize: bool = True
) -> Tuple[float, Dict]:
    """
    Compute Character Error Rate.
    
    Args:
        reference: Ground truth transcription
        hypothesis: Predicted transcription
        normalize: Whether to normalize texts
        
    Returns:
        Tuple of (CER, details dict)
    """
    if normalize:
        reference = normalize_text(reference)
        hypothesis = normalize_text(hypothesis)
    
    ref_chars = list(reference)
    hyp_chars = list(hypothesis)
    
    # Handle empty cases
    if len(ref_chars) == 0:
        if len(hyp_chars) == 0:
            return 0.0, {'insertions': 0, 'deletions': 0, 'substitutions': 0}
        else:
            return float(len(hyp_chars)), {'insertions': len(hyp_chars), 'deletions': 0, 'substitutions': 0}
    
    distance = editdistance.eval(ref_chars, hyp_chars)
    
    cer = distance / len(ref_chars)
    
    details = {
        'distance': distance,
        'reference_chars': len(ref_chars),
        'hypothesis_chars': len(hyp_chars)
    }
    
    return cer, details


def _get_edit_operations(ref: List[str], hyp: List[str]) -> Dict[str, int]:
    """
    Get counts of edit operations using dynamic programming.
    
    Returns dict with insertions, deletions, substitutions.
    """
    m, n = len(ref), len(hyp)
    
    # DP table: (cost, operation)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    ops = [[None] * (n + 1) for _ in range(m + 1)]
    
    # Initialize
    for i in range(m + 1):
        dp[i][0] = i
        if i > 0:
            ops[i][0] = 'delete'
    
    for j in range(n + 1):
        dp[0][j] = j
        if j > 0:
            ops[0][j] = 'insert'
    
    # Fill table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i-1] == hyp[j-1]:
                dp[i][j] = dp[i-1][j-1]
                ops[i][j] = 'match'
            else:
                costs = [
                    (dp[i-1][j] + 1, 'delete'),      # Deletion
                    (dp[i][j-1] + 1, 'insert'),      # Insertion
                    (dp[i-1][j-1] + 1, 'substitute') # Substitution
                ]
                dp[i][j], ops[i][j] = min(costs, key=lambda x: x[0])
    
    # Backtrack to count operations
    insertions = deletions = substitutions = 0
    i, j = m, n
    
    while i > 0 or j > 0:
        op = ops[i][j]
        if op == 'match':
            i -= 1
            j -= 1
        elif op == 'substitute':
            substitutions += 1
            i -= 1
            j -= 1
        elif op == 'delete':
            deletions += 1
            i -= 1
        elif op == 'insert':
            insertions += 1
            j -= 1
        else:
            break
    
    return {
        'insertions': insertions,
        'deletions': deletions,
        'substitutions': substitutions
    }


def compute_metrics(
    reference: str,
    hypothesis: str,
    normalize: bool = True
) -> TranscriptionMetrics:
    """
    Compute all transcription metrics.
    
    Args:
        reference: Ground truth transcription
        hypothesis: Predicted transcription
        normalize: Whether to normalize texts
        
    Returns:
        TranscriptionMetrics object
    """
    wer, wer_details = compute_wer(reference, hypothesis, normalize)
    cer, cer_details = compute_cer(reference, hypothesis, normalize)
    
    return TranscriptionMetrics(
        wer=wer,
        cer=cer,
        word_accuracy=max(0, 1 - wer),
        char_accuracy=max(0, 1 - cer),
        insertions=wer_details['insertions'],
        deletions=wer_details['deletions'],
        substitutions=wer_details['substitutions'],
        reference_words=wer_details['reference_words'],
        hypothesis_words=wer_details['hypothesis_words'],
        reference_chars=cer_details['reference_chars'],
        hypothesis_chars=cer_details['hypothesis_chars']
    )


def compute_batch_metrics(
    references: List[str],
    hypotheses: List[str],
    normalize: bool = True
) -> Dict:
    """
    Compute metrics over a batch of samples.
    
    Args:
        references: List of ground truth transcriptions
        hypotheses: List of predicted transcriptions
        normalize: Whether to normalize texts
        
    Returns:
        Dictionary with aggregate metrics
    """
    assert len(references) == len(hypotheses), "Mismatched lengths"
    
    total_wer = 0.0
    total_cer = 0.0
    total_ref_words = 0
    total_hyp_words = 0
    total_ref_chars = 0
    total_hyp_chars = 0
    total_word_errors = 0
    total_char_errors = 0
    
    sample_metrics = []
    
    for ref, hyp in zip(references, hypotheses):
        metrics = compute_metrics(ref, hyp, normalize)
        sample_metrics.append(metrics)
        
        total_ref_words += metrics.reference_words
        total_hyp_words += metrics.hypothesis_words
        total_ref_chars += metrics.reference_chars
        total_hyp_chars += metrics.hypothesis_chars
        total_word_errors += metrics.insertions + metrics.deletions + metrics.substitutions
        total_char_errors += int(metrics.cer * metrics.reference_chars)
    
    # Micro-average (total errors / total words)
    micro_wer = total_word_errors / max(1, total_ref_words)
    micro_cer = total_char_errors / max(1, total_ref_chars)
    
    # Macro-average (average of per-sample metrics)
    macro_wer = sum(m.wer for m in sample_metrics) / len(sample_metrics)
    macro_cer = sum(m.cer for m in sample_metrics) / len(sample_metrics)
    
    return {
        'micro_wer': micro_wer,
        'micro_cer': micro_cer,
        'macro_wer': macro_wer,
        'macro_cer': macro_cer,
        'total_samples': len(references),
        'total_reference_words': total_ref_words,
        'total_hypothesis_words': total_hyp_words,
        'total_word_errors': total_word_errors,
        'word_accuracy': 1 - micro_wer,
        'char_accuracy': 1 - micro_cer,
        'sample_metrics': sample_metrics
    }


class MetricsTracker:
    """
    Tracks metrics over time during training/evaluation.
    """
    
    def __init__(self):
        self.history = []
        self.running_wer = 0.0
        self.running_cer = 0.0
        self.count = 0
    
    def update(self, reference: str, hypothesis: str):
        """Add a sample to tracking."""
        metrics = compute_metrics(reference, hypothesis)
        self.history.append(metrics)
        
        # Update running averages
        self.count += 1
        self.running_wer += (metrics.wer - self.running_wer) / self.count
        self.running_cer += (metrics.cer - self.running_cer) / self.count
    
    def get_current_metrics(self) -> Dict:
        """Get current running metrics."""
        return {
            'wer': self.running_wer,
            'cer': self.running_cer,
            'samples': self.count
        }
    
    def reset(self):
        """Reset tracker."""
        self.history = []
        self.running_wer = 0.0
        self.running_cer = 0.0
        self.count = 0


if __name__ == "__main__":
    print("✅ Metrics module loaded successfully!")
    print("")
    
    # Test WER computation
    print("Testing WER computation...")
    
    test_cases = [
        ("i'm finna get this bread", "i'm finna get this bread"),  # Perfect
        ("i'm finna get this bread", "im gonna get this bread"),   # Some errors
        ("she bussin it down", "she busting it down yeah"),        # Substitution + insertion
        ("no cap", ""),                                             # All deleted
    ]
    
    for ref, hyp in test_cases:
        wer, details = compute_wer(ref, hyp)
        print(f"   Ref: '{ref}'")
        print(f"   Hyp: '{hyp}'")
        print(f"   WER: {wer:.2%} (I:{details['insertions']} D:{details['deletions']} S:{details['substitutions']})")
        print("")
    
    # Test batch metrics
    print("Testing batch metrics...")
    refs = ["i'm on my grind", "she bussin it down", "no cap this fire"]
    hyps = ["i'm on my grind", "she busting down", "no cap is fire"]
    
    batch = compute_batch_metrics(refs, hyps)
    print(f"   Samples: {batch['total_samples']}")
    print(f"   Micro WER: {batch['micro_wer']:.2%}")
    print(f"   Macro WER: {batch['macro_wer']:.2%}")
    print(f"   Word Accuracy: {batch['word_accuracy']:.2%}")
