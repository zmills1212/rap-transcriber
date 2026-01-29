"""
Beam search decoder for rap transcription.
Provides better decoding than greedy search by exploring multiple hypotheses.
"""
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import heapq


@dataclass
class Hypothesis:
    """A single beam search hypothesis."""
    tokens: List[int]
    score: float
    
    def __lt__(self, other):
        # For heap operations (higher score = better)
        return self.score > other.score


class BeamSearchDecoder:
    """
    Beam search decoder for CTC outputs.
    
    Explores multiple hypotheses to find better transcriptions.
    """
    
    def __init__(
        self,
        blank_id: int = 0,
        beam_size: int = 10,
        length_penalty: float = 1.0,
        score_threshold: float = -float('inf')
    ):
        """
        Initialize beam search decoder.
        
        Args:
            blank_id: CTC blank token ID
            beam_size: Number of beams to keep
            length_penalty: Penalty/bonus for sequence length
            score_threshold: Minimum score to keep hypothesis
        """
        self.blank_id = blank_id
        self.beam_size = beam_size
        self.length_penalty = length_penalty
        self.score_threshold = score_threshold
    
    def decode(
        self,
        log_probs: torch.Tensor,
        return_all_beams: bool = False
    ) -> List[List[int]]:
        """
        Decode log probabilities using beam search.
        
        Args:
            log_probs: Log probabilities [batch, time, vocab]
            return_all_beams: Whether to return all beams or just best
            
        Returns:
            List of decoded token sequences (one per batch)
        """
        batch_size = log_probs.size(0)
        results = []
        
        for b in range(batch_size):
            if return_all_beams:
                beams = self._beam_search_single(log_probs[b])
                results.append([h.tokens for h in beams])
            else:
                best = self._beam_search_single(log_probs[b])[0]
                results.append(best.tokens)
        
        return results
    
    def _beam_search_single(
        self,
        log_probs: torch.Tensor
    ) -> List[Hypothesis]:
        """
        Beam search for a single sequence.
        
        Args:
            log_probs: Log probabilities [time, vocab]
            
        Returns:
            List of hypotheses sorted by score
        """
        time_steps, vocab_size = log_probs.shape
        
        # Initialize with empty hypothesis
        beams = [Hypothesis(tokens=[], score=0.0)]
        
        for t in range(time_steps):
            new_beams = []
            
            for hyp in beams:
                # Get top-k tokens for this timestep
                top_k = min(self.beam_size * 2, vocab_size)
                scores, indices = log_probs[t].topk(top_k)
                
                for score, idx in zip(scores.tolist(), indices.tolist()):
                    new_score = hyp.score + score
                    
                    if new_score < self.score_threshold:
                        continue
                    
                    if idx == self.blank_id:
                        # Blank: keep same tokens
                        new_hyp = Hypothesis(
                            tokens=hyp.tokens.copy(),
                            score=new_score
                        )
                    elif len(hyp.tokens) > 0 and idx == hyp.tokens[-1]:
                        # Repeat: keep same tokens (CTC collapse)
                        new_hyp = Hypothesis(
                            tokens=hyp.tokens.copy(),
                            score=new_score
                        )
                    else:
                        # New token
                        new_hyp = Hypothesis(
                            tokens=hyp.tokens + [idx],
                            score=new_score
                        )
                    
                    new_beams.append(new_hyp)
            
            # Keep top beams
            beams = sorted(new_beams, key=lambda h: h.score, reverse=True)
            beams = self._merge_equivalent(beams)[:self.beam_size]
        
        # Apply length penalty
        for hyp in beams:
            length = max(1, len(hyp.tokens))
            hyp.score = hyp.score / (length ** self.length_penalty)
        
        return sorted(beams, key=lambda h: h.score, reverse=True)
    
    def _merge_equivalent(self, beams: List[Hypothesis]) -> List[Hypothesis]:
        """Merge beams with identical token sequences."""
        merged = {}
        
        for hyp in beams:
            key = tuple(hyp.tokens)
            if key not in merged or hyp.score > merged[key].score:
                merged[key] = hyp
        
        return list(merged.values())


class PrefixBeamSearchDecoder:
    """
    Prefix beam search decoder.
    More efficient variant that groups hypotheses by prefix.
    """
    
    def __init__(
        self,
        blank_id: int = 0,
        beam_size: int = 10,
        prune_threshold: float = 0.001
    ):
        self.blank_id = blank_id
        self.beam_size = beam_size
        self.prune_threshold = prune_threshold
    
    def decode(self, log_probs: torch.Tensor) -> List[List[int]]:
        """
        Decode using prefix beam search.
        
        Args:
            log_probs: Log probabilities [batch, time, vocab]
            
        Returns:
            List of decoded token sequences
        """
        batch_size = log_probs.size(0)
        results = []
        
        for b in range(batch_size):
            result = self._prefix_beam_search(log_probs[b])
            results.append(result)
        
        return results
    
    def _prefix_beam_search(self, log_probs: torch.Tensor) -> List[int]:
        """
        Prefix beam search for single sequence.
        
        Args:
            log_probs: Log probabilities [time, vocab]
            
        Returns:
            Best token sequence
        """
        time_steps, vocab_size = log_probs.shape
        
        # Probabilities: prefix -> (p_blank, p_non_blank)
        # p_blank: prob of prefix ending in blank
        # p_non_blank: prob of prefix ending in non-blank
        
        # Initialize
        probs = {(): (1.0, 0.0)}  # Empty prefix
        
        for t in range(time_steps):
            new_probs = {}
            
            # Convert log probs to probs for this timestep
            frame_probs = torch.exp(log_probs[t])
            
            for prefix, (p_b, p_nb) in probs.items():
                p_total = p_b + p_nb
                
                if p_total < self.prune_threshold:
                    continue
                
                # Extend with blank
                blank_prob = frame_probs[self.blank_id].item()
                new_p_b = p_total * blank_prob
                
                if prefix in new_probs:
                    new_probs[prefix] = (
                        new_probs[prefix][0] + new_p_b,
                        new_probs[prefix][1]
                    )
                else:
                    new_probs[prefix] = (new_p_b, 0.0)
                
                # Extend with non-blank tokens
                for c in range(vocab_size):
                    if c == self.blank_id:
                        continue
                    
                    c_prob = frame_probs[c].item()
                    
                    if len(prefix) > 0 and c == prefix[-1]:
                        # Repeat character: only extend from blank
                        new_p_nb = p_b * c_prob
                    else:
                        # New character
                        new_p_nb = p_total * c_prob
                    
                    new_prefix = prefix + (c,)
                    
                    if new_prefix in new_probs:
                        new_probs[new_prefix] = (
                            new_probs[new_prefix][0],
                            new_probs[new_prefix][1] + new_p_nb
                        )
                    else:
                        new_probs[new_prefix] = (0.0, new_p_nb)
            
            # Prune to top beams
            scored = [
                (prefix, p_b + p_nb) 
                for prefix, (p_b, p_nb) in new_probs.items()
            ]
            scored.sort(key=lambda x: x[1], reverse=True)
            
            probs = {}
            for prefix, _ in scored[:self.beam_size]:
                probs[prefix] = new_probs[prefix]
        
        # Return best prefix
        if not probs:
            return []
        
        best_prefix = max(probs.keys(), key=lambda p: sum(probs[p]))
        return list(best_prefix)


class LanguageModelDecoder:
    """
    Decoder with optional language model rescoring.
    Combines acoustic model scores with language model probabilities.
    """
    
    def __init__(
        self,
        beam_decoder: BeamSearchDecoder,
        lm_weight: float = 0.5,
        word_bonus: float = 0.0
    ):
        """
        Initialize LM decoder.
        
        Args:
            beam_decoder: Base beam search decoder
            lm_weight: Weight for language model scores
            word_bonus: Bonus for each word (encourages longer outputs)
        """
        self.beam_decoder = beam_decoder
        self.lm_weight = lm_weight
        self.word_bonus = word_bonus
        self.lm = None  # Language model (to be loaded)
    
    def set_language_model(self, lm):
        """Set language model for rescoring."""
        self.lm = lm
    
    def decode(
        self,
        log_probs: torch.Tensor,
        tokenizer=None
    ) -> List[List[int]]:
        """
        Decode with optional LM rescoring.
        
        Args:
            log_probs: Log probabilities [batch, time, vocab]
            tokenizer: Tokenizer for converting to text (for LM)
            
        Returns:
            List of decoded token sequences
        """
        # Get beam hypotheses
        beams = self.beam_decoder.decode(log_probs, return_all_beams=True)
        
        results = []
        
        for batch_beams in beams:
            if self.lm is None or tokenizer is None:
                # No LM: return best beam
                results.append(batch_beams[0] if batch_beams else [])
            else:
                # Rescore with LM
                best_score = float('-inf')
                best_tokens = []
                
                for tokens in batch_beams:
                    # Get text
                    text = tokenizer.decode(tokens)
                    
                    # Score with LM
                    lm_score = self._score_with_lm(text)
                    
                    # Combined score (acoustic already in beam)
                    # Add word bonus
                    num_words = len(text.split()) if text else 0
                    combined = lm_score * self.lm_weight + num_words * self.word_bonus
                    
                    if combined > best_score:
                        best_score = combined
                        best_tokens = tokens
                
                results.append(best_tokens)
        
        return results
    
    def _score_with_lm(self, text: str) -> float:
        """Score text with language model."""
        if self.lm is None:
            return 0.0
        
        # Placeholder - implement based on your LM
        # Could use a simple n-gram model or neural LM
        return 0.0


if __name__ == "__main__":
    print("✅ Decoder module loaded successfully!")
    print("")
    
    # Test beam search decoder
    print("Testing BeamSearchDecoder...")
    
    decoder = BeamSearchDecoder(beam_size=5)
    
    # Dummy log probs [batch=2, time=50, vocab=100]
    log_probs = torch.randn(2, 50, 100)
    log_probs = F.log_softmax(log_probs, dim=-1)
    
    results = decoder.decode(log_probs)
    
    print(f"   ✅ Batch size: {len(results)}")
    print(f"   ✅ Decoded lengths: {[len(r) for r in results]}")
    print("")
    
    # Test prefix beam search
    print("Testing PrefixBeamSearchDecoder...")
    
    prefix_decoder = PrefixBeamSearchDecoder(beam_size=5)
    results = prefix_decoder.decode(log_probs)
    
    print(f"   ✅ Batch size: {len(results)}")
    print(f"   ✅ Decoded lengths: {[len(r) for r in results]}")
