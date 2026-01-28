"""
Phoneme Head for rap transcription.
Uses CTC (Connectionist Temporal Classification) for phoneme prediction.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


class PhonemeHead(nn.Module):
    """
    Phoneme prediction head using CTC.
    
    Predicts ARPAbet phoneme sequences from encoder output.
    """
    
    def __init__(
        self,
        encoder_dim: int = 512,
        hidden_dim: int = 256,
        vocab_size: int = 84,  # ARPAbet + special tokens
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.vocab_size = vocab_size
        
        # Projection layers
        self.layers = nn.ModuleList()
        
        in_dim = encoder_dim
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else hidden_dim
            self.layers.append(nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ))
            in_dim = out_dim
        
        # Output projection to phoneme vocabulary
        self.output_proj = nn.Linear(hidden_dim, vocab_size)
        
        # CTC loss (blank token is index 0)
        self.ctc_loss = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
        
        # Phoneme vocabulary (ARPAbet)
        self.phonemes = self._build_phoneme_vocab()
    
    def _build_phoneme_vocab(self) -> List[str]:
        """Build ARPAbet phoneme vocabulary."""
        phonemes = [
            '<blank>',  # CTC blank token (index 0)
            '<sos>',    # Start of sequence
            '<eos>',    # End of sequence
            '<unk>',    # Unknown
            # Vowels
            'AA', 'AE', 'AH', 'AO', 'AW', 'AY',
            'EH', 'ER', 'EY',
            'IH', 'IY',
            'OW', 'OY',
            'UH', 'UW',
            # Consonants
            'B', 'CH', 'D', 'DH',
            'F', 'G', 'HH', 'JH',
            'K', 'L', 'M', 'N', 'NG',
            'P', 'R', 'S', 'SH',
            'T', 'TH', 'V', 'W', 'Y', 'Z', 'ZH',
            # Stress markers (optional, can be combined with vowels)
            'AA0', 'AA1', 'AA2',
            'AE0', 'AE1', 'AE2',
            'AH0', 'AH1', 'AH2',
            'AO0', 'AO1', 'AO2',
            'AW0', 'AW1', 'AW2',
            'AY0', 'AY1', 'AY2',
            'EH0', 'EH1', 'EH2',
            'ER0', 'ER1', 'ER2',
            'EY0', 'EY1', 'EY2',
            'IH0', 'IH1', 'IH2',
            'IY0', 'IY1', 'IY2',
            'OW0', 'OW1', 'OW2',
            'OY0', 'OY1', 'OY2',
            'UH0', 'UH1', 'UH2',
            'UW0', 'UW1', 'UW2',
        ]
        return phonemes
    
    def forward(
        self, 
        encoder_out: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            encoder_out: [batch, time, encoder_dim]
            encoder_mask: Optional [batch, time] padding mask
            
        Returns:
            logits: [batch, time, vocab_size]
        """
        x = encoder_out
        
        for layer in self.layers:
            x = layer(x)
        
        logits = self.output_proj(x)
        return logits
    
    def compute_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute CTC loss.
        
        Args:
            logits: [batch, time, vocab_size]
            targets: [batch, max_target_len] phoneme indices
            input_lengths: [batch] actual input lengths
            target_lengths: [batch] actual target lengths
            
        Returns:
            CTC loss value
        """
        # CTC expects [time, batch, vocab]
        log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
        
        loss = self.ctc_loss(
            log_probs,
            targets,
            input_lengths,
            target_lengths
        )
        return loss
    
    def decode_greedy(self, logits: torch.Tensor) -> List[List[str]]:
        """
        Greedy decoding (argmax at each timestep).
        
        Args:
            logits: [batch, time, vocab_size]
            
        Returns:
            List of phoneme sequences (one per batch item)
        """
        predictions = logits.argmax(dim=-1)  # [batch, time]
        
        results = []
        for pred in predictions:
            # Remove blanks and consecutive duplicates
            decoded = []
            prev_idx = -1
            for idx in pred.tolist():
                if idx != 0 and idx != prev_idx:  # Skip blank (0) and duplicates
                    if idx < len(self.phonemes):
                        decoded.append(self.phonemes[idx])
                prev_idx = idx
            results.append(decoded)
        
        return results
    
    def decode_beam(
        self, 
        logits: torch.Tensor, 
        beam_size: int = 10
    ) -> List[List[str]]:
        """
        Beam search decoding.
        
        Args:
            logits: [batch, time, vocab_size]
            beam_size: Number of beams
            
        Returns:
            List of phoneme sequences
        """
        # For simplicity, use greedy for now
        # Full beam search implementation would go here
        return self.decode_greedy(logits)


class PhonemeEncoder(nn.Module):
    """
    Encodes phoneme sequences to embeddings.
    Useful for phoneme-text alignment.
    """
    
    def __init__(
        self,
        vocab_size: int = 84,
        embed_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.output_proj = nn.Linear(hidden_dim * 2, hidden_dim)
    
    def forward(
        self, 
        phoneme_ids: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            phoneme_ids: [batch, seq_len]
            lengths: Optional [batch] actual lengths
            
        Returns:
            encoded: [batch, seq_len, hidden_dim]
        """
        embedded = self.embedding(phoneme_ids)
        
        if lengths is not None:
            embedded = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
        
        output, _ = self.lstm(embedded)
        
        if lengths is not None:
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        
        output = self.output_proj(output)
        return output


if __name__ == "__main__":
    print("✅ PhonemeHead module loaded successfully!")
    print("")
    
    # Test phoneme head
    batch_size = 2
    time_steps = 125
    encoder_dim = 512
    
    head = PhonemeHead(encoder_dim=encoder_dim)
    
    # Dummy encoder output
    encoder_out = torch.randn(batch_size, time_steps, encoder_dim)
    
    # Forward pass
    logits = head(encoder_out)
    
    # Test decoding
    decoded = head.decode_greedy(logits)
    
    print(f"   Encoder output: {encoder_out.shape}")
    print(f"   Phoneme logits: {logits.shape}")
    print(f"   Vocab size:     {head.vocab_size}")
    print(f"   Decoded sample: {decoded[0][:10]}...")
