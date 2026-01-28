"""
Rap Transcriber - Multi-Head ASR Model.
Combines Conformer encoder with phoneme and text prediction heads.
"""
import torch
import torch.nn as nn
from typing import Optional, Dict, List, Tuple

from src.models.encoder import ConformerEncoder
from src.models.phoneme_head import PhonemeHead
from src.models.text_head import TextHead


class RapTranscriber(nn.Module):
    """
    Multi-headed ASR model for rap lyrics transcription.
    
    Architecture:
        Audio -> Conformer Encoder -> [Phoneme Head, Text Head]
        
    The dual-head design allows:
        1. Phoneme head: Captures pronunciation patterns, handles slang
        2. Text head: Direct text prediction for standard words
        3. Fusion: Combine both for better accuracy
    """
    
    def __init__(
        self,
        # Encoder params
        input_dim: int = 80,
        encoder_dim: int = 512,
        encoder_layers: int = 12,
        encoder_heads: int = 8,
        # Phoneme head params
        phoneme_vocab_size: int = 84,
        phoneme_hidden_dim: int = 256,
        # Text head params
        text_vocab_size: int = 8192,
        text_hidden_dim: int = 512,
        # General params
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.encoder_dim = encoder_dim
        
        # Shared encoder
        self.encoder = ConformerEncoder(
            input_dim=input_dim,
            d_model=encoder_dim,
            num_layers=encoder_layers,
            num_heads=encoder_heads,
            dropout=dropout
        )
        
        # Phoneme prediction head
        self.phoneme_head = PhonemeHead(
            encoder_dim=encoder_dim,
            hidden_dim=phoneme_hidden_dim,
            vocab_size=phoneme_vocab_size,
            dropout=dropout
        )
        
        # Text prediction head
        self.text_head = TextHead(
            encoder_dim=encoder_dim,
            hidden_dim=text_hidden_dim,
            vocab_size=text_vocab_size,
            dropout=dropout
        )
        
        # Store config
        self.config = {
            'input_dim': input_dim,
            'encoder_dim': encoder_dim,
            'encoder_layers': encoder_layers,
            'encoder_heads': encoder_heads,
            'phoneme_vocab_size': phoneme_vocab_size,
            'text_vocab_size': text_vocab_size,
            'dropout': dropout
        }
    
    def forward(
        self,
        features: torch.Tensor,
        feature_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through encoder and both heads.
        
        Args:
            features: Mel spectrogram [batch, time, n_mels] or [batch, n_mels, time]
            feature_mask: Optional padding mask [batch, time]
            
        Returns:
            Dictionary with:
                - encoder_out: Encoder output [batch, time//4, encoder_dim]
                - phoneme_logits: Phoneme predictions [batch, time//4, phoneme_vocab]
                - text_logits: Text predictions [batch, time//4, text_vocab]
        """
        # Encode
        encoder_out, mask = self.encoder(features, feature_mask)
        
        # Phoneme head
        phoneme_logits = self.phoneme_head(encoder_out, mask)
        
        # Text head
        text_logits = self.text_head(encoder_out, mask)
        
        return {
            'encoder_out': encoder_out,
            'encoder_mask': mask,
            'phoneme_logits': phoneme_logits,
            'text_logits': text_logits
        }
    
    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        phoneme_targets: torch.Tensor,
        text_targets: torch.Tensor,
        input_lengths: torch.Tensor,
        phoneme_lengths: torch.Tensor,
        text_lengths: torch.Tensor,
        phoneme_weight: float = 0.3,
        text_weight: float = 0.7
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss from both heads.
        
        Args:
            outputs: Forward pass outputs
            phoneme_targets: Target phoneme sequences
            text_targets: Target text token sequences
            input_lengths: Encoder output lengths
            phoneme_lengths: Phoneme target lengths
            text_lengths: Text target lengths
            phoneme_weight: Weight for phoneme loss
            text_weight: Weight for text loss
            
        Returns:
            Dictionary with individual and combined losses
        """
        # Phoneme loss
        phoneme_loss = self.phoneme_head.compute_loss(
            outputs['phoneme_logits'],
            phoneme_targets,
            input_lengths,
            phoneme_lengths
        )
        
        # Text loss
        text_loss = self.text_head.compute_loss(
            outputs['text_logits'],
            text_targets,
            input_lengths,
            text_lengths
        )
        
        # Combined loss
        total_loss = phoneme_weight * phoneme_loss + text_weight * text_loss
        
        return {
            'loss': total_loss,
            'phoneme_loss': phoneme_loss,
            'text_loss': text_loss
        }
    
    def decode(
        self,
        features: torch.Tensor,
        feature_mask: Optional[torch.Tensor] = None,
        decode_phonemes: bool = True,
        decode_text: bool = True
    ) -> Dict[str, List]:
        """
        Decode audio features to phonemes and/or text.
        
        Args:
            features: Mel spectrogram
            feature_mask: Optional padding mask
            decode_phonemes: Whether to decode phonemes
            decode_text: Whether to decode text
            
        Returns:
            Dictionary with decoded sequences
        """
        # Forward pass
        outputs = self.forward(features, feature_mask)
        
        results = {}
        
        if decode_phonemes:
            results['phonemes'] = self.phoneme_head.decode_greedy(
                outputs['phoneme_logits']
            )
        
        if decode_text:
            results['text_tokens'] = self.text_head.decode_greedy(
                outputs['text_logits']
            )
        
        return results
    
    def transcribe(
        self,
        features: torch.Tensor,
        tokenizer=None
    ) -> List[str]:
        """
        Full transcription pipeline.
        
        Args:
            features: Mel spectrogram [batch, time, n_mels]
            tokenizer: Text tokenizer for decoding
            
        Returns:
            List of transcribed text strings
        """
        outputs = self.forward(features)
        
        if tokenizer is not None:
            return self.text_head.decode_to_text(
                outputs['text_logits'],
                tokenizer
            )
        else:
            # Return raw token IDs if no tokenizer
            return self.text_head.decode_greedy(outputs['text_logits'])
    
    def get_encoder_output(
        self,
        features: torch.Tensor,
        feature_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Get encoder output for external use (e.g., fusion module)."""
        return self.encoder(features, feature_mask)
    
    @property
    def num_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def parameter_breakdown(self) -> Dict[str, int]:
        """Get parameter count by component."""
        return {
            'encoder': sum(p.numel() for p in self.encoder.parameters()),
            'phoneme_head': sum(p.numel() for p in self.phoneme_head.parameters()),
            'text_head': sum(p.numel() for p in self.text_head.parameters()),
            'total': self.num_parameters
        }
    
    def save(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'config': self.config,
            'state_dict': self.state_dict()
        }, path)
    
    @classmethod
    def load(cls, path: str, device: str = 'cpu') -> 'RapTranscriber':
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=device)
        model = cls(**checkpoint['config'])
        model.load_state_dict(checkpoint['state_dict'])
        return model


def create_model(config: Dict = None) -> RapTranscriber:
    """
    Factory function to create model from config.
    
    Args:
        config: Optional config dict (uses defaults if None)
        
    Returns:
        RapTranscriber model
    """
    default_config = {
        'input_dim': 80,
        'encoder_dim': 512,
        'encoder_layers': 12,
        'encoder_heads': 8,
        'phoneme_vocab_size': 84,
        'text_vocab_size': 8192,
        'dropout': 0.1
    }
    
    if config:
        default_config.update(config)
    
    return RapTranscriber(**default_config)


if __name__ == "__main__":
    print("✅ RapTranscriber model loaded successfully!")
    print("")
    
    # Create model
    model = create_model()
    
    # Test forward pass
    batch_size = 2
    time_frames = 500
    n_mels = 80
    
    dummy_input = torch.randn(batch_size, time_frames, n_mels)
    
    outputs = model(dummy_input)
    
    # Parameter breakdown
    params = model.parameter_breakdown()
    
    print(f"   Input shape:      {dummy_input.shape}")
    print(f"   Encoder output:   {outputs['encoder_out'].shape}")
    print(f"   Phoneme logits:   {outputs['phoneme_logits'].shape}")
    print(f"   Text logits:      {outputs['text_logits'].shape}")
    print("")
    print(f"   Parameters:")
    print(f"     Encoder:        {params['encoder']:,}")
    print(f"     Phoneme head:   {params['phoneme_head']:,}")
    print(f"     Text head:      {params['text_head']:,}")
    print(f"     Total:          {params['total']:,}")
