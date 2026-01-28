"""
Text Head for rap transcription.
Predicts text tokens (BPE subwords) from encoder output.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict


class TextHead(nn.Module):
    """
    Text prediction head using CTC.
    
    Predicts BPE subword sequences from encoder output.
    """
    
    def __init__(
        self,
        encoder_dim: int = 512,
        hidden_dim: int = 512,
        vocab_size: int = 8192,  # BPE vocabulary size
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.vocab_size = vocab_size
        
        # Projection layers
        self.layers = nn.ModuleList()
        
        in_dim = encoder_dim
        for i in range(num_layers):
            out_dim = hidden_dim
            self.layers.append(nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ))
            in_dim = out_dim
        
        # Output projection to text vocabulary
        self.output_proj = nn.Linear(hidden_dim, vocab_size)
        
        # CTC loss
        self.ctc_loss = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
        
        # Special tokens
        self.special_tokens = {
            '<blank>': 0,
            '<unk>': 1,
            '<sos>': 2,
            '<eos>': 3,
            '<pad>': 4,
        }
    
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
            targets: [batch, max_target_len] token indices
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
    
    def decode_greedy(
        self,
        logits: torch.Tensor,
        tokenizer=None
    ) -> List[List[int]]:
        """
        Greedy decoding.
        
        Args:
            logits: [batch, time, vocab_size]
            tokenizer: Optional tokenizer for converting to text
            
        Returns:
            List of token ID sequences
        """
        predictions = logits.argmax(dim=-1)  # [batch, time]
        
        results = []
        for pred in predictions:
            # Remove blanks and consecutive duplicates
            decoded = []
            prev_idx = -1
            for idx in pred.tolist():
                if idx != 0 and idx != prev_idx:  # Skip blank and duplicates
                    decoded.append(idx)
                prev_idx = idx
            results.append(decoded)
        
        return results
    
    def decode_to_text(
        self,
        logits: torch.Tensor,
        tokenizer
    ) -> List[str]:
        """
        Decode logits to text strings.
        
        Args:
            logits: [batch, time, vocab_size]
            tokenizer: Tokenizer with decode() method
            
        Returns:
            List of decoded text strings
        """
        token_ids = self.decode_greedy(logits)
        
        texts = []
        for ids in token_ids:
            # Filter out special tokens
            filtered_ids = [
                i for i in ids 
                if i not in self.special_tokens.values()
            ]
            text = tokenizer.decode(filtered_ids)
            texts.append(text)
        
        return texts


class AttentionTextDecoder(nn.Module):
    """
    Autoregressive text decoder with attention.
    Alternative to CTC for better sequence modeling.
    """
    
    def __init__(
        self,
        encoder_dim: int = 512,
        decoder_dim: int = 512,
        vocab_size: int = 8192,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_len: int = 500
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.decoder_dim = decoder_dim
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, decoder_dim, padding_idx=0)
        
        # Positional embedding
        self.pos_embedding = nn.Embedding(max_len, decoder_dim)
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=decoder_dim,
            nhead=num_heads,
            dim_feedforward=decoder_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Encoder projection (if dimensions differ)
        self.encoder_proj = nn.Linear(encoder_dim, decoder_dim) if encoder_dim != decoder_dim else nn.Identity()
        
        # Output projection
        self.output_proj = nn.Linear(decoder_dim, vocab_size)
        
        # Loss
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
    
    def forward(
        self,
        encoder_out: torch.Tensor,
        targets: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass (teacher forcing).
        
        Args:
            encoder_out: [batch, src_len, encoder_dim]
            targets: [batch, tgt_len] target token IDs
            encoder_mask: Optional encoder padding mask
            target_mask: Optional target padding mask
            
        Returns:
            logits: [batch, tgt_len, vocab_size]
        """
        batch_size, tgt_len = targets.shape
        
        # Project encoder output
        memory = self.encoder_proj(encoder_out)
        
        # Embed targets
        positions = torch.arange(tgt_len, device=targets.device).unsqueeze(0).expand(batch_size, -1)
        tgt_embed = self.token_embedding(targets) + self.pos_embedding(positions)
        
        # Create causal mask
        causal_mask = self._generate_causal_mask(tgt_len, targets.device)
        
        # Decode
        decoded = self.decoder(
            tgt=tgt_embed,
            memory=memory,
            tgt_mask=causal_mask,
            memory_key_padding_mask=encoder_mask
        )
        
        # Project to vocabulary
        logits = self.output_proj(decoded)
        
        return logits
    
    def _generate_causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        """Generate causal attention mask."""
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def compute_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss.
        
        Args:
            logits: [batch, tgt_len, vocab_size]
            targets: [batch, tgt_len] target token IDs
            
        Returns:
            Loss value
        """
        # Shift for next-token prediction
        logits = logits[:, :-1, :].contiguous()
        targets = targets[:, 1:].contiguous()
        
        loss = self.loss_fn(
            logits.view(-1, self.vocab_size),
            targets.view(-1)
        )
        return loss
    
    @torch.no_grad()
    def generate(
        self,
        encoder_out: torch.Tensor,
        max_len: int = 200,
        sos_id: int = 2,
        eos_id: int = 3
    ) -> List[List[int]]:
        """
        Autoregressive generation.
        
        Args:
            encoder_out: [batch, src_len, encoder_dim]
            max_len: Maximum generation length
            sos_id: Start of sequence token ID
            eos_id: End of sequence token ID
            
        Returns:
            List of generated token ID sequences
        """
        batch_size = encoder_out.size(0)
        device = encoder_out.device
        
        # Start with SOS token
        generated = torch.full((batch_size, 1), sos_id, dtype=torch.long, device=device)
        
        # Project encoder output once
        memory = self.encoder_proj(encoder_out)
        
        for _ in range(max_len):
            # Forward pass
            tgt_len = generated.size(1)
            positions = torch.arange(tgt_len, device=device).unsqueeze(0).expand(batch_size, -1)
            tgt_embed = self.token_embedding(generated) + self.pos_embedding(positions)
            
            causal_mask = self._generate_causal_mask(tgt_len, device)
            
            decoded = self.decoder(
                tgt=tgt_embed,
                memory=memory,
                tgt_mask=causal_mask
            )
            
            # Get next token
            logits = self.output_proj(decoded[:, -1, :])
            next_token = logits.argmax(dim=-1, keepdim=True)
            
            generated = torch.cat([generated, next_token], dim=1)
            
            # Check if all sequences have generated EOS
            if (next_token == eos_id).all():
                break
        
        # Convert to lists
        results = []
        for seq in generated.tolist():
            # Remove SOS and everything after EOS
            if eos_id in seq:
                seq = seq[:seq.index(eos_id)]
            if seq and seq[0] == sos_id:
                seq = seq[1:]
            results.append(seq)
        
        return results


if __name__ == "__main__":
    print("✅ TextHead module loaded successfully!")
    print("")
    
    # Test CTC text head
    batch_size = 2
    time_steps = 125
    encoder_dim = 512
    
    head = TextHead(encoder_dim=encoder_dim)
    
    # Dummy encoder output
    encoder_out = torch.randn(batch_size, time_steps, encoder_dim)
    
    # Forward pass
    logits = head(encoder_out)
    
    # Test decoding
    decoded = head.decode_greedy(logits)
    
    print(f"   CTC Text Head:")
    print(f"   Encoder output: {encoder_out.shape}")
    print(f"   Text logits:    {logits.shape}")
    print(f"   Vocab size:     {head.vocab_size}")
    print(f"   Decoded tokens: {decoded[0][:10]}...")
    print("")
    
    # Test attention decoder
    att_decoder = AttentionTextDecoder(encoder_dim=encoder_dim)
    
    # Dummy targets for teacher forcing
    targets = torch.randint(0, 100, (batch_size, 50))
    
    att_logits = att_decoder(encoder_out, targets)
    
    print(f"   Attention Decoder:")
    print(f"   Target shape:   {targets.shape}")
    print(f"   Output logits:  {att_logits.shape}")
