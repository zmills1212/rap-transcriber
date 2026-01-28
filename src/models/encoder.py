"""
Conformer Encoder for rap transcription.
Based on: "Conformer: Convolution-augmented Transformer for Speech Recognition"
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class ConvSubsampling(nn.Module):
    """
    Convolutional subsampling layer.
    Reduces time dimension by factor of 4.
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(1, out_channels, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.out_proj = nn.Linear(out_channels * (in_channels // 4), out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, time, features]
        Returns:
            [batch, time//4, out_channels]
        """
        # Add channel dimension: [batch, 1, time, features]
        x = x.unsqueeze(1)
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        # Reshape: [batch, channels, time, features] -> [batch, time, channels * features]
        batch, channels, time, features = x.size()
        x = x.permute(0, 2, 1, 3).contiguous().view(batch, time, channels * features)
        
        x = self.out_proj(x)
        return x


class FeedForwardModule(nn.Module):
    """Feed-forward module with expansion and dropout."""
    
    def __init__(self, d_model: int, expansion_factor: int = 4, dropout: float = 0.1):
        super().__init__()
        d_ff = d_model * expansion_factor
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.layer_norm(x)
        x = F.silu(self.linear1(x))  # SiLU (Swish) activation
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return residual + 0.5 * x  # Half-step residual


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention module."""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        residual = x
        x = self.layer_norm(x)
        x, _ = self.attention(x, x, x, key_padding_mask=mask)
        x = self.dropout(x)
        return residual + x


class ConvolutionModule(nn.Module):
    """
    Convolution module with gating.
    Captures local patterns in audio.
    """
    
    def __init__(self, d_model: int, kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Pointwise conv -> GLU -> Depthwise conv -> BatchNorm -> Swish -> Pointwise
        self.pointwise1 = nn.Linear(d_model, d_model * 2)
        self.depthwise = nn.Conv1d(
            d_model, d_model, 
            kernel_size=kernel_size, 
            padding=kernel_size // 2,
            groups=d_model
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.pointwise2 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.layer_norm(x)
        
        # Pointwise with GLU
        x = self.pointwise1(x)
        x = F.glu(x, dim=-1)
        
        # Depthwise conv (need to transpose for Conv1d)
        x = x.transpose(1, 2)  # [batch, d_model, time]
        x = self.depthwise(x)
        x = self.batch_norm(x)
        x = F.silu(x)
        x = x.transpose(1, 2)  # [batch, time, d_model]
        
        # Pointwise
        x = self.pointwise2(x)
        x = self.dropout(x)
        
        return residual + x


class ConformerBlock(nn.Module):
    """
    Single Conformer block.
    Structure: FFN -> MHSA -> Conv -> FFN
    """
    
    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        ff_expansion: int = 4,
        conv_kernel_size: int = 31,
        dropout: float = 0.1
    ):
        super().__init__()
        self.ffn1 = FeedForwardModule(d_model, ff_expansion, dropout)
        self.attention = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.conv = ConvolutionModule(d_model, conv_kernel_size, dropout)
        self.ffn2 = FeedForwardModule(d_model, ff_expansion, dropout)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = self.ffn1(x)
        x = self.attention(x, mask)
        x = self.conv(x)
        x = self.ffn2(x)
        x = self.layer_norm(x)
        return x


class ConformerEncoder(nn.Module):
    """
    Full Conformer encoder.
    Converts mel spectrograms to encoded representations.
    """
    
    def __init__(
        self,
        input_dim: int = 80,
        d_model: int = 512,
        num_layers: int = 12,
        num_heads: int = 8,
        ff_expansion: int = 4,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
        subsampling_factor: int = 4
    ):
        super().__init__()
        self.d_model = d_model
        self.subsampling_factor = subsampling_factor
        
        # Input projection with subsampling
        self.subsampling = ConvSubsampling(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)
        
        # Conformer blocks
        self.layers = nn.ModuleList([
            ConformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                ff_expansion=ff_expansion,
                conv_kernel_size=conv_kernel_size,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
    
    def forward(
        self, 
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: Mel spectrogram [batch, time, features] or [batch, features, time]
            mask: Optional padding mask [batch, time]
        
        Returns:
            encoder_out: [batch, time//4, d_model]
            mask: Updated mask for subsampled sequence
        """
        # Handle both [B, T, F] and [B, F, T] inputs
        if x.size(1) == 80:  # Likely [B, F, T]
            x = x.transpose(1, 2)  # -> [B, T, F]
        
        # Subsampling
        x = self.subsampling(x)
        
        # Update mask for subsampled sequence
        if mask is not None:
            # Subsample mask by factor of 4
            mask = mask[:, ::self.subsampling_factor]
        
        # Positional encoding
        x = self.pos_encoding(x)
        
        # Conformer blocks
        for layer in self.layers:
            x = layer(x, mask)
        
        return x, mask
    
    def get_output_length(self, input_length: int) -> int:
        """Calculate output sequence length."""
        return input_length // self.subsampling_factor


if __name__ == "__main__":
    print("✅ ConformerEncoder module loaded successfully!")
    print("")
    
    # Test with dummy input
    batch_size = 2
    time_frames = 500
    mel_bins = 80
    
    model = ConformerEncoder(
        input_dim=mel_bins,
        d_model=512,
        num_layers=12,
        num_heads=8
    )
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    
    dummy_input = torch.randn(batch_size, time_frames, mel_bins)
    output, _ = model(dummy_input)
    
    print(f"   Input shape:  {dummy_input.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Parameters:   {num_params:,}")
