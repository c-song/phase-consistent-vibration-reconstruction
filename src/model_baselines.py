"""
Baseline Models for Comparison: CNN and Transformer

Shares the same interface as ResidualCorrector (Mamba):
- Input:  [B, 2, K] complex spectrum (Re + Im channels)
- Output: [B, 2, K] residual spectrum
"""

import torch
import torch.nn as nn
import math


class CNNResidualBlock(nn.Module):
    """Conv1D + BatchNorm + ReLU residual block."""
    def __init__(self, channels, kernel_size=5, dilation=1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out


class CNNResidualCorrector(nn.Module):
    """
    CNN-based Residual Corrector.

    Architecture:
    - Input Projection: 2 -> d_model
    - Residual CNN blocks with multi-scale dilated convolutions (dilation 1, 2, 4, ...)
    - Output Projection: d_model -> 2

    Strengths: efficient local feature extraction.
    Weakness: limited receptive field; struggles with long-range dependencies.
    """
    def __init__(self, d_model=64, n_layer=6):
        super().__init__()

        # Input projection
        self.input_proj = nn.Conv1d(2, d_model, kernel_size=1)

        self.blocks = nn.ModuleList()
        for i in range(n_layer):
            dilation = 2 ** (i % 3)  # 1, 2, 4, 1, 2, 4
            self.blocks.append(CNNResidualBlock(d_model, kernel_size=5, dilation=dilation))

        # Output projection
        self.output_proj = nn.Conv1d(d_model, 2, kernel_size=1)

        # Zero-init so residual output starts at zero
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x):
        """
        Args:
            x: [B, 2, K] input spectrum (Re + Im)
        Returns:
            out: [B, 2, K] residual spectrum
        """
        out = self.input_proj(x)
        for block in self.blocks:
            out = block(out)
        out = self.output_proj(out)
        return out


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""
    def __init__(self, d_model, max_len=2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TransformerResidualCorrector(nn.Module):
    """
    Transformer-based Residual Corrector.

    Architecture:
    - Input Projection: 2 -> d_model
    - Positional Encoding
    - Transformer Encoder layers
    - Output Projection: d_model -> 2

    Strengths: global self-attention for long-range modeling.
    Weakness: O(L^2) complexity; slow for low-latency applications.
    """
    def __init__(self, d_model=64, n_layer=2, n_head=8, dim_feedforward=256, dropout=0.1):
        super().__init__()

        # Input projection
        self.input_proj = nn.Linear(2, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)

        # Output projection
        self.output_proj = nn.Linear(d_model, 2)

        # Initialize output projection to zero
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x):
        """
        Args:
            x: [B, 2, K] input spectrum (Re + Im)
        Returns:
            out: [B, 2, K] residual spectrum
        """
        B, C, K = x.shape
        x = x.transpose(1, 2)       # [B, K, 2]
        x = self.input_proj(x)      # [B, K, d_model]
        x = self.pos_encoder(x)
        x = self.transformer(x)     # [B, K, d_model]
        out = self.output_proj(x)   # [B, K, 2]
        out = out.transpose(1, 2)   # [B, 2, K]
        return out


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test input
    B, C, K = 4, 2, 257  # Batch, Channels (Real+Imag), Freq bins
    x = torch.randn(B, C, K)

    print("="*60)
    print("Baseline Models Comparison")
    print("="*60)

    # 1. CNN
    print("\n[1] CNN-based Residual Corrector")
    cnn_model = CNNResidualCorrector(d_model=64, n_layer=6)
    cnn_out = cnn_model(x)
    cnn_params = count_parameters(cnn_model)
    print(f"   Input shape:  {x.shape}")
    print(f"   Output shape: {cnn_out.shape}")
    print(f"   Parameters:   {cnn_params:,} ({cnn_params/1e6:.2f}M)")

    # 2. Transformer
    print("\n[2] Transformer-based Residual Corrector")
    transformer_model = TransformerResidualCorrector(d_model=64, n_layer=2, n_head=8)
    transformer_out = transformer_model(x)
    transformer_params = count_parameters(transformer_model)
    print(f"   Input shape:  {x.shape}")
    print(f"   Output shape: {transformer_out.shape}")
    print(f"   Parameters:   {transformer_params:,} ({transformer_params/1e6:.2f}M)")

    # 3. Mamba (for comparison)
    try:
        from model_mamba import ResidualCorrector
        print("\n[3] Mamba-based Residual Corrector")
        mamba_model = ResidualCorrector(d_model=64, n_layer=2)
        mamba_out = mamba_model(x)
        mamba_params = count_parameters(mamba_model)
        print(f"   Input shape:  {x.shape}")
        print(f"   Output shape: {mamba_out.shape}")
        print(f"   Parameters:   {mamba_params:,} ({mamba_params/1e6:.2f}M)")
    except ImportError:
        print("\n[3] Mamba model not available (requires mamba_ssm)")

    print("\n" + "="*60)
    print("Summary:")
    print(f"  CNN Params:         {cnn_params:,}")
    print(f"  Transformer Params: {transformer_params:,}")
    print(f"  Mamba Params:       {mamba_params:,}" if 'mamba_params' in locals() else "  Mamba Params:       N/A")
    print("="*60)
