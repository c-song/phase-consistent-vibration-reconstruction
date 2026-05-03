# model_mamba.py
import torch
import torch.nn as nn
try:
    from mamba_ssm import Mamba
except ImportError:
    print("Error: mamba_ssm not found. Install it on Linux with:")
    print("pip install mamba-ssm")
    exit()

class BiMambaBlock(nn.Module):
    """
    Bidirectional Mamba block.

    Spectra are non-causal, so both forward and backward scans are needed
    to capture global spectral leakage.
    """
    def __init__(self, d_model):
        super().__init__()
        self.mamba_fwd = Mamba(
            d_model=d_model,
            d_state=16,
            d_conv=4,
            expand=2
        )
        self.mamba_bwd = Mamba(
            d_model=d_model,
            d_state=16,
            d_conv=4,
            expand=2
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: [Batch, Length, Dim]
        out_fwd = self.mamba_fwd(x)

        # Reverse along the sequence axis, scan, then flip back
        x_rev = torch.flip(x, dims=[1])
        out_bwd = self.mamba_bwd(x_rev)
        out_bwd = torch.flip(out_bwd, dims=[1])

        return self.norm(out_fwd + out_bwd + x)

class ResidualCorrector(nn.Module):
    def __init__(self, d_model=64, n_layers=2):
        super(ResidualCorrector, self).__init__()
        
        # Input projection: 2 channels (Re/Im) -> d_model
        self.input_proj = nn.Linear(2, d_model)

        self.layers = nn.ModuleList([
            BiMambaBlock(d_model) for _ in range(n_layers)
        ])

        # Output projection: d_model -> 2 channels
        self.output_proj = nn.Linear(d_model, 2)

        # Zero-init output projection so the residual starts at zero,
        # preserving the FD baseline accuracy at the start of training.
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.input_proj.weight, mode='fan_in', nonlinearity='linear')
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x):
        # x: [B, 2, L] — permute to [B, L, 2] for Mamba/Linear layers
        x = x.permute(0, 2, 1)
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x)
        x = self.output_proj(x)
        x = x.permute(0, 2, 1)  # [B, L, 2] -> [B, 2, L]
        return x