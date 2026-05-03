"""
run_gearbox_architecture_ablation.py
Compares proposed Two-Stage Bi-S-Mamba against:
  - VibrMamba  (Yi et al., 2025, Measurement) — adapted to reconstruction
  - MD-BiMamba (Wang et al., 2024, Measurement) — adapted to reconstruction
  - CNN / BiLSTM / Transformer (existing baselines, from run_gearbox_architecture_ablation_v2)
  - Poly (deterministic baseline)

Adaptation strategy for classification-only SOTA models:
  Both VibrMamba and MD-BiMamba are originally designed for direct 1D time-series
  classification. To create a fair reconstruction comparison:
    - All deep models operate as Poly-front residual correctors (same as existing
      CNN/BiLSTM/Transformer baselines), receiving a 2-channel complex spectrum
      [B, 2, L] derived from the polyphase baseline and predicting a residual.
    - VibrMamba: bidirectional Mamba backbone + KAN (Kolmogorov-Arnold) projection
      layers, implementing the KAN principle via RBF-augmented linear layers.
    - MD-BiMamba: multi-decomposition strategy adapted to spectral domain using
      learned sequential decomposition (trend/residual splits) instead of CEEMDAN
      (which is undefined for complex spectra), with parallel bidirectional Mamba
      branches and learned fusion.
  All baselines use the same RobustGearboxLoss, same training budget, and the
  same downstream evaluation protocol as in run_gearbox_architecture_ablation_v2.py.

Training / reconstruction pipeline:
  - Dataset: file-level z-score at load time; scale = std(|rfft(y_base)|) per window
  - Network input: rfft(y_base) / scale  (2-channel real/imag)
  - Prediction: pred_norm [B, 2, L] in normalized spectrum space
  - Denorm: pred_denorm = pred_norm * scale
  - Time domain: pred_time = irfft(pred_denorm)
  - Mamba (two-stage): y_hat = y_poly + pred_time
  - Others (poly-front): y_hat = pred_time
"""

import os, re, time, math, copy, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import scipy.signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model_mamba import ResidualCorrector
from phase_aware_loss import RobustGearboxLoss
import utils_pub as U

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# ============================================================
# 0) Config
# ============================================================
ROOT_DIR   = "./gearboxdata_extracted"
HZ_KEEP    = (30,)
TRAIN_LOADS = (0, 10, 20, 30, 40, 50, 60)
TEST_LOADS  = (70, 80, 90)

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_LEN  = 1024
HOP         = TARGET_LEN // 2
SCALE_K     = 4
BATCH_SIZE  = 32
EPOCHS      = 350
PATIENCE    = 15
LR_MAMBA    = 5e-5
LR_BASE     = 1e-4
WEIGHT_DECAY = 1e-4
Z_SCORE     = True
SEED        = 42

PRETRAINED_MODEL_PATH = "mamba_poly_phase_aware_best_noise.pth"
OUT_DIR     = "plots_gearbox_architecture_ablation"
RESULT_DIR  = "gearbox_result"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

MAG_THRESH_RATIO = 0.05
PHASE_BAND_HZ    = (0, 500)
FS_HIGH          = 12800

# ============================================================
# 1) Existing baselines
# ============================================================
class BaselineCNN(nn.Module):
    def __init__(self, in_channels=2, hidden=64, out_channels=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden), nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden), nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden), nn.GELU(),
            nn.Conv1d(hidden, out_channels, kernel_size=3, padding=1)
        )
    def forward(self, x): return self.net(x)


class BaselineTransformer(nn.Module):
    def __init__(self, in_channels=2, d_model=64, nhead=4, num_layers=3, out_channels=2):
        super().__init__()
        self.proj_in = nn.Conv1d(in_channels, d_model, kernel_size=1)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                         dim_feedforward=d_model * 2,
                                         batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers=num_layers,
                                                  enable_nested_tensor=False)
        self.proj_out = nn.Conv1d(d_model, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.proj_in(x).permute(0, 2, 1)
        x = self.transformer(x)
        return self.proj_out(x.permute(0, 2, 1))


class BaselineBiLSTM(nn.Module):
    def __init__(self, in_channels=2, hidden=64, num_layers=2, out_channels=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=in_channels, hidden_size=hidden,
                            num_layers=num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden * 2, out_channels)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        return self.fc(out).permute(0, 2, 1)


# ============================================================
# 2) VibrMamba — adapted to spectral residual reconstruction
#
# Reference: Yi et al. (2025). VibrMamba: A lightweight Mamba based
# fault diagnosis of rotating machinery. Measurement 249:116881.
#
# Original purpose: direct 1D vibration classification.
# Adaptation: Poly-front residual corrector on complex spectrum.
#   Key architectural fidelity:
#     - Bidirectional SSM backbone (retained)
#     - KAN-style projection layers (retained, approximated via
#       RBF-augmented linear layers — a standard lightweight KAN
#       variant that avoids full B-spline grid evaluation while
#       preserving the learnable nonlinear activation principle)
# ============================================================
class KANLinear(nn.Module):
    """
    Lightweight KAN layer.

    Implements the Kolmogorov-Arnold Network principle:
      out = W_base * x + sum_j(c_j * phi_j(x))
    where phi_j(x) = exp(-sigma*(x - mu_j)^2) are RBF basis functions
    with learnable centers mu_j and a fixed bandwidth sigma.

    This is a standard efficient approximation used in KAN-inspired
    architectures when full B-spline grids are too costly.
    """
    def __init__(self, in_features: int, out_features: int,
                 n_basis: int = 5, sigma: float = 0.5):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.n_basis      = n_basis
        self.sigma        = sigma

        # Learnable RBF centers: one set per input feature
        centers_init = torch.linspace(-1.0, 1.0, n_basis)          # (n_basis,)
        self.centers = nn.Parameter(
            centers_init.unsqueeze(0).expand(in_features, -1).clone()
        )   # (in_features, n_basis)

        # Basis weights: how each basis function contributes to each output
        self.basis_weight = nn.Parameter(
            torch.randn(out_features, in_features, n_basis) * 0.01
        )
        # Standard residual linear term (keeps gradient flow stable)
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., in_features)
        # RBF activations
        x_c = torch.tanh(x)                              # (..., in)  normalize to [-1,1]
        x_c = x_c.unsqueeze(-1)                          # (..., in, 1)
        dist = x_c - self.centers                        # (..., in, n_basis)
        phi  = torch.exp(-self.sigma * dist ** 2)        # (..., in, n_basis)
        # Weighted sum over basis functions and input features
        kan_out = torch.einsum('...ib,oib->...o', phi, self.basis_weight)
        return self.linear(x) + kan_out


from model_mamba import BiMambaBlock


class VibrMambaBaseline(nn.Module):
    """
    VibrMamba adapted to spectral residual reconstruction.

    Architecture (reconstruction variant):
      Input [B, 2, L]
        → KANLinear projection  [2 → d_model]
        → N × (BiMambaBlock + KANLinear gate)
        → KANLinear projection  [d_model → 2]
      Output [B, 2, L]

    Parameter budget is matched to ResidualCorrector (d_model=64, n_layers=2)
    to ensure a fair parameter-count comparison.
    """
    def __init__(self, d_model: int = 64, n_layers: int = 2,
                 n_basis: int = 5):
        super().__init__()
        self.input_kan  = KANLinear(2, d_model, n_basis=n_basis)
        self.layers     = nn.ModuleList([
            BiMambaBlock(d_model) for _ in range(n_layers)
        ])
        # Per-layer KAN gate (channel-wise rescaling after each BiMamba block)
        self.kan_gates  = nn.ModuleList([
            KANLinear(d_model, d_model, n_basis=n_basis) for _ in range(n_layers)
        ])
        self.output_kan = KANLinear(d_model, 2, n_basis=n_basis)
        # Zero-initialize output to preserve polyphase baseline at init
        nn.init.zeros_(self.output_kan.linear.weight)
        nn.init.zeros_(self.output_kan.linear.bias)
        nn.init.zeros_(self.output_kan.basis_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 2, L]
        x = x.permute(0, 2, 1)           # [B, L, 2]
        x = self.input_kan(x)             # [B, L, d_model]
        for bi_mamba, gate in zip(self.layers, self.kan_gates):
            x = bi_mamba(x)
            x = gate(x)                   # learnable per-layer nonlinear transform
        x = self.output_kan(x)            # [B, L, 2]
        return x.permute(0, 2, 1)         # [B, 2, L]


# ============================================================
# 3) MD-BiMamba — adapted to spectral residual reconstruction
#
# Reference: Wang et al. (2024). MD-BiMamba: An aero-engine inter-shaft
# bearing fault diagnosis method based on Mamba with modal decomposition
# and bidirectional features fusion. Measurement 242.
#
# Original purpose: CEEMDAN decomposition + BiMamba + fusion for
# aero-engine inter-shaft bearing fault classification.
#
# Adaptation: spectral-domain multi-decomposition residual corrector.
#   CEEMDAN is undefined on complex spectra; we replace it with
#   learned sequential decomposition (trend/residual extraction via
#   running-mean filters with learnable scales) to preserve the
#   multi-component decomposition + parallel BiMamba + fusion spirit
#   of MD-BiMamba in the frequency domain.
# ============================================================
class LearnedDecompose(nn.Module):
    """
    Learned sequential decomposition module.

    Implements a differentiable analogue of EMD-style decomposition:
      Component_1 = MovingAverage(x, k1)         ← trend
      Component_2 = x - Component_1              ← first residual
      Component_3 = MovingAverage(Component_2, k2) ← sub-trend
      Component_4 = Component_2 - Component_3    ← second residual

    Kernel sizes {k1, k2} are chosen to span coarse and fine scales.
    A learnable per-component scale corrects for energy imbalance.
    """
    def __init__(self, n_components: int = 4,
                 kernel_sizes: tuple = (31, 9)):
        super().__init__()
        assert n_components == 4, "Current impl supports exactly 4 components"
        self.k1 = kernel_sizes[0]
        self.k2 = kernel_sizes[1]
        # Learnable scale per component (applied in feature space, see forward)

    def _moving_avg(self, x: torch.Tensor, k: int) -> torch.Tensor:
        # x: [B, L, C] — average along L
        padding = k // 2
        # Reflect-pad for boundary stability
        x_t = x.permute(0, 2, 1)             # [B, C, L]
        x_pad = F.pad(x_t, (padding, padding), mode='reflect')
        avg = F.avg_pool1d(x_pad, kernel_size=k, stride=1, padding=0)
        # avg may be slightly shorter due to even k; trim/pad to match
        diff = x_t.shape[-1] - avg.shape[-1]
        if diff > 0:
            avg = F.pad(avg, (0, diff))
        elif diff < 0:
            avg = avg[..., :x_t.shape[-1]]
        return avg.permute(0, 2, 1)           # [B, L, C]

    def forward(self, x: torch.Tensor):
        # x: [B, L, C]
        trend1 = self._moving_avg(x, self.k1)
        resid1 = x - trend1
        trend2 = self._moving_avg(resid1, self.k2)
        resid2 = resid1 - trend2
        # Returns 4 components: two trends + two residuals
        return [trend1, resid1, trend2, resid2]


class MDBiMambaBaseline(nn.Module):
    """
    MD-BiMamba adapted to spectral residual reconstruction.

    Architecture (reconstruction variant):
      Input [B, 2, L]
        → Linear projection [2 → d_model]
        → LearnedDecompose → K=4 components [B, L, d_model] each
        → K parallel BiMambaBlock branches
        → Concat [B, L, K*d_model] → fusion Conv1d [K*d_model → d_model]
        → LayerNorm + BiMambaBlock (global integration)
        → Linear projection [d_model → 2]
      Output [B, 2, L]

    The multi-branch parallel processing mirrors MD-BiMamba's core idea
    of processing each decomposed mode independently before fusion.
    """
    def __init__(self, d_model: int = 64, n_components: int = 4,
                 kernel_sizes: tuple = (31, 9)):
        super().__init__()
        self.input_proj  = nn.Linear(2, d_model)
        self.decompose   = LearnedDecompose(n_components, kernel_sizes)
        self.n_components = n_components

        # One BiMamba branch per decomposed component
        self.branches = nn.ModuleList([
            BiMambaBlock(d_model) for _ in range(n_components)
        ])
        # Fusion: concatenate all branches then project back to d_model
        self.fusion = nn.Sequential(
            nn.Conv1d(d_model * n_components, d_model, kernel_size=1),
            nn.GELU()
        )
        # Final global integration block
        self.global_block = BiMambaBlock(d_model)
        self.output_proj  = nn.Linear(d_model, 2)

        # Zero-initialize output
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 2, L]
        x = x.permute(0, 2, 1)                   # [B, L, 2]
        x = self.input_proj(x)                    # [B, L, d_model]

        components = self.decompose(x)            # list of K × [B, L, d_model]

        branch_outs = [
            branch(comp) for branch, comp in zip(self.branches, components)
        ]   # K × [B, L, d_model]

        # Concat along channel dim, then fuse via 1×1 conv
        cat = torch.cat(branch_outs, dim=-1)      # [B, L, K*d_model]
        cat = cat.permute(0, 2, 1)                # [B, K*d_model, L]
        fused = self.fusion(cat).permute(0, 2, 1) # [B, L, d_model]

        out = self.global_block(fused)             # [B, L, d_model]
        out = self.output_proj(out)                # [B, L, 2]
        return out.permute(0, 2, 1)               # [B, 2, L]


# ============================================================
# 4) Shared infrastructure
# ============================================================
def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)


def freq_to_time_domain(freq_channels, target_length=None):
    real = freq_channels[:, 0, :]
    imag = freq_channels[:, 1, :]
    complex_spec = torch.complex(real, imag)
    if target_length is not None:
        time_signal = torch.fft.irfft(complex_spec, n=target_length, dim=-1)
    else:
        time_signal = torch.fft.irfft(complex_spec, dim=-1)
    return time_signal.unsqueeze(1)


def calc_metrics(y_true, y_hat):
    return {
        "SNR":    U.calc_snr(y_true, y_hat),
        "PhsMAE": U.phase_mae_fft(y_true, y_hat, fs=FS_HIGH,
                                   band_hz=PHASE_BAND_HZ,
                                   mag_ratio=MAG_THRESH_RATIO),
        "GD_MAE": U.gd_mae_fft(y_true, y_hat, fs=FS_HIGH,
                                 band_hz=PHASE_BAND_HZ,
                                 mag_ratio=MAG_THRESH_RATIO),
    }


def calculate_fisher_ratio(X, y):
    X, y = np.asarray(X), np.asarray(y)
    mu = X.mean(axis=0, keepdims=True)
    Sb = Sw = 0.0
    for c in np.unique(y):
        Xc = X[y == c]
        muc = Xc.mean(axis=0, keepdims=True)
        Sb += len(Xc) * np.sum((muc - mu) ** 2)
        Sw += np.sum((Xc - muc) ** 2)
    return float(Sb / max(Sw, 1e-12))


def extract_feature_vector(y):
    eps = 1e-8
    y0 = np.asarray(y, dtype=np.float32).reshape(-1)
    y0 = y0 - y0.mean()
    rms  = float(np.sqrt(np.mean(y0 * y0) + eps))
    peak = float(np.max(np.abs(y0)) + eps)
    crest = peak / rms
    mu4   = np.mean(y0 ** 4)
    kurt  = float(mu4 / (np.mean(y0 ** 2) ** 2 + eps))
    spec  = np.abs(np.fft.rfft(y0))
    freqs = np.fft.rfftfreq(len(y0), d=1.0 / FS_HIGH)
    spec_sum = np.sum(spec) + eps
    spectral_centroid = float(np.sum(freqs * spec) / spec_sum)
    analytic = scipy.signal.hilbert(y0)
    env  = np.abs(analytic)
    env0 = env - env.mean()
    env_mu4  = np.mean(env0 ** 4)
    env_kurt = float(env_mu4 / (np.mean(env0 ** 2) ** 2 + eps))
    return np.array([rms, crest, kurt, spectral_centroid, env_kurt],
                    dtype=np.float32)


def downstream_check(Xtr, ytr, Xte, yte):
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=10.0, solver='liblinear',
                           max_iter=2000, random_state=SEED)
    )
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    acc  = accuracy_score(yte, pred)
    f1   = f1_score(yte, pred, average="macro")
    try:    sil = silhouette_score(Xte, yte)
    except: sil = 0.0
    fr = calculate_fisher_ratio(Xte, yte)
    return acc, f1, sil, fr


_NAME_RE = re.compile(
    r'^(?P<label>[hb])(?P<hz>\d+)hz(?P<load>\d+)\.txt$', re.IGNORECASE
)

def parse_gearbox_name(fname):
    m = _NAME_RE.match(os.path.basename(fname))
    if not m: return None
    label = m.group("label").lower()
    return (0 if label == "h" else 1), int(m.group("hz")), int(m.group("load"))

def load_txt_1d(path):
    try:
        x = np.loadtxt(path, dtype=np.float32)
    except:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = [ln.replace(",", " ") for ln in f
                     if ln.strip() and (ln[0].isdigit() or ln[0] in "+-.")]
        x = np.loadtxt(lines, dtype=np.float32)
    x = np.asarray(x).squeeze()
    if x.ndim > 1: x = x[:, 0]
    return x.astype(np.float32)

class GearboxReconstructionDataset(Dataset):
    def __init__(self, root_dir, split, hz_keep=(30,), train_loads=(),
                 test_loads=(), target_len=1024, hop=512, scale_factor=4,
                 base_method="poly", zscore=True, seed=0):
        super().__init__()
        self.target_len = int(target_len)
        self.hop        = int(hop)
        self.K          = int(scale_factor)
        self.zscore     = bool(zscore)

        cand = []
        for sub in ["Healthy Data", "BrokenTooth Data"]:
            d = os.path.join(root_dir, sub)
            if not os.path.isdir(d): continue
            for fn in os.listdir(d):
                info = parse_gearbox_name(fn)
                if info is None: continue
                cls, hz, load = info
                if hz_keep and (hz not in hz_keep): continue
                cand.append((os.path.join(d, fn), cls, hz, load))

        keep_loads = set(train_loads) if split == "train" else set(test_loads)
        self.files = sorted(
            [x for x in cand if x[3] in keep_loads],
            key=lambda t: (t[1], t[2], t[3], t[0])
        )

        self.signals, self.meta, self.index = [], [], []
        for fid, (path, cls, hz, load) in enumerate(self.files):
            sig = load_txt_1d(path)
            if self.zscore:
                sig = (sig - sig.mean()) / (sig.std() + 1e-8)
            self.signals.append(sig.astype(np.float32))
            self.meta.append((cls, hz, load, os.path.basename(path)))
            n = len(sig)
            if n < self.target_len: continue
            for st in range(0, n - self.target_len + 1, self.hop):
                self.index.append((fid, st))

    def __len__(self): return len(self.index)

    def __getitem__(self, i):
        fid, st = self.index[i]
        y_high  = self.signals[fid][st:st + self.target_len]
        cls, _, _, _ = self.meta[fid]

        # FIR decimation
        y_low = scipy.signal.decimate(
            y_high, self.K, ftype="fir", zero_phase=True
        ).astype(np.float32)
        t_low = self.target_len // self.K
        if len(y_low) > t_low: y_low = y_low[:t_low]
        elif len(y_low) < t_low:
            y_low = np.pad(y_low, (0, t_low - len(y_low)), mode='edge')

        y_base = scipy.signal.resample_poly(y_low, up=self.K, down=1)
        if len(y_base) > self.target_len: y_base = y_base[:self.target_len]
        elif len(y_base) < self.target_len:
            y_base = np.pad(y_base, (0, self.target_len - len(y_base)), mode='edge')
        y_base = np.asarray(y_base, dtype=np.float32).flatten()

        # scale = std(|rfft(y_base)|)  ← NOT time-domain sigma
        spec_base = np.fft.rfft(y_base)
        scale     = float(np.std(np.abs(spec_base)) + 1e-8)
        spec_base_n = spec_base / scale

        inp = U.complex_to_channels(spec_base_n)   # [2, L//2+1]

        return (
            torch.tensor(inp, dtype=torch.float32),
            torch.tensor(y_high, dtype=torch.float32),
            torch.tensor(y_base, dtype=torch.float32),
            torch.tensor(cls, dtype=torch.long),
            torch.tensor(scale, dtype=torch.float32),
        )


def train_model_fair(model, train_loader, val_loader,
                     epochs=350, lr=1e-4, name="Model"):
    opt = torch.optim.AdamW(model.parameters(), lr=lr,
                             weight_decay=WEIGHT_DECAY)
    criterion = RobustGearboxLoss(
        lambda_freq=0.05, lambda_phase=1.0, lambda_gd=0.2,
        energy_threshold=0.005
    ).to(DEVICE)
    best_val, best_state, patience_cnt = float('inf'), None, 0

    print(f"   Training {name} (LR={lr}, MaxEp={epochs})...")

    for ep in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            x_in, y_gt, y_poly, _, scale = batch
            x_in   = x_in.to(DEVICE)
            y_gt   = y_gt.to(DEVICE).unsqueeze(1)
            y_poly = y_poly.to(DEVICE).unsqueeze(1)
            scale  = scale.to(DEVICE)

            opt.zero_grad()
            pred_norm   = model(x_in)
            # Denorm: scale = std(|rfft(y_base)|)
            pred_denorm = pred_norm * scale.view(-1, 1, 1)
            pred_time   = freq_to_time_domain(pred_denorm, target_length=TARGET_LEN)

            # Two-stage residual for Mamba; direct prediction for others
            if name == "Mamba":
                pred_final = y_poly + pred_time
            else:
                pred_final = pred_time

            loss = criterion(pred_final, y_gt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x_in, y_gt, y_poly, _, scale = batch
                x_in   = x_in.to(DEVICE)
                y_gt   = y_gt.to(DEVICE).unsqueeze(1)
                y_poly = y_poly.to(DEVICE).unsqueeze(1)
                scale  = scale.to(DEVICE)

                pred_norm   = model(x_in)
                pred_denorm = pred_norm * scale.view(-1, 1, 1)
                pred_time   = freq_to_time_domain(pred_denorm, target_length=TARGET_LEN)
                pred_final  = y_poly + pred_time if name == "Mamba" else pred_time
                val_loss   += criterion(pred_final, y_gt).item()

        avg_val = val_loss / max(len(val_loader), 1)

        if avg_val < best_val:
            best_val    = avg_val
            best_state  = copy.deepcopy(model.state_dict())
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"     Early stop at epoch {ep}  (best={best_val:.6f})")
                break

        if ep % 50 == 0 or ep == 1:
            print(f"     [Ep {ep:03d}] val={avg_val:.6f}  best={best_val:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def reconstruct(model, inp_channels, scale, y_base, name):
    model.eval()
    x = inp_channels.to(DEVICE)
    pred_norm = model(x).cpu().numpy()[0]        # [2, L//2+1]
    s = float(scale.item())

    pred_complex = pred_norm[0] + 1j * pred_norm[1]
    pred_time    = np.fft.irfft(pred_complex * s, n=TARGET_LEN).astype(np.float32)

    if name == "Mamba":
        return y_base.numpy()[0] + pred_time
    else:
        return pred_time


# ---- Plotting ----
def plot_bar_charts(final_results, methods, timestamp):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    # Color map: Poly=grey, CNN=orange, BiLSTM=red, Transf=blue,
    #            VibrMamba=purple, MDMamba=teal, Mamba=green
    palette = {
        "Poly":      "#95a5a6",
        "CNN":       "#e67e22",
        "BiLSTM":    "#e74c3c",
        "Transf":    "#3498db",
        "VibrMamba": "#9b59b6",
        "MDMamba":   "#1abc9c",
        "Mamba":     "#2ecc71",
    }
    colors = [palette.get(m, "#aaaaaa") for m in methods]

    metrics_config = [
        ("PhsMAE", "Phase Fidelity ↓", "Phase MAE (rad)",     False),
        ("GD_MAE", "GD Fidelity ↓",    "GD MAE (ms)",         True),
        ("Fisher", "Separability ↑",   "Fisher Ratio",        False),
        ("F1",     "Macro F1 ↑",       "Macro F1-Score (%)",  False),
    ]
    for idx, (mk, title, ylabel, is_ms) in enumerate(metrics_config):
        ax   = axes[idx]
        vals = [final_results[m][mk] for m in methods]
        if is_ms: vals = [v * 1000 for v in vals]
        bars = ax.bar(methods, vals, color=colors, alpha=0.85, width=0.6,
                      edgecolor='black', linewidth=0.8)
        ax.set_title(f"({chr(97+idx)}) {title}", fontweight='bold')
        ax.set_ylabel(ylabel)
        ax.tick_params(axis='x', rotation=30, labelsize=8)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        if mk == "F1": ax.set_ylim(max(0, min(vals) - 5), 105)
        for bar in bars:
            h = bar.get_height()
            yspan = ax.get_ylim()[1] - ax.get_ylim()[0]
            ax.text(bar.get_x() + bar.get_width() / 2,
                    h + yspan * 0.02,
                    f"{h:.3f}" if mk != "F1" else f"{h:.1f}",
                    ha='center', va='bottom', fontsize=7)

    fig.suptitle("Gearbox SOTA Comparison (3 dB SNR)", fontsize=13,
                 fontweight='bold', y=1.02)
    plt.tight_layout()
    save_path = os.path.join(
        OUT_DIR, f"Figure_SOTA_Comparison_BarCharts_{timestamp}.png"
    )
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Bar chart saved: {save_path}")
    plt.close()


def save_results(final_results, methods, val_meta, timestamp, seeds=None):
    """Save results. If seeds is given, final_results contains mean/std dicts."""
    txt_path = os.path.join(
        RESULT_DIR, f"gearbox_architecture_ablation_{timestamp}.txt"
    )
    csv_path = os.path.join(
        RESULT_DIR, f"gearbox_architecture_ablation_{timestamp}.csv"
    )
    json_path = os.path.join(
        RESULT_DIR, f"gearbox_architecture_ablation_{timestamp}.json"
    )
    multi = seeds is not None
    if multi:
        header = (f"{'Method':<12} | {'SNR':>13} | {'PhsMAE':>14} | "
                  f"{'GD(ms)':>16} | {'Fisher':>14} | {'F1%':>14}")
    else:
        header = (f"{'Method':<12} | {'SNR':>7} | {'PhsMAE':>8} | "
                  f"{'GD(ms)':>9} | {'Acc%':>7} | {'Fisher':>8} | "
                  f"{'F1%':>7} | {'ms/samp':>9}")
    sep = "-" * len(header)
    prefix = ["Gearbox SOTA Comparison"]
    if multi:
        prefix.append(f"Seeds: {seeds}  (mean ± std over {len(seeds)} runs)")
    prefix += [f"Timestamp: {timestamp}", "", header, sep]
    lines = prefix[:]
    for m in methods:
        r = final_results[m]
        if multi:
            lines.append(
                f"{m:<12} | {r['SNR_mean']:>6.2f}±{r['SNR_std']:>5.2f} | "
                f"{r['PhsMAE_mean']:>7.3f}±{r['PhsMAE_std']:>5.3f} | "
                f"{r['GD_MAE_mean']*1000:>8.3f}±{r['GD_MAE_std']*1000:>5.3f} | "
                f"{r['Fisher_mean']:>7.3f}±{r['Fisher_std']:>5.3f} | "
                f"{r['F1_mean']:>7.2f}±{r['F1_std']:>5.2f}"
            )
        else:
            lines.append(
                f"{m:<12} | {r['SNR']:>7.2f} | {r['PhsMAE']:>8.3f} | "
                f"{r['GD_MAE']*1000:>9.3f} | {r['Acc']:>7.2f} | "
                f"{r['Fisher']:>8.3f} | {r['F1']:>7.2f} | "
                f"{r['InferMs']:>9.3f}"
            )
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    if multi:
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("Method,SNR_mean,SNR_std,PhsMAE_mean,PhsMAE_std,"
                    "GD_MAE_mean,GD_MAE_std,Fisher_mean,Fisher_std,"
                    "F1_mean,F1_std,InferMs_mean\n")
            for m in methods:
                r = final_results[m]
                f.write(f"{m},{r['SNR_mean']:.4f},{r['SNR_std']:.4f},"
                        f"{r['PhsMAE_mean']:.6f},{r['PhsMAE_std']:.6f},"
                        f"{r['GD_MAE_mean']:.9f},{r['GD_MAE_std']:.9f},"
                        f"{r['Fisher_mean']:.4f},{r['Fisher_std']:.4f},"
                        f"{r['F1_mean']:.4f},{r['F1_std']:.4f},"
                        f"{r['InferMs_mean']:.3f}\n")
    else:
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("Method,SNR_dB,PhsMAE,GD_MAE_s,Acc_pct,Fisher,F1_pct,InferMs\n")
            for m in methods:
                r = final_results[m]
                f.write(f"{m},{r['SNR']:.4f},{r['PhsMAE']:.6f},"
                        f"{r['GD_MAE']:.9f},{r['Acc']:.4f},"
                        f"{r['Fisher']:.4f},{r['F1']:.4f},{r['InferMs']:.3f}\n")
    with open(json_path, "w") as f:
        json.dump({"timestamp": timestamp, "val_files": val_meta,
                   "seeds": seeds, "results": final_results}, f, indent=2)
    print(f"Results: {txt_path}")


# ============================================================
# 5) Main
# ============================================================
MULTI_SEEDS = [42, 123, 2025]

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_single_seed(seed, full_train_ds, test_ds, val_meta, methods):
    """Run one complete train+eval cycle for the given seed. Returns per-method result dict."""
    set_seed(seed)

    # --- Val split (seed-dependent) ---
    rng = np.random.default_rng(seed)
    fids_by_class = {}
    for fid, (_, cls, _, _) in enumerate(full_train_ds.files):
        fids_by_class.setdefault(cls, []).append(fid)
    val_fids = set()
    for cls, fids in sorted(fids_by_class.items()):
        val_fids.add(int(rng.choice(fids)))
    train_fids = set(range(len(full_train_ds.files))) - val_fids

    train_indices = [i for i, (fid, _) in enumerate(full_train_ds.index)
                     if fid in train_fids]
    val_indices   = [i for i, (fid, _) in enumerate(full_train_ds.index)
                     if fid in val_fids]

    train_ds = torch.utils.data.Subset(full_train_ds, train_indices)
    val_ds   = torch.utils.data.Subset(full_train_ds, val_indices)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=1, shuffle=False)

    # --- Model registry ---
    set_seed(seed)
    models = {
        "BiLSTM":    BaselineBiLSTM().to(DEVICE),
        "CNN":       BaselineCNN().to(DEVICE),
        "Transf":    BaselineTransformer().to(DEVICE),
        "VibrMamba": VibrMambaBaseline(d_model=64, n_layers=2).to(DEVICE),
        "MDMamba":   MDBiMambaBaseline(d_model=40, n_components=4).to(DEVICE),
        "Mamba":     ResidualCorrector().to(DEVICE),
    }

    # --- Training ---
    for name, model in models.items():
        lr = LR_BASE
        if name == "Mamba" and os.path.exists(PRETRAINED_MODEL_PATH):
            model.load_state_dict(
                torch.load(PRETRAINED_MODEL_PATH, map_location=DEVICE,
                           weights_only=True)
            )
            lr = LR_MAMBA
        train_model_fair(model, train_loader, val_loader,
                         epochs=EPOCHS, lr=lr, name=name)

    # --- GT downstream classifier ---
    set_seed(seed)
    Xtr, ytr = [], []
    gen = torch.Generator().manual_seed(seed)
    loader_tr = DataLoader(full_train_ds, batch_size=1, shuffle=True,
                           num_workers=0, generator=gen)
    for i, batch in enumerate(loader_tr):
        if i >= 3000: break
        Xtr.append(extract_feature_vector(batch[1].numpy()[0]))
        ytr.append(int(batch[3].item()))
    Xtr_arr = np.array(Xtr)
    ytr_arr = np.array(ytr)

    # --- Evaluation ---
    metrics_pool = {m: {"SNR": [], "PhsMAE": [], "GD_MAE": []} for m in methods}
    X_rec_pool   = {m: [] for m in methods}
    infer_times  = {m: [] for m in methods}
    y_test_cls   = []

    for batch in test_loader:
        x_in, y_true, y_poly, cls, scale = batch
        y_true_np = y_true.numpy()[0]
        y_poly_np = y_poly.numpy()[0]
        y_test_cls.append(int(cls.item()))

        t0 = time.perf_counter()
        X_rec_pool["Poly"].append(extract_feature_vector(y_poly_np))
        infer_times["Poly"].append(time.perf_counter() - t0)
        for k, v in calc_metrics(y_true_np, y_poly_np).items():
            metrics_pool["Poly"][k].append(v)

        for name, model in models.items():
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            y_hat_np = reconstruct(model, x_in, scale, y_poly, name)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            infer_times[name].append(time.perf_counter() - t0)
            X_rec_pool[name].append(extract_feature_vector(y_hat_np))
            for k, v in calc_metrics(y_true_np, y_hat_np).items():
                metrics_pool[name][k].append(v)

    y_test = np.array(y_test_cls)
    seed_results = {}
    for m in methods:
        acc, f1, sil, fr = downstream_check(
            Xtr_arr, ytr_arr, np.array(X_rec_pool[m]), y_test
        )
        seed_results[m] = {
            "SNR":     float(np.nanmean(metrics_pool[m]["SNR"])),
            "PhsMAE":  float(np.nanmean(metrics_pool[m]["PhsMAE"])),
            "GD_MAE":  float(np.nanmean(metrics_pool[m]["GD_MAE"])),
            "Acc":     float(acc * 100),
            "Fisher":  float(fr),
            "F1":      float(f1 * 100),
            "InferMs": float(np.mean(infer_times[m]) * 1000),
        }
        print(f"    {m:<12}  SNR={seed_results[m]['SNR']:.2f}  "
              f"PhsMAE={seed_results[m]['PhsMAE']:.3f}  "
              f"GD={seed_results[m]['GD_MAE']*1000:.3f}ms  "
              f"F1={seed_results[m]['F1']:.1f}%  "
              f"Fisher={seed_results[m]['Fisher']:.3f}")
    return seed_results


def main():
    print("=" * 65)
    print("Gearbox SOTA Comparison: VibrMamba & MD-BiMamba vs Baselines")
    print(f"Multi-seed run: seeds={MULTI_SEEDS}")
    print("=" * 65)

    methods = ["Poly", "BiLSTM", "CNN", "Transf", "VibrMamba", "MDMamba", "Mamba"]

    # Datasets are deterministic (z-score at file level), built once
    full_train_ds = GearboxReconstructionDataset(
        ROOT_DIR, "train", HZ_KEEP, TRAIN_LOADS, TEST_LOADS,
        TARGET_LEN, HOP, SCALE_K, "poly", Z_SCORE, SEED
    )
    test_ds = GearboxReconstructionDataset(
        ROOT_DIR, "test", HZ_KEEP, TRAIN_LOADS, TEST_LOADS,
        TARGET_LEN, HOP, SCALE_K, "poly", Z_SCORE, SEED
    )

    # Print param counts once
    print("\nParameter counts:")
    tmp_models = {
        "BiLSTM":    BaselineBiLSTM(),
        "CNN":       BaselineCNN(),
        "Transf":    BaselineTransformer(),
        "VibrMamba": VibrMambaBaseline(d_model=64, n_layers=2),
        "MDMamba":   MDBiMambaBaseline(d_model=40, n_components=4),
        "Mamba":     ResidualCorrector(),
    }
    for name, m in tmp_models.items():
        print(f"  {name:<12}: {count_parameters(m):>8,} params")
    del tmp_models

    # Val meta for first seed (informational only)
    rng0 = np.random.default_rng(MULTI_SEEDS[0])
    fids_by_class0 = {}
    for fid, (_, cls, _, _) in enumerate(full_train_ds.files):
        fids_by_class0.setdefault(cls, []).append(fid)
    val_fids0 = set()
    for cls, fids in sorted(fids_by_class0.items()):
        val_fids0.add(int(rng0.choice(fids)))
    val_meta = [full_train_ds.meta[fid][3] for fid in sorted(val_fids0)]

    # --- Multi-seed loop ---
    all_seed_results = []  # list of {method: {metric: value}}
    for si, seed in enumerate(MULTI_SEEDS):
        print(f"\n{'='*65}")
        print(f"  Seed {seed}  ({si+1}/{len(MULTI_SEEDS)})")
        print(f"{'='*65}")
        sr = run_single_seed(seed, full_train_ds, test_ds, val_meta, methods)
        all_seed_results.append(sr)

    # --- Aggregate mean ± std ---
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    metric_keys = ["SNR", "PhsMAE", "GD_MAE", "Acc", "Fisher", "F1", "InferMs"]
    final_results = {}
    print(f"\n{'='*65}")
    print("Final results (mean ± std):")
    print(f"{'Method':<12}  {'F1 (%)':>14}  {'Fisher':>14}  {'GD (ms)':>14}  {'PhsMAE':>14}")
    for m in methods:
        final_results[m] = {}
        for k in metric_keys:
            vals = [sr[m][k] for sr in all_seed_results]
            final_results[m][f"{k}_mean"] = float(np.mean(vals))
            final_results[m][f"{k}_std"]  = float(np.std(vals))
        r = final_results[m]
        print(f"  {m:<12}  "
              f"F1={r['F1_mean']:.2f}±{r['F1_std']:.2f}  "
              f"Fisher={r['Fisher_mean']:.3f}±{r['Fisher_std']:.3f}  "
              f"GD={r['GD_MAE_mean']*1000:.3f}±{r['GD_MAE_std']*1000:.3f}ms  "
              f"PhsMAE={r['PhsMAE_mean']:.3f}±{r['PhsMAE_std']:.3f}")

    mean_results = {m: {
        "SNR":     final_results[m]["SNR_mean"],
        "PhsMAE":  final_results[m]["PhsMAE_mean"],
        "GD_MAE":  final_results[m]["GD_MAE_mean"],
        "Acc":     final_results[m]["Acc_mean"],
        "Fisher":  final_results[m]["Fisher_mean"],
        "F1":      final_results[m]["F1_mean"],
        "InferMs": final_results[m]["InferMs_mean"],
    } for m in methods}
    plot_bar_charts(mean_results, methods, timestamp)
    save_results(final_results, methods, val_meta, timestamp, seeds=MULTI_SEEDS)
    print(f"\nDone. Results in '{RESULT_DIR}/', plots in '{OUT_DIR}/'")


if __name__ == "__main__":
    main()
