"""
CWRU Paradigm Comparison: Proposed Two-Stage Framework vs. Poly-front Deep Baselines
[Comparison under High SNR Domain Generalization]

Core Design:
1. All deep models are optimized with the SAME proposed `PhaseAwareLoss`.
2. Proposed Paradigm (Mamba): Poly interpolation baseline + Pretrained Phase-Aware residual compensation.
3. Conventional Deep Baselines (CNN, BiLSTM, Transformer): Poly-front deep networks trained from scratch.
4. Evaluates PhsMAE, GD_MAE, F1, and Fisher Ratio with a stratified file-level hold-out split.
"""

import os, re, time, math, copy, json, random
import numpy as np
import scipy.io
import scipy.signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import Dict, List, Tuple

# ---- Local Imports ----
from model_mamba import ResidualCorrector, BiMambaBlock
from phase_aware_loss import PhaseAwareLoss  

# Sklearn / Scipy
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from scipy.interpolate import interp1d

# =========================
# 0) Config
# =========================
DATA_DIR = "./cwru_mat"
FS_HIGH = 48000
SCALE_K = 4
TARGET_LEN = 1024
HOP = TARGET_LEN // 2

BATCH_SIZE = 32
EPOCHS = 500       
PATIENCE = 15      
LR_MAMBA = 5e-5     # Mamba 微调 LR (使用预训练)
LR_BASE = 1e-4      # 传统深度基准从头训练 LR
WEIGHT_DECAY = 1e-4
SEED = 42
MULTI_SEEDS = [42, 123, 2025]

PRETRAINED_MODEL_PATH = "mamba_poly_phase_aware_best_noise.pth" 
OUT_DIR = "plots_cwru_architecture_comparison"
RESULT_DIR = "cwru_result"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loss Parameters
LAMBDA_FREQ = 0.05
LAMBDA_PHASE = 1.0
LAMBDA_GD = 0.1
ENERGY_THRESHOLD = 0.005

# Downstream Config
MAX_TRAIN_FEATS_PER_CLASS = 4000

# CWRU Loads
TRAIN_IDS = {
    "Normal": [97, 98, 99],
    "IR":     [109, 110, 111],
    "Ball":   [122, 123, 124],
    "OR":     [135, 136, 137],
}
TEST_IDS = {
    "Normal": [100],
    "IR":     [112],
    "Ball":   [125],
    "OR":     [138],
}

# =========================
# 1) Baseline Architectures
# =========================
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
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*2, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers, enable_nested_tensor=False)
        self.proj_out = nn.Conv1d(d_model, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.proj_in(x).permute(0, 2, 1) 
        x = self.transformer(x)
        return self.proj_out(x.permute(0, 2, 1))

class BaselineBiLSTM(nn.Module):
    def __init__(self, in_channels=2, hidden=64, num_layers=2, out_channels=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=in_channels, hidden_size=hidden, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden * 2, out_channels)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out.permute(0, 2, 1)

class KANLinear(nn.Module):
    def __init__(self, in_features, out_features, n_basis=5, sigma=0.5):
        super().__init__()
        self.sigma = sigma
        centers_init = torch.linspace(-1.0, 1.0, n_basis)
        self.centers = nn.Parameter(centers_init.unsqueeze(0).expand(in_features, -1).clone())
        self.basis_weight = nn.Parameter(torch.randn(out_features, in_features, n_basis) * 0.01)
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        x_c = torch.tanh(x).unsqueeze(-1)
        phi = torch.exp(-self.sigma * (x_c - self.centers) ** 2)
        return self.linear(x.squeeze(-1) if x.dim() > 2 else x) + torch.einsum('...ib,oib->...o', phi, self.basis_weight)

class VibrMambaBaseline(nn.Module):
    def __init__(self, d_model=64, n_layers=2, n_basis=5):
        super().__init__()
        self.input_kan  = KANLinear(2, d_model, n_basis=n_basis)
        self.layers     = nn.ModuleList([BiMambaBlock(d_model) for _ in range(n_layers)])
        self.kan_gates  = nn.ModuleList([KANLinear(d_model, d_model, n_basis=n_basis) for _ in range(n_layers)])
        self.output_kan = KANLinear(d_model, 2, n_basis=n_basis)
        nn.init.zeros_(self.output_kan.linear.weight)
        nn.init.zeros_(self.output_kan.linear.bias)
        nn.init.zeros_(self.output_kan.basis_weight)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.input_kan(x)
        for bi_mamba, gate in zip(self.layers, self.kan_gates):
            x = bi_mamba(x)
            x = gate(x)
        return self.output_kan(x).permute(0, 2, 1)

class LearnedDecompose(nn.Module):
    def __init__(self, n_components=4, kernel_sizes=(31, 9)):
        super().__init__()
        self.k1, self.k2 = kernel_sizes

    def _moving_avg(self, x, k):
        padding = k // 2
        x_t = x.permute(0, 2, 1)
        x_pad = F.pad(x_t, (padding, padding), mode='reflect')
        avg = F.avg_pool1d(x_pad, kernel_size=k, stride=1, padding=0)
        diff = x_t.shape[-1] - avg.shape[-1]
        if diff > 0: avg = F.pad(avg, (0, diff))
        elif diff < 0: avg = avg[..., :x_t.shape[-1]]
        return avg.permute(0, 2, 1)

    def forward(self, x):
        trend1 = self._moving_avg(x, self.k1)
        resid1 = x - trend1
        trend2 = self._moving_avg(resid1, self.k2)
        resid2 = resid1 - trend2
        return [trend1, resid1, trend2, resid2]

class MDBiMambaBaseline(nn.Module):
    def __init__(self, d_model=64, n_components=4, kernel_sizes=(31, 9)):
        super().__init__()
        self.input_proj  = nn.Linear(2, d_model)
        self.decompose   = LearnedDecompose(n_components, kernel_sizes)
        self.branches    = nn.ModuleList([BiMambaBlock(d_model) for _ in range(n_components)])
        self.fusion      = nn.Sequential(nn.Conv1d(d_model * n_components, d_model, kernel_size=1), nn.GELU())
        self.global_block = BiMambaBlock(d_model)
        self.output_proj = nn.Linear(d_model, 2)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x):
        x = self.input_proj(x.permute(0, 2, 1))
        components = self.decompose(x)
        cat = torch.cat([b(c) for b, c in zip(self.branches, components)], dim=-1)
        fused = self.fusion(cat.permute(0, 2, 1)).permute(0, 2, 1)
        return self.output_proj(self.global_block(fused)).permute(0, 2, 1)

# =========================
# 2) Helper Functions & Metrics
# =========================
def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def freq_to_time_domain(freq_channels, target_length=None):
    real = freq_channels[:, 0, :]  
    imag = freq_channels[:, 1, :]  
    complex_spec = torch.complex(real, imag)  
    if target_length is not None:
        time_signal = torch.fft.irfft(complex_spec, n=target_length, dim=-1)
    else:
        time_signal = torch.fft.irfft(complex_spec, dim=-1)
    return time_signal.unsqueeze(1) 

def compute_snr_db(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mse = float(np.mean((y_true - y_pred) ** 2))
    if mse < 1e-12: return 100.0
    sig_pow = float(np.mean(y_true ** 2))
    if sig_pow < 1e-12: return -100.0
    return float(10.0 * np.log10(sig_pow / (mse + 1e-12)))

def wrap_phase_diff(p_pred: np.ndarray, p_true: np.ndarray) -> np.ndarray:
    return np.angle(np.exp(1j * (p_pred - p_true)))

def compute_phase_mae(y_true: np.ndarray, y_pred: np.ndarray, cutoff_hz=500.0, fs=48000.0, thresh=0.05) -> float:
    spec_true = np.fft.rfft(y_true)
    spec_pred = np.fft.rfft(y_pred)
    freq = np.fft.rfftfreq(len(y_true), 1.0 / fs)
    mask_freq = (freq <= cutoff_hz)
    mag_true = np.abs(spec_true)
    peak = np.max(mag_true[mask_freq]) if np.any(mask_freq) else np.max(mag_true)
    mask_mag = (mag_true >= thresh * peak)
    mask = mask_freq & mask_mag
    if not np.any(mask): return float("nan")
    p_true = np.angle(spec_true[mask])
    p_pred = np.angle(spec_pred[mask])
    return float(np.mean(np.abs(wrap_phase_diff(p_pred, p_true))))

def group_delay_from_spectrum(S: np.ndarray, fs: int) -> np.ndarray:
    n_bins = len(S)
    if n_bins < 2: return np.full((n_bins,), np.nan, dtype=np.float32)
    phi = np.unwrap(np.angle(S)).astype(np.float64)
    dphi = np.diff(phi)
    N = (n_bins - 1) * 2
    df = fs / float(N)
    tau = np.full((n_bins,), np.nan, dtype=np.float64)
    tau[1:] = -dphi / (2.0 * np.pi * df + 1e-12)
    return tau.astype(np.float32)

def compute_gd_mae(y_true: np.ndarray, y_pred: np.ndarray, cutoff_hz=500.0, fs=48000.0, thresh=0.05) -> float:
    """🚨 完美的连续掩码校正版 (Consecutive Masking) 彻底杜绝黑哨"""
    spec_true = np.fft.rfft(y_true)
    spec_pred = np.fft.rfft(y_pred)
    freq = np.fft.rfftfreq(len(y_true), 1.0 / fs)
    mask_freq = (freq <= cutoff_hz)
    
    mag_true = np.abs(spec_true)
    peak = np.max(mag_true[mask_freq]) if np.any(mask_freq) else np.max(mag_true)
    mask_mag = (mag_true >= thresh * peak)
    base_mask = mask_freq & mask_mag
    
    # 连续有效性检查
    gd_mask = np.zeros_like(base_mask, dtype=bool)
    gd_mask[1:] = base_mask[1:] & base_mask[:-1]
    
    if not np.any(gd_mask): return float("nan")

    tau_true = group_delay_from_spectrum(spec_true, fs)
    tau_pred = group_delay_from_spectrum(spec_pred, fs)
    return float(np.mean(np.abs(tau_true[gd_mask] - tau_pred[gd_mask])))

def calc_metrics(y_true, y_hat):
    return {
        "SNR": compute_snr_db(y_true, y_hat),
        "PhsMAE": compute_phase_mae(y_true, y_hat, cutoff_hz=500.0, fs=FS_HIGH, thresh=0.05),
        "GD_MAE": compute_gd_mae(y_true, y_hat, cutoff_hz=500.0, fs=FS_HIGH, thresh=0.05)
    }

def calculate_fisher_ratio(X, y):
    X = np.asarray(X)
    y = np.asarray(y)
    classes = np.unique(y)
    mu = X.mean(axis=0, keepdims=True)
    Sb, Sw = 0.0, 0.0
    for c in classes:
        Xc = X[y == c]
        if len(Xc) == 0: continue
        muc = Xc.mean(axis=0, keepdims=True)
        Sb += len(Xc) * np.sum((muc - mu)**2)
        Sw += np.sum((Xc - muc)**2)
    return float(Sb / max(Sw, 1e-12))

def extract_feature_vector(y):
    eps = 1e-8
    y0 = np.asarray(y, dtype=np.float32).reshape(-1)
    y0 = y0 - y0.mean()

    rms = float(np.sqrt(np.mean(y0 * y0) + eps))
    peak = float(np.max(np.abs(y0)) + eps)
    crest = peak / rms
    mu4 = np.mean(y0 ** 4)
    kurt = float(mu4 / (np.mean(y0 ** 2) ** 2 + eps))

    spec = np.abs(np.fft.rfft(y0))
    freqs = np.fft.rfftfreq(len(y0), d=1.0/FS_HIGH)
    spec_sum = np.sum(spec) + eps
    spectral_centroid = float(np.sum(freqs * spec) / spec_sum)
    
    analytic_sig = scipy.signal.hilbert(y0)
    env = np.abs(analytic_sig)
    env0 = env - env.mean()
    env_mu4 = np.mean(env0 ** 4)
    env_kurt = float(env_mu4 / (np.mean(env0 ** 2) ** 2 + eps))

    return np.array([rms, crest, kurt, spectral_centroid, env_kurt], dtype=np.float32)

def downstream_check(Xtr, ytr, Xte, yte):
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=10.0, solver='liblinear', max_iter=2000, random_state=SEED))
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    acc = accuracy_score(yte, pred)
    f1 = f1_score(yte, pred, average="macro")
    try: sil = silhouette_score(Xte, yte)
    except: sil = 0.0
    fr = calculate_fisher_ratio(Xte, yte)
    return acc, f1, sil, fr

# =========================
# 3) Dataset Loading
# =========================
def find_de_time_key(mat_dict: dict) -> str:
    keys = [k for k in mat_dict.keys() if isinstance(k, str)]
    cand = [k for k in keys if ("DE_time" in k) and (not k.startswith("__"))]
    if cand: return cand[0]
    for k in keys:
        if not k.startswith("__") and isinstance(mat_dict[k], np.ndarray):
            return k
    raise KeyError("Cannot find DE_time-like key.")

def load_cwru_file(fname: str) -> np.ndarray:
    mat = scipy.io.loadmat(fname, struct_as_record=False, squeeze_me=True)
    key = find_de_time_key(mat)
    sig = np.asarray(mat[key], dtype=np.float32).flatten()
    return sig

def complex_to_channels(spec: np.ndarray) -> np.ndarray:
    return np.stack([spec.real, spec.imag], axis=0).astype(np.float32)

def fault_to_int(fault: str) -> int:
    mp = {"Normal": 0, "IR": 1, "Ball": 2, "OR": 3}
    return mp.get(fault, -1)

@dataclass
class SegmentMeta:
    file_id: int
    fault: str
    start_idx: int

class CWRUReconstructionDataset(Dataset):
    def __init__(self, data_dir: str, file_ids: Dict[str, List[int]], scale: int, target_len: int, hop: int):
        super().__init__()
        self.scale = scale
        self.target_len = target_len
        self.signals = []
        self.metas = []

        for fault, ids in file_ids.items():
            for fid in ids:
                fname = os.path.join(data_dir, f"{fid}.mat")
                if not os.path.exists(fname): continue
                sig = load_cwru_file(fname)
                
                # Z-score aligning
                sig = (sig - np.mean(sig)) / (np.std(sig) + 1e-8)
                
                for start in range(0, len(sig) - target_len + 1, hop):
                    self.signals.append(sig[start:start + target_len])
                    self.metas.append(SegmentMeta(fid, fault, start))

    def __len__(self): return len(self.signals)

    def __getitem__(self, idx):
        y_true = self.signals[idx]
        
        # Downsample
        if self.scale == 1: y_low = y_true
        else: y_low = scipy.signal.decimate(y_true, self.scale, ftype="fir", zero_phase=True).astype(np.float32)
        
        # Poly upsample
        if self.scale == 1: y_poly = y_low.astype(np.float32)
        else: y_poly = scipy.signal.resample_poly(y_low, up=self.scale, down=1)
        
        if len(y_poly) >= self.target_len: y_poly = y_poly[:self.target_len].astype(np.float32)
        else: y_poly = np.pad(y_poly, (0, self.target_len - len(y_poly)), mode='edge').astype(np.float32)

        spec_rect = np.fft.rfft(y_poly)
        scale_mag = float(np.std(np.abs(spec_rect)) + 1e-12)
        spec_rect_n = spec_rect / scale_mag
        
        inp = complex_to_channels(spec_rect_n)
        cls_idx = fault_to_int(self.metas[idx].fault)

        # 🚨 返回格式严格对齐 Gearbox 比较脚本
        return (torch.tensor(inp, dtype=torch.float32),
                torch.tensor(y_true, dtype=torch.float32), 
                torch.tensor(y_poly, dtype=torch.float32), 
                torch.tensor(cls_idx, dtype=torch.long), 
                torch.tensor(scale_mag, dtype=torch.float32))

# =========================
# 4) Training & Recon (Paradigm Aware)
# =========================
def train_model_fair(model, train_loader, val_loader, epochs=50, lr=1e-3, name="Model"):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    
    criterion = PhaseAwareLoss(lambda_freq=LAMBDA_FREQ, lambda_phase=LAMBDA_PHASE, lambda_gd=LAMBDA_GD, energy_threshold=ENERGY_THRESHOLD).to(DEVICE)
    
    best_loss = float('inf')
    patience_cnt = 0
    best_weights = None
    
    print(f"   🚀 Training {name} with Phase-Aware Loss (LR={lr}, MaxEp={epochs})...")
    
    for ep in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            x_in, y_gt_time, y_poly_time, _, scale = batch
            x_in, scale = x_in.to(DEVICE), scale.to(DEVICE)
            y_gt_time = y_gt_time.to(DEVICE).unsqueeze(1)
            y_poly_time = y_poly_time.to(DEVICE).unsqueeze(1)
            
            opt.zero_grad()
            pred_norm = model(x_in)
            
            pred_denorm = pred_norm * scale.view(-1, 1, 1)
            pred_time = freq_to_time_domain(pred_denorm, target_length=TARGET_LEN)
            
            # 🚨 核心逻辑：区分两步范式与 Poly 前端深度重构范式
            if name == "Mamba":
                pred_final = y_poly_time + pred_time 
            else:
                pred_final = pred_time 
            
            loss, _ = criterion(pred_final, y_gt_time, return_components=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x_in, y_gt_time, y_poly_time, _, scale = batch
                x_in, scale = x_in.to(DEVICE), scale.to(DEVICE)
                y_gt_time, y_poly_time = y_gt_time.to(DEVICE).unsqueeze(1), y_poly_time.to(DEVICE).unsqueeze(1)
                
                pred_norm = model(x_in)
                pred_time = freq_to_time_domain(pred_norm * scale.view(-1, 1, 1), target_length=TARGET_LEN)
                
                if name == "Mamba":
                    pred_final = y_poly_time + pred_time
                else:
                    pred_final = pred_time
                    
                l, _ = criterion(pred_final, y_gt_time, return_components=True)
                val_loss += l.item()
                
        avg_val = val_loss / len(val_loader)
        
        if avg_val < best_loss:
            best_loss = avg_val
            patience_cnt = 0
            best_weights = copy.deepcopy(model.state_dict())
        else:
            patience_cnt += 1
            
        if ep % 5 == 0 or ep==1:
            print(f"     [Ep {ep:03d}] Val Loss: {avg_val:.6f} | Best: {best_loss:.6f}")
        if patience_cnt >= PATIENCE:
            print(f"     🛑 Early stopping at epoch {ep}")
            break
            
    if best_weights: model.load_state_dict(best_weights)
    return model

@torch.no_grad()
def reconstruct(model, inp_channels, scale, y_base, name):
    model.eval()
    x = inp_channels.to(DEVICE)
    pred_norm = model(x).cpu().numpy()[0]
    s = float(scale.item())
    
    pred_complex = pred_norm[0] + 1j * pred_norm[1]
    pred_time = np.fft.irfft(pred_complex * s, n=TARGET_LEN).astype(np.float32)
    
    if name == "Mamba":
        return y_base.numpy()[0] + pred_time
    else:
        return pred_time

# =========================
# 5) Plotting & Export
# =========================
def plot_academic_bar_charts(final_results, methods, timestamp):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    colors = ["#95a5a6", "#e67e22", "#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#1abc9c"]
    metrics_config = [
        ("PhsMAE", "Phase Fidelity (Lower is Better)", "Phase MAE (rad)", False),
        ("GD_MAE", "Physical Latency (Lower is Better)", "Group Delay MAE (ms)", True),
        ("Fisher", "Feature Separability (Higher is Better)", "Fisher Ratio", False),
        ("F1", "Diagnostic Performance (Higher is Better)", "Macro F1-Score (%)", False)
    ]
    
    for idx, (m_key, title, ylabel, is_ms) in enumerate(metrics_config):
        ax = axes[idx]
        vals = [final_results[m][m_key] for m in methods]
        if is_ms: vals = [v * 1000 for v in vals] 
        
        bars = ax.bar(methods, vals, color=colors[:len(methods)], alpha=0.85, width=0.6, edgecolor='black', linewidth=1)
        ax.set_title(f"({chr(97+idx)}) {title}", fontweight='bold')
        ax.set_ylabel(ylabel)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        
        if m_key == "F1": ax.set_ylim(min(vals)-5, 100)
        
        for bar in bars:
            height = bar.get_height()
            text_y = height + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02
            ax.text(bar.get_x() + bar.get_width()/2, text_y, f"{height:.3f}" if m_key!="F1" else f"{height:.1f}", 
                    ha='center', va='bottom', fontsize=9)

    fig.suptitle("CWRU Method/Paradigm Comparison", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_path = os.path.join(OUT_DIR, f"Figure_CWRU_Method_Paradigm_Comparison_BarCharts_{timestamp}.png")
    plt.savefig(save_path, dpi=300)
    print(f"📊 4-Panel Bar Chart saved: {save_path}")
    plt.close()

def save_results(final_results, methods, val_meta, timestamp, seeds=None):
    txt_path  = os.path.join(RESULT_DIR, f"cwru_method_paradigm_comparison_{timestamp}.txt")
    csv_path  = os.path.join(RESULT_DIR, f"cwru_method_paradigm_comparison_{timestamp}.csv")
    json_path = os.path.join(RESULT_DIR, f"cwru_method_paradigm_comparison_{timestamp}.json")

    seed_str = str(seeds) if seeds else "single"
    lines = [
        "CWRU Method/Paradigm Comparison",
        f"Seeds: {seed_str}  (mean ± std over {len(seeds) if seeds else 1} runs)",
        f"Timestamp: {timestamp}", "",
        f"{'Method':<12} | {'SNR':>14} | {'PhsMAE':>14} | {'GD(ms)':>16} | {'Fisher':>14} | {'F1%':>14}",
        "-" * 90,
    ]
    for m in methods:
        r = final_results[m]
        if "SNR_mean" in r:
            lines.append(
                f"{m:<12} | {r['SNR_mean']:>6.2f}±{r['SNR_std']:>5.2f} | "
                f"{r['PhsMAE_mean']:>6.3f}±{r['PhsMAE_std']:>5.3f} | "
                f"{r['GD_MAE_mean']*1000:>7.3f}±{r['GD_MAE_std']*1000:>5.3f} | "
                f"{r['Fisher_mean']:>6.3f}±{r['Fisher_std']:>5.3f} | "
                f"{r['F1_mean']:>6.2f}±{r['F1_std']:>5.2f}"
            )
        else:
            lines.append(
                f"{m:<12} | {r['SNR']:>14.2f} | {r['PhsMAE']:>14.3f} | "
                f"{r['GD_MAE']*1000:>14.3f} | {r['Fisher']:>14.3f} | {r['F1']:>14.2f}"
            )
    with open(txt_path, "w") as f: f.write("\n".join(lines) + "\n")
    with open(json_path, "w") as f:
        json.dump({"timestamp": timestamp, "seeds": seeds,
                   "validation_files": val_meta, "results": final_results}, f, indent=2)
    with open(csv_path, "w") as f:
        f.write("Method,SNR_mean,SNR_std,PhsMAE_mean,PhsMAE_std,GD_mean,GD_std,Fisher_mean,Fisher_std,F1_mean,F1_std\n")
        for m in methods:
            r = final_results[m]
            if "SNR_mean" in r:
                f.write(f"{m},{r['SNR_mean']:.4f},{r['SNR_std']:.4f},{r['PhsMAE_mean']:.4f},{r['PhsMAE_std']:.4f},"
                        f"{r['GD_MAE_mean']:.6f},{r['GD_MAE_std']:.6f},{r['Fisher_mean']:.4f},{r['Fisher_std']:.4f},"
                        f"{r['F1_mean']:.4f},{r['F1_std']:.4f}\n")
    print(f"Results saved: {txt_path}")

# =========================
# 6) Single-seed runner
# =========================
def run_single_seed(seed, full_train_ds, test_ds, methods):
    set_seed(seed)

    # Seed-dependent val split: randomly pick one file per fault class
    rng = np.random.default_rng(seed)
    fids_by_fault = {}
    for meta in full_train_ds.metas:
        fids_by_fault.setdefault(meta.fault, set()).add(meta.file_id)
    val_fids = set()
    for fault, fids in sorted(fids_by_fault.items()):
        val_fids.add(int(rng.choice(sorted(fids))))

    train_indices = [i for i, m in enumerate(full_train_ds.metas) if m.file_id not in val_fids]
    val_indices   = [i for i, m in enumerate(full_train_ds.metas) if m.file_id in val_fids]
    train_loader = DataLoader(torch.utils.data.Subset(full_train_ds, train_indices),
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader   = DataLoader(torch.utils.data.Subset(full_train_ds, val_indices),
                              batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=1, shuffle=False)

    # Models
    set_seed(seed)
    models = {
        "BiLSTM":    BaselineBiLSTM().to(DEVICE),
        "CNN":       BaselineCNN().to(DEVICE),
        "Transf":    BaselineTransformer().to(DEVICE),
        "VibrMamba": VibrMambaBaseline(d_model=64, n_layers=2).to(DEVICE),
        "MDMamba":   MDBiMambaBaseline(d_model=40, n_components=4).to(DEVICE),
        "Mamba":     ResidualCorrector().to(DEVICE),
    }
    for name, model in models.items():
        lr = LR_BASE
        if name == "Mamba" and os.path.exists(PRETRAINED_MODEL_PATH):
            model.load_state_dict(torch.load(PRETRAINED_MODEL_PATH, map_location=DEVICE, weights_only=True))
            lr = LR_MAMBA
        train_model_fair(model, train_loader, val_loader, epochs=EPOCHS, lr=lr, name=name)

    # GT downstream classifier
    set_seed(seed)
    Xtr, ytr = [], []
    gen = torch.Generator().manual_seed(seed)
    loader_tr = DataLoader(full_train_ds, batch_size=1, shuffle=True, num_workers=0, generator=gen)
    cls_count = {0:0, 1:0, 2:0, 3:0}
    for batch in loader_tr:
        c = int(batch[3].item())
        if cls_count[c] < MAX_TRAIN_FEATS_PER_CLASS:
            Xtr.append(extract_feature_vector(batch[1].numpy()[0]))
            ytr.append(c)
            cls_count[c] += 1
    Xtr_arr, ytr_arr = np.array(Xtr), np.array(ytr)

    # Evaluation
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
            if torch.cuda.is_available(): torch.cuda.synchronize()
            t0 = time.perf_counter()
            y_hat = reconstruct(model, x_in, scale, batch[2], name)
            if torch.cuda.is_available(): torch.cuda.synchronize()
            infer_times[name].append(time.perf_counter() - t0)
            X_rec_pool[name].append(extract_feature_vector(y_hat))
            for k, v in calc_metrics(y_true_np, y_hat).items():
                metrics_pool[name][k].append(v)

    y_test = np.array(y_test_cls)
    seed_results = {}
    for m in methods:
        acc, f1, sil, fr = downstream_check(Xtr_arr, ytr_arr, np.array(X_rec_pool[m]), y_test)
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

# =========================
# 7) Main Execution
# =========================
def main():
    print("=" * 70)
    print("CWRU Method/Paradigm Comparison")
    print(f"Multi-seed run: seeds={MULTI_SEEDS}")
    print("Methods: Poly, BiLSTM, CNN, Transf, VibrMamba, MDMamba, Mamba")
    print("=" * 70)

    methods = ["Poly", "BiLSTM", "CNN", "Transf", "VibrMamba", "MDMamba", "Mamba"]

    full_train_ds = CWRUReconstructionDataset(DATA_DIR, TRAIN_IDS, SCALE_K, TARGET_LEN, HOP)
    test_ds       = CWRUReconstructionDataset(DATA_DIR, TEST_IDS,  SCALE_K, TARGET_LEN, HOP)

    # Val meta for reporting (seed 0)
    rng0 = np.random.default_rng(MULTI_SEEDS[0])
    fids_by_fault0 = {}
    for meta in full_train_ds.metas:
        fids_by_fault0.setdefault(meta.fault, set()).add(meta.file_id)
    val_fids0 = set()
    for fault, fids in sorted(fids_by_fault0.items()):
        val_fids0.add(int(rng0.choice(sorted(fids))))
    val_meta = sorted(val_fids0)

    all_seed_results = []
    for si, seed in enumerate(MULTI_SEEDS):
        print(f"\n{'='*70}")
        print(f"  Seed {seed}  ({si+1}/{len(MULTI_SEEDS)})")
        print(f"{'='*70}")
        sr = run_single_seed(seed, full_train_ds, test_ds, methods)
        all_seed_results.append(sr)

    # Aggregate mean ± std
    metric_keys = ["SNR", "PhsMAE", "GD_MAE", "Acc", "Fisher", "F1", "InferMs"]
    final_results = {}
    print(f"\n{'='*70}")
    print("Final results (mean ± std):")
    print(f"{'Method':<12}  {'SNR(dB)':>14}  {'F1 (%)':>14}  {'Fisher':>14}  {'GD (ms)':>14}  {'PhsMAE':>14}")
    for m in methods:
        final_results[m] = {}
        for k in metric_keys:
            vals = [sr[m][k] for sr in all_seed_results]
            final_results[m][f"{k}_mean"] = float(np.mean(vals))
            final_results[m][f"{k}_std"]  = float(np.std(vals))
        r = final_results[m]
        print(f"  {m:<12}  "
              f"SNR={r['SNR_mean']:.2f}±{r['SNR_std']:.2f}  "
              f"F1={r['F1_mean']:.2f}±{r['F1_std']:.2f}  "
              f"Fisher={r['Fisher_mean']:.3f}±{r['Fisher_std']:.3f}  "
              f"GD={r['GD_MAE_mean']*1000:.3f}±{r['GD_MAE_std']*1000:.3f}ms  "
              f"PhsMAE={r['PhsMAE_mean']:.3f}±{r['PhsMAE_std']:.3f}")

    mean_results = {m: {k: final_results[m][f"{k}_mean"] for k in metric_keys} for m in methods}
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    plot_academic_bar_charts(mean_results, methods, timestamp)
    save_results(final_results, methods, val_meta, timestamp, seeds=MULTI_SEEDS)
    print(f"\nDone. Results in '{RESULT_DIR}/', plots in '{OUT_DIR}/'")

if __name__ == "__main__":
    main()