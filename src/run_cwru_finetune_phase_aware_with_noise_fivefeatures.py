“””
run_cwru_finetune_phase_aware_with_noise_fivefeatures.py

Fine-tune the pretrained Mamba corrector on CWRU data using PhaseAwareLoss.
Dataset segments are pre-cached in RAM to avoid repeated CPU downsampling during training.
GD_MAE uses local complex differences to prevent phase-unwrap failure on real CWRU noise.
“””

import os
import sys
import time
import random
import numpy as np
import scipy.io
import scipy.signal
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import Dict, List, Tuple

from phase_aware_loss import PhaseAwareLoss

# -------------------------
# 0. Configuration
# -------------------------
DATA_DIR = "./cwru_mat"
FS_HIGH = 48000
SCALE = 4
TARGET_LEN = 1024
HOP = TARGET_LEN // 2

# Training Setup
EPOCHS = 1000
PATIENCE = 15
BATCH = 32
LR = 1e-4
WEIGHT_DECAY = 1e-4
SEED = 42

# Metrics Config
PHASE_CUTOFF = 500
PHASE_THRESH_RATIO = 0.05
GD_CUTOFF = 500

# Phase-Aware Loss Config
LAMBDA_FREQ = 0.05
LAMBDA_PHASE = 1.0
LAMBDA_GD = 0.0
ENERGY_THRESHOLD = 0.005  

# Downstream Config
MAX_TRAIN_FEATS_PER_CLASS = 4000

# Paths
PRETRAINED_MODEL_PATH = "mamba_poly_phase_aware_best_noise.pth"
OUT_DIR = "plots_cwru_phase_aware_finetune"
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Train (Load 2)
TRAIN_IDS = {
    "Normal": [97, 98, 99],
    "IR":     [109, 110, 111],
    "Ball":   [122, 123, 124],
    "OR":     [135, 136, 137],
}
# Test (Load 3 - cross-load generalization)
TEST_IDS = {
    "Normal": [100],
    "IR":     [112],
    "Ball":   [125],
    "OR":     [138],
}

# -------------------------
# 1. Utils & Logging
# -------------------------
class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def find_de_time_key(mat_dict: dict) -> str:
    keys = [k for k in mat_dict.keys() if isinstance(k, str)]
    cand = [k for k in keys if ("DE_time" in k) and (not k.startswith("__"))]
    if cand: return cand[0]
    best_k, best_len = None, 0
    for k in keys:
        if k.startswith("__"): continue
        v = mat_dict[k]
        if isinstance(v, np.ndarray) and v.size > best_len and v.ndim in (1, 2):
            best_k, best_len = k, v.size
    if best_k is None: raise KeyError("Cannot find DE_time-like key in .mat.")
    return best_k

def load_cwru_file(fname: str) -> np.ndarray:
    mat = scipy.io.loadmat(fname, struct_as_record=False, squeeze_me=True)
    key = find_de_time_key(mat)
    sig = np.asarray(mat[key], dtype=np.float32).flatten()
    return sig

def downsample_antialias(x: np.ndarray, scale: int) -> np.ndarray:
    if scale == 1: return x
    return scipy.signal.decimate(x, scale, ftype="fir", zero_phase=True).astype(np.float32)

def upsample_rect_fft_resample(y_low: np.ndarray, target_len: int) -> np.ndarray:
    if len(y_low) == target_len: return y_low.astype(np.float32)
    return scipy.signal.resample(y_low, target_len).astype(np.float32)

def upsample_poly(y_low: np.ndarray, scale: int, target_len: int) -> np.ndarray:
    if scale == 1:
        y = y_low.astype(np.float32)
    else:
        y = scipy.signal.resample_poly(y_low, up=scale, down=1)
    if len(y) >= target_len:
        return y[:target_len].astype(np.float32)
    return np.pad(y, (0, target_len - len(y)), mode='edge').astype(np.float32)

def upsample_cubic(y_low: np.ndarray, scale: int, target_len: int) -> np.ndarray:
    if len(y_low) == target_len: return y_low.astype(np.float32)
    n_low = len(y_low)
    x_low = np.arange(n_low, dtype=np.float32)
    x_hi = np.arange(target_len, dtype=np.float32) / scale
    f = interp1d(x_low, y_low, kind="cubic", fill_value="extrapolate")
    return f(x_hi).astype(np.float32)

def complex_to_channels(spec: np.ndarray) -> np.ndarray:
    return np.stack([spec.real, spec.imag], axis=0).astype(np.float32)

def fault_to_int(fault: str) -> int:
    mp = {"Normal": 0, "IR": 1, "Ball": 2, "OR": 3}
    return mp.get(fault, -1)

# -------------------------
# 2. Physics Metrics
# -------------------------
def compute_snr_db(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mse = float(np.mean((y_true - y_pred) ** 2))
    if mse < 1e-12: return 100.0
    sig_pow = float(np.mean(y_true ** 2))
    if sig_pow < 1e-12: return -100.0
    return float(10.0 * np.log10(sig_pow / (mse + 1e-12)))

def wrap_phase_diff(p_pred: np.ndarray, p_true: np.ndarray) -> np.ndarray:
    return np.angle(np.exp(1j * (p_pred - p_true)))

def compute_phase_mae(y_true: np.ndarray, y_pred: np.ndarray,
                      cutoff_hz: float = 500.0, fs: float = 48000.0, thresh_ratio: float = 0.05) -> float:
    spec_true = np.fft.rfft(y_true)
    spec_pred = np.fft.rfft(y_pred)
    freq = np.fft.rfftfreq(len(y_true), 1.0 / fs)
    mask_freq = (freq <= cutoff_hz)
    mag_true = np.abs(spec_true)
    peak = np.max(mag_true[mask_freq]) if np.any(mask_freq) else np.max(mag_true)
    mask_mag = (mag_true >= thresh_ratio * peak)
    mask = mask_freq & mask_mag
    if not np.any(mask): return float("nan")
    p_true = np.angle(spec_true[mask])
    p_pred = np.angle(spec_pred[mask])
    return float(np.mean(np.abs(wrap_phase_diff(p_pred, p_true))))

def compute_gd_mae(y_true: np.ndarray, y_pred: np.ndarray,
                   cutoff_hz: float = 500.0, fs: float = 48000.0, thresh_ratio: float = 0.05) -> float:
    spec_true = np.fft.rfft(y_true)
    spec_pred = np.fft.rfft(y_pred)
    freq = np.fft.rfftfreq(len(y_true), 1.0 / fs)
    mask_freq = (freq <= cutoff_hz)
    
    mag_true = np.abs(spec_true)
    peak = np.max(mag_true[mask_freq]) if np.any(mask_freq) else np.max(mag_true)
    mask_mag = (mag_true >= thresh_ratio * peak)
    
    base_mask = mask_freq & mask_mag
    
    gd_mask = np.zeros_like(base_mask, dtype=bool)
    base_mask_copy = base_mask.copy()
    base_mask_copy[0] = False 
    gd_mask[1:] = base_mask_copy[1:] & base_mask_copy[:-1]
    
    if not np.any(gd_mask): return float("nan")

    d_omega = (freq[1] - freq[0]) * 2 * np.pi
    diff_phi_true = np.angle(np.exp(1j * np.diff(np.angle(spec_true))))
    diff_phi_pred = np.angle(np.exp(1j * np.diff(np.angle(spec_pred))))
    
    tau_true = np.zeros_like(freq)
    tau_pred = np.zeros_like(freq)
    tau_true[1:] = -diff_phi_true / d_omega
    tau_pred[1:] = -diff_phi_pred / d_omega
    
    return float(np.mean(np.abs(tau_true[gd_mask] - tau_pred[gd_mask])))

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, fs: float = 48000.0) -> dict:
    snr = compute_snr_db(y_true, y_pred)
    phase_mae = compute_phase_mae(y_true, y_pred, PHASE_CUTOFF, fs, PHASE_THRESH_RATIO)
    gd_mae = compute_gd_mae(y_true, y_pred, GD_CUTOFF, fs, PHASE_THRESH_RATIO)
    return {"snr": snr, "phase_mae": phase_mae, "gd_mae": gd_mae}

# -------------------------
# 3. Plotting
# -------------------------
def plot_signal_processing_style(y_true, y_poly, y_mamba, fs, out_path):
    def znorm(x): return (x - np.mean(x)) / (np.std(x) + 1e-8)

    t = np.arange(len(y_true)) / fs * 1000 
    y_gt_z = znorm(y_true)
    y_poly_z = znorm(y_poly)
    y_mam_z = znorm(y_mamba)
    err_poly = np.abs(y_true - y_poly)
    err_mamba = np.abs(y_true - y_mamba)

    N = len(y_true)
    freq = np.fft.rfftfreq(N, 1/fs)
    S_gt = np.fft.rfft(y_true)
    S_poly = np.fft.rfft(y_poly)
    S_mam = np.fft.rfft(y_mamba)

    def get_gd_smooth(spec, freqs):
        d_omega = (freqs[1] - freqs[0]) * 2 * np.pi
        diff_phi = np.angle(np.exp(1j * np.diff(np.angle(spec))))
        tau = np.zeros_like(freqs)
        tau[1:] = -diff_phi / d_omega
        return tau * 1000  # convert to ms

    def get_phase_err_wrapped(s_pred, s_true):
        return np.angle(np.exp(1j * (np.angle(s_pred) - np.angle(s_true))))

    gd_gt = get_gd_smooth(S_gt, freq)
    gd_poly = get_gd_smooth(S_poly, freq)
    gd_mam = get_gd_smooth(S_mam, freq)

    ph_err_poly = get_phase_err_wrapped(S_poly, S_gt)
    ph_err_mam = get_phase_err_wrapped(S_mam, S_gt)

    mag_gt = np.abs(S_gt)
    plot_freq_max = 2000 
    freq_mask = (freq <= plot_freq_max) & (freq > 0)
    peak_mag = np.max(mag_gt[freq_mask])
    
    valid_bins = freq_mask & (mag_gt >= 0.01 * peak_mag) 
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    plt.subplots_adjust(hspace=0.3, wspace=0.25)

    # --- (a) Time Domain ---
    ax = axes[0, 0]
    ax.plot(t, y_gt_z, 'k-', linewidth=3.0, alpha=0.3, label='Ground Truth')
    ax.plot(t, y_poly_z, 'g--', linewidth=1.5, alpha=0.8, label='Poly (Baseline)')
    ax.plot(t, y_mam_z, 'r-', linewidth=1.5, alpha=0.9, label='Mamba (Phase-Aware)')
    ax.set_title("(a) Time-Domain Reconstruction (Zoomed to impacts)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Normalized Amplitude (σ)")
    ax.set_xlabel("Time (ms)")
    ax.set_xlim(2, 6) 
    ax.legend(loc='upper right', frameon=True, fontsize=9)
    ax.grid(True, linestyle=':', alpha=0.6)

    # --- (b) Absolute Error ---
    ax = axes[0, 1]
    ax.semilogy(t, err_poly, 'g-', linewidth=1.0, alpha=0.5, label='|Poly - GT|')
    ax.semilogy(t, err_mamba, 'r-', linewidth=1.0, alpha=0.7, label='|Mamba - GT|')
    ax.set_title("(b) Absolute Reconstruction Error (Log Scale)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Error Magnitude (log)")
    ax.set_xlabel("Time (ms)")
    ax.set_xlim(2, 6)
    ax.set_ylim(1e-3, 1e0) 
    ax.legend(loc='upper right', frameon=True, fontsize=9)
    ax.grid(True, which='both', linestyle=':', alpha=0.6)

    # --- (c) Absolute Group Delay Error ---
    ax = axes[1, 0]
    
    gd_err_poly = np.abs(gd_poly - gd_gt)
    gd_err_mam = np.abs(gd_mam - gd_gt)
    
    ax.axhline(0, color='k', linestyle='-', linewidth=1.5, alpha=0.3)
    
    ax.scatter(freq[valid_bins], gd_err_poly[valid_bins], color='g', marker='x', s=40, alpha=0.8, label='Poly GD Error')
    ax.scatter(freq[valid_bins], gd_err_mam[valid_bins], color='r', marker='o', s=40, alpha=0.9, label='Mamba GD Error')
    
    ax.set_title("(c) Absolute Group Delay Error at Physical Harmonics (|Δτ|)", fontsize=11, fontweight='bold')
    ax.set_ylabel("GD Error (ms)")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_xlim(0, plot_freq_max)
    
    max_gd_err = max(np.max(gd_err_poly[valid_bins]), np.max(gd_err_mam[valid_bins]))
    ax.set_ylim(-0.05 * max_gd_err, max_gd_err * 1.2)
    
    ax.legend(loc='upper right', frameon=True, fontsize=9)
    ax.grid(True, linestyle=':', alpha=0.6)

    # --- (d) Phase Error ---
    ax = axes[1, 1]
    ax.axhline(0, color='k', linestyle='-', linewidth=1.5, alpha=0.3) 
    
    ax.scatter(freq[valid_bins], ph_err_poly[valid_bins], color='g', marker='x', s=40, alpha=0.8, label='Poly Phase Error')
    ax.scatter(freq[valid_bins], ph_err_mam[valid_bins], color='r', marker='o', s=40, alpha=0.8, label='Mamba Phase Error')
    
    ax.set_title("(d) Phase Alignment Error at Physical Harmonics (Δφ)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Phase Difference (rad)")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_xlim(0, plot_freq_max)
    max_ph_err = np.max(np.abs(ph_err_poly[valid_bins]))
    ax.set_ylim(-max_ph_err * 1.2 - 0.05, max_ph_err * 1.2 + 0.05) 
    ax.legend(loc='upper right', frameon=True, fontsize=9)
    ax.grid(True, linestyle=':', alpha=0.6)

    fig.suptitle(f"Phase-Aware Reconstruction on Fault Sample (Load 3, Scale={SCALE}x)", y=1.02, fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Saved VALIDATED signal processing style comparison plot: {out_path}")

# -------------------------
# 4. Downstream Feature Engineering
# -------------------------
def extract_features(y: np.ndarray, fs: int) -> np.ndarray:
    eps = 1e-8
    y0 = np.asarray(y, dtype=np.float32).reshape(-1)
    y0 = y0 - y0.mean()

    rms = float(np.sqrt(np.mean(y0 * y0) + eps))
    peak = float(np.max(np.abs(y0)) + eps)
    crest = peak / rms
    mu4 = np.mean(y0 ** 4)
    kurt = float(mu4 / (np.mean(y0 ** 2) ** 2 + eps))

    spec = np.abs(np.fft.rfft(y0))
    freqs = np.fft.rfftfreq(len(y0), d=1.0/fs)
    spec_sum = np.sum(spec) + eps
    spectral_centroid = float(np.sum(freqs * spec) / spec_sum)
    
    analytic_sig = scipy.signal.hilbert(y0)
    env = np.abs(analytic_sig)
    env0 = env - env.mean()
    env_mu4 = np.mean(env0 ** 4)
    env_kurt = float(env_mu4 / (np.mean(env0 ** 2) ** 2 + eps))

    return np.array([rms, crest, kurt, spectral_centroid, env_kurt], dtype=np.float32)

def fisher_ratio(X: np.ndarray, y: np.ndarray) -> float:
    eps = 1e-12
    classes = np.unique(y)
    if len(classes) < 2: return float("nan")
    mu = np.mean(X, axis=0, keepdims=True)
    Sb_trace = 0.0
    Sw_trace = 0.0
    for c in classes:
        Xc = X[y == c]
        if Xc.shape[0] < 2: continue
        muc = np.mean(Xc, axis=0, keepdims=True)
        Sb_trace += float(Xc.shape[0]) * float(np.sum((muc - mu) ** 2))
        Sw_trace += float(np.sum((Xc - muc) ** 2))
    return float(Sb_trace / (Sw_trace + eps))

# -------------------------
# 5. Dataset (pre-computed RAM cache)
# -------------------------
@dataclass
class SegmentMeta:
    file_id: int
    fault: str
    start_idx: int

class CWRUSegDataset(Dataset):
    def __init__(self, data_dir: str, file_ids: Dict[str, List[int]], scale: int, target_len: int, hop: int):
        super().__init__()
        self.scale = scale
        self.target_len = target_len
        self.metas = []
        self.cached_data = []

        print(f"⏳ Pre-computing and caching dataset into RAM (Scale={scale}x)... Please wait.")
        start_time = time.time()
        
        for fault, ids in file_ids.items():
            for fid in ids:
                fname = os.path.join(data_dir, f"{fid}.mat")
                if not os.path.exists(fname): continue
                sig = load_cwru_file(fname)
                
                sig = (sig - np.mean(sig)) / (np.std(sig) + 1e-8)
                
                for start in range(0, len(sig) - target_len, hop):
                    self.metas.append(SegmentMeta(fid, fault, start))
                    y_true = sig[start:start + target_len]
                    
                    y_low = downsample_antialias(y_true, self.scale)
                    y_poly = upsample_poly(y_low, self.scale, self.target_len)

                    spec_rect = np.fft.rfft(y_poly)
                    scale_mag = float(np.std(np.abs(spec_rect)) + 1e-12)
                    
                    spec_rect_n = spec_rect / scale_mag
                    x_in = complex_to_channels(spec_rect_n)

                    spec_true = np.fft.rfft(y_true)
                    spec_true_n = spec_true / scale_mag
                    y_lab = complex_to_channels(spec_true_n - spec_rect_n)

                    self.cached_data.append({
                        "x_freq": torch.from_numpy(x_in).float(),
                        "y_freq": torch.from_numpy(y_lab).float(),
                        "gt_time": torch.from_numpy(y_true).float(),
                        "poly_time": torch.from_numpy(y_poly).float(),
                        "norm_factor": torch.tensor(scale_mag, dtype=torch.float32),
                        "target_length": self.target_len,
                        "fault": fault
                    })
                    
        print(f"✅ Cache complete! Cached {len(self.cached_data)} segments in {time.time() - start_time:.2f} seconds.")

    def __len__(self):
        return len(self.cached_data)

    def __getitem__(self, idx):
        return self.cached_data[idx]

# -------------------------
# 6. Model
# -------------------------
def build_model() -> nn.Module:
    from model_mamba import ResidualCorrector
    return ResidualCorrector()

def freq_to_time_domain(freq_residual: torch.Tensor, target_length: int) -> torch.Tensor:
    complex_spec = freq_residual[:, 0, :] + 1j * freq_residual[:, 1, :]
    time_signal = torch.fft.irfft(complex_spec, n=target_length)
    return time_signal.unsqueeze(1)

# -------------------------
# 7. Evaluation Protocol
# -------------------------
@torch.no_grad()
def evaluate_full_suite(model: nn.Module, test_loader: DataLoader, device: torch.device, desc: str, timestamp: str) -> Tuple[Dict, List]:
    model.eval()
    
    metrics = {m: {"snr": [], "phase_mae": [], "gd_mae": []} 
               for m in ["Rect", "Poly", "Cubic", "Mamba"]}
    
    downstream_recs = []
    has_plotted = False

    for batch in test_loader:
        batch_x = batch["x_freq"].to(device)
        batch_gt = batch["gt_time"].cpu().numpy()
        norm_factor = batch["norm_factor"].view(-1, 1, 1).to(device)
        target_len = batch["target_length"][0].item()
        fault_labels = batch["fault"]

        pred_res = model(batch_x)
        pred_res_denorm = pred_res * norm_factor
        pred_res_time = freq_to_time_domain(pred_res_denorm, target_len).squeeze(1).cpu().numpy()
        
        B = batch_gt.shape[0]
        batch_poly = batch["poly_time"].cpu().numpy()
        
        for i in range(B):
            gt = batch_gt[i]
            poly = batch_poly[i]
            mamba = poly + pred_res_time[i]
            
            if not has_plotted and fault_labels[i] != "Normal":
                plot_filename = f"signal_processing_style_comparison_{timestamp}.png"
                plot_signal_processing_style(gt, poly, mamba, FS_HIGH, 
                                   os.path.join(OUT_DIR, plot_filename))
                has_plotted = True
            
            y_low = downsample_antialias(gt, SCALE)
            rect = upsample_rect_fft_resample(y_low, TARGET_LEN)
            cub = upsample_cubic(y_low, SCALE, TARGET_LEN)

            for m_name, sig in zip(["Rect", "Poly", "Cubic", "Mamba"], [rect, poly, cub, mamba]):
                met = compute_metrics(gt, sig, FS_HIGH)
                metrics[m_name]["snr"].append(met["snr"])
                metrics[m_name]["phase_mae"].append(met["phase_mae"])
                metrics[m_name]["gd_mae"].append(met["gd_mae"])
            
            lbl = fault_to_int(fault_labels[i])
            downstream_recs.append({
                "y": lbl, 
                "poly": poly, 
                "mamba": mamba
            })

    return metrics, downstream_recs

def print_full_results(metrics: Dict, title: str):
    print("\n" + "="*120)
    print(title)
    print("-"*120)
    print(f"{'Method':<10} | {'SNR (dB)':<15} | {'PhsMAE':<15} | {'GD_MAE (s)':<15} | {'ΔSNR(M-P)':<10} | {'ΔPhs(M-P)':<10} | {'ΔPhs%↓':<8}")
    print("-"*120)

    means = {}
    for m in metrics:
        means[m] = {k: np.mean([v for v in vals if not np.isnan(v)]) 
                   for k, vals in metrics[m].items()}
        means[m]["snr_std"] = np.std(metrics[m]["snr"])
        means[m]["phs_std"] = np.std([v for v in metrics[m]["phase_mae"] if not np.isnan(v)])
        means[m]["gd_std"] = np.std([v for v in metrics[m]["gd_mae"] if not np.isnan(v)])

    ref = means["Poly"]
    tgt = means["Mamba"]
    
    d_snr = tgt["snr"] - ref["snr"]
    d_phs = tgt["phase_mae"] - ref["phase_mae"]
    d_phs_pct = (d_phs / (ref["phase_mae"] + 1e-12)) * 100

    for m in ["Rect", "Poly", "Cubic", "Mamba"]:
        s = means[m]
        snr_str = f"{s['snr']:.2f}±{s['snr_std']:.2f}"
        phs_str = f"{s['phase_mae']:.3f}±{s['phs_std']:.3f}"
        gd_str = f"{s['gd_mae']:.6f}±{s['gd_std']:.6f}"
        
        diffs = ""
        if m == "Mamba":
            diffs = f"| {d_snr:>+9.2f}  | {d_phs:>+9.3f}  | {d_phs_pct:>+7.1f}%"
        
        print(f"{m:<10} | {snr_str:<15} | {phs_str:<15} | {gd_str:<15} {diffs}")
    print("="*120)
    return means

def perform_downstream_eval(train_ds, recs):
    print("\n[Downstream PHM Evaluation - Domain Generalization Settings]")
    print("-" * 60)
    
    Xtr, ytr = [], []
    cls_count = {0:0, 1:0, 2:0, 3:0}
    for i in range(len(train_ds)):
        d = train_ds[i]
        gt = d["gt_time"].numpy()
        lbl = fault_to_int(d["fault"])
        if cls_count[lbl] < MAX_TRAIN_FEATS_PER_CLASS:
            Xtr.append(extract_features(gt, FS_HIGH))
            ytr.append(lbl)
            cls_count[lbl] += 1
    
    Xtr = np.array(Xtr)
    ytr = np.array(ytr)
    
    X_poly, y_poly = [], []
    X_mam, y_mam = [], []
    
    for r in recs:
        lbl = r["y"]
        X_poly.append(extract_features(r["poly"], FS_HIGH))
        y_poly.append(lbl)
        X_mam.append(extract_features(r["mamba"], FS_HIGH))
        y_mam.append(lbl)
        
    X_poly, y_poly = np.array(X_poly), np.array(y_poly)
    X_mam, y_mam = np.array(X_mam), np.array(y_mam)

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import accuracy_score, f1_score
    
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=10.0, solver='liblinear', max_iter=2000, random_state=SEED))
    clf.fit(Xtr, ytr)
    
    def eval_set(X, y):
        pred = clf.predict(X)
        acc = accuracy_score(y, y_pred=pred)
        f1 = f1_score(y, pred, average="macro")
        fr = fisher_ratio(X, y)
        return acc, f1, fr

    acc_p, f1_p, fr_p = eval_set(X_poly, y_poly)
    acc_m, f1_m, fr_m = eval_set(X_mam, y_mam)
    
    print(f"{'Method':<10} | {'Acc':<8} | {'F1':<8} | {'Fisher':<8}")
    print("-" * 60)
    print(f"{'Poly':<10} | {acc_p:.4f}   | {f1_p:.4f}   | {fr_p:.3f}")
    print(f"{'Mamba':<10} | {acc_m:.4f}   | {f1_m:.4f}   | {fr_m:.3f}")
    print("-" * 60)
    print(f"Δ F1 Score: {f1_m - f1_p:+.4f}")
    print(f"Δ Fisher:   {fr_m - fr_p:+.3f}")

# -------------------------
# 8. Training Loop & Main
# -------------------------
def train_with_phase_aware_loss(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, device: torch.device):
    criterion = PhaseAwareLoss(
        lambda_freq=LAMBDA_FREQ,
        lambda_phase=LAMBDA_PHASE,
        lambda_gd=LAMBDA_GD,
        energy_threshold=ENERGY_THRESHOLD
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best_val_loss = float("inf")
    patience_counter = 0

    print("\n🚀 Phase-Aware Fine-tuning Started...")
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        comps = {}
        n_train = 0
        
        for batch in train_loader:
            batch_x = batch["x_freq"].to(device)
            batch_gt = batch["gt_time"].unsqueeze(1).to(device)
            batch_poly = batch["poly_time"].unsqueeze(1).to(device)
            norm = batch["norm_factor"].view(-1, 1, 1).to(device)
            tgt_len = batch["target_length"][0].item()

            pred_res = model(batch_x)
            pred_denorm = pred_res * norm
            pred_time = freq_to_time_domain(pred_denorm, tgt_len)
            pred_final = batch_poly + pred_time

            loss, loss_dict = criterion(pred_final, batch_gt, return_components=True)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            bs = batch_x.size(0)
            train_loss += loss.item() * bs
            
            if loss_dict:
                for k, v in loss_dict.items():
                    comps[k] = comps.get(k, 0.0) + v * bs
            n_train += bs

        train_loss /= n_train
        avg_comps = {k: v / n_train for k,v in comps.items()}

        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                batch_x = batch["x_freq"].to(device)
                batch_gt = batch["gt_time"].unsqueeze(1).to(device)
                batch_poly = batch["poly_time"].unsqueeze(1).to(device)
                norm = batch["norm_factor"].view(-1, 1, 1).to(device)
                tgt_len = batch["target_length"][0].item()

                pred_res = model(batch_x)
                pred_time = freq_to_time_domain(pred_res * norm, tgt_len)
                pred_final = batch_poly + pred_time
                loss, _ = criterion(pred_final, batch_gt, return_components=True)
                
                bs = batch_x.size(0)
                val_loss += loss.item() * bs
                n_val += bs

        val_loss /= n_val

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(OUT_DIR, "mamba_cwru_finetuned_best.pth"))
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1:
            phase_val = avg_comps.get('phase', 0.0)
            print(f"Ep {epoch:3d} | Train: {train_loss:.4f} (Phs:{phase_val:.4f}) | Val: {val_loss:.4f}")

        if patience_counter >= PATIENCE:
            print(f"🛑 Early stopping at Epoch {epoch}")
            break
            
    print(f"✅ Loaded best model (Val Loss: {best_val_loss:.6f})")
    model.load_state_dict(torch.load(os.path.join(OUT_DIR, "mamba_cwru_finetuned_best.pth"), map_location=device, weights_only=True))

def main():
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(OUT_DIR, f"run_log_cwru_{timestamp}.txt")
    sys.stdout = Logger(log_file)
    
    print(f"📝 Experiment log: {log_file}")
    print(f"🕒 Run timestamp: {timestamp}")
    print("="*120)
    print("CWRU Fine-tuning with Phase-Aware Loss (ULTIMATE ENGINEERING FIX)")
    print("========================================================================================================================")

    set_seed(SEED)
    
    train_ds = CWRUSegDataset(DATA_DIR, TRAIN_IDS, SCALE, TARGET_LEN, HOP)
    
    val_fids = set([TRAIN_IDS[k][-1] for k in TRAIN_IDS]) 
    print(f"Validation file IDs (Hold-out): {val_fids}")
    
    train_indices = [i for i, m in enumerate(train_ds.metas) if m.file_id not in val_fids]
    val_indices = [i for i, m in enumerate(train_ds.metas) if m.file_id in val_fids]
    
    train_sub = torch.utils.data.Subset(train_ds, train_indices)
    val_sub = torch.utils.data.Subset(train_ds, val_indices)
    
    test_ds = CWRUSegDataset(DATA_DIR, TEST_IDS, SCALE, TARGET_LEN, HOP)
    
    tr_loader = DataLoader(train_sub, batch_size=BATCH, shuffle=True)
    val_loader = DataLoader(val_sub, batch_size=BATCH, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    
    print(f"Dataset: Train={len(train_sub)}, Val={len(val_sub)}, Test={len(test_ds)}")

    model = build_model().to(DEVICE)
    if os.path.exists(PRETRAINED_MODEL_PATH):
        model.load_state_dict(torch.load(PRETRAINED_MODEL_PATH, map_location=DEVICE, weights_only=True))
        print(f"Loaded Pretrained: {PRETRAINED_MODEL_PATH}")
    else:
        print("Pretrained model not found, training from scratch...")

    train_with_phase_aware_loss(model, tr_loader, val_loader, DEVICE)

    final_model_path = os.path.join(OUT_DIR, f"mamba_cwru_finetuned_final_{timestamp}.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"💾 Final model saved to: {final_model_path}")

    print("\n📊 Running Full Evaluation Suite...")
    metrics, downstream_recs = evaluate_full_suite(model, test_loader, DEVICE, "Fine-tuned Model", timestamp)
    
    print_full_results(metrics, "Phase-Aware Fine-tuned Results")
    perform_downstream_eval(train_ds, downstream_recs)
    
    print(f"\n✅ Done. All logs, models, and HD plots are saved to '{OUT_DIR}' with timestamp '{timestamp}'.")

if __name__ == "__main__":
    main()