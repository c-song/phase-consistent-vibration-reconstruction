"""
run_gearbox_finetune_phase_aware_with_noise_fivefeatures.py
"""

import os, re, time, math, copy, hashlib, sys
from datetime import datetime
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import numpy as np
import scipy.io
import scipy.signal
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model_mamba import ResidualCorrector
from phase_aware_loss import RobustGearboxLoss
import utils_pub as U

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import silhouette_score

# =========================
# 0) Config
# =========================
ROOT_DIR = "./gearboxdata_extracted"
HZ_KEEP = (30,)
TRAIN_LOADS = (0, 10, 20, 30, 40, 50, 60)
TEST_LOADS  = (70, 80, 90)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TARGET_LEN = 1024
HOP = TARGET_LEN // 2
SCALE_K = 4

BATCH_SIZE = 32
EPOCHS = 300
PATIENCE = 10
LR = 5e-5
WEIGHT_DECAY = 1e-4

Z_SCORE = True
SEED = 42

PRETRAINED_MODEL_PATH = "mamba_poly_phase_aware_best_noise.pth"

LAMBDA_FREQ = 0.05
LAMBDA_PHASE = 1.0
LAMBDA_GD = 0.2
ENERGY_THRESHOLD = 0.005

OUT_DIR = "plots_gearbox_phase_aware_finetune"
os.makedirs(OUT_DIR, exist_ok=True)
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

LOG_PATH = os.path.join(OUT_DIR, f"run_log_{LAMBDA_FREQ}_{LAMBDA_PHASE}_{LAMBDA_GD}_{ENERGY_THRESHOLD}_{RUN_TIMESTAMP}.txt")

# Metrics config
MAG_THRESH_RATIO = 0.05
PHASE_BAND_HZ = (0, 500)
FS_HIGH = 12800

# =========================
# 1) Helper Functions & Safe Replacements
# =========================
def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.best_epoch = None
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, val_loss, model, epoch=None):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.best_model_state = copy.deepcopy(model.state_dict())
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.counter = 0

class Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, data):
        for s in self.streams: s.write(data)
        return len(data)
    def flush(self):
        for s in self.streams: s.flush()

def upsample_cubic_safe(y_low: np.ndarray, scale: int, target_len: int) -> np.ndarray:
    if len(y_low) == target_len: return y_low.astype(np.float32)
    n_low = len(y_low)
    x_low = np.arange(n_low, dtype=np.float32)
    x_hi = np.arange(target_len, dtype=np.float32) / scale 
    f = interp1d(x_low, y_low, kind="cubic", fill_value="extrapolate")
    return f(x_hi).astype(np.float32)

def poly_interpolation_local(y_low, target_len, scale=4):
    y_poly = scipy.signal.resample_poly(y_low, up=scale, down=1)
    if len(y_poly) > target_len: return y_poly[:target_len]
    elif len(y_poly) < target_len: return np.pad(y_poly, (0, target_len - len(y_poly)), mode='edge')
    return y_poly.astype(np.float32)

def freq_to_time_domain(freq_channels, target_length=None):
    real = freq_channels[:, 0, :]  
    imag = freq_channels[:, 1, :]  
    complex_spec = torch.complex(real, imag)  
    if target_length is not None:
        time_signal = torch.fft.irfft(complex_spec, n=target_length, dim=-1)
    else:
        time_signal = torch.fft.irfft(complex_spec, dim=-1)
    return time_signal.unsqueeze(1)  

# =========================
# 2) Dataset Class (RAM Caching)
# =========================
_NAME_RE = re.compile(r'^(?P<label>[hb])(?P<hz>\d+)hz(?P<load>\d+)\.txt$', re.IGNORECASE)

def parse_gearbox_name(fname: str):
    m = _NAME_RE.match(os.path.basename(fname))
    if not m: return None
    return 0 if m.group("label").lower() == "h" else 1, int(m.group("hz")), int(m.group("load"))

def load_txt_1d(path: str):
    try: x = np.loadtxt(path, dtype=np.float32)
    except Exception:
        lines = []
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for ln in f:
                ln = ln.strip()
                if not ln: continue
                if ln[0].isdigit() or ln[0] in "+-.": lines.append(ln.replace(",", " "))
        x = np.loadtxt(lines, dtype=np.float32)
    x = np.asarray(x).squeeze()
    if x.ndim > 1: x = x[:, 0]
    return x.astype(np.float32)

class GearboxReconstructionDataset(Dataset):
    def __init__(self, root_dir, split, hz_keep=(30,), train_loads=(), test_loads=(),
                 target_len=1024, hop=512, scale_factor=4, base_method="poly",
                 zscore=True, seed=0):
        super().__init__()
        self.root_dir = root_dir
        self.split = split.lower()
        self.target_len = int(target_len)
        self.hop = int(hop)
        self.K = int(scale_factor)
        self.base_method = base_method.lower()
        self.zscore = bool(zscore)

        cand = []
        for sub in ["Healthy Data", "BrokenTooth Data"]:
            d = os.path.join(root_dir, sub)
            if not os.path.isdir(d): continue
            for fn in os.listdir(d):
                if not fn.lower().endswith(".txt"): continue
                info = parse_gearbox_name(fn)
                if info is None: continue
                cls, hz, load = info
                if hz_keep and (hz not in hz_keep): continue
                cand.append((os.path.join(d, fn), cls, hz, load))

        if self.split == "train": keep_loads = set(train_loads)
        elif self.split in ("test", "val", "eval"): keep_loads = set(test_loads)
        else: raise ValueError("split error")

        self.files = [x for x in cand if x[3] in keep_loads]
        self.files.sort(key=lambda t: (t[1], t[2], t[3], t[0]))

        self.cached_data = []
        self.original_files_info = []

        print(f"⏳ Pre-computing and caching {self.split} dataset (Method={self.base_method})... Please wait.")
        start_time = time.time()

        for fid, (path, cls, hz, load) in enumerate(self.files):
            sig = load_txt_1d(path)
            if self.zscore:
                sig = (sig - sig.mean()) / (sig.std() + 1e-8)
            sig = sig.astype(np.float32)
            self.original_files_info.append((cls, hz, load, os.path.basename(path)))

            n = len(sig)
            if n < self.target_len: continue
            
            for st in range(0, n - self.target_len + 1, self.hop):
                y_high = sig[st:st + self.target_len]
                
                if self.K > 1:
                    y_low = scipy.signal.decimate(y_high, self.K, ftype="fir", zero_phase=True).astype(np.float32)
                else:
                    y_low = y_high.astype(np.float32)

                if self.base_method == "poly": y_base = poly_interpolation_local(y_low, self.target_len, self.K)
                elif self.base_method == "cubic": y_base = upsample_cubic_safe(y_low, self.K, self.target_len)
                elif self.base_method == "fd": y_base, _ = U.fd_interpolation(y_low, self.target_len)
                else: y_base = poly_interpolation_local(y_low, self.target_len, self.K)
                
                y_base = np.asarray(y_base, dtype=np.float32).flatten()
                y_fd, _ = U.fd_interpolation(y_low, self.target_len)
                y_fd = np.asarray(y_fd, dtype=np.float32).flatten()

                spec_base = np.fft.rfft(y_base)
                spec_true = np.fft.rfft(y_high)
                res_spec = spec_true - spec_base

                scale = float(np.std(np.abs(spec_base)) + 1e-8)
                spec_base_n = spec_base / scale
                res_spec_n = res_spec / scale

                inp = U.complex_to_channels(spec_base_n)
                lab = U.complex_to_channels(res_spec_n)

                self.cached_data.append((
                    torch.tensor(inp, dtype=torch.float32),
                    torch.tensor(lab, dtype=torch.float32),
                    torch.tensor(y_high, dtype=torch.float32),
                    torch.tensor(y_fd, dtype=torch.float32),
                    torch.tensor(y_base, dtype=torch.float32),
                    torch.tensor(cls, dtype=torch.long),
                    torch.tensor(hz, dtype=torch.long),
                    torch.tensor(load, dtype=torch.long),
                    torch.tensor(scale, dtype=torch.float32),
                    fid 
                ))

        print(f"✅ Cache complete! Cached {len(self.cached_data)} segments in {time.time() - start_time:.2f} s.")

    def __len__(self):
        return len(self.cached_data)

    def __getitem__(self, i):
        return self.cached_data[i]

# =========================
# 3) Metrics & Utils & Plotting
# =========================
def compute_gd_mae_safe(y_true: np.ndarray, y_pred: np.ndarray, cutoff_hz: float, fs: float, thresh_ratio: float) -> float:
    spec_true, spec_pred = np.fft.rfft(y_true), np.fft.rfft(y_pred)
    freq = np.fft.rfftfreq(len(y_true), 1.0 / fs)
    mask_freq = (freq <= cutoff_hz)
    
    mag_true = np.abs(spec_true)
    peak = np.max(mag_true[mask_freq]) if np.any(mask_freq) else np.max(mag_true)
    mask_mag = (mag_true >= thresh_ratio * peak)
    
    base_mask = mask_freq & mask_mag
    gd_mask = np.zeros_like(base_mask, dtype=bool)
    gd_mask[1:] = base_mask[1:] & base_mask[:-1]
    
    if not np.any(gd_mask): return float("nan")

    d_omega = (freq[1] - freq[0]) * 2 * np.pi
    diff_phi_true = np.angle(np.exp(1j * np.diff(np.angle(spec_true))))
    diff_phi_pred = np.angle(np.exp(1j * np.diff(np.angle(spec_pred))))
    
    tau_true, tau_pred = np.zeros_like(freq), np.zeros_like(freq)
    tau_true[1:] = -diff_phi_true / d_omega
    tau_pred[1:] = -diff_phi_pred / d_omega
    return float(np.mean(np.abs(tau_true[gd_mask] - tau_pred[gd_mask])))

def calc_metrics(y_true, y_hat):
    out = {}
    out["SNR"] = U.calc_snr(y_true, y_hat)
    out["PhsMAE"] = U.phase_mae_fft(y_true, y_hat, fs=FS_HIGH, band_hz=PHASE_BAND_HZ, mag_ratio=MAG_THRESH_RATIO)
    out["GD_MAE"] = compute_gd_mae_safe(y_true, y_hat, cutoff_hz=500, fs=FS_HIGH, thresh_ratio=MAG_THRESH_RATIO)
    return out

@torch.no_grad()
def mamba_reconstruct(model, inp_channels, scale=None):
    model.eval()
    x = inp_channels.to(DEVICE)
    pred_res = model(x).detach().cpu().numpy()
    base = inp_channels.detach().cpu().numpy()
    B = base.shape[0]
    y_hat = []
    for i in range(B):
        spec_base = base[i, 0] + 1j * base[i, 1]
        res = pred_res[i, 0] + 1j * pred_res[i, 1]
        if scale is not None:
            s = float(scale[i])
            spec_base *= s
            res *= s
        y = np.fft.irfft(spec_base + res, n=TARGET_LEN)
        y_hat.append(y.astype(np.float32))
    return np.stack(y_hat, axis=0)

def extract_feature_vector(y, fs: int):
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
    
    env = np.abs(scipy.signal.hilbert(y0))
    env0 = env - env.mean()
    env_kurt = float(np.mean(env0 ** 4) / (np.mean(env0 ** 2) ** 2 + eps))
    return np.array([rms, crest, kurt, spectral_centroid, env_kurt], dtype=np.float32)

def downstream_check(Xtr, ytr, Xte, yte):
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=10,solver='liblinear', max_iter=2000, random_state=SEED))
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    acc = accuracy_score(yte, pred)
    f1 = f1_score(yte, pred, average="macro")
    try: sil = silhouette_score(Xte, yte)
    except: sil = 0.0
    classes = np.unique(yte)
    mu = Xte.mean(axis=0, keepdims=True)
    Sb, Sw = 0.0, 0.0
    for c in classes:
        Xc = Xte[yte==c]
        if len(Xc) == 0: continue
        muc = Xc.mean(axis=0, keepdims=True)
        Sb += len(Xc) * np.sum((muc - mu)**2)
        Sw += np.sum((Xc - muc)**2)
    fr = float(Sb / max(Sw, 1e-12))
    return acc, f1, sil, fr

def plot_academic_style(y_true, y_poly, y_mamba, fs, out_path):
    def znorm(x): return (x - np.mean(x)) / (np.std(x) + 1e-8)

    t = np.arange(len(y_true)) / fs * 1000 
    y_gt_z = znorm(y_true)
    y_poly_z = znorm(y_poly)
    y_mam_z = znorm(y_mamba)

    N = len(y_true)
    freq = np.fft.rfftfreq(N, 1/fs)
    S_gt = np.fft.rfft(y_true)
    S_poly = np.fft.rfft(y_poly)
    S_mam = np.fft.rfft(y_mamba)

    def get_gd_safe(spec, freqs):
        d_omega = (freqs[1] - freqs[0]) * 2 * np.pi
        diff_phi = np.angle(np.exp(1j * np.diff(np.angle(spec))))
        tau = np.zeros_like(freqs)
        tau[1:] = -diff_phi / d_omega
        return tau * 1000 

    def get_phase_err_wrapped(s_pred, s_true):
        return np.angle(np.exp(1j * (np.angle(s_pred) - np.angle(s_true))))

    gd_gt = get_gd_safe(S_gt, freq)
    gd_poly = get_gd_safe(S_poly, freq)
    gd_mam = get_gd_safe(S_mam, freq)

    ph_err_poly = get_phase_err_wrapped(S_poly, S_gt)
    ph_err_mam = get_phase_err_wrapped(S_mam, S_gt)

    mag_gt = np.abs(S_gt)
    plot_freq_max = 2000 
    freq_mask = (freq <= plot_freq_max) & (freq > 0)
    peak_mag = np.max(mag_gt[freq_mask])
    
    valid_bins = freq_mask & (mag_gt >= 0.01 * peak_mag) 
    
    fig = plt.figure(figsize=(14, 8), dpi=150)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.0], hspace=0.35, wspace=0.2)

    # (a) Time Domain
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(t, y_gt_z, 'k-', linewidth=3.5, alpha=0.3, label='Ground Truth')
    ax1.plot(t, y_poly_z, 'g--', linewidth=1.5, alpha=0.8, label='Poly (Baseline)')
    ax1.plot(t, y_mam_z, 'r-', linewidth=1.5, alpha=0.9, label='Bi-Mamba (Phase-Aware)')
    ax1.set_title("(a) Time-Domain Reconstruction (Macroscopic View)", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Normalized Amplitude (σ)", fontsize=11)
    ax1.set_xlabel("Time (ms)", fontsize=11)
    ax1.set_xlim(5, 15) 
    ax1.legend(loc='upper right', frameon=True, fontsize=10, ncol=3)
    ax1.grid(True, linestyle=':', alpha=0.6)

    # (b) Absolute Group Delay Error
    ax2 = fig.add_subplot(gs[1, 0])
    gd_err_poly = np.abs(gd_poly - gd_gt)
    gd_err_mam = np.abs(gd_mam - gd_gt)
    ax2.axhline(0, color='k', linestyle='-', linewidth=1.5, alpha=0.3)
    ax2.scatter(freq[valid_bins], gd_err_poly[valid_bins], color='g', marker='x', s=45, alpha=0.8, label='Poly GD Error')
    ax2.scatter(freq[valid_bins], gd_err_mam[valid_bins], color='r', marker='o', s=18, alpha=0.7, label='Bi-Mamba GD Error')
    ax2.set_title("(b) Absolute Group Delay Error (|Δτ|)", fontsize=12, fontweight='bold')
    ax2.set_ylabel("GD Error (ms)", fontsize=11)
    ax2.set_xlabel("Frequency (Hz)", fontsize=11)
    ax2.set_xlim(0, plot_freq_max)
    
    if np.any(valid_bins):
        robust_max = max(np.percentile(gd_err_poly[valid_bins], 95), np.percentile(gd_err_mam[valid_bins], 95))
        plot_max = np.clip(robust_max * 2.0, 0.5, 5.0) 
        ax2.set_ylim(-0.05, plot_max)
    ax2.legend(loc='upper right', frameon=True, fontsize=10)
    ax2.grid(True, linestyle=':', alpha=0.6)

    # (c) Phase Error
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axhline(0, color='k', linestyle='-', linewidth=1.5, alpha=0.3) 
    ax3.scatter(freq[valid_bins], ph_err_poly[valid_bins], color='g', marker='x', s=45, alpha=0.8, label='Poly Phase Error')
    ax3.scatter(freq[valid_bins], ph_err_mam[valid_bins], color='r', marker='o', s=18, alpha=0.7, label='Bi-Mamba Phase Error')
    ax3.set_title("(c) Phase Alignment Error (Δφ)", fontsize=12, fontweight='bold')
    ax3.set_ylabel("Phase Difference (rad)", fontsize=11)
    ax3.set_xlabel("Frequency (Hz)", fontsize=11)
    ax3.set_xlim(0, plot_freq_max)
    if np.any(valid_bins):
        max_ph_err = np.max(np.abs(ph_err_poly[valid_bins]))
        ax3.set_ylim(-max_ph_err * 1.2 - 0.05, max_ph_err * 1.2 + 0.05) 
    ax3.legend(loc='upper right', frameon=True, fontsize=10)
    ax3.grid(True, linestyle=':', alpha=0.6)

    fig.suptitle(f"Gearbox Phase-Aware Reconstruction Superiority (Scale={SCALE_K}x)", y=0.98, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Saved Academic style plot: {out_path}")

# =========================
# 4) Reporting Functions
# =========================
def report_table_v5_style(title, metrics_by_method):
    methods = list(metrics_by_method.keys())
    order = [m for m in ["FDRect", "Poly", "Cubic", "Mamba"] if m in methods]
    stats = {m: {} for m in methods}
    keys = ["SNR", "PhsMAE", "GD_MAE"]

    for m in methods:
        for k in keys:
            vals = [d[k] for d in metrics_by_method[m] if not np.isnan(d[k])]
            stats[m][k] = (float(np.mean(vals)), float(np.std(vals))) if vals else (0.0, 0.0)

    dSNR = dPhs = dGD = phs_pct = gd_pct = 0.0
    if "Mamba" in stats and "Poly" in stats:
        dSNR = stats["Mamba"]["SNR"][0] - stats["Poly"]["SNR"][0]
        dPhs = stats["Mamba"]["PhsMAE"][0] - stats["Poly"]["PhsMAE"][0]
        dGD  = stats["Mamba"]["GD_MAE"][0] - stats["Poly"]["GD_MAE"][0]
        denom_phs = max(abs(stats["Poly"]["PhsMAE"][0]), 1e-12)
        denom_gd = max(abs(stats["Poly"]["GD_MAE"][0]), 1e-12)
        phs_pct = (dPhs / denom_phs) * 100
        gd_pct  = (dGD / denom_gd) * 100

    print("\n" + "="*120)
    print(f"{title} (Mean±Std)")
    print("-" * 120)
    print(f"{'Method':<10} | {'SNR (dB)':<15} | {'PhsMAE':<15} | {'GD_MAE (s)':<15} | {'ΔSNR(M-P)':<10} | {'ΔPhs(M-P)':<10} | {'ΔPhs%↓':<8} | {'ΔGD%↓':<8}")
    print("-" * 120)

    for m in order:
        s = stats[m]
        snr_str = f"{s['SNR'][0]:.2f}±{s['SNR'][1]:.2f}"
        phs_str = f"{s['PhsMAE'][0]:.3f}±{s['PhsMAE'][1]:.3f}"
        gd_str  = f"{s['GD_MAE'][0]:.6f}±{s['GD_MAE'][1]:.6f}"
        line = f"{m:<10} | {snr_str:<15} | {phs_str:<15} | {gd_str:<15} |"
        if m == "Mamba" and "Poly" in stats:
            line += f" {dSNR:+.2f}      | {dPhs:+.3f}      | {phs_pct:+.1f}%    | {gd_pct:+.1f}%"
        else:
            line += "             |             |          |        "
        print(line)
    print("="*120)

# =========================
# 5) Main Fine-tuning Loop
# =========================
def finetune(model, train_loader, val_loader, epochs=50):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    criterion = RobustGearboxLoss(
        lambda_freq=LAMBDA_FREQ,
        lambda_phase=LAMBDA_PHASE,
        lambda_gd=LAMBDA_GD,
        energy_threshold=ENERGY_THRESHOLD
    ).to(DEVICE)

    early_stopping = EarlyStopping(patience=PATIENCE, min_delta=1e-5)

    print(f"🚀 Fine-tuning with RobustGearboxLoss (Solution A)")
    print(f"   Energy Threshold={ENERGY_THRESHOLD} (Hard Gating)")

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        comps = {}
        n = 0

        for batch in train_loader:
            batch_x = batch[0].to(DEVICE)       
            batch_gt_time = batch[2].to(DEVICE).unsqueeze(1) if batch[2].dim()==2 else batch[2].to(DEVICE)
            batch_poly_time = batch[4].to(DEVICE).unsqueeze(1) if batch[4].dim()==2 else batch[4].to(DEVICE)
            batch_scale = batch[8].to(DEVICE).view(-1, 1, 1)

            opt.zero_grad()
            pred_residual_denorm = model(batch_x) * batch_scale  
            pred_residual_time = freq_to_time_domain(pred_residual_denorm, target_length=batch_gt_time.shape[2])  
            pred_final = batch_poly_time + pred_residual_time  

            loss, loss_dict = criterion(pred_final, batch_gt_time, return_components=True)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            bs = batch_x.size(0)
            total_loss += float(loss.item()) * bs
            
            if loss_dict:
                for k, v in loss_dict.items(): 
                    comps[k] = comps.get(k, 0.0) + float(v) * bs
            n += bs

        train_loss = total_loss / max(n, 1)
        train_comps = {k: v / max(n, 1) for k, v in comps.items()}

        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                batch_x = batch[0].to(DEVICE)       
                batch_gt_time = batch[2].to(DEVICE).unsqueeze(1) if batch[2].dim()==2 else batch[2].to(DEVICE)
                batch_poly_time = batch[4].to(DEVICE).unsqueeze(1) if batch[4].dim()==2 else batch[4].to(DEVICE)
                batch_scale = batch[8].to(DEVICE).view(-1, 1, 1)

                pred_residual_denorm = model(batch_x) * batch_scale  
                pred_residual_time = freq_to_time_domain(pred_residual_denorm, target_length=batch_gt_time.shape[2])  
                pred_final = batch_poly_time + pred_residual_time  

                loss, _ = criterion(pred_final, batch_gt_time, return_components=True)
                bs = batch_x.size(0)
                val_loss += float(loss.item()) * bs
                n_val += bs

        val_loss /= max(n_val, 1)

        if ep % 5 == 0 or ep == 1:
            phase_v = train_comps.get('phase', 0.0)
            gd_v = train_comps.get('gd', 0.0)
            print(f"Epoch {ep:02d} | Train Loss={train_loss:.5f} (Phs={phase_v:.4f}, GD={gd_v:.4f}) | Val Loss={val_loss:.5f}")

        early_stopping(val_loss, model, epoch=ep)
        if early_stopping.early_stop:
            print(f"🛑 Early stopping triggered at Epoch {ep}!")
            break

    print(f"✅ Training finished. Best Val Loss: {early_stopping.best_loss:.6f}")
    if early_stopping.best_model_state is not None:
        model.load_state_dict(early_stopping.best_model_state)

    save_path = f"mamba_poly_phase_aware_gearbox_finetuned_best_{LAMBDA_FREQ}_{LAMBDA_PHASE}_{LAMBDA_GD}_{ENERGY_THRESHOLD}_{RUN_TIMESTAMP}.pth"
    torch.save(model.state_dict(), os.path.join(OUT_DIR, save_path))

def main():
    print("="*120)
    print("Gearbox Fine-tuning with Phase-Aware Loss (ULTIMATE ENGINEERING FIX)")
    print("="*120)

    train_ds_full = GearboxReconstructionDataset(ROOT_DIR, "train", HZ_KEEP, TRAIN_LOADS, TEST_LOADS, TARGET_LEN, HOP, SCALE_K, "poly", Z_SCORE, seed=SEED)

    rng = np.random.default_rng(SEED)
    fids_by_class = {}
    for fid, (cls,_, _, _) in enumerate(train_ds_full.original_files_info):
        fids_by_class.setdefault(cls, []).append(fid)

    val_fids = set()
    for cls, fids in sorted(fids_by_class.items()):
        val_fids.add(int(rng.choice(fids)))
    
    train_indices = [i for i, data in enumerate(train_ds_full.cached_data) if data[9] not in val_fids]
    val_indices = [i for i, data in enumerate(train_ds_full.cached_data) if data[9] in val_fids]

    train_ds = torch.utils.data.Subset(train_ds_full, train_indices)
    val_ds = torch.utils.data.Subset(train_ds_full, val_indices)
    print(f"Dataset Split: Train={len(train_ds)}, Val={len(val_ds)}")

    test_ds_poly = GearboxReconstructionDataset(ROOT_DIR, "test", HZ_KEEP, TRAIN_LOADS, TEST_LOADS, TARGET_LEN, HOP, SCALE_K, "poly", Z_SCORE, seed=SEED)
    test_ds_cubic = GearboxReconstructionDataset(ROOT_DIR, "test", HZ_KEEP, TRAIN_LOADS, TEST_LOADS, TARGET_LEN, HOP, SCALE_K, "cubic", Z_SCORE, seed=SEED)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader_poly  = DataLoader(test_ds_poly, batch_size=1, shuffle=False)
    test_loader_cubic = DataLoader(test_ds_cubic, batch_size=1, shuffle=False)

    model = ResidualCorrector().to(DEVICE)
    if os.path.exists(PRETRAINED_MODEL_PATH):
        model.load_state_dict(torch.load(PRETRAINED_MODEL_PATH, map_location=DEVICE, weights_only=True))
        print(f"✅ Loaded Phase-Aware pretrained weights from: {PRETRAINED_MODEL_PATH}")
    else:
        print("⚠️ Pretrained model not found, training from scratch!")

    finetune(model, train_loader, val_loader, epochs=EPOCHS)

    print("\n📊 Evaluating on Test loads...")
    methods = ["FDRect", "Poly", "Cubic", "Mamba"]
    groups = ["Overall", "Healthy", "Broken"]
    metrics = {g: {m: [] for m in methods} for g in groups}
    downstream_records = []
    has_plotted = False

    model.eval()
    with torch.no_grad():
        for batch_p, batch_c in zip(test_loader_poly, test_loader_cubic):
            x_m, _, y_true_t, y_fd_t, y_poly_t, cls_t, _, _, scale_t, _ = batch_p
            _, _, _, _, y_cub_t, _, _, _, _, _ = batch_c

            y_true = y_true_t.numpy()[0]
            y_poly = y_poly_t.numpy()[0]
            y_fd   = y_fd_t.numpy()[0]
            y_cub  = y_cub_t.numpy()[0]

            y_m_rec = mamba_reconstruct(model, x_m, scale_t.numpy())[0]
            
            if not has_plotted and int(cls_t.item()) != 0:
                plot_filename = f"Figure_Gearbox_Academic_{LAMBDA_FREQ}_{LAMBDA_PHASE}_{LAMBDA_GD}_{ENERGY_THRESHOLD}_{RUN_TIMESTAMP}.png"
                plot_academic_style(y_true, y_poly, y_m_rec, FS_HIGH, 
                                   os.path.join(OUT_DIR, plot_filename))
                has_plotted = True

            mp = calc_metrics(y_true, y_poly)
            mc = calc_metrics(y_true, y_cub)
            mf = calc_metrics(y_true, y_fd)
            mm = calc_metrics(y_true, y_m_rec)

            cls_val = int(cls_t.item())
            g = "Healthy" if cls_val == 0 else "Broken"
            for G in ["Overall", g]:
                metrics[G]["FDRect"].append(mf)
                metrics[G]["Poly"].append(mp)
                metrics[G]["Cubic"].append(mc)
                metrics[G]["Mamba"].append(mm)

            downstream_records.append({"y": cls_val, "poly": y_poly, "mamba": y_m_rec})

    for G in groups: report_table_v5_style(G, metrics[G])

    print("\n[Downstream Check]")
    Xtr, ytr = [], []
    for i, data in enumerate(train_ds_full.cached_data):
        if i >= 3000: break
        y_true = data[2].numpy()
        label = int(data[5].item())
        Xtr.append(extract_feature_vector(y_true, FS_HIGH))
        ytr.append(label)

    Xte_p, yte_p, Xte_m, yte_m = [], [], [], []
    for rec in downstream_records:
        Xte_p.append(extract_feature_vector(rec["poly"], FS_HIGH))
        yte_p.append(rec["y"])
        Xte_m.append(extract_feature_vector(rec["mamba"], FS_HIGH))
        yte_m.append(rec["y"])

    res_p = downstream_check(np.array(Xtr), np.array(ytr), np.array(Xte_p), np.array(yte_p))
    res_m = downstream_check(np.array(Xtr), np.array(ytr), np.array(Xte_m), np.array(yte_m))

    print(f"{'Method':<10} | {'Acc':<8} | {'F1':<8} | {'Fisher':<8}")
    print(f"{'Poly':<10} | {res_p[0]:.4f}   | {res_p[1]:.4f}   | {res_p[3]:.3f}")
    print(f"{'Mamba':<10} | {res_m[0]:.4f}   | {res_m[1]:.4f}   | {res_m[3]:.3f}")
    print(f"Δ F1 Score: {res_m[1] - res_p[1]:+.4f}")

    print(f"\n✅ All done. Logs, models, and HD plots are in '{OUT_DIR}'.")

def run_with_logging():
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with open(LOG_PATH, "w", encoding="utf-8") as log_fp:
        sys.stdout = Tee(original_stdout, log_fp)
        sys.stderr = Tee(original_stderr, log_fp)
        try:
            print(f"📝 Experiment log: {LOG_PATH}")
            main()
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout = original_stdout
            sys.stderr = original_stderr

if __name__ == "__main__":
    run_with_logging()