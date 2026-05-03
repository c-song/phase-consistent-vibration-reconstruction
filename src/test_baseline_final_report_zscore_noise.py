“””
test_baseline_final_report_zscore_noise.py

Synthetic baseline evaluation: Rect(FD) / Poly / Hann-sinc / LASSO / Mamba
over 100 Monte Carlo trials with 5% additive noise.
“””

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import config
import utils
import data_generator
import os
import sys
import time
import json
import scipy.signal
from sklearn.linear_model import Lasso

from model_mamba import ResidualCorrector as Mamba_Model

# =========================
# 0. Config
# =========================
N_MONTE_CARLO = 100
FS_HIGH = getattr(config, 'FS_HIGH', 12800)
PHASE_BAND_HZ = (0, 500)
MAG_THRESH_RATIO = 0.05
SCALE_K = getattr(config, 'SCALE_K', 4)

# noise level as a fraction of the clean signal's standard deviation
NOISE_LEVEL = 0.05

MAMBA_PATH = "mamba_poly_phase_aware_best_noise.pth"
OUT_DIR = "baseline_result"
os.makedirs(OUT_DIR, exist_ok=True)

# =========================
# 1. Utilities
# =========================
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

# =========================
# 2. Core Metrics
# =========================
def calc_snr(y_true, y_pred):
    diff = y_true - y_pred
    mse = np.mean(diff ** 2)
    if mse < 1e-12:
        return 100.0
    energy = np.mean(y_true ** 2)
    return float(10.0 * np.log10(energy / mse))

def compute_metrics_scalar(y_true, y_pred):
    """Compute phase MAE and group-delay MAE using consecutive-bin masking and local complex differences."""
    N = len(y_true)
    freqs = np.fft.rfftfreq(N, d=1.0 / FS_HIGH)

    S_true = np.fft.rfft(y_true)
    S_pred = np.fft.rfft(y_pred)

    mag_true = np.abs(S_true)
    thresh = MAG_THRESH_RATIO * np.max(mag_true)
    mask_base = (mag_true > thresh) & (freqs >= PHASE_BAND_HZ[0]) & (freqs <= PHASE_BAND_HZ[1])

    if np.sum(mask_base) == 0:
        return float('nan'), float('nan')

    phi_true = np.angle(S_true)
    phi_pred = np.angle(S_pred)

    phase_diff = np.angle(np.exp(1j * (phi_pred - phi_true)))
    phase_mae = float(np.mean(np.abs(phase_diff)[mask_base]))

    # ── Group Delay MAE ──
    mask_base_copy = mask_base.copy()
    mask_base_copy[0] = False  
    mask_gd = np.zeros_like(mask_base, dtype=bool)
    mask_gd[1:] = mask_base_copy[1:] & mask_base_copy[:-1]

    if not np.any(mask_gd):
        return phase_mae, float('nan')

    d_omega = (freqs[1] - freqs[0]) * 2 * np.pi

    # Local complex differences instead of np.unwrap, to avoid global phase collapse in noisy bins
    diff_phi_true = np.angle(np.exp(1j * np.diff(phi_true)))
    diff_phi_pred = np.angle(np.exp(1j * np.diff(phi_pred)))

    gd_true = np.zeros_like(phi_true)
    gd_pred = np.zeros_like(phi_pred)
    gd_true[1:] = -diff_phi_true / d_omega
    gd_pred[1:] = -diff_phi_pred / d_omega

    gd_mae = float(np.mean(np.abs(gd_true[mask_gd] - gd_pred[mask_gd])))

    return phase_mae, gd_mae

def get_phase_error_spectrum(y_true, y_pred, plot_thresh_ratio=0.01):
    S_true = np.fft.rfft(y_true)
    S_pred = np.fft.rfft(y_pred)
    phi_true = np.angle(S_true)
    phi_pred = np.angle(S_pred)
    phase_diff = np.angle(np.exp(1j * (phi_pred - phi_true)))
    phase_err = np.abs(phase_diff)

    mag_true = np.abs(S_true)
    thresh = plot_thresh_ratio * np.max(mag_true)
    mask = mag_true > thresh
    phase_err[~mask] = np.nan
    return phase_err

# =========================
# 3. Interpolation Methods
# =========================
def poly_interpolation(y_low, target_len):
    y_poly = scipy.signal.resample_poly(y_low, up=SCALE_K, down=1)
    if len(y_poly) > target_len:
        y_poly = y_poly[:target_len]
    elif len(y_poly) < target_len:
        y_poly = np.pad(y_poly, (0, target_len - len(y_poly)), mode='edge')
    return y_poly.astype(np.float32)

def lasso_reconstruction(y_low, target_len, alpha=0.001):
    """
    Spectral LASSO: recover sparse high-rate spectrum from aliased low-rate DFT.
    Aliasing model: Y_low[k] = (1/K) * sum_{m=0}^{K-1} X_high[k + m*N_low]
    Solves real and imaginary parts independently (no training data required).
    """
    N_low = len(y_low)
    K = round(target_len / N_low)
    n_lo = N_low // 2 + 1
    n_hi = target_len // 2 + 1

    # Build aliasing sensing matrix (n_lo x n_hi)
    A = np.zeros((n_lo, n_hi))
    for k in range(n_lo):
        for m in range(K + 1):
            j = k + m * N_low
            if j < n_hi:
                A[k, j] = 1.0 / K

    Y_low = np.fft.rfft(y_low)

    lasso_re = Lasso(alpha=alpha, max_iter=10000, fit_intercept=False, tol=1e-5)
    lasso_im = Lasso(alpha=alpha, max_iter=10000, fit_intercept=False, tol=1e-5)
    lasso_re.fit(A, Y_low.real)
    lasso_im.fit(A, Y_low.imag)

    X_high = lasso_re.coef_ + 1j * lasso_im.coef_
    y_high = np.fft.irfft(X_high, n=target_len)
    return y_high.astype(np.float32)

def windowed_interpolation(y_low, target_len):
    """Time-domain Hann-windowed sinc FIR interpolation."""
    N_low = len(y_low)
    ratio = target_len / N_low 
    scale_k = int(round(ratio))
    half_w = int(ratio * 3)
    t_kernel = np.arange(-half_w, half_w + 1, dtype=np.float64)
    sinc_kernel = np.sinc(t_kernel / ratio)
    hann_win = np.hanning(len(sinc_kernel))
    kernel = sinc_kernel * hann_win
    kernel /= kernel.sum() 
    
    y_up = np.zeros(N_low * scale_k, dtype=np.float64)
    y_up[::scale_k] = y_low
    y_up *= scale_k  
    
    y_interp = np.convolve(y_up, kernel, mode='same')
    if len(y_interp) > target_len:
        y_interp = y_interp[:target_len]
    elif len(y_interp) < target_len:
        y_interp = np.pad(y_interp, (0, target_len - len(y_interp)), mode='edge')
    return y_interp.astype(np.float32)

# =========================
# 4. Reporting & Storage
# =========================
def report_table(metrics_dict):
    methods = ["Rect(FD)", "Poly", "Hann", "LASSO", "Mamba"]
    stats = {m: {} for m in methods}
    keys = ["SNR", "PhsMAE", "GD_MAE"]

    for m in methods:
        for k in keys:
            vals = [v for v in metrics_dict[m][k] if not np.isnan(v)]
            stats[m][k] = (float(np.mean(vals)), float(np.std(vals))) if vals else (0.0, 0.0)

    print("\n" + "=" * 130)
    print(f"Synthetic Ultimate Fusion Evaluation (N={N_MONTE_CARLO}, Mean±Std) | Noise={NOISE_LEVEL}")
    print("-" * 130)
    print(f"{'Method':<12} | {'SNR (dB)':<15} | {'PhsMAE':<15} | {'GD_MAE (s)':<20} | "
          f"{'ΔSNR vs Poly':<13} | {'ΔPhs%↓':<8} | {'ΔGD%↓':<8}")
    print("-" * 130)

    poly_snr = stats["Poly"]["SNR"][0]
    poly_phs = stats["Poly"]["PhsMAE"][0] or 1e-12
    poly_gd  = stats["Poly"]["GD_MAE"][0] or 1e-12

    for m in methods:
        s = stats[m]
        snr_str = f"{s['SNR'][0]:.2f}±{s['SNR'][1]:.2f}"
        phs_str = f"{s['PhsMAE'][0]:.3f}±{s['PhsMAE'][1]:.3f}"
        gd_str  = f"{s['GD_MAE'][0]:.6f}±{s['GD_MAE'][1]:.6f}"
        if m == "Poly":
            suffix = "  (baseline)  |          |        "
        else:
            dsnr = s['SNR'][0] - poly_snr
            dphs_pct = (s['PhsMAE'][0] - poly_phs) / poly_phs * 100
            dgd_pct  = (s['GD_MAE'][0] - poly_gd)  / poly_gd  * 100
            suffix = f"  {dsnr:+.2f}         | {dphs_pct:+.1f}%    | {dgd_pct:+.1f}%"
        print(f"{m:<12} | {snr_str:<15} | {phs_str:<15} | {gd_str:<20} | {suffix}")
    print("=" * 130)
    return stats

def save_results_with_timestamp(stats, timestamp):
    noise_tag = str(NOISE_LEVEL).replace(".", "p")
    txt_path  = os.path.join(OUT_DIR, f"synthetic_baseline_noise_{noise_tag}_{timestamp}.txt")
    csv_path  = os.path.join(OUT_DIR, f"synthetic_baseline_noise_{noise_tag}_{timestamp}.csv")
    json_path = os.path.join(OUT_DIR, f"synthetic_baseline_noise_{noise_tag}_{timestamp}.json")

    methods = ["Rect(FD)", "Poly", "Hann", "LASSO", "Mamba"]
    header = (f"{'Method':<12} | {'SNR_mean':<12} | {'SNR_std':<12} | "
              f"{'PhsMAE_mean':<12} | {'PhsMAE_std':<12} | {'GD_mean':<12} | {'GD_std':<12}")
    lines = [
        "Synthetic Ultimate Fusion Evaluation",
        f"timestamp={timestamp}", f"noise_level={NOISE_LEVEL}",
        f"n_monte_carlo={N_MONTE_CARLO}", f"fs_high={FS_HIGH}",
        f"phase_band_hz={PHASE_BAND_HZ}", f"mag_thresh_ratio={MAG_THRESH_RATIO}",
        "", header, "-" * 110,
    ]
    for m in methods:
        s = stats[m]
        lines.append(f"{m:<12} | {s['SNR'][0]:<12.6f} | {s['SNR'][1]:<12.6f} | "
                     f"{s['PhsMAE'][0]:<12.6f} | {s['PhsMAE'][1]:<12.6f} | "
                     f"{s['GD_MAE'][0]:<12.6f} | {s['GD_MAE'][1]:<12.6f}")

    with open(txt_path, "w", encoding="utf-8") as f: f.write("\n".join(lines) + "\n")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("Method,SNR_mean,SNR_std,PhsMAE_mean,PhsMAE_std,GD_MAE_mean,GD_MAE_std\n")
        for m in methods:
            s = stats[m]
            f.write(f"{m},{s['SNR'][0]:.6f},{s['SNR'][1]:.6f},"
                    f"{s['PhsMAE'][0]:.6f},{s['PhsMAE'][1]:.6f},"
                    f"{s['GD_MAE'][0]:.6f},{s['GD_MAE'][1]:.6f}\n")

    print(f"✅ Results saved to: {OUT_DIR}  [{timestamp}]")

# =========================
# 5. Plotting
# =========================
def plot_comprehensive(y_ref_clean, y_dict, save_path):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 14), dpi=150)
    colors  = {'Rect(FD)': 'red',  'Poly': 'blue', 'Hann': 'cyan', 'LASSO': 'orange', 'Mamba': 'green'}
    styles  = {'Rect(FD)': '--',   'Poly': '-.',   'Hann': ':',    'LASSO': '--',     'Mamba': '-'}
    alphas  = {'Rect(FD)': 0.6,    'Poly': 0.7,    'Hann': 0.7,    'LASSO': 0.8,      'Mamba': 1.0}
    zorders = {'Rect(FD)': 3,      'Poly': 4,      'Hann': 3,      'LASSO': 5,        'Mamba': 10}

    view_len = 300
    t = np.arange(view_len)

    for name, y_pred in y_dict.items():
        ax1.plot(t, y_pred[:view_len], color=colors[name], linestyle=styles[name],
                 linewidth=1.5, alpha=alphas[name], zorder=zorders[name], label=name)
                 
    ax1.plot(t, y_ref_clean[:view_len], 'k', linewidth=3.5, alpha=0.3,
             label='Clean Noiseless Target', zorder=0)
             
    ax1.set_title(f"(a) Time-Domain Reconstruction (Representative Sample)", fontweight='bold')
    ax1.set_ylabel("Amplitude")
    ax1.set_xlim(0, view_len)
    ax1.legend(loc='upper right', ncol=3, fontsize=9)
    ax1.grid(True, linestyle=':', alpha=0.6)

    for name, y_pred in y_dict.items():
        err = np.abs(y_ref_clean - y_pred)
        ax2.plot(t, err[:view_len], color=colors[name], linewidth=1.2,
                 alpha=alphas[name], zorder=zorders[name], label=f'{name} Error')
                 
    ax2.set_title("(b) Absolute Reconstruction Error - Representative Sample (Log Scale)", fontweight='bold')
    ax2.set_ylabel("Abs Error (Log)")
    ax2.set_yscale('log')
    ax2.set_xlim(0, view_len)
    ax2.set_ylim(bottom=1e-5, top=1)
    ax2.legend(loc='upper right', ncol=3, fontsize=9)
    ax2.grid(True, linestyle=':', alpha=0.6)

    N = len(y_ref_clean)
    freqs = np.fft.rfftfreq(N, d=1.0 / FS_HIGH)
    zoom_idx = np.where(freqs <= 500)[0]
    phase_err_max = 0.0

    for name, y_pred in y_dict.items():
        pe = get_phase_error_spectrum(y_ref_clean, y_pred, plot_thresh_ratio=0.01)
        finite_pe = pe[zoom_idx][np.isfinite(pe[zoom_idx])]
        if finite_pe.size > 0:
            phase_err_max = max(phase_err_max, float(np.max(finite_pe)))
        
        ax3.scatter(freqs[zoom_idx], pe[zoom_idx], color=colors[name], 
                    alpha=alphas[name], s=25, marker='o' if name=="Mamba" else 'x', zorder=zorders[name], label=name)

    ax3.set_title("(c) Wrapped Phase Error at Valid Physical Harmonics", fontweight='bold')
    ax3.set_xlabel("Frequency (Hz)")
    ax3.set_ylabel("|Δφ| (rad)")
    ax3.set_xlim(0, 500)
    phase_ylim_top = max(0.05, phase_err_max * 1.15)
    ax3.set_ylim(0, phase_ylim_top)
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"✅ Figure saved to: {save_path}")
    plt.close()

# =========================
# 6. Main
# =========================
def main():
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    noise_tag = str(NOISE_LEVEL).replace(".", "p")
    log_path = os.path.join(OUT_DIR, f"run_log_synthetic_noise_{noise_tag}_{timestamp}.txt")
    sys.stdout = Logger(log_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🕒 Run timestamp: {timestamp}")
    print(f"🚀 Running ULTIMATE Fusion Benchmark | noise={NOISE_LEVEL} | device={device}")

    if not os.path.exists(MAMBA_PATH):
        print(f"❌ Error: {MAMBA_PATH} not found.")
        return
    net_mamba = Mamba_Model().to(device)
    net_mamba.load_state_dict(torch.load(MAMBA_PATH, map_location=device))
    net_mamba.eval()

    methods = ["Rect(FD)", "Poly", "Hann", "LASSO", "Mamba"]
    metrics = {m: {"SNR": [], "PhsMAE": [], "GD_MAE": []} for m in methods}

    history_y_clean = []
    history_y_dict = []

    np.random.seed(42)

    print(f"🔄 Processing {N_MONTE_CARLO} Monte Carlo samples...")
    for i in range(N_MONTE_CARLO):

        # ── 1. Clean signal ──
        _, y_true_raw = data_generator.create_pair_data()
        y_true_raw = np.asarray(y_true_raw, dtype=np.float32).flatten()
        target_len = len(y_true_raw)
        y_true_clean = y_true_raw - np.mean(y_true_raw)

        # ── 2. Add noise then Z-score (must match Mamba training pipeline) ──
        noise_std = NOISE_LEVEL * np.std(y_true_clean)
        y_noisy = y_true_clean + np.random.normal(0, noise_std, target_len)

        mean_noisy = np.mean(y_noisy)
        std_noisy = np.std(y_noisy) + 1e-8
        y_input_zscore = ((y_noisy - mean_noisy) / std_noisy).astype(np.float32)

        # ── 3. Simulate low-rate acquisition ──
        y_low = scipy.signal.decimate(y_input_zscore, SCALE_K, ftype="fir", zero_phase=True)

        # ── 4. Interpolation methods ──
        y_rect, _ = utils.fd_interpolation(y_low, target_len)
        y_poly = poly_interpolation(y_low, target_len)
        spec_poly = np.fft.rfft(y_poly)
        y_hann = windowed_interpolation(y_low, target_len)

        # Mamba spectral reconstruction
        scale = float(np.std(np.abs(spec_poly)) + 1e-8)
        spec_poly_norm = spec_poly / scale
        inp_t = torch.tensor(utils.complex_to_channels(spec_poly_norm), dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            res_norm = net_mamba(inp_t).cpu().numpy()[0]
            
        res = (res_norm[0] + 1j * res_norm[1]) * scale
        y_residual = np.fft.irfft(res, n=target_len)
        y_mamba = (y_poly + y_residual).astype(np.float32)

        y_lasso = lasso_reconstruction(y_low, target_len, alpha=0.001)

        y_dict = {
            "Rect(FD)": np.asarray(y_rect, dtype=np.float32),
            "Poly": y_poly,
            "Hann": y_hann,
            "LASSO": y_lasso,
            "Mamba": y_mamba,
        }

        # ── 5. Reference: scale the noiseless signal with the same normalization parameters ──
        y_eval_target = ((y_true_clean - mean_noisy) / std_noisy).astype(np.float32)

        # ── 6. Score each method ──
        for m, y_pred in y_dict.items():
            snr = calc_snr(y_eval_target, y_pred)
            p_mae, gd_mae = compute_metrics_scalar(y_eval_target, y_pred)
            metrics[m]["SNR"].append(snr)
            metrics[m]["PhsMAE"].append(p_mae)
            metrics[m]["GD_MAE"].append(gd_mae)

        history_y_clean.append(y_eval_target)
        history_y_dict.append(y_dict)

    stats = report_table(metrics)
    save_results_with_timestamp(stats, timestamp)
    
    # Select the trial whose Mamba SNR is closest to the mean for representative plotting
    mean_mamba_snr = stats["Mamba"]["SNR"][0]
    best_idx = np.argmin(np.abs(np.array(metrics["Mamba"]["SNR"]) - mean_mamba_snr))
    print(f"\n📊 Auto-selected Sample #{best_idx} (SNR={metrics['Mamba']['SNR'][best_idx]:.2f}dB) for representative plotting.")
    
    plot_path = os.path.join(OUT_DIR, f"Figure_Synthetic_Report_Noise_{noise_tag}_zscore_{timestamp}.png")
    plot_comprehensive(history_y_clean[best_idx], history_y_dict[best_idx], plot_path)

if __name__ == "__main__":
    main()