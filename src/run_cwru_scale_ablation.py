"""
CWRU Scale-Factor Ablation: ×4 vs ×8 vs ×16
Shows how reconstruction quality and downstream F1 change with compression ratio.
Only compares Poly vs Mamba (two-stage) to isolate the scale effect.
Single seed per scale (fast).
"""
import os, time, copy, json
import numpy as np
import scipy.io, scipy.signal
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import Dict, List

from model_mamba import ResidualCorrector
from phase_aware_loss import PhaseAwareLoss
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR   = "./cwru_mat"
FS_HIGH    = 48000
TARGET_LEN = 1024
HOP        = TARGET_LEN // 2
BATCH_SIZE = 32
EPOCHS     = 500
PATIENCE   = 15
LR_MAMBA   = 5e-5
LR_BASE    = 1e-4
WEIGHT_DECAY = 1e-4
SEED        = 42
MULTI_SEEDS = [42, 123, 2025]
SCALE_LIST  = [4, 8, 16]

PRETRAINED_MODEL_PATH = "mamba_poly_phase_aware_best_noise.pth"
RESULT_DIR = "cwru_result"
os.makedirs(RESULT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LAMBDA_FREQ = 0.05
LAMBDA_PHASE = 1.0
LAMBDA_GD = 0.1
ENERGY_THRESHOLD = 0.005

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
MAX_TRAIN_FEATS_PER_CLASS = 4000

# ── Helpers ────────────────────────────────────────────────────────────────────
def set_seed(s):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def complex_to_channels(spec): return np.stack([spec.real, spec.imag], axis=0).astype(np.float32)
def fault_to_int(f): return {"Normal":0,"IR":1,"Ball":2,"OR":3}.get(f,-1)

def freq_to_time(fc, n):
    c = torch.complex(fc[:,0,:], fc[:,1,:]); return torch.fft.irfft(c, n=n, dim=-1).unsqueeze(1)

def find_de_key(d):
    cand = [k for k in d if "DE_time" in k and not k.startswith("__")]
    if cand: return cand[0]
    return next(k for k in d if not k.startswith("__") and isinstance(d[k], np.ndarray))

def load_cwru(fname):
    mat = scipy.io.loadmat(fname, struct_as_record=False, squeeze_me=True)
    return np.asarray(mat[find_de_key(mat)], dtype=np.float32).flatten()

def compute_snr(yt, yp):
    mse = np.mean((yt-yp)**2); sp = np.mean(yt**2)
    return 10*np.log10(sp/(mse+1e-12)) if mse>1e-12 and sp>1e-12 else (100. if mse<1e-12 else -100.)

def compute_phase_mae(yt, yp, cutoff=500., thresh=0.05):
    st, sp = np.fft.rfft(yt), np.fft.rfft(yp)
    freq = np.fft.rfftfreq(len(yt), 1./FS_HIGH)
    mf = freq <= cutoff; mt = np.abs(st); pk = mt[mf].max() if mf.any() else mt.max()
    mask = mf & (mt >= thresh*pk)
    if not mask.any(): return float("nan")
    return float(np.mean(np.abs(np.angle(np.exp(1j*(np.angle(sp[mask])-np.angle(st[mask])))))))

def extract_feature(y):
    y0 = np.asarray(y, dtype=np.float32).flatten() - np.mean(y)
    eps = 1e-8
    rms = float(np.sqrt(np.mean(y0**2)+eps)); peak = float(np.max(np.abs(y0))+eps)
    crest = peak/rms; mu4 = np.mean(y0**4); kurt = float(mu4/(np.mean(y0**2)**2+eps))
    spec = np.abs(np.fft.rfft(y0)); freqs = np.fft.rfftfreq(len(y0), 1./FS_HIGH)
    sc = float(np.sum(freqs*spec)/(np.sum(spec)+eps))
    env = np.abs(scipy.signal.hilbert(y0)); env0 = env - env.mean()
    ek = float(np.mean(env0**4)/(np.mean(env0**2)**2+eps))
    return np.array([rms, crest, kurt, sc, ek], dtype=np.float32)

# ── Dataset ────────────────────────────────────────────────────────────────────
@dataclass
class Meta: file_id:int; fault:str; start:int

class CWRUDataset(Dataset):
    def __init__(self, data_dir, file_ids, scale, target_len, hop):
        self.scale = scale; self.target_len = target_len
        self.signals, self.metas = [], []
        for fault, ids in file_ids.items():
            for fid in ids:
                fname = os.path.join(data_dir, f"{fid}.mat")
                if not os.path.exists(fname): continue
                sig = load_cwru(fname); sig = (sig-sig.mean())/(sig.std()+1e-8)
                for start in range(0, len(sig)-target_len+1, hop):
                    self.signals.append(sig[start:start+target_len])
                    self.metas.append(Meta(fid, fault, start))

    def __len__(self): return len(self.signals)

    def __getitem__(self, idx):
        yt = self.signals[idx]
        yl = scipy.signal.decimate(yt, self.scale, ftype='fir', zero_phase=True).astype(np.float32)
        yp = scipy.signal.resample_poly(yl, up=self.scale, down=1)
        yp = yp[:self.target_len] if len(yp)>=self.target_len else np.pad(yp,(0,self.target_len-len(yp)),'edge')
        yp = yp.astype(np.float32)
        spec = np.fft.rfft(yp); s = float(np.std(np.abs(spec))+1e-12)
        return (torch.tensor(complex_to_channels(spec/s), dtype=torch.float32),
                torch.tensor(yt, dtype=torch.float32),
                torch.tensor(yp, dtype=torch.float32),
                torch.tensor(fault_to_int(self.metas[idx].fault), dtype=torch.long),
                torch.tensor(s, dtype=torch.float32))

# ── Training ────────────────────────────────────────────────────────────────────
def train_model(model, tr_loader, val_loader, epochs, lr, name):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    crit = PhaseAwareLoss(LAMBDA_FREQ, LAMBDA_PHASE, LAMBDA_GD, ENERGY_THRESHOLD).to(DEVICE)
    best_loss, patience_cnt, best_w = float('inf'), 0, None
    print(f"  Training {name} (lr={lr})...")
    for ep in range(1, epochs+1):
        model.train()
        for batch in tr_loader:
            x, ygt, ypo, _, sc = [b.to(DEVICE) for b in batch]
            opt.zero_grad()
            pred_n = model(x)
            pt = freq_to_time(pred_n * sc.view(-1,1,1), TARGET_LEN)
            pf = ypo.unsqueeze(1) + pt if name=="Mamba" else pt
            loss, _ = crit(pf, ygt.unsqueeze(1), return_components=True)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval(); vl = 0.
        with torch.no_grad():
            for batch in val_loader:
                x, ygt, ypo, _, sc = [b.to(DEVICE) for b in batch]
                pn = model(x); pt = freq_to_time(pn*sc.view(-1,1,1), TARGET_LEN)
                pf = ypo.unsqueeze(1)+pt if name=="Mamba" else pt
                l,_ = crit(pf, ygt.unsqueeze(1), return_components=True); vl += l.item()
        vl /= len(val_loader)
        if vl < best_loss: best_loss=vl; patience_cnt=0; best_w=copy.deepcopy(model.state_dict())
        else: patience_cnt += 1
        if ep%20==0 or ep==1: print(f"    [Ep{ep:03d}] val={vl:.5f}")
        if patience_cnt >= PATIENCE: print(f"    Early stop ep{ep}"); break
    if best_w: model.load_state_dict(best_w)

@torch.no_grad()
def reconstruct(model, x, sc, yp_t, name):
    model.eval()
    pn = model(x.to(DEVICE)).cpu().numpy()[0]
    pt = np.fft.irfft(complex(pn[0,0],pn[1,0]) if False else (pn[0]+1j*pn[1])*float(sc), n=TARGET_LEN)
    return yp_t.numpy()[0] + pt.astype(np.float32) if name=="Mamba" else pt.astype(np.float32)

# ── Single seed runner ────────────────────────────────────────────────────────
def run_scale_seed(scale_k, seed):
    set_seed(seed)

    full_tr = CWRUDataset(DATA_DIR, TRAIN_IDS, scale_k, TARGET_LEN, HOP)
    test_ds = CWRUDataset(DATA_DIR, TEST_IDS,  scale_k, TARGET_LEN, HOP)

    # Seed-dependent val split
    rng = np.random.default_rng(seed)
    fids_by_fault = {}
    for meta in full_tr.metas:
        fids_by_fault.setdefault(meta.fault, set()).add(meta.file_id)
    val_fids = set(int(rng.choice(sorted(fids))) for fids in fids_by_fault.values())

    tri = [i for i,m in enumerate(full_tr.metas) if m.file_id not in val_fids]
    vli = [i for i,m in enumerate(full_tr.metas) if m.file_id in val_fids]
    tr_loader  = DataLoader(torch.utils.data.Subset(full_tr, tri), BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(torch.utils.data.Subset(full_tr, vli), BATCH_SIZE, shuffle=False)
    te_loader  = DataLoader(test_ds, batch_size=1, shuffle=False)

    # GT classifier features
    Xtr, ytr, cnt = [], [], {0:0,1:0,2:0,3:0}
    gen = torch.Generator().manual_seed(seed)
    for batch in DataLoader(full_tr, 1, shuffle=True, num_workers=0, generator=gen):
        c = int(batch[3].item())
        if cnt[c] < MAX_TRAIN_FEATS_PER_CLASS:
            Xtr.append(extract_feature(batch[1].numpy()[0])); ytr.append(c); cnt[c] += 1
    Xtr_arr, ytr_arr = np.array(Xtr), np.array(ytr)

    # Poly (deterministic, seed-independent)
    snrs_p, phs_p, feats_p, cls_te = [], [], [], []
    for batch in te_loader:
        x, yt, yp, cl, sc = batch
        yt_np, yp_np = yt.numpy()[0], yp.numpy()[0]
        snrs_p.append(compute_snr(yt_np, yp_np))
        phs_p.append(compute_phase_mae(yt_np, yp_np))
        feats_p.append(extract_feature(yp_np))
        cls_te.append(int(cl.item()))
    y_test = np.array(cls_te)
    clf_p = make_pipeline(StandardScaler(), LogisticRegression(C=10., solver='liblinear', max_iter=2000, random_state=seed))
    clf_p.fit(Xtr_arr, ytr_arr)
    f1_p = f1_score(y_test, clf_p.predict(np.array(feats_p)), average='macro') * 100

    # Mamba
    set_seed(seed)
    model = ResidualCorrector().to(DEVICE)
    if os.path.exists(PRETRAINED_MODEL_PATH):
        model.load_state_dict(torch.load(PRETRAINED_MODEL_PATH, map_location=DEVICE, weights_only=True))
    train_model(model, tr_loader, val_loader, EPOCHS, LR_MAMBA, "Mamba")

    snrs_m, phs_m, feats_m = [], [], []
    for batch in te_loader:
        x, yt, yp, cl, sc = batch
        yhat = reconstruct(model, x, sc, yp, "Mamba")
        snrs_m.append(compute_snr(yt.numpy()[0], yhat))
        phs_m.append(compute_phase_mae(yt.numpy()[0], yhat))
        feats_m.append(extract_feature(yhat))
    clf_m = make_pipeline(StandardScaler(), LogisticRegression(C=10., solver='liblinear', max_iter=2000, random_state=seed))
    clf_m.fit(Xtr_arr, ytr_arr)
    f1_m = f1_score(y_test, clf_m.predict(np.array(feats_m)), average='macro') * 100

    print(f"    seed={seed}  Poly SNR={np.nanmean(snrs_p):.2f} F1={f1_p:.1f}%  "
          f"Mamba SNR={np.nanmean(snrs_m):.2f} F1={f1_m:.1f}%")
    return {
        "Poly":  {"SNR": float(np.nanmean(snrs_p)), "PhsMAE": float(np.nanmean(phs_p)), "F1": f1_p},
        "Mamba": {"SNR": float(np.nanmean(snrs_m)), "PhsMAE": float(np.nanmean(phs_m)), "F1": f1_m},
    }

def main():
    print(f"CWRU Scale-Factor Ablation: ×4 / ×8 / ×16  (seeds={MULTI_SEEDS})")
    # all_results[scale_k] = {metric: {mean, std}}
    all_results = {}

    for k in SCALE_LIST:
        print(f"\n{'='*60}")
        print(f"  SCALE_K = ×{k}")
        print(f"{'='*60}")
        seed_records = []
        for seed in MULTI_SEEDS:
            sr = run_scale_seed(k, seed)
            seed_records.append(sr)

        # Aggregate
        scale_res = {}
        for method in ["Poly", "Mamba"]:
            for metric in ["SNR", "PhsMAE", "F1"]:
                vals = [sr[method][metric] for sr in seed_records]
                scale_res.setdefault(method, {})[f"{metric}_mean"] = float(np.mean(vals))
                scale_res.setdefault(method, {})[f"{metric}_std"]  = float(np.std(vals))
        all_results[k] = scale_res

    # Summary table
    print("\n" + "="*80)
    print(f"{'Scale':<5}  {'Poly SNR':>14}  {'Mamba SNR':>14}  {'Poly F1':>13}  {'Mamba F1':>13}")
    print("-"*80)
    for k in SCALE_LIST:
        r = all_results[k]
        print(f"×{k:<4}  "
              f"{r['Poly']['SNR_mean']:>5.2f}±{r['Poly']['SNR_std']:>4.2f}  "
              f"{r['Mamba']['SNR_mean']:>5.2f}±{r['Mamba']['SNR_std']:>4.2f}  "
              f"{r['Poly']['F1_mean']:>5.1f}±{r['Poly']['F1_std']:>4.1f}%  "
              f"{r['Mamba']['F1_mean']:>5.1f}±{r['Mamba']['F1_std']:>4.1f}%")

    # Save JSON
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = os.path.join(RESULT_DIR, f"cwru_scale_ablation_{ts}.json")
    with open(out, "w") as f: json.dump({str(k): v for k,v in all_results.items()}, f, indent=2)
    print(f"\nSaved: {out}")

    # Generate line plot with error bars
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    scales = SCALE_LIST
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    metrics_cfg = [
        ("SNR",    "SNR (dB)",           "higher is better"),
        ("PhsMAE", "Phase MAE (rad)",    "lower is better"),
        ("F1",     "Macro F1-Score (%)", "higher is better"),
    ]
    for ax, (key, ylabel, note) in zip(axes, metrics_cfg):
        poly_m  = [all_results[k]["Poly"][f"{key}_mean"]  for k in scales]
        poly_s  = [all_results[k]["Poly"][f"{key}_std"]   for k in scales]
        mamba_m = [all_results[k]["Mamba"][f"{key}_mean"] for k in scales]
        mamba_s = [all_results[k]["Mamba"][f"{key}_std"]  for k in scales]

        ax.errorbar(scales, poly_m,  yerr=poly_s,  fmt='o--', color='#95a5a6',
                    linewidth=2, markersize=7, capsize=5, label='Poly')
        ax.errorbar(scales, mamba_m, yerr=mamba_s, fmt='s-',  color='#1abc9c',
                    linewidth=2, markersize=7, capsize=5, label='Mamba (ours)')

        for k, pm, mm in zip(scales, poly_m, mamba_m):
            delta = mm - pm
            ax.annotate(f'{delta:+.2f}', xy=(k, (pm+mm)/2),
                        ha='center', va='bottom', fontsize=8.5,
                        color='#e74c3c', fontweight='bold')

        ax.set_xticks(scales)
        ax.set_xticklabels([f'×{k}' for k in scales])
        ax.set_xlabel('Downsampling Factor', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f'{ylabel}\n({note})', fontsize=11, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(axis='y', linestyle='--', alpha=0.5)

    fig.suptitle('CWRU Scale-Factor Ablation: Poly vs Mamba (mean±std, 3 seeds)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plot_out = os.path.join(RESULT_DIR, f"cwru_scale_ablation_lineplot_{ts}.png")
    plt.savefig(plot_out, dpi=300, bbox_inches='tight')
    print(f"Plot saved: {plot_out}")

if __name__ == "__main__":
    main()
