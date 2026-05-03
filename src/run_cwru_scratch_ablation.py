"""
run_cwru_scratch_ablation.py
Runs only Mamba_scratch (Bi-S-Mamba, no pretraining, trained from scratch) over 3 seeds.
Results are merged into the existing cwru_method_paradigm_comparison_*.json,
for Scheme C + mini-ablation: quantifying pretraining contribution vs. architecture contribution.

Reuses all utility functions from run_cwru_architecture_ablation.py;
only the initialization strategy differs (no pretrained weights loaded, lr=LR_BASE).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Reuse all constants, models, and utility functions from the main script ────
from run_cwru_architecture_ablation import (
    DEVICE, MULTI_SEEDS, EPOCHS, LR_BASE, BATCH_SIZE,
    DATA_DIR, SCALE_K as SCALE, TARGET_LEN, HOP,
    TRAIN_IDS, TEST_IDS,
    RESULT_DIR, OUT_DIR,
    set_seed, CWRUReconstructionDataset,
    ResidualCorrector,
    train_model_fair, reconstruct,
    calc_metrics, extract_feature_vector, downstream_check,
)
import json, time, copy, glob
import numpy as np
import torch
from torch.utils.data import DataLoader

# Auto-find the latest architecture-ablation JSON so new users don't need to
# edit this file manually after running run_cwru_architecture_ablation.py.
_matches = sorted(glob.glob(os.path.join(RESULT_DIR, "cwru_method_paradigm_comparison_*.json")))
EXISTING_JSON = _matches[-1] if _matches else os.path.join(RESULT_DIR, "cwru_method_paradigm_comparison_notfound.json")
NEW_KEY = "Mamba_scratch"   # display name in tables


def run_scratch_seed(seed, full_train_ds, test_ds):
    set_seed(seed)

    # Identical seed-dependent validation split as in the main script
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

    # Mamba_scratch: random initialization, lr=LR_BASE (same as other baselines)
    set_seed(seed)
    model = ResidualCorrector().to(DEVICE)
    print(f"  [Seed {seed}] Training {NEW_KEY} from scratch (lr={LR_BASE}, epochs={EPOCHS})...")
    train_model_fair(model, train_loader, val_loader, epochs=EPOCHS, lr=LR_BASE, name=NEW_KEY)

    # Downstream classifier using ground-truth features (same as main script)
    set_seed(seed)
    Xtr, ytr = [], []
    gen = torch.Generator().manual_seed(seed)
    MAX_TRAIN_FEATS_PER_CLASS = 4000
    loader_tr = DataLoader(full_train_ds, batch_size=1, shuffle=True,
                           num_workers=0, generator=gen)
    cls_count = {0:0, 1:0, 2:0, 3:0}
    for batch in loader_tr:
        c = int(batch[3].item())
        if cls_count[c] < MAX_TRAIN_FEATS_PER_CLASS:
            Xtr.append(extract_feature_vector(batch[1].numpy()[0]))
            ytr.append(c)
            cls_count[c] += 1
    Xtr_arr, ytr_arr = np.array(Xtr), np.array(ytr)

    # Evaluation
    metrics_pool = {"SNR": [], "PhsMAE": [], "GD_MAE": []}
    X_rec_pool   = []
    infer_times  = []
    y_test_cls   = []

    for batch in test_loader:
        x_in, y_true, y_poly, cls, scale = batch
        y_true_np = y_true.numpy()[0]
        y_test_cls.append(int(cls.item()))

        if torch.cuda.is_available(): torch.cuda.synchronize()
        t0 = time.perf_counter()
        y_hat = reconstruct(model, x_in, scale, batch[2], NEW_KEY)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        infer_times.append(time.perf_counter() - t0)
        X_rec_pool.append(extract_feature_vector(y_hat))
        for k, v in calc_metrics(y_true_np, y_hat).items():
            metrics_pool[k].append(v)

    y_test = np.array(y_test_cls)
    acc, f1, sil, fr = downstream_check(Xtr_arr, ytr_arr,
                                        np.array(X_rec_pool), y_test)
    result = {
        "SNR":     float(np.nanmean(metrics_pool["SNR"])),
        "PhsMAE":  float(np.nanmean(metrics_pool["PhsMAE"])),
        "GD_MAE":  float(np.nanmean(metrics_pool["GD_MAE"])),
        "Acc":     float(acc * 100),
        "Fisher":  float(fr),
        "F1":      float(f1 * 100),
        "InferMs": float(np.mean(infer_times) * 1000),
    }
    print(f"  [Seed {seed}] SNR={result['SNR']:.2f}  PhsMAE={result['PhsMAE']:.3f}  "
          f"F1={result['F1']:.1f}%  Fisher={result['Fisher']:.3f}")
    return result


def main():
    print("=" * 60)
    print(f"CWRU Mamba_scratch Ablation  seeds={MULTI_SEEDS}")
    print("=" * 60)

    full_train_ds = CWRUReconstructionDataset(DATA_DIR, TRAIN_IDS, SCALE, TARGET_LEN, HOP)
    test_ds       = CWRUReconstructionDataset(DATA_DIR, TEST_IDS,  SCALE, TARGET_LEN, HOP)
    print(f"Train segments: {len(full_train_ds)}  Test segments: {len(test_ds)}")

    per_seed = []
    for seed in MULTI_SEEDS:
        per_seed.append(run_scratch_seed(seed, full_train_ds, test_ds))

    # Aggregate mean ± std
    metric_keys = ["SNR", "PhsMAE", "GD_MAE", "Acc", "Fisher", "F1", "InferMs"]
    scratch_agg = {}
    for k in metric_keys:
        vals = [s[k] for s in per_seed]
        scratch_agg[f"{k}_mean"] = float(np.mean(vals))
        scratch_agg[f"{k}_std"]  = float(np.std(vals))

    print(f"\n=== {NEW_KEY} 3-seed summary ===")
    print(f"SNR={scratch_agg['SNR_mean']:.2f}±{scratch_agg['SNR_std']:.2f}  "
          f"PhsMAE={scratch_agg['PhsMAE_mean']:.3f}±{scratch_agg['PhsMAE_std']:.3f}  "
          f"F1={scratch_agg['F1_mean']:.1f}±{scratch_agg['F1_std']:.1f}%")

    # ── Merge into existing JSON ──────────────────────────────────────────────
    if os.path.exists(EXISTING_JSON):
        with open(EXISTING_JSON) as f:
            base = json.load(f)
    else:
        print(f"WARNING: {EXISTING_JSON} not found, creating new file.")
        base = {"timestamp": "merged", "seeds": MULTI_SEEDS, "results": {}}

    merged = copy.deepcopy(base)
    merged["results"][NEW_KEY] = scratch_agg

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(RESULT_DIR, f"cwru_arch_with_scratch_{ts}.json")
    with open(out_path, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"\nMerged JSON saved: {out_path}")

    # ── Print comparison table ────────────────────────────────────────────────
    r = merged["results"]
    order = ["Poly", "BiLSTM", "CNN", "Transf", "VibrMamba", "MDMamba",
             "Mamba_scratch", "Mamba"]
    print(f"\n{'Method':<20} {'SNR':>12} {'PhsMAE':>12} {'GD(ms)':>12} {'F1%':>12}")
    print("-" * 72)
    for m in order:
        if m not in r: continue
        d = r[m]
        print(f"{m:<20} "
              f"{d['SNR_mean']:>5.2f}±{d['SNR_std']:>4.2f}  "
              f"{d['PhsMAE_mean']:>5.3f}±{d['PhsMAE_std']:>5.3f}  "
              f"{d['GD_MAE_mean']*1000:>5.3f}±{d['GD_MAE_std']*1000:>5.3f}  "
              f"{d['F1_mean']:>5.1f}±{d['F1_std']:>4.1f}")

    return out_path


if __name__ == "__main__":
    main()
