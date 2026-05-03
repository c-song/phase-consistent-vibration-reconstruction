"""
run_gearbox_scratch_ablation.py
Trains Mamba_scratch (Bi-S-Mamba, no pretrained weights) over 3 seeds
and merges results into an existing gearbox_architecture_ablation_*.json.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_gearbox_architecture_ablation import (
    DEVICE, MULTI_SEEDS, EPOCHS, LR_BASE, BATCH_SIZE,
    ROOT_DIR, HZ_KEEP, TRAIN_LOADS, TEST_LOADS,
    SCALE_K, TARGET_LEN, HOP,
    RESULT_DIR, OUT_DIR,
    set_seed, GearboxReconstructionDataset, ResidualCorrector,
    train_model_fair, reconstruct,
    calc_metrics, extract_feature_vector, downstream_check,
)
import json, time, copy, glob
import numpy as np
import torch
from torch.utils.data import DataLoader

_matches = sorted(glob.glob(os.path.join(RESULT_DIR, "gearbox_architecture_ablation_*.json")))
EXISTING_JSON = _matches[-1] if _matches else os.path.join(RESULT_DIR, "gearbox_architecture_ablation_notfound.json")
NEW_KEY = "Mamba_scratch"


def run_scratch_seed(seed, full_train_ds, test_ds):
    set_seed(seed)

    # Seed-dependent val split
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

    train_loader = DataLoader(torch.utils.data.Subset(full_train_ds, train_indices),
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader   = DataLoader(torch.utils.data.Subset(full_train_ds, val_indices),
                              batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=1, shuffle=False)

    # Random init, lr=LR_BASE
    set_seed(seed)
    model = ResidualCorrector().to(DEVICE)
    print(f"  [Seed {seed}] Training {NEW_KEY} from scratch "
          f"(lr={LR_BASE}, epochs={EPOCHS})...")
    train_model_fair(model, train_loader, val_loader,
                     epochs=EPOCHS, lr=LR_BASE, name=NEW_KEY)

    # GT downstream classifier
    set_seed(seed)
    Xtr, ytr = [], []
    gen = torch.Generator().manual_seed(seed)
    MAX_FEATS = 4000
    loader_tr = DataLoader(full_train_ds, batch_size=1, shuffle=True,
                           num_workers=0, generator=gen)
    cls_count = {}
    for batch in loader_tr:
        inp, gt, base, cls_t, scale_t = batch
        c = int(cls_t.item())
        if cls_count.get(c, 0) < MAX_FEATS:
            Xtr.append(extract_feature_vector(gt.numpy()[0]))
            ytr.append(c)
            cls_count[c] = cls_count.get(c, 0) + 1
    Xtr_arr, ytr_arr = np.array(Xtr), np.array(ytr)

    # Evaluation
    metrics_pool = {"SNR": [], "PhsMAE": [], "GD_MAE": []}
    X_rec_pool, infer_times, y_test_cls = [], [], []

    for batch in test_loader:
        inp, gt, base, cls_t, scale_t = batch
        gt_np = gt.numpy()[0]
        y_test_cls.append(int(cls_t.item()))

        if torch.cuda.is_available(): torch.cuda.synchronize()
        t0 = time.perf_counter()
        y_hat = reconstruct(model, inp, scale_t, base, NEW_KEY)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        infer_times.append(time.perf_counter() - t0)
        X_rec_pool.append(extract_feature_vector(y_hat))
        for k, v in calc_metrics(gt_np, y_hat).items():
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
    print(f"  [Seed {seed}] SNR={result['SNR']:.2f}  "
          f"PhsMAE={result['PhsMAE']:.3f}  "
          f"F1={result['F1']:.1f}%  Fisher={result['Fisher']:.3f}")
    return result


def main():
    print("=" * 60)
    print(f"Gearbox Mamba_scratch Ablation  seeds={MULTI_SEEDS}")
    print("=" * 60)

    full_train_ds = GearboxReconstructionDataset(
        ROOT_DIR, "train", HZ_KEEP, TRAIN_LOADS, TEST_LOADS,
        TARGET_LEN, HOP, SCALE_K)
    test_ds = GearboxReconstructionDataset(
        ROOT_DIR, "test", HZ_KEEP, TRAIN_LOADS, TEST_LOADS,
        TARGET_LEN, HOP, SCALE_K)
    print(f"Train segments: {len(full_train_ds)}  Test segments: {len(test_ds)}")

    per_seed = []
    for seed in MULTI_SEEDS:
        per_seed.append(run_scratch_seed(seed, full_train_ds, test_ds))

    # Aggregate
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

    # Merge
    if os.path.exists(EXISTING_JSON):
        with open(EXISTING_JSON) as f:
            base = json.load(f)
    else:
        print(f"WARNING: {EXISTING_JSON} not found.")
        base = {"timestamp": "merged", "seeds": MULTI_SEEDS, "results": {}}

    merged = copy.deepcopy(base)
    merged["results"][NEW_KEY] = scratch_agg

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(RESULT_DIR, f"gearbox_arch_with_scratch_{ts}.json")
    with open(out_path, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"\nMerged JSON saved: {out_path}")

    # Print table
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
