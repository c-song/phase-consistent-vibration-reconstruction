# Bi-S-Mamba: Phase-Consistent Signal Reconstruction

Official code for the paper:  
**"Phase-Consistent Signal Reconstruction for Bandwidth-Constrained Vibration Signals"**

---

## Overview

Industrial vibration sensors are often bandwidth-limited. This work recovers high-fidelity, phase-coherent signals from low-rate measurements using a two-stage framework:

1. **Synthetic Pre-training** — the Bi-S-Mamba model learns general phase equalization on physics-based synthetic data with a theory-driven Phase-Aware Loss.
2. **Real-data Fine-tuning** — the pretrained model is fine-tuned on target datasets (CWRU / OEDI Gearbox), closing the domain gap with minimal data.

At inference, the model outputs a complex-spectrum *residual* that corrects the polyphase interpolation baseline, then converts back to the time domain.

```
Low-rate input
   │
   ├─► FIR Decimate ──► Polyphase Upsample (Poly baseline)
   │                            │
   │         rfft               │
   └───────────────────► [B, 2, K] ──► Bi-S-Mamba ──► residual
                                                           │
                                          y_poly + irfft(residual) = y_hat
```

---

## File Structure

```
src/
├── model_mamba.py              # Bi-S-Mamba (BiMambaBlock + ResidualCorrector)
├── model_baselines.py          # Comparison baselines: CNN, Transformer, BiLSTM
├── phase_aware_loss.py         # Phase-Aware Loss (with GD term and energy masking)
├── data_generator.py           # Synthetic signal generator for pre-training
├── config.py                   # Shared configuration (fs, duration, batch size, …)
├── utils.py                    # Signal utilities (SNR, complex↔channel, windowed interp)
├── utils_pub.py                # Extended metrics (PhsMAE, GD_MAE, Fisher ratio)
│
├── train_poly_bimamba_phase_aware_noise.py      # Stage 1: synthetic pre-training
│                                                #   → mamba_poly_phase_aware_best_noise.pth
├── test_baseline_final_report_zscore_noise.py   # Synthetic data evaluation
│                                                   #   Poly/Hann/LASSO/Mamba comparison
│
├── run_cwru_finetune_phase_aware_with_noise_fivefeatures.py   # Stage 2a: CWRU single-run fine-tune
│                                                               #   + full visual report (Fig. 4)
├── run_cwru_architecture_ablation.py    # CWRU multi-seed (3×) architecture comparison (Table 9)
├── run_cwru_scale_ablation.py           # Scale-factor (×4/×8/×16) ablation on CWRU (Table 10)
├── run_cwru_scratch_ablation.py         # CWRU scratch vs. pretrain ablation
│
├── run_gearbox_finetune_phase_aware_with_noise_fivefeatures.py           # Stage 2a: Gearbox fine-tune
│                                                                          #   + full visual report (Fig. 5)
├── run_gearbox_architecture_ablation.py # Gearbox multi-seed (3×) architecture comparison (Table 8)
└── run_gearbox_scratch_ablation.py      # Gearbox scratch vs. pretrain ablation
```

---

## Requirements

```
Python >= 3.10
PyTorch >= 2.0
mamba-ssm >= 1.2.0   # requires CUDA + Linux
numpy, scipy, scikit-learn, matplotlib
```

Install `mamba-ssm` (CUDA required):
```bash
pip install mamba-ssm
```

Install remaining dependencies:
```bash
pip install torch scipy scikit-learn matplotlib
```

---

## Quick Start

### Step 1 — Pre-train on synthetic data

```bash
python train_poly_bimamba_phase_aware_noise.py
# Output: mamba_poly_phase_aware_best_noise.pth
```

Trains Bi-S-Mamba with Phase-Aware Loss on dynamically generated synthetic signals with 5% additive noise (150 epochs, early stopping patience=20). No dataset download needed.

### Step 1b — Evaluate on synthetic data (optional)

```bash
python test_baseline_final_report_zscore_noise.py
# Output: baseline_result/
```

Compares Rect(FD) / Poly / Hann-sinc / LASSO / Mamba on 100 Monte Carlo synthetic samples. Requires `mamba_poly_phase_aware_best_noise.pth`.

### Step 2 — Fine-tune on CWRU

Place the CWRU `.mat` files in `./cwru_mat/` (files: `97–138.mat`, drive-end DE_time channel, 48 kHz).

```bash
# Single-run fine-tune with full visual report
python run_cwru_finetune_phase_aware_with_noise_fivefeatures.py
# Outputs → plots_cwru_phase_aware_finetune/

# Multi-seed (3×) architecture comparison — quantitative tables
python run_cwru_architecture_ablation.py
# Results → cwru_result/

# Scale-factor ablation (×4 / ×8 / ×16)
python run_cwru_scale_ablation.py

# Scratch vs. pretrain ablation
python run_cwru_scratch_ablation.py
```

### Step 3 — Fine-tune on OEDI Gearbox

Place the OEDI gearbox `.txt` files in `./gearboxdata_extracted/` with subdirectories `Healthy Data/` and `BrokenTooth Data/`.

```bash
# Single-run fine-tune with full visual report
python run_gearbox_finetune_phase_aware_with_noise_fivefeatures.py
# Outputs → plots_gearbox_phase_aware_finetune/

# Multi-seed (3×) architecture comparison
python run_gearbox_architecture_ablation.py
# Results → gearbox_result/

# Scratch vs. pretrain ablation
python run_gearbox_scratch_ablation.py
```

---

## Datasets (not included)

| Dataset | Source | Notes |
|---|---|---|
| CWRU Bearing | [Case Western Reserve University](https://engineering.case.edu/bearingdatacenter) | Drive-end, 48 kHz, files 97–138 |
| OEDI Gearbox | [OEDI Data Lake](https://data.openei.org/submissions/4448) | 30 Hz rotation, load 0–90 Nm, healthy / broken tooth |

---

## Pre-trained Weights

The pre-trained checkpoint `mamba_poly_phase_aware_best_noise.pth` (~1.5 MB) is released alongside this repository. Fine-tuned checkpoints for CWRU and Gearbox are generated by the fine-tuning scripts.

---

## Citation

```bibtex
@article{song2026bimamba,
  title   = {Phase-Consistent Signal Reconstruction for Bandwidth-Constrained Vibration Monitoring},
  author  = {...},
  journal = {…},
  year    = {2026}
}
```
