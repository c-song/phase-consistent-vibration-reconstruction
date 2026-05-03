"""
train_poly_bimamba_phase_aware.py

Pre-training stage: learn general physical phase equalization on high-SNR synthetic data.
1. Task: Pure PhaseAwareLoss (aligned with Theorem 13).
2. Config: energy_threshold = 0.01 (as specified in the paper for pretraining).
3. Data: Dynamic stream with exact downstream Z-Score normalization.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset
import scipy.signal as signal
import os
import scipy.signal as signal
import config
import utils
import data_generator
from model_mamba import ResidualCorrector

from phase_aware_loss import PhaseAwareLoss

# Config
BATCH_SIZE = getattr(config, 'BATCH_SIZE', 32)
LEARNING_RATE = 1e-4  # lower LR stabilizes phase trigonometric optimization
EPOCHS = 150
STEPS_PER_EPOCH = 600 
SCALE_K = getattr(config, 'SCALE_K', 4)
SEED = getattr(config, 'SEED', 42)

# Loss hyperparameters (Section 3.4/4.2)
LAMBDA_FREQ = 0.05
LAMBDA_PHASE = 1.0 
LAMBDA_GD = 0.1
ENERGY_THRESHOLD = 0.01  # 0.01 for high-SNR pretraining; 0.0 for low-SNR fine-tuning
LAMBDA_AUX = 0.05        # auxiliary spectral residual supervision, stabilizes early gradients

# === Data Generation Functions ===
def poly_interpolation(y_low, target_len):
    """ Simulate industrial Polyphase interpolation """
    y_poly = signal.resample_poly(y_low, up=SCALE_K, down=1)

    if len(y_poly) > target_len:
        y_poly = y_poly[:target_len]
    elif len(y_poly) < target_len:
        y_poly = np.pad(y_poly, (0, target_len - len(y_poly)), mode='edge')
    return y_poly.astype(np.float32)

class SyntheticStreamDataset(IterableDataset):
    """Stream dataset that generates samples on-the-fly to prevent memorization."""
    def __init__(self, steps_per_epoch, batch_size):
        self.steps_per_epoch = steps_per_epoch
        self.batch_size = batch_size

    def __iter__(self):
        for _ in range(self.steps_per_epoch):
            batch_x, batch_y, batch_gt_time, batch_poly_time, batch_sigma = [], [], [], [], []
            
            for _ in range(self.batch_size):
                _, y_true = data_generator.create_pair_data()
                target_len = len(y_true)

                # Z-score normalization must match the downstream evaluation pipeline
                noise_level = 0.05 * np.std(y_true)
                y_true = y_true + np.random.normal(0, noise_level, y_true.shape)

                y_true = (y_true - np.mean(y_true)) / (np.std(y_true) + 1e-8)
                y_low = signal.decimate(y_true, SCALE_K, ftype="fir", zero_phase=True)
                y_poly = poly_interpolation(y_low, target_len)

                spec_poly = np.fft.rfft(y_poly)
                spec_true = np.fft.rfft(y_true)
                residual_spec = spec_true - spec_poly

                # Spectral normalization
                sigma = float(np.std(np.abs(spec_poly)) + 1e-8)
                spec_poly_norm = spec_poly / sigma
                residual_spec_norm = residual_spec / sigma

                batch_x.append(utils.complex_to_channels(spec_poly_norm))
                batch_y.append(utils.complex_to_channels(residual_spec_norm))
                batch_gt_time.append(y_true.astype(np.float32))
                batch_poly_time.append(y_poly.astype(np.float32))
                batch_sigma.append(sigma)

            yield (
                torch.tensor(np.array(batch_x), dtype=torch.float32),
                torch.tensor(np.array(batch_y), dtype=torch.float32), 
                torch.tensor(np.array(batch_gt_time)[:, np.newaxis, :], dtype=torch.float32),
                torch.tensor(np.array(batch_poly_time)[:, np.newaxis, :], dtype=torch.float32),
                torch.tensor(np.array(batch_sigma), dtype=torch.float32)
            )

class EarlyStopping:
    def __init__(self, patience=15, min_delta=1e-5, path='mamba_poly_phase_aware_best.pth'):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        print(f'   ✅ [Save] Val loss decreased ({self.best_loss:.6f} --> {val_loss:.6f}).')
        torch.save(model.state_dict(), self.path)
        self.best_loss = val_loss

def freq_to_time_domain(freq_channels, target_length=None):
    real = freq_channels[:, 0, :]
    imag = freq_channels[:, 1, :]
    complex_spec = torch.complex(real, imag)
    if target_length is not None:
        time_signal = torch.fft.irfft(complex_spec, n=target_length, dim=-1)
    else:
        time_signal = torch.fft.irfft(complex_spec, dim=-1)
    return time_signal.unsqueeze(1)

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")
    print(f"📊 Pre-training with Standard Phase-Aware Loss")
    print(f"   Config: threshold={ENERGY_THRESHOLD} (High-SNR mode), lambda_gd={LAMBDA_GD}")

    train_dataset = SyntheticStreamDataset(steps_per_epoch=STEPS_PER_EPOCH, batch_size=BATCH_SIZE)
    val_dataset = SyntheticStreamDataset(steps_per_epoch=int(STEPS_PER_EPOCH * 0.2), batch_size=BATCH_SIZE)

    train_loader = DataLoader(train_dataset, batch_size=None)
    val_loader = DataLoader(val_dataset, batch_size=None)

    model = ResidualCorrector().to(device)

    criterion = PhaseAwareLoss(
        lambda_freq=LAMBDA_FREQ,
        lambda_phase=LAMBDA_PHASE,
        lambda_gd=LAMBDA_GD,
        energy_threshold=ENERGY_THRESHOLD
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    save_path = "mamba_poly_phase_aware_best_noise.pth"
    early_stopping = EarlyStopping(patience=20, path=save_path)

    print("=" * 100)

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        train_loss_components = {'time': 0.0, 'freq_mag': 0.0, 'phase': 0.0, 'gd': 0.0, 'aux': 0.0}

        for batch_data in train_loader:
            batch_x, batch_y, batch_gt_time, batch_poly_time, batch_sigma = [b.to(device) for b in batch_data]

            optimizer.zero_grad()
            pred_norm = model(batch_x) 
            
            # L1 auxiliary loss in the normalized spectral domain
            loss_aux = F.l1_loss(pred_norm, batch_y)
            
            sigma_expanded = batch_sigma.view(-1, 1, 1)
            pred_residual_denorm = pred_norm * sigma_expanded 

            target_length = batch_gt_time.shape[2] 
            pred_residual_time = freq_to_time_domain(pred_residual_denorm, target_length=target_length) 
            pred_final = batch_poly_time + pred_residual_time 

            loss_main, loss_dict = criterion(pred_final, batch_gt_time, return_components=True)
            
            loss_total = loss_main + LAMBDA_AUX * loss_aux
            
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss_total.item()
            train_loss_components['time'] += loss_dict.get('time', 0.0)
            train_loss_components['freq_mag'] += loss_dict.get('freq_mag', loss_dict.get('freq', 0.0))
            train_loss_components['phase'] += loss_dict.get('phase', 0.0)
            train_loss_components['gd'] += loss_dict.get('gd', 0.0)
            train_loss_components['aux'] += loss_aux.item()

        train_loss /= STEPS_PER_EPOCH
        train_loss_components = {k: v / STEPS_PER_EPOCH for k, v in train_loss_components.items()}

        # ========== Validation ==========
        model.eval()
        val_loss = 0.0
        val_loss_components = {'time': 0.0, 'freq_mag': 0.0, 'phase': 0.0, 'gd': 0.0, 'aux': 0.0}
        val_steps = int(STEPS_PER_EPOCH * 0.2)

        with torch.no_grad():
            for batch_data in val_loader:
                batch_x, batch_y, batch_gt_time, batch_poly_time, batch_sigma = [b.to(device) for b in batch_data]

                pred_norm = model(batch_x)
                loss_aux = F.l1_loss(pred_norm, batch_y)
                
                sigma_expanded = batch_sigma.view(-1, 1, 1)
                pred_residual_denorm = pred_norm * sigma_expanded
                pred_residual_time = freq_to_time_domain(pred_residual_denorm, target_length=batch_gt_time.shape[2])
                pred_final = batch_poly_time + pred_residual_time

                loss_main, loss_dict = criterion(pred_final, batch_gt_time, return_components=True)
                loss_total = loss_main + LAMBDA_AUX * loss_aux
                
                val_loss += loss_total.item()
                val_loss_components['time'] += loss_dict.get('time', 0.0)
                val_loss_components['freq_mag'] += loss_dict.get('freq_mag', loss_dict.get('freq', 0.0))
                val_loss_components['phase'] += loss_dict.get('phase', 0.0)
                val_loss_components['gd'] += loss_dict.get('gd', 0.0)
                val_loss_components['aux'] += loss_aux.item()

        val_loss /= val_steps
        val_loss_components = {k: v / val_steps for k, v in val_loss_components.items()}

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Ep {epoch+1:03d} [LR: {current_lr:.2e}] | "
              f"Tr: {train_loss:.4f} (Phs:{train_loss_components['phase']:.4f}, GD:{train_loss_components['gd']:.4f}) | "
              f"Val: {val_loss:.4f} (Phs:{val_loss_components['phase']:.4f}, GD:{val_loss_components['gd']:.4f})")

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("🛑 Early stopping triggered!")
            break

    print("=" * 100)
    print(f"✅ Pre-training completed! Base model saved to: {save_path}")
    print("   Now you can use this model for downstream fine-tuning (CWRU/Gearbox).")

if __name__ == "__main__":
    main()