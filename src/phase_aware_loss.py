"""
Phase-Aware Loss Function for Signal Super-Resolution
Based on Theorem 13 in the paper

Implements theory-driven loss function derived from CRLB analysis
"""

import torch
import torch.nn as nn
import torch.fft


class PhaseAwareLoss(nn.Module):
    """
    Phase-Aware Loss for Signal Super-Resolution

    L_total = L_time + λ_freq * L_freq_mag + λ_phase * L_phase

    Theorem 13: Traditional MSE weights phase errors by magnitude^2, causing
    negligible gradient flow for high-frequency components. This loss provides
    uniform phase error weighting across all frequency bins, aligning with
    CRLB's uniform variance requirement.

    Args:
        lambda_freq (float): Weight for frequency-domain magnitude loss (L1)
        lambda_phase (float): Weight for phase loss (cosine distance)

    Reference:
        Section 3.6 "Theory-Driven Loss Function Design"
        Paper: "Mamba-Based Residual Correction for Phase-Preserving
               Signal Reconstruction in Digital Twin Systems"
    """

    def __init__(self, lambda_freq=0.1, lambda_phase=1.0, lambda_gd = 0.5, energy_threshold=0.01):
        super().__init__()
        self.lambda_freq = lambda_freq
        self.lambda_phase = lambda_phase
        self.lambda_gd = lambda_gd
        self.energy_threshold = energy_threshold  # Energy masking threshold
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()

    def forward(self, pred, target, return_components=False):
        """
        Args:
            pred: [B, 1, N] predicted signal (time domain)
            target: [B, 1, N] ground truth signal (time domain)
            return_components: If True, return dict with loss components

        Returns:
            loss_total: scalar tensor
            loss_dict (optional): dict with individual loss components
        """
        # 1. Time-domain MSE: L_time = (1/N) Σ |y_GT[n] - ŷ[n]|²
        loss_time = self.mse(pred, target)

        # 2. Frequency-domain magnitude loss (L1)
        pred_fft = torch.fft.rfft(pred, dim=-1)  # [B, 1, N//2+1] complex
        target_fft = torch.fft.rfft(target, dim=-1)

        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)

        loss_freq_mag = self.l1(pred_mag, target_mag)

        # 3. Phase loss with energy masking
        # Normalize to unit circle to isolate phase: e^{jφ} = FFT / |FFT|
        eps = 1e-8
        pred_phase_complex = pred_fft / (pred_mag + eps)
        target_phase_complex = target_fft / (target_mag + eps)

        # Re(e^{jφ_GT} · conj(e^{jφ_pred})) = cos(φ_GT - φ_pred)
        phase_cosine = (target_phase_complex * torch.conj(pred_phase_complex)).real

        target_energy = target_mag ** 2
        threshold_ratio = getattr(self, 'energy_threshold', 0.01)
        max_energy = target_energy.max(dim=-1, keepdim=True)[0]
        energy_mask = target_energy > (max_energy * threshold_ratio)

        raw_phase_loss = 2.0 * (1.0 - phase_cosine)
        loss_phase = (raw_phase_loss * energy_mask).sum() / (energy_mask.sum() + eps)

        # 4. Group Delay loss with masking
        # Approximate GD via conjugate product of adjacent bins: X[k] * conj(X[k+1])
        pred_gd_proxy = pred_fft[..., :-1] * torch.conj(pred_fft[..., 1:])
        target_gd_proxy = target_fft[..., :-1] * torch.conj(target_fft[..., 1:])

        pred_gd_norm = pred_gd_proxy / (torch.abs(pred_gd_proxy) + eps)
        target_gd_norm = target_gd_proxy / (torch.abs(target_gd_proxy) + eps)

        gd_cosine = (target_gd_norm * torch.conj(pred_gd_norm)).real
        raw_gd_loss = 2.0 * (1.0 - gd_cosine)

        # GD is valid only when both adjacent bins have significant energy
        mask_gd = energy_mask[..., :-1] & energy_mask[..., 1:]
        loss_gd = (raw_gd_loss * mask_gd).sum() / (mask_gd.sum() + eps)

        # 5. Total loss
        weight_gd = getattr(self, 'lambda_gd', 0.1)
        
        loss_total = (loss_time +
                     self.lambda_freq * loss_freq_mag +
                     self.lambda_phase * loss_phase +
                     weight_gd * loss_gd)

        if return_components:
            loss_dict = {
                'total': loss_total.item(),
                'time': loss_time.item(),
                'freq_mag': loss_freq_mag.item(),
                'phase': loss_phase.item(),
                'gd': loss_gd.item()
            }
            return loss_total, loss_dict
        else:
            return loss_total


class SimplifiedPhaseAwareLoss(nn.Module):
    """
    Simplified version: Only MSE + Phase Loss (no separate freq magnitude)

    L = L_time + λ_phase * L_phase

    Use this if you want a simpler baseline or if frequency magnitude
    is implicitly captured by time-domain MSE.
    """

    def __init__(self, lambda_phase=1.0, energy_threshold=0.01):
        super().__init__()
        self.lambda_phase = lambda_phase
        self.energy_threshold = energy_threshold
        self.mse = nn.MSELoss()

    def forward(self, pred, target, return_components=False):
        """
        Args:
            pred: [B, 1, N] predicted signal
            target: [B, 1, N] ground truth signal
        """
        # Time-domain MSE
        loss_time = self.mse(pred, target)

        # Phase loss with energy masking
        pred_fft = torch.fft.rfft(pred, dim=-1)
        target_fft = torch.fft.rfft(target, dim=-1)

        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)

        pred_phase = pred_fft / (pred_mag + 1e-8)
        target_phase = target_fft / (target_mag + 1e-8)

        phase_cosine = (target_phase * torch.conj(pred_phase)).real

        # Energy masking
        target_energy = target_mag ** 2
        max_energy = target_energy.max(dim=-1, keepdim=True)[0]
        energy_mask = target_energy > (max_energy * self.energy_threshold)

        raw_phase_loss = 2.0 * (1.0 - phase_cosine)
        masked_phase_loss = raw_phase_loss * energy_mask
        num_valid_freqs = energy_mask.sum() + 1e-8
        loss_phase = masked_phase_loss.sum() / num_valid_freqs

        loss_total = loss_time + self.lambda_phase * loss_phase

        if return_components:
            return loss_total, {
                'total': loss_total.item(),
                'time': loss_time.item(),
                'phase': loss_phase.item()
            }
        else:
            return loss_total
# ==========================================
# 2. 新增 Loss (用于 Gearbox 微调) - 升级版！
# ==========================================
# class RobustGearboxLoss(nn.Module):
#     """
#     [New] Robust Adaptation for Low-SNR Gearbox
    
#     Improvements over original:
#     1. Time Domain: Huber Loss (Handles impulsive noise better than MSE)
#     2. Phase/GD: Differentiable Surrogate (More stable gradients)
#     3. Masking: Hard Gating (0.5%) to strictly filter broadband noise
#     """
#     def __init__(self, lambda_freq=0.02, lambda_phase=0.7, lambda_gd=0.1, energy_threshold=0.005):
#         super().__init__()
#         self.lambda_freq = lambda_freq
#         self.lambda_phase = lambda_phase
#         self.lambda_gd = lambda_gd
#         self.energy_threshold = energy_threshold
        
#         # 变化 1: 时域用 Huber 或 MSE (推荐先用 MSE 保持物理一致性，若不稳定再换 Huber)
#         self.mse = nn.MSELoss() 
#         self.l1 = nn.L1Loss()

#     def forward(self, pred, target, return_components=False):
#         # 1. Time Domain
#         loss_time = self.mse(pred, target)
#         #loss_time = self.l1(pred, target)
#         # FFT
#         pred_fft = torch.fft.rfft(pred, dim=-1)
#         target_fft = torch.fft.rfft(target, dim=-1)
#         pred_mag = torch.abs(pred_fft)
#         target_mag = torch.abs(target_fft)

#         # 2. Freq Domain (L1 Sparsity - 核心去噪)
#         loss_freq = self.l1(pred_mag, target_mag)

#         # 3. Hard Gating (关键!)
#         target_energy = target_mag ** 2
#         max_energy = target_energy.max(dim=-1, keepdim=True)[0]
#         # 阈值 0.005
#         mask = (target_energy > (max_energy * self.energy_threshold)).float()
#         num_valid = mask.sum() + 1e-8

#         # 4. Phase Loss (Masked)
#         pred_phase = pred_fft / (pred_mag + 1e-8)
#         target_phase = target_fft / (target_mag + 1e-8)
#         phase_cosine = (target_phase * torch.conj(pred_phase)).real
#         loss_phase = (2.0 * (1.0 - phase_cosine) * mask).sum() / num_valid

#         # 5. Differentiable GD (可微 GD 代理)
#         # 用共轭乘积算相邻频点相位差，数值更稳
#         pred_gd_vec = pred_fft[..., 1:] * torch.conj(pred_fft[..., :-1])
#         target_gd_vec = target_fft[..., 1:] * torch.conj(target_fft[..., :-1])
        
#         # Normalize
#         pred_gd_vec = pred_gd_vec / (torch.abs(pred_gd_vec) + 1e-8)
#         target_gd_vec = target_gd_vec / (torch.abs(target_gd_vec) + 1e-8)
        
#         # GD Cosine Distance
#         gd_cosine = (target_gd_vec * torch.conj(pred_gd_vec)).real
        
#         # GD Mask (相邻两点都要有效)
#         mask_gd = mask[..., 1:] * mask[..., :-1]
#         loss_gd = (2.0 * (1.0 - gd_cosine) * mask_gd).sum() / (mask_gd.sum() + 1e-8)

#         # Total
#         loss_total = (loss_time + 
#                       self.lambda_freq * loss_freq + 
#                       self.lambda_phase * loss_phase + 
#                       self.lambda_gd * loss_gd)

#         if return_components:
#             return loss_total, {
#                 'total': loss_total.item(),
#                 'time': loss_time.item(),
#                 'freq': loss_freq.item(),
#                 'phase': loss_phase.item(),
#                 'gd': loss_gd.item()
#             }
#         else:
#             return loss_total

import torch
import torch.nn as nn

# ==========================================
# 2. 新增 Loss (用于 Gearbox 微调) - 终极物理加权版 (Bug-Free)
# ==========================================
class RobustGearboxLoss(nn.Module):
    """
    [New] Robust Adaptation for Low-SNR Gearbox
    
    Improvements:
    1. Time: MSE anchors macroscopic energy.
    2. Phase: Sample-wise masked absolute phase alignment.
    3. GD (Route B): Symmetric Amplitude-Weighted Group Delay Regularization.
       (Ensures batch fairness and physical symmetry)
    """
    def __init__(self, lambda_freq=0.05, lambda_phase=1.0, lambda_gd=0.1, energy_threshold=0.01):
        super().__init__()
        self.lambda_freq = lambda_freq
        self.lambda_phase = lambda_phase
        self.lambda_gd = lambda_gd
        self.energy_threshold = energy_threshold
        
        self.mse = nn.MSELoss() 
        self.l1 = nn.L1Loss()

    def forward(self, pred, target, return_components=False):
        # 1. Time Domain (MSE 保底)
        loss_time = self.mse(pred, target)

        # FFT 转换
        pred_fft = torch.fft.rfft(pred, dim=-1)
        target_fft = torch.fft.rfft(target, dim=-1)
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)

        # 2. Freq Domain (幅值 L1 稀疏)
        loss_freq = self.l1(pred_mag, target_mag)

        # 3. Hard Gating (0.01 能量硬门控)
        target_energy = target_mag ** 2
        max_energy = target_energy.max(dim=-1, keepdim=True)[0]
        mask = (target_energy > (max_energy * self.energy_threshold)).float()
        
        # ---------------------------------------------------------
        # 4. Phase Loss - 逐样本平均（与旧版一致，控制变量）
        # ---------------------------------------------------------
        pred_phase = pred_fft / (pred_mag + 1e-8)
        target_phase = target_fft / (target_mag + 1e-8)
        phase_cosine = (target_phase * torch.conj(pred_phase)).real

        raw_phase_loss = 2.0 * (1.0 - phase_cosine)
        phase_loss_per_sample = (raw_phase_loss * mask).sum(dim=-1) / (mask.sum(dim=-1) + 1e-8)
        loss_phase = phase_loss_per_sample.mean()

        # ---------------------------------------------------------
        # 5. Differentiable GD - 全局几何平均软加权
        # 移除硬掩码，用几何平均幅值作为全局权重
        # 高SNR场景：高能量频点对权重大，GD loss有效
        # 低SNR场景：噪声频点幅值趋零，在全局求和中被自然压制
        # ---------------------------------------------------------
        pred_gd_vec = pred_fft[..., 1:] * torch.conj(pred_fft[..., :-1])
        target_gd_vec = target_fft[..., 1:] * torch.conj(target_fft[..., :-1])

        pred_gd_vec = pred_gd_vec / (torch.abs(pred_gd_vec) + 1e-8)
        target_gd_vec = target_gd_vec / (torch.abs(target_gd_vec) + 1e-8)

        gd_error = 2.0 * (1.0 - (target_gd_vec * torch.conj(pred_gd_vec)).real)

        weight_gd = torch.sqrt(target_mag[..., 1:] * target_mag[..., :-1])
        loss_gd = (gd_error * weight_gd).sum() / (weight_gd.sum() + 1e-8)
        # ---------------------------------------------------------

        # Total Loss
        loss_total = (loss_time + 
                      self.lambda_freq * loss_freq + 
                      self.lambda_phase * loss_phase + 
                      self.lambda_gd * loss_gd)

        if return_components:
            return loss_total, {
                'total': loss_total.item(),
                'time': loss_time.item(),
                'freq': loss_freq.item(),
                'phase': loss_phase.item(),
                'gd': loss_gd.item()
            }
        else:
            return loss_total
# Example usage
if __name__ == "__main__":
    # Test the loss function
    import numpy as np

    # Create dummy signals
    batch_size = 4
    signal_length = 1024

    # Ground truth: clean sinusoid
    t = torch.linspace(0, 1, signal_length).unsqueeze(0).unsqueeze(0)  # [1, 1, N]
    target = torch.sin(2 * np.pi * 30 * t).repeat(batch_size, 1, 1)  # [B, 1, N]

    # Prediction 1: Same amplitude, phase shift (should have high phase loss)
    pred_phase_shift = torch.sin(2 * np.pi * 30 * t + 0.1).repeat(batch_size, 1, 1)

    # Prediction 2: Same phase, amplitude error (should have low phase loss)
    pred_amp_error = 0.8 * torch.sin(2 * np.pi * 30 * t).repeat(batch_size, 1, 1)

    # Initialize loss
    criterion = PhaseAwareLoss(lambda_freq=0.1, lambda_phase=1.0)

    # Test 1: Phase shift
    loss1, dict1 = criterion(pred_phase_shift, target, return_components=True)
    print("Test 1: Phase shift (Δφ=0.1 rad)")
    print(f"  Total Loss: {dict1['total']:.6f}")
    print(f"  Time Loss:  {dict1['time']:.6f}")
    print(f"  Freq Loss:  {dict1['freq_mag']:.6f}")
    print(f"  Phase Loss: {dict1['phase']:.6f}")
    print()

    # Test 2: Amplitude error
    loss2, dict2 = criterion(pred_amp_error, target, return_components=True)
    print("Test 2: Amplitude error (0.8x)")
    print(f"  Total Loss: {dict2['total']:.6f}")
    print(f"  Time Loss:  {dict2['time']:.6f}")
    print(f"  Freq Loss:  {dict2['freq_mag']:.6f}")
    print(f"  Phase Loss: {dict2['phase']:.6f}")
    print()

    # Verify: Phase loss should be much higher in Test 1
    assert dict1['phase'] > dict2['phase'], "Phase loss should detect phase shift!"
    print("✓ Phase loss correctly distinguishes phase vs amplitude errors")
