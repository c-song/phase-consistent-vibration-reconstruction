
"""
Utility functions for gearbox/CWRU benchmark scripts.

Provides baseline interpolation helpers (FD / windowed / poly / cubic)
and phase/group-delay metrics. Accepts both `band_hz` and `fmin`/`fmax`
keyword forms, and both `mag_ratio` and `rel_mag_th`, so callers with
either naming convention work without modification.
"""

from __future__ import annotations

import numpy as np

# Optional SciPy baselines
try:
    import scipy.signal
    import scipy.interpolate
except Exception:  # pragma: no cover
    scipy = None


# ----------------------------
# Baselines / helpers
# ----------------------------

def complex_to_channels(spec: np.ndarray) -> np.ndarray:
    """Convert complex spectrum -> (2, Nf) float32 channels: [Re, Im]."""
    spec = np.asarray(spec)
    return np.stack([spec.real, spec.imag], axis=0).astype(np.float32)


def fd_interpolation(y_low: np.ndarray, target_len: int):
    """
    Frequency-domain (FD) zero-padding interpolation baseline.
    Returns:
      y_fd: (target_len,)
      spec_fd: rFFT spectrum of y_fd
    """
    y_low = np.asarray(y_low, dtype=np.float32).reshape(-1)
    # rfft at low resolution
    spec_low = np.fft.rfft(y_low)
    # zero pad in frequency to match target_len
    spec_fd = np.zeros(target_len // 2 + 1, dtype=np.complex64)
    m = min(spec_fd.shape[0], spec_low.shape[0])
    spec_fd[:m] = spec_low[:m]
    y_fd = np.fft.irfft(spec_fd, n=target_len).astype(np.float32)
    return y_fd, spec_fd


def windowed_interpolation(y_low: np.ndarray, target_len: int, window: str = "hann"):
    """
    FD interpolation with a window applied in time domain before FFT (Rect/Hann/Blackman).
    Returns y_win, spec_win
    """
    y_low = np.asarray(y_low, dtype=np.float32).reshape(-1)
    if window.lower() == "rect":
        w = np.ones_like(y_low, dtype=np.float32)
    elif window.lower() == "hann":
        w = np.hanning(len(y_low)).astype(np.float32)
    elif window.lower() == "blackman":
        w = np.blackman(len(y_low)).astype(np.float32)
    else:
        raise ValueError(f"Unknown window: {window}")
    y_w = y_low * w
    spec_low = np.fft.rfft(y_w)
    spec_win = np.zeros(target_len // 2 + 1, dtype=np.complex64)
    m = min(spec_win.shape[0], spec_low.shape[0])
    spec_win[:m] = spec_low[:m]
    y_win = np.fft.irfft(spec_win, n=target_len).astype(np.float32)
    return y_win, spec_win


def poly_interpolation(y_low: np.ndarray, target_len: int, up: int):
    """
    Engineering baseline: scipy.signal.resample_poly.
    Returns y_poly, spec_poly
    """
    if scipy is None or not hasattr(scipy, "signal"):
        raise ImportError("scipy is required for poly_interpolation")
    y_low = np.asarray(y_low, dtype=np.float32).reshape(-1)
    # upsample by `up`, then crop/pad to target_len
    y_up = scipy.signal.resample_poly(y_low, up=up, down=1).astype(np.float32)
    if y_up.shape[0] > target_len:
        y_up = y_up[:target_len]
    elif y_up.shape[0] < target_len:
        y_up = np.pad(y_up, (0, target_len - y_up.shape[0]), mode="constant")
    spec = np.fft.rfft(y_up)
    return y_up, spec


def cubic_interpolation(y_low: np.ndarray, target_len: int, up: int):
    """
    Baseline: cubic spline interpolation in time domain.
    Returns y_cubic, spec_cubic
    """
    if scipy is None or not hasattr(scipy, "interpolate"):
        raise ImportError("scipy is required for cubic_interpolation")
    y_low = np.asarray(y_low, dtype=np.float32).reshape(-1)

    n_low = y_low.shape[0]
    n_high = n_low * up
    x_low = np.arange(n_low, dtype=np.float32)
    x_high = np.linspace(0, n_low - 1, num=n_high, dtype=np.float32)

    cs = scipy.interpolate.CubicSpline(x_low, y_low, bc_type="natural")
    y_up = cs(x_high).astype(np.float32)

    if y_up.shape[0] > target_len:
        y_up = y_up[:target_len]
    elif y_up.shape[0] < target_len:
        y_up = np.pad(y_up, (0, target_len - y_up.shape[0]), mode="constant")
    spec = np.fft.rfft(y_up)
    return y_up, spec


def calc_snr(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> float:
    """SNR in dB. Higher is better."""
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    num = np.mean(y_true ** 2)
    den = np.mean((y_true - y_pred) ** 2) + eps
    return float(10.0 * np.log10((num + eps) / den))


# ----------------------------
# Phase / Group delay metrics
# ----------------------------

def _parse_band(band_hz=None, fmin=None, fmax=None):
    if band_hz is not None:
        if not (isinstance(band_hz, (tuple, list)) and len(band_hz) == 2):
            raise ValueError("band_hz must be (fmin,fmax)")
        fmin2, fmax2 = float(band_hz[0]), float(band_hz[1])
        return fmin2, fmax2
    fmin2 = 0.0 if fmin is None else float(fmin)
    fmax2 = np.inf if fmax is None else float(fmax)
    return fmin2, fmax2


def _parse_mag_th(mag_ratio=None, rel_mag_th=None, default=0.05):
    # Accept either keyword name for the relative magnitude threshold.
    if mag_ratio is not None:
        return float(mag_ratio)
    if rel_mag_th is not None:
        return float(rel_mag_th)
    return float(default)



def get_scale(
    spec: np.ndarray,
    *,
    fs: float = None,
    n: int = None,
    freqs: np.ndarray = None,
    band_hz=None,
    fmin=None,
    fmax=None,
    method: str = "p99",
    q: float = 0.99,
    eps: float = 1e-12,
) -> float:
    """Robust per-sample spectral scale for amplitude-invariant training/evaluation.

    Parameters
    ----------
    spec : complex np.ndarray
        Complex spectrum (rfft) of the *baseline/coarse* reconstruction.
    fs, n : used to derive rfftfreq if `freqs` is not provided.
    band_hz / fmin,fmax : restrict scale estimation to a frequency band.
    method : {"p99","max","rms"}
        - "p99": percentile(|spec|, q) within band (robust, recommended)
        - "max": max(|spec|) within band
        - "rms": sqrt(mean(|spec|^2)) within band
    q : float
        Percentile for "p99" method.
    eps : float
        Numerical floor to avoid zero scale.

    Returns
    -------
    float
        Positive scale factor.
    """
    spec = np.asarray(spec)
    mag = np.abs(spec).astype(np.float64)

    if freqs is None:
        if fs is None or n is None:
            # fall back to full spectrum
            freqs = None
        else:
            freqs = np.fft.rfftfreq(int(n), d=1.0 / float(fs))

    if freqs is not None:
        fmin2, fmax2 = _parse_band(band_hz=band_hz, fmin=fmin, fmax=fmax)
        band_mask = (freqs >= fmin2) & (freqs <= fmax2)
        if np.any(band_mask):
            mag = mag[band_mask]

    if mag.size == 0:
        return float(1.0)

    method = str(method).lower()
    if method in ("p99", "p", "percentile"):
        s = float(np.quantile(mag, float(q)))
    elif method == "max":
        s = float(np.max(mag))
    elif method == "rms":
        s = float(np.sqrt(np.mean(mag * mag)))
    else:
        raise ValueError(f"Unknown get_scale method: {method}")
    return float(max(s, eps))

def phase_mae_fft(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    fs: float,
    *,
    band_hz=None,
    fmin=None,
    fmax=None,
    mag_ratio=None,
    rel_mag_th=None,
    eps: float = 1e-12,
):
    """
    Full-sequence FFT phase MAE (radians), using circular phase difference.

    Bin selection:
      - restrict to band (band_hz or fmin/fmax)
      - only count bins where |Y_true| >= mag_th * max(|Y_true|) (relative threshold)

    Lower is better.
    """
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    n = min(len(y_true), len(y_pred))
    y_true = y_true[:n]
    y_pred = y_pred[:n]

    spec_t = np.fft.rfft(y_true)
    spec_p = np.fft.rfft(y_pred)
    freqs = np.fft.rfftfreq(n, d=1.0 / float(fs))

    fmin2, fmax2 = _parse_band(band_hz=band_hz, fmin=fmin, fmax=fmax)
    mag_th = _parse_mag_th(mag_ratio=mag_ratio, rel_mag_th=rel_mag_th, default=0.05)

    band_mask = (freqs >= fmin2) & (freqs <= fmax2)
    mag = np.abs(spec_t)
    # Use peak within band for relative threshold (more stable across datasets)
    mag_band = mag[band_mask] if np.any(band_mask) else mag
    peak = float(np.max(mag_band) + eps)
    mag_mask = mag >= (mag_th * peak)

    mask = band_mask & mag_mask
    # exclude DC for phase/GD metrics
    if mask.shape[0] > 0:
        mask[0] = False
    if not np.any(mask):
        return float("nan")

    ph_t = np.angle(spec_t[mask])
    ph_p = np.angle(spec_p[mask])
    # circular difference in [-pi,pi]
    dphi = np.angle(np.exp(1j * (ph_p - ph_t)))
    return float(np.mean(np.abs(dphi)))


def group_delay_curve(
    y: np.ndarray,
    fs: float,
    *,
    band_hz=None,
    fmin=None,
    fmax=None,
    mag_ratio=None,
    rel_mag_th=None,
    eps: float = 1e-12,
):
    """
    Compute group delay curve τ(f) = - dφ / dω (seconds) from full-sequence rFFT.
    Returns (freqs, gd_seconds) after masking.
    """
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    n = len(y)
    spec = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(n, d=1.0 / float(fs))
    phase = np.unwrap(np.angle(spec))
    omega = 2.0 * np.pi * freqs

    # numerical derivative
    dphi = np.gradient(phase, omega + eps)
    gd = -dphi  # seconds

    fmin2, fmax2 = _parse_band(band_hz=band_hz, fmin=fmin, fmax=fmax)
    mag_th = _parse_mag_th(mag_ratio=mag_ratio, rel_mag_th=rel_mag_th, default=0.05)

    band_mask = (freqs >= fmin2) & (freqs <= fmax2)
    mag = np.abs(spec)
    mag_band = mag[band_mask] if np.any(band_mask) else mag
    peak = float(np.max(mag_band) + eps)
    mag_mask = mag >= (mag_th * peak)

    mask = band_mask & mag_mask
    if mask.shape[0] > 0:
        mask[0] = False
    return freqs[mask], gd[mask]


def phase_error_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    fs: float,
    *,
    band_hz=None,
    fmin=None,
    fmax=None,
    mag_ratio=None,
    rel_mag_th=None,
    eps: float = 1e-12,
):
    """Return (freqs, |Δphase|) using full-sequence rFFT and robust masking.

    The mask keeps bins within the specified band and above a relative magnitude threshold
    defined w.r.t. the peak magnitude *within the band* (to avoid dataset-dependent DC dominance).
    """
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    n = min(len(y_true), len(y_pred))
    y_true = y_true[:n]
    y_pred = y_pred[:n]

    spec_t = np.fft.rfft(y_true)
    spec_p = np.fft.rfft(y_pred)
    freqs = np.fft.rfftfreq(n, d=1.0 / float(fs))

    fmin2, fmax2 = _parse_band(band_hz=band_hz, fmin=fmin, fmax=fmax)
    mag_th = _parse_mag_th(mag_ratio=mag_ratio, rel_mag_th=rel_mag_th, default=0.05)

    band_mask = (freqs >= fmin2) & (freqs <= fmax2)
    mag = np.abs(spec_t)
    mag_band = mag[band_mask] if np.any(band_mask) else mag
    peak = float(np.max(mag_band) + eps)
    mag_mask = mag >= (mag_th * peak)

    mask = band_mask & mag_mask
    if mask.shape[0] > 0:
        mask[0] = False
    if not np.any(mask):
        return freqs[band_mask], np.full(np.sum(band_mask), np.nan)

    ph_t = np.unwrap(np.angle(spec_t[mask]))
    ph_p = np.unwrap(np.angle(spec_p[mask]))
    dph = np.abs(ph_p - ph_t)
    return freqs[mask], dph

def gd_mae_fft(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    fs: float,
    *,
    band_hz=None,
    fmin=None,
    fmax=None,
    mag_ratio=None,
    rel_mag_th=None,
    eps: float = 1e-12,
):
    """
    Group-delay MAE (seconds) between y_true and y_pred.

    Uses a consecutive-bin mask: a bin contributes only when both it and its
    neighbour have significant energy, preventing noise-floor bins from
    polluting the gradient.
    """
    if fmin is None or fmax is None:
        if band_hz is not None:
            fmin, fmax = band_hz
        else:
            fmin, fmax = 0, fs / 2
    
    mag_th = rel_mag_th if rel_mag_th is not None else (mag_ratio if mag_ratio is not None else 0.05)

    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    n = min(len(y_true), len(y_pred))
    y_true = y_true[:n]
    y_pred = y_pred[:n]

    spec_t = np.fft.rfft(y_true)
    spec_p = np.fft.rfft(y_pred)
    freqs = np.fft.rfftfreq(n, d=1.0 / float(fs))
    omega = 2.0 * np.pi * freqs

    band_mask = (freqs >= fmin) & (freqs <= fmax)
    mag = np.abs(spec_t)
    mag_band = mag[band_mask] if np.any(band_mask) else mag
    peak = float(np.max(mag_band) + eps)

    mag_mask = mag >= (mag_th * peak)
    base_mask = band_mask & mag_mask
    if base_mask.shape[0] > 0:
        base_mask[0] = False  # exclude DC

    # Consecutive mask: both bin k and bin k-1 must be above threshold
    gd_mask = np.zeros_like(base_mask, dtype=bool)
    gd_mask[1:] = base_mask[1:] & base_mask[:-1]

    if not np.any(gd_mask):
        return float(“nan”)

    ph_t = np.unwrap(np.angle(spec_t))
    ph_p = np.unwrap(np.angle(spec_p))

    gd_t = np.zeros_like(ph_t)
    gd_p = np.zeros_like(ph_p)

    d_omega = np.diff(omega)
    # GD = -d(phi)/d(omega), using backward finite difference
    gd_t[1:] = -np.diff(ph_t) / (d_omega + eps)
    gd_p[1:] = -np.diff(ph_p) / (d_omega + eps)

    return float(np.mean(np.abs(gd_p[gd_mask] - gd_t[gd_mask])))


phase_mae = phase_mae_fft
gd_mae = gd_mae_fft

def group_delay_mae_fft(*args, **kwargs):
    return gd_mae_fft(*args, **kwargs)
