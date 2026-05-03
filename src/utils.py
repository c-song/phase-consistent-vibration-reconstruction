# utils.py
import numpy as np
from scipy.signal import resample
import numpy as np
import scipy.signal

def windowed_interpolation(y_low, target_len):
    """
    Windowed Sinc interpolation baseline.

    Applies a Hanning window before FFT-based resampling to suppress Gibbs
    ringing at the signal boundaries, then compensates for the amplitude
    reduction by dividing by the window's coherent gain (0.5 for Hann).

    Args:
        y_low: low-rate input signal (numpy array)
        target_len: target length after upsampling (int)
    Returns:
        y_final: reconstructed high-rate signal
    """
    window = scipy.signal.windows.hann(len(y_low))
    y_windowed = y_low * window

    # scipy.signal.resample performs FFT zero-padding (Sinc interpolation)
    y_interp = scipy.signal.resample(y_windowed, target_len)

    # Restore amplitude: divide by Hann window coherent gain (= 0.5)
    coherent_gain = np.sum(window) / len(window)
    y_final = y_interp / coherent_gain

    return y_final
class StandardScaler:
    """
    Standard (z-score) scaler: (x - mean) / std.
    Supports save/load so inference uses the same distribution as training.
    """
    def __init__(self):
        self.mean = 0.0
        self.std = 1.0
        
    def fit(self, data):
        self.mean = np.mean(data)
        self.std = np.std(data)
        
    def transform(self, data):
        return (data - self.mean) / (self.std + 1e-8)
        
    def inverse_transform(self, data):
        return data * (self.std + 1e-8) + self.mean
    
    def save(self, path):
        np.savez(path, mean=self.mean, std=self.std)
        
    def load(self, path):
        data = np.load(path)
        self.mean = data['mean']
        self.std = data['std']

def fd_interpolation(y_input, target_len):
    """Frequency-domain interpolation baseline (scipy resample + rFFT)."""
    y_interp = resample(y_input, target_len)
    spectrum = np.fft.rfft(y_interp)
    return y_interp, spectrum


def calc_snr(signal_true, signal_pred):
    """Signal-to-noise ratio in dB."""
    noise = signal_true - signal_pred
    power_signal = np.sum(signal_true**2)
    power_noise = np.sum(noise**2)
    if power_noise == 0: return 100.0
    return 10 * np.log10(power_signal / power_noise)

def complex_to_channels(c_data):
    """Convert complex array to [2, Length] real array (Re, Im channels)."""
    return np.stack([c_data.real, c_data.imag], axis=0).astype(np.float32)