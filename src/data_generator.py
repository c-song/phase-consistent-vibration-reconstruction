# data_generator.py
import numpy as np
import config

def generate_fixed_signal(t, f0, phi1, phi2):
    """Generate a multi-harmonic signal with the given parameters."""
    signal = 1.0 * np.sin(2 * np.pi * f0 * t + phi1) + \
             0.5 * np.sin(2 * np.pi * 2 * f0 * t + phi2) + \
             0.2 * np.cos(2 * np.pi * 3 * f0 * t)
    return signal

def create_pair_data():
    """
    Generate a physically aligned (low, high) sample pair via decimation.
    The low-rate signal is obtained by downsampling the high-rate ground truth,
    ensuring perfect time alignment.
    """
    f0 = np.random.uniform(50, 200)
    phi1 = np.random.uniform(0, 2*np.pi)
    phi2 = np.random.uniform(0, 2*np.pi)

    N_high = int(config.DURATION * config.FS_HIGH)
    t_high = np.linspace(0, config.DURATION, N_high, endpoint=False)
    y_high = generate_fixed_signal(t_high, f0, phi1, phi2)

    ratio = int(config.FS_HIGH // config.FS_LOW)
    y_low = y_high[::ratio]

    expected_low = int(config.DURATION * config.FS_LOW)
    y_low = y_low[:expected_low]

    # Add Gaussian white noise to the low-rate input (noise_level ~5%)
    noise_level = 0.05
    noise = np.random.normal(0, noise_level, y_low.shape)
    y_low = y_low + noise

    return y_low, y_high