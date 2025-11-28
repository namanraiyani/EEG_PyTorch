import numpy as np
import scipy.signal
import mne

def compute_psd(data, fs, nperseg=256):
    freqs, psd = scipy.signal.welch(data, fs, nperseg=nperseg)
    return freqs, psd

def compute_de(data):
    variance = np.var(data, axis=-1)
    return 0.5 * np.log(2 * np.pi * np.e * variance)

def compute_band_power(data, fs, bands):
    freqs, psd = scipy.signal.welch(data, fs, nperseg=min(256, data.shape[-1]))
    band_powers = {}
    for band_name, (low, high) in bands.items():
        idx_band = np.logical_and(freqs >= low, freqs <= high)
        band_power = scipy.integrate.simps(psd[..., idx_band], freqs[idx_band], axis=-1)
        band_powers[band_name] = band_power
    return band_powers

def extract_features(data, fs):
    bands = {
        'Delta': (0.5, 4),
        'Theta': (4, 8),
        'Alpha': (8, 13),
        'Beta': (13, 30),
        'Gamma': (30, 100)
    }
    
    features = {}
    features['DE'] = compute_de(data)
    
    band_powers = compute_band_power(data, fs, bands)
    features.update(band_powers)
    
    return features
