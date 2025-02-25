import numpy as np
from scipy.stats import skew, iqr, kurtosis
from scipy.signal import find_peaks

def apply_fft_to_windows(windows, sampling_rate=100):
    """
    Applies FFT transformation to windows.
    """
    num_windows, window_size, num_features = windows.shape
    fft_transformed = np.fft.rfft(windows, axis=1)  
    fft_magnitude = np.abs(fft_transformed)  
    freq_bins = np.fft.rfftfreq(window_size, d=1/sampling_rate)  

    return fft_magnitude, freq_bins  


def extract_time_domain_features(windows):
    """
    Extracts time-domain features from windowed sensor data.
    """
    feature_list = []

    for window in windows:
        features = []

        for i in range(window.shape[1]):  
            sensor_signal = window[:, i]

            # === Basic Statistical Features ===
            features.append(np.mean(sensor_signal))  # Mean
            features.append(np.std(sensor_signal))   # Standard Deviation
            features.append(skew(sensor_signal))  # Skewness
            features.append(kurtosis(sensor_signal))  # Kurtosis (tail heaviness)
            features.append(iqr(sensor_signal))  # Interquartile Range (IQR)

            # === Signal Power & Intensity ===
            features.append(np.sum(sensor_signal**2))  # Signal Energy
            features.append(np.sqrt(np.mean(sensor_signal**2)))  # RMS (Root Mean Square)
            features.append(np.var(sensor_signal))  # Variance

            # === Peak-Based Features ===
            peaks, _ = find_peaks(sensor_signal)
            features.append(len(peaks))  # Peak Count
            features.append(np.max(sensor_signal) - np.min(sensor_signal))  # Peak-to-Peak Amplitude

            # === Signal Complexity & Activity ===
            zero_crossings = np.where(np.diff(np.sign(sensor_signal)))[0]
            features.append(len(zero_crossings))  # Zero-Crossing Rate
            features.append(np.mean(np.abs(np.diff(sensor_signal))))  # Mean Absolute Change
            features.append(np.std(np.diff(sensor_signal)))  # Standard Deviation of Change
            
        feature_list.append(features)

    return np.array(feature_list)


def extract_fft_features(fft_windows, freq_bins):
    """
    Extracts frequency-domain features from FFT-transformed sensor data.
    """
    feature_list = []

    for fft_win in fft_windows:
        features = []

        for i in range(fft_win.shape[1]):  
            sensor_fft = np.abs(fft_win[:, i])
            power = sensor_fft**2  # Compute power spectrum

            # === Key Features ===
            normal_power = np.sum(power[(freq_bins >= 3) & (freq_bins < 4)])  # Normal Gait Power (3-4 Hz)
            fog_power = np.sum(power[(freq_bins >= 5) & (freq_bins <= 8)])  # Freezing of Gait Power (5-8 Hz)

            # Spectral Entropy (captures tremor irregularity)
            prob = power / np.sum(power)
            spectral_entropy = -np.sum(prob * np.log2(prob + 1e-10))

            # Dominant Frequency (main movement frequency)
            dominant_freq = freq_bins[np.argmax(power)]

            # Spectral Centroid (where the energy is concentrated)
            spectral_centroid = np.sum(freq_bins * power) / np.sum(power)

            
            features.extend([normal_power, fog_power, spectral_entropy, dominant_freq, spectral_centroid])

        feature_list.append(features)

    return np.array(feature_list)

def extract_metadata_per_window(sub_ids, sensor_df):
    """
    Extracts metadata (Age, Gender, UPDRS3) for each window.
    """
    unique_meta = sensor_df[['SubID', 'Age', 'Gender', 'UPDRS3']].drop_duplicates().set_index('SubID')

    meta_features = [unique_meta.loc[sub_id].values for sub_id in sub_ids] 
    return np.array(meta_features)


