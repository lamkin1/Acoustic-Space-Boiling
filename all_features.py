import numpy as np
import pandas as pd
import scipy.signal as signal
import pywt
import hurst
from pathlib import Path


# Feature Extraction Functions
def unpack_data(data):
    """
    Unpacks dataframe into time and signal
    """
    time = data.index
    signal = data.iloc[:, 0]
    return time, signal


def compute_spectral_entropy(signal_data, fs=1000):
    """
    Compute spectral entropy from the power spectral density (PSD).
    """
    freqs, psd = signal.welch(signal_data, fs=fs, nperseg=256)
    psd_norm = psd / np.sum(psd)  # Normalize power spectrum
    entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))  # Compute entropy
    return entropy


def compute_spectral_centroid(signal_data, fs=1000):
    """
    Computes the spectral centroid, indicating the center of mass of the frequency spectrum.
    """
    freqs, psd = signal.welch(signal_data, fs=fs, nperseg=256)
    centroid = np.sum(freqs * psd) / np.sum(psd)
    return centroid


def compute_spectral_flatness(signal_data, fs=1000):
    """
    Computes spectral flatness, which measures the noisiness of the signal in the frequency domain.
    """
    freqs, psd = signal.welch(signal_data, fs=fs, nperseg=256)
    geometric_mean = np.exp(np.mean(np.log(psd + 1e-12)))  # Avoid log(0)
    arithmetic_mean = np.mean(psd)
    flatness = geometric_mean / arithmetic_mean
    return flatness


def compute_spectral_bandwidth(signal_data, fs=1000):
    """
    Computes spectral bandwidth, which quantifies the spread of the frequency spectrum.
    """
    freqs, psd = signal.welch(signal_data, fs=fs, nperseg=256)
    centroid = np.sum(freqs * psd) / np.sum(psd)
    bandwidth = np.sqrt(np.sum(psd * (freqs - centroid) ** 2) / np.sum(psd))
    return bandwidth


def extract_peaks(file):
    """
    Extracts peak-related features from the signal.
    """
    data = pd.read_csv(file, index_col="Time")
    _, signal_data = unpack_data(data)

    peaks, _ = signal.find_peaks(signal_data, distance=200, height=0.1)
    peaks = peaks / 10000  # Convert to time domain

    if len(peaks) == 0:
        return None

    magnitude = np.array(signal_data.loc[peaks])

    time_differences = np.diff(peaks) if len(peaks) > 1 else [np.nan]
    std_dev_time_diff = np.std(time_differences)
    mean_time_diff = np.mean(time_differences)
    median_time_diff = np.median(time_differences)

    max_peak = np.max(magnitude) if len(magnitude) > 0 else np.nan
    median_peak = np.median(magnitude) if len(magnitude) > 0 else np.nan
    std_peak = np.std(magnitude) if len(magnitude) > 0 else np.nan
    avg_peaks_per_second = np.mean(peaks) if len(peaks) > 0 else np.nan
    sum_peak_magnitude = np.sum(magnitude) if len(magnitude) > 0 else np.nan

    threshold = 0.1
    percent_time_above_threshold = np.mean(signal_data > threshold)

    post_peaks = [peak + 100 for peak in peaks]
    post_peak_avgs = [np.mean(signal_data.loc[post_peak:post_peak + 100]) for post_peak in post_peaks if
                      post_peak + 100 in signal_data.index]
    # mean_post_peak_magnitude = np.mean(post_peak_avgs) if len(post_peak_avgs) > 0 else np.nan
    # std_dev_post_peak_magnitude = np.std(post_peak_avgs) if len(post_peak_avgs) > 0 else np.nan

    return {
        "file_name": file,
        "std_dev_time_diff": std_dev_time_diff,
        "mean_time_diff": mean_time_diff,
        "median_time_diff": median_time_diff,
        "max_peak": max_peak,
        "median_peak": median_peak,
        "std_peak": std_peak,
        "avg_peaks_per_second": avg_peaks_per_second,
        "sum_peak_magnitude": sum_peak_magnitude,
        "percent_time_above_threshold": percent_time_above_threshold,
        # "mean_post_peak_magnitude": mean_post_peak_magnitude,
        # "std_dev_post_peak_magnitude": std_dev_post_peak_magnitude
    }


def extract_all_features(file):
    """
    Extracts both peak-related and spectral features from the given file.
    """
    data = pd.read_csv(file, index_col="Time")
    _, signal_data = unpack_data(data)

    # Extract spectral and statistical features
    features = {
        "file_name": file,
        "spectral_entropy": compute_spectral_entropy(signal_data),
        "spectral_centroid": compute_spectral_centroid(signal_data),
        "spectral_flatness": compute_spectral_flatness(signal_data),
        "spectral_bandwidth": compute_spectral_bandwidth(signal_data)
    }

    # Extract peak-based features
    peak_features = extract_peaks(file)
    if peak_features:
        features.update(peak_features)

    return features


def process_directory(directory_name="Data/After_May/"):
    """
    Processes all CSV files in a directory and extracts all features from each.
    """
    directory = Path(directory_name)
    file_names = [f for f in directory.iterdir() if f.suffix == '.csv']

    extracted_features = [extract_all_features(file) for file in file_names]

    # Convert to DataFrame
    feature_df = pd.DataFrame(extracted_features)


if __name__ == "__main__":
    process_directory()