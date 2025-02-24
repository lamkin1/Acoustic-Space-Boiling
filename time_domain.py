import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal
from pathlib import Path
import scipy.signal as signal
import scipy.stats as stats
import pywt
import hurst


def plot_file(file):
    # 1. Load the CSV file
    # Replace 'file.csv' with the path to your file
    data = pd.read_csv(file, index_col="Time")

    # Takes every 10th row
    #data_grouped = data.groupby(data.index // 50).max()

    threshold = 0.1

    # Extract 'Time' and acceleration columns from the grouped data
    time = data.index
    data0 = data['0']
    acceleration0 = data['0']
    # acceleration1 = data['1']

    peaks, _ = scipy.signal.find_peaks(acceleration0, distance = 200, height=.1)
    peaks = peaks / 10000
    print("Peaks: ", peaks)
    print("Peak values: ", data.loc[peaks])

    # 2. Plot the time domain signals
    plt.figure(figsize=(10, 6))
    plt.plot(time, acceleration0, label='Accelerator 0', color='blue', alpha=0.7)
    #plt.plot(time, acceleration1, label='Accelerator 1', color='orange', alpha=0.7)
    plt.plot(peaks, data0.loc[peaks], "x", color="orange")
    plt.title('Time Domain Signals')
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration')
    plt.legend()
    plt.grid(True)
    plt.show()

    return

def add_feature(df, feature, values):
    df[feature] = values
    return df


def extract_peaks(file):
    data = pd.read_csv(file, index_col="Time")

    acceleration0 = data['0'] #Using only accelerometer 0

    peaks, _ = scipy.signal.find_peaks(acceleration0, distance = 200, height=.1) #This height needs to be configured
    peaks = peaks / 10000 #time of peaks
    match (len(peaks)):
        case 0:
            return None, None, None
        case 1 | 2:
            # NEEDS TO BE UPDATED TO INCLUDE NEW FEATURES
            magnitude = np.array(acceleration0.loc[peaks])
            return peaks, magnitude, None

    magnitude = np.array(acceleration0.loc[peaks]) #magnitude of peaks

    #Get std, mean of distance between peaks
    time_differences = [(peaks[i] - peaks[i - 1]) for i in range(1, len(peaks))] 
    std_dev = np.std(time_differences)
    mean_time_diff = np.mean(time_differences)
    median_time_diff = np.median(time_differences)

    max_peak = max(magnitude)
    median_peak = np.median(magnitude)
    std_peak = np.std(magnitude)
    
    avg_peaks_per_second = np.mean(peaks)
    sum_peak_magnitude = np.sum(magnitude)

    threshold = 0.1
    percent_time_above_threshold = np.mean([amplitude > threshold for amplitude in acceleration0])

    post_peaks = [peak + 100 for peak in peaks]
    post_peak_avgs = [np.mean(acceleration0.loc[post_peak:post_peak+100]) for post_peak in post_peaks]
    mean_post_peak_magnitude = np.mean(post_peak_avgs)
    std_dev_post_peak_magnitude = np.std(post_peak_avgs)

    results = {
        "file_name": file,
        "std_dev": std_dev,
        "mean_time_diff": mean_time_diff,
        "median_time_diff": median_time_diff,
        "max_peak": max_peak,
        "median_peak": median_peak,
        "std_peak": std_peak,
        "avg_peaks_per_second": avg_peaks_per_second,
        "sum_peak_magnitude": sum_peak_magnitude,
        "percent_time_above_threshold": percent_time_above_threshold,
        "mean_post_peak_magnitude": mean_post_peak_magnitude,
        "std_dev_post_peak_magnitude": std_dev_post_peak_magnitude,
    }

    return results


def get_signal_energy(data):
    '''
    :param data: acceleration data
    :return: average signal energy per second
    '''
    time, signal = unpack_data(data)
    total = np.sum(np.square(signal))
    return total / time[-1]


def get_zero_crossing_rate(data):
    """
    :param data: pandas Series containing acceleration data
    :return: Zero crossing rate (average absolute slope at zero crossings)
    """
    # Identify zero crossings and compute differences
    _, signal = unpack_data(data)
    cross_slopes = [
        np.abs(signal.iloc[i + 1] - signal.iloc[i])
        for i in range(len(signal) - 1)
        if signal.iloc[i] * signal.iloc[i + 1] <= 0
    ]

    # Return the average of the differences (zero crossing rate)
    return np.mean(cross_slopes) if cross_slopes else 0


def spectral_entropy(signal_data, fs=1000):
    """
    Compute spectral entropy from the power spectral density (PSD).
    """
    freqs, psd = signal.welch(signal_data, fs=fs, nperseg=256)
    psd_norm = psd / np.sum(psd)  # Normalize power spectrum
    entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))  # Compute entropy
    return entropy


def compute_hurst(signal_data):
    """
    Computes the Hurst Exponent to measure long-term correlations.
    """
    return hurst.compute_Hc(signal_data, kind='price', simplified=True)[0]


def compute_higuchi_fd(signal_data, k_max=10):
    """
    Computes Higuchi's Fractal Dimension (HFD).
    """
    N = len(signal_data)
    Lmk = np.zeros((k_max, k_max))

    for k in range(1, k_max + 1):
        for m in range(k):
            Lm = 0
            count = 0
            for i in range(1, (N - m) // k):
                Lm += abs(signal_data[m + i * k] - signal_data[m + (i - 1) * k])
                count += 1
            Lmk[k - 1, m] = (Lm / count) * ((N - 1) / (k * count))

    Lk = np.mean(Lmk, axis=1)
    log_Lk = np.log(Lk)
    log_k = np.log(np.arange(1, k_max + 1))

    return -np.polyfit(log_k, log_Lk, 1)[0]


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


def unpack_data(data):
    '''
    Unpacks dataframe into time and signal
    '''
    time = data.index
    signal = data.iloc[:, 0]
    return time, signal


def main():
    directory_name = "Data/After_May/"
    directory = Path(directory_name)

    # Get all file names in the directory
    file_names = [directory_name+f.name for f in directory.iterdir() if f.is_file()]


    extract_all_files = [extract_peaks(f) for f in file_names]
    [plot_file(f) for f in file_names] #Plot each file individually, hit q to progress through plots

    zipped_list = [(row[2],row[3]) for row in extract_all_files if row[2] is not None] #get standard deviations
    stds, mean_time_diffs = zip(*zipped_list)

    # valid_std_row = [row for row in extract_all_files if row[2] is not None]
    # sorted_stds = sorted(valid_std_row, key=lambda x: x[2])
    # for i in range(5):
    #     plot_file(sorted_stds[i][4])


    plt.hist(stds, bins=30)
    plt.xlabel('Standard Deviations')
    plt.ylabel('Frequency')
    plt.show()

    plt.hist(mean_time_diffs, bins=30)
    plt.xlabel('Mean Time Differences')
    plt.ylabel('Frequency')
    plt.show()

    mean_time_diffs_lt1 = [mtd for mtd in mean_time_diffs if mtd < 1] #Getting mean time diffs less than 1 to plot
    plt.hist(mean_time_diffs_lt1, bins=30)
    plt.xlabel('Mean Time Differences < 1 sec')
    plt.ylabel('Frequency')
    plt.show()

    mean_std = np.mean(stds)
    print("Mean Std:", mean_std)

if __name__ == "__main__":
    main()