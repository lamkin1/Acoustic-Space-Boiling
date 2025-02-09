import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq
from scipy.signal import butter, filtfilt, welch, find_peaks
from scipy.stats import entropy

def butter_highpass_filter(data, cutoff_freq, sample_rate, order=4):
    """Apply a high-pass Butterworth filter to remove low-frequency noise."""
    nyquist = 0.5 * sample_rate  # Nyquist frequency
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, data)

def clear_directory(directory):
    """Remove all files from the specified directory."""
    if os.path.exists(directory):
        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
    else:
        os.makedirs(directory, exist_ok=True)

def spectral_entropy(psd):
    """Compute Spectral Entropy from the Power Spectral Density (PSD)."""
    if np.sum(psd) == 0: return 0
    psd_prob = psd / np.sum(psd)  # Convert PSD to a probability distribution
    return entropy(psd_prob)  # Compute entropy

def process_file(input_file, output_feature_folder, output_plot_folder, column_index, highpass_cutoff=50):
    df = pd.read_csv(input_file)
    
    # Extract time and accelerometer data
    time = df.iloc[:, 0].values  # (Time)
    accel = df.iloc[:, column_index].values  # Selected Accelerometer Data

    # Compute sampling interval and frequency axis
    N = len(time)
    dt = np.median(np.diff(time))  # Sampling interval
    sample_rate = 1 / dt  # Sampling rate
    freq = fftfreq(N, d=dt)[:N//2]  # Frequency axis

    # Apply high-pass filtering to remove low-frequency noise
    filtered_accel = butter_highpass_filter(accel, highpass_cutoff, sample_rate)

    # Fast Fourier Transformation calc - Used bc comp faster than FDT and boiling regimes are constant per run 
    fft_values = fft(filtered_accel)
    power_spectrum = np.abs(fft_values[:N//2])  # Magnitude
    log_power_spectrum = np.log10(power_spectrum + 1)

    # Power Spectral Density - Analyzes energy spread, rhythmic would be distinct freq bands, irregular power distrubtion would be far more spread out
    frequencies, psd = welch(filtered_accel, sample_rate, nperseg=1024)

    # Spectral Entropy Calculation - low entropy rhythmic (less randomness) opposite for irregular 
    spec_entropy = spectral_entropy(psd)

    # Peak Extraction
    peaks, properties = find_peaks(power_spectrum, height=np.max(power_spectrum) * 0.2)
    
    # Feature Extraction
    top_peak_freq = freq[peaks][np.argmax(properties['peak_heights'])] if len(peaks) > 0 else 0
    num_peaks = len(peaks)
    mean_power = np.mean(power_spectrum)
    std_power = np.std(power_spectrum)
    frequency_centroid = np.sum(freq * power_spectrum) / np.sum(power_spectrum) if np.sum(power_spectrum) > 0 else 0

    # Save Extracted Features 
    features_csv_path = os.path.join(output_feature_folder, os.path.basename(input_file).replace('.csv', '_features.csv'))
    feature_data = {
        "File": os.path.basename(input_file),
        "Top Peak Frequency (Hz)": top_peak_freq,
        "Number of Peaks": num_peaks,
        "Mean Power": mean_power,
        "Std Power": std_power,
        "Frequency Centroid": frequency_centroid,
        "Spectral Entropy": spec_entropy
    }
    
    pd.DataFrame([feature_data]).to_csv(features_csv_path, index=False)

    # FFT Spectrum
    plt.figure(figsize=(10, 5))
    plt.plot(freq, log_power_spectrum, color='b', label="Log-Scaled FFT")
    plt.scatter(freq[peaks], log_power_spectrum[peaks], color='r', marker='o', label="Peaks")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Log Power")
    plt.title(f"FFT of {os.path.basename(input_file)}")
    plt.legend()
    plt.grid(True)

    # Save FFT
    fft_plot_path = os.path.join(output_plot_folder, os.path.basename(input_file).replace('.csv', '_fft.png'))
    plt.savefig(fft_plot_path)
    plt.close()

    # Plot PSD
    plt.figure(figsize=(10, 5))
    plt.semilogy(frequencies, psd, color='g', label="PSD")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power Spectral Density")
    plt.title(f"PSD of {os.path.basename(input_file)}")
    plt.legend()
    plt.grid(True)

    # Save PSD 
    psd_plot_path = os.path.join(output_plot_folder, os.path.basename(input_file).replace('.csv', '_psd.png'))
    plt.savefig(psd_plot_path)
    plt.close()

    print(f"Processed: {input_file} -> {features_csv_path}, {fft_plot_path}, {psd_plot_path}")

def main():
    if len(sys.argv) != 5:
        print("Usage: python fourier_transform.py <input_folder> <output_feature_folder> <output_plot_folder> <column_index>")
        sys.exit(1)

    input_folder = sys.argv[1]
    output_feature_folder = sys.argv[2]
    output_plot_folder = sys.argv[3]
    column_index = int(sys.argv[4])

    # Clear previous outputs
    clear_directory(output_feature_folder)
    clear_directory(output_plot_folder)

    # Process each CSV in the input folder
    for file in os.listdir(input_folder):
        if file.endswith('.csv'):
            process_file(os.path.join(input_folder, file), output_feature_folder, output_plot_folder, column_index)

if __name__ == "__main__":
    main()
