import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq
from scipy.signal import butter, filtfilt, find_peaks

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

def process_file(input_file, fft_output_folder, cluster_output_folder, column_index, highpass_cutoff=50):
    # Read CSV file
    df = pd.read_csv(input_file)
    
    # Extract time and acceleration data
    time = df.iloc[:, 0].values  # First column (Time)
    accel = df.iloc[:, column_index].values  # Selected Accelerometer Data

    # Compute sampling interval and frequency axis
    N = len(time)
    dt = np.mean(np.diff(time))  # Sampling interval
    sample_rate = 1 / dt  # Sampling rate
    freq = fftfreq(N, d=dt)[:N//2]  # Frequency axis

    # Apply high-pass filtering to remove low-frequency noise
    filtered_accel = butter_highpass_filter(accel, highpass_cutoff, sample_rate)

    # Perform FFT on filtered signal
    fft_values = fft(filtered_accel)
    power_spectrum = np.abs(fft_values[:N//2])  # Take magnitude (real part)

    # Apply log-scaling
    log_power_spectrum = np.log10(power_spectrum + 1)  # Avoid log(0) issue

    # Peak extraction
    peaks, properties = find_peaks(power_spectrum, height=np.max(power_spectrum) * 0.2)
    
    # Extract features for clustering
    top_peak_freq = freq[peaks][np.argmax(properties['peak_heights'])] if len(peaks) > 0 else 0
    num_peaks = len(peaks)
    mean_power = np.mean(power_spectrum)
    std_power = np.std(power_spectrum)
    frequency_centroid = np.sum(freq * power_spectrum) / np.sum(power_spectrum) if np.sum(power_spectrum) > 0 else 0

    # Save FFT data to CSV
    output_fft_csv = os.path.join(fft_output_folder, os.path.basename(input_file).replace('.csv', '_fft.csv'))
    fft_df = pd.DataFrame({'Frequency (Hz)': freq, 'Power': power_spectrum, 'Log Power': log_power_spectrum})
    fft_df.to_csv(output_fft_csv, index=False)

    # Save extracted features for clustering
    output_features_csv = os.path.join(cluster_output_folder, os.path.basename(input_file).replace('.csv', 'extracted_features.csv'))
    feature_data = {
        "File": os.path.basename(input_file),
        "Top Peak Frequency (Hz)": top_peak_freq,
        "Number of Peaks": num_peaks,
        "Mean Power": mean_power,
        "Std Power": std_power,
        "Frequency Centroid": frequency_centroid
    }
    
    pd.DataFrame([feature_data]).to_csv(output_features_csv, index=False)

    # Plot FFT Spectrum
    plt.figure(figsize=(10, 5))
    plt.plot(freq, log_power_spectrum, color='b', label="Log-Scaled FFT")
    plt.scatter(freq[peaks], log_power_spectrum[peaks], color='r', marker='o', label="Peaks")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Log Power")
    plt.title(f"FFT of {os.path.basename(input_file)}")
    plt.legend()
    plt.grid(True)

    # Save Plot
    output_plot = os.path.join(fft_output_folder, os.path.basename(input_file).replace('.csv', '_fft.png'))
    plt.savefig(output_plot)
    plt.close()

    print(f"Processed: {input_file} -> {output_fft_csv}, {output_plot}, {output_features_csv}")

def main():
    if len(sys.argv) != 5:
        print("Usage: python fourier_transform.py <input_folder> <fft_output_folder> <cluster_output_folder> <column_index>")
        sys.exit(1)

    input_folder = sys.argv[1]
    fft_output_folder = sys.argv[2]
    cluster_output_folder = sys.argv[3]
    column_index = int(sys.argv[4])

    # Clear previous outputs
    clear_directory(fft_output_folder)
    clear_directory(cluster_output_folder)


    # Process each CSV in the input folder
    for file in os.listdir(input_folder):
        if file.endswith('.csv'):
            process_file(os.path.join(input_folder, file), fft_output_folder, cluster_output_folder, column_index)

if __name__ == "__main__":
    main()
