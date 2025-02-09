# Accelerometer Data Processing and Feature Extraction Script

## Features of the Script:
- High-pass filtering of accelerometer data to remove low-frequency noise. (Would definitely be worthwile to investigate different cutoff frequenices for future analysis!)
- Fast Fourier Transform (Computationally optimized version of FDT) to convert time-domain data into the frequency domain. Reasoning: No boiling regime transitions occur so Wavelet or alternative transformations not needed.
- Power Spectral Density calculation to analyze the energy distribution across frequencies. Reasoning: rhythmic would be distinct freq bands, irregular power distrubtion would be far more spread out
- Extraction of key features such as peak frequencies, power statistics, and spectral entropy.
- Visualization of the FFT and PSD through plots.
- Saving extracted features into a CSV file for Clustering.

## Script Breakdown

### 1. **`butter_highpass_filter` Function**
   - **Purpose**: Applies a high-pass Butterworth filter to remove low-frequency noise from accelerometer data. (More investigation into this functionality is required.... This may cause data loss so it's very finicky)
   - **Parameters**:
     - `data`: The accelerometer data to filter.
     - `cutoff_freq`: The cutoff frequency for the high-pass filter.
     - `sample_rate`: The sample rate of the data.
     - `order`: The order of the Butterworth filter.
   - **How it works**: Designs a high-pass Butterworth filter and applies it to the data using `filtfilt` to ensure zero-phase distortion.

### 2. **`clear_directory` Function**
   - **Purpose**: Clears all files from the specified directory or creates the directory if it doesn’t exist.
   - **How it works**: Checks if the directory exists. If it does, it removes all files inside. If not, it creates the directory.

### 3. **`spectral_entropy` Function**
   - **Purpose**: Computes the Spectral Entropy of the signal from the Power Spectral Density (PSD).
   - **How it works**: Normalizes the PSD to a probability distribution and computes the entropy, which measures the randomness or unpredictability of the signal. Lower entropy indicates rhythmic patterns, while higher entropy suggests irregular signals.

### 4. **`process_file` Function**
   - **Purpose**: Processes each CSV file containing accelerometer data.
   - **Steps**:
     1. **Read CSV file**: Extracts the time and accelerometer data.
     2. **Compute Sampling Rate**: Calculates the sampling rate from the time differences.
     3. **Apply High-pass Filter**: Filters the accelerometer data to remove low-frequency noise.
     4. **FFT Calculation**: Converts the accelerometer data from the time domain to the frequency domain using FFT.
     5. **Power Spectrum**: Calculates the power spectrum from the FFT.
     6. **Power Spectral Density (PSD)**: Estimates the power distribution across frequencies using the `welch` function.
     7. **Spectral Entropy**: Computes the spectral entropy from the PSD.
     8. **Peak Extraction**: Identifies significant peaks in the power spectrum using the `find_peaks` function.
     9. **Feature Extraction**: Extracts the following features:
        - **Top Peak Frequency (Hz)**: Frequency with the highest power.
        - **Number of Peaks**: Count of significant peaks.
        - **Mean Power**: Average power across all frequencies.
        - **Std Power**: Standard deviation of power values.
        - **Frequency Centroid**: Weighted average frequency of the power spectrum.
        - **Spectral Entropy**: Measure of signal randomness.
     10. **Save Features**: Saves the extracted features to a CSV file.
     11. **FFT Spectrum Plot**: Plots the log-transformed FFT spectrum with peaks highlighted and saves it as an image.
     12. **PSD Plot**: Plots the PSD and saves it as an image.
  
### 5. **`main` Function**
   - **Purpose**: The main function manages the workflow of the script.
   - **Steps**:
     1. **Argument Validation**: Ensures the correct number of command-line arguments are provided.
     2. **Clear Output Directories**: Clears any existing files from the output directories (features and plots).
     3. **Process Files**: Loops through all CSV files in the input directory and processes them using the `process_file` function.

## Output Files:
1. **Feature CSV**: For each input file, the script generates a CSV file containing the extracted features (e.g., peak frequency, number of peaks, spectral entropy).
2. **FFT Plot**: A plot of the log-scaled FFT spectrum for each input file.
3. **PSD Plot**: A plot of the Power Spectral Density for each input file.

## Feature Overview:
- **Top Peak Frequency (Hz)**: The frequency corresponding to the peak with the highest power.
- **Number of Peaks**: The number of significant peaks identified in the power spectrum.
- **Mean Power**: The average power across the frequency range.
- **Standard Deviation of Power**: The variability of the power in the frequency domain.
- **Frequency Centroid**: The "center" of the frequency spectrum, weighted by the power values.
- **Spectral Entropy**: A measure of the signal's complexity or regularity in the frequency domain.

## Usage:
```bash
python fourier_transform.py <input_folder> <output_feature_folder> <output_plot_folder> <column_index>
```