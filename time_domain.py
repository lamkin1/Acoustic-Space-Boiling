import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal
from pathlib import Path
import math
from collections import Counter

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

    

    results = {
        "file_name": file,
        "std_dev": std_dev,
        "mean_time_diff": mean_time_diff,
        "median_time_diff": median_time_diff,
        "max_peak": max_peak,
        "median_peak": median_peak,
        "std_peak": std_peak,
        "avg_peaks_per_second": avg_peaks_per_second,
        "sum_peak_magnitude": sum_peak_magnitude
    }

    return results

def main():
    directory_name = "Data/After_May/"
    directory = Path(directory_name)

    df = pd.DataFrame()

    # Get all file names in the directory
    file_names = [directory_name+f.name for f in directory.iterdir() if f.is_file()]

    extract_all_files = [peaks(f) for f in file_names]
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