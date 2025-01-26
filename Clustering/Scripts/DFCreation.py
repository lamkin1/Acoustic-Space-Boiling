import numpy as np
import pandas as pd
import scipy.signal

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