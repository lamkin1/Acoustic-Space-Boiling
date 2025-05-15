import pandas as pd
import numpy as np
import scipy.signal as signal
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import binom


def get_file(file):
    data = pd.read_csv(file, index_col="Time")
    acceleration0 = data['0']
    return acceleration0


def unpack_data(data):
    """
    Unpacks dataframe into time and signal
    """
    time = data.index
    signal = data.iloc[:, 0]
    return time, signal


def clamp(value, lower=0.015, upper=0.1):
    return max(lower, min(value, upper))

def get_diffs(x, margin=0.003, p_null=0.1, alpha=0.05):
    x = np.sort(x)
    range_x = x[-1] - x[0]
    cutoff = range_x / 2

    diffs = []
    for i in range(len(x) - 1):
        for j in range(i + 1, len(x)):
            d = round(float(x[j] - x[i]), 4)  # round to get rid of floating point error
            if d > cutoff:
                break

            anchor = x[i]
            hits = 0
            tries = 0
            t = anchor
            while t <= x[-1]:
                if np.any(np.abs(x - t) <= margin):
                    hits += 1
                tries += 1
                t += d

            p_value = 1 - binom.cdf(hits - 1, tries, p_null) if hits < tries else 0.0
            if p_value < alpha:
                diffs.append(d)

    return sorted(diffs)


def get_peaks(acceleration0):
    """
    acceleration0: accelerometer data, measured to the 10,000th of a second
    Returns: x and y coordinates of peaks
    """
    percentile = np.percentile(acceleration0, 99.5)
    percent_of_max = 0.1*np.max(acceleration0)

    height = clamp(max(percentile, percent_of_max))
    distance = 350 + 5 / height

    x, y = signal.find_peaks(acceleration0, distance=distance, height=height)
    x = x / 10000
    y = y["peak_heights"]  # unpack data
    return x, y


def plot_file(file):
    data = pd.read_csv(file, index_col="Time")
    time = data.index
    acceleration0 = data['0']

    x, y = get_peaks(acceleration0)

    plt.figure(figsize=(10, 6))
    plt.plot(time, acceleration0, label='Accelerator 0', color='blue', alpha=0.7)
    #plt.plot(time, acceleration1, label='Accelerator 1', color='orange', alpha=0.7)
    plt.plot(x, y, "x", color="orange")
    plt.title(f'Time Domain Signals for {file}')
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration')
    plt.legend()
    plt.grid(True)
    plt.show()
    return