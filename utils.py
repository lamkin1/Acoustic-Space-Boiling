import pandas as pd
import numpy as np
import scipy.signal as signal
from pathlib import Path
import matplotlib.pyplot as plt


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