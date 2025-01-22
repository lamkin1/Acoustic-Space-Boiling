import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the CSV file
# Replace 'file.csv' with the path to your file
data = pd.read_csv('Data/After_May/MATLAB 12-47 PM Thu, Nov 7, 2024 Run6 .csv')

# Assuming the CSV has columns named 'Time' and 'Acceleration'
time = data['Time']
acceleration0 = data['0']
acceleration1 = data['1']

# 2. Ensure the time is uniformly sampled
# Calculate the sampling frequency (assuming uniform sampling)
dt = np.mean(np.diff(time))
fs = 1 / dt  # Sampling frequency

# 3. Perform FFT
print(acceleration0)
acceleration0_filtered = [acc for acc in acceleration0 if acc > 0.01]
n = len(acceleration0_filtered)
frequencies = np.fft.rfftfreq(n, d=dt)  # Frequencies for positive half
fft_values = np.fft.rfft(acceleration0_filtered)  # Compute FFT

# Compute magnitude of FFT (spectral power)
spectral_power = np.abs(fft_values)

n2 = len(acceleration1)
frequencies1 = np.fft.rfftfreq(n2, d=dt)
fft_values1 = np.fft.rfft(acceleration1)
spectral_power1 = np.abs(fft_values1)


# 4. Plot the spectral graph
plt.figure(figsize=(10, 6))

start_index = (frequencies >= 250).argmax()  # Find the index where frequencies >= 1
filtered_frequencies = frequencies[start_index:]
filtered_spectral_power = spectral_power[start_index:]

# Plot the filtered data
plt.plot(filtered_frequencies, filtered_spectral_power, label='Accelerator 0', color='blue', alpha=0.7)
plt.xlabel('Frequency')
plt.ylabel('Spectral Power')
plt.legend()
plt.show()

#plt.plot(frequencies, spectral_power, label='Accelerator 0', color='blue', alpha=0.7)
#plt.plot(frequencies1, spectral_power1, label='Accelerator 1', color='orange', alpha=0.7)
#
plt.grid(True)
plt.show()
