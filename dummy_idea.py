import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import obspy
from obspy import read
from scipy.signal import butter, filtfilt, spectrogram

# Define the directory containing your data
data_dir = r'C:\Users\ahmed\Downloads\continous_waveform\elyh0\2018\334'

# Function to apply a band-pass filter to the data
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

# Function to preprocess the mseed data
def preprocess_mseed(file_path, lowcut=0.1, highcut=10.0, fs=100.0):
    # Read the mseed file
    st = read(file_path)
    tr = st[0]
    
    # Apply band-pass filter
    filtered_data = bandpass_filter(tr.data, lowcut, highcut, fs)
    
    return tr.times(), filtered_data

# Function to visualize the waveform and its spectrogram
def visualize_waveform(times, data, title, fs=100.0):
    plt.figure(figsize=(12, 8))
    
    # Plot waveform
    plt.subplot(2, 1, 1)
    plt.plot(times, data)
    plt.title(f'Waveform: {title}')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    
    # Adjust nperseg based on the length of the data
    nperseg = min(256, len(data))
    
    # Plot spectrogram
    plt.subplot(2, 1, 2)
    f, t, Sxx = spectrogram(data, fs=fs, nperseg=nperseg, noverlap=nperseg//2)
    
    if Sxx.size == 0:
        print(f"Warning: Spectrogram for {title} could not be generated due to insufficient data.")
    else:
        plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
        plt.title('Spectrogram')
        plt.xlabel('Time [s]')
        plt.ylabel('Frequency [Hz]')
        plt.colorbar(label='dB')

    plt.tight_layout()
    plt.show()

# Function to process CSV files
def process_csv(file_path):
    df = pd.read_csv(file_path)
    print(df.head())
    return df

# Main function to process all files in the directory
def process_directory(data_dir):
    for file_name in os.listdir(data_dir):
        if file_name.endswith('.mseed'):
            file_path = os.path.join(data_dir, file_name)
            print(f'Processing {file_path}...')
            times, filtered_data = preprocess_mseed(file_path)
            visualize_waveform(times, filtered_data, title=file_name)
        elif file_name.endswith('.csv'):
            file_path = os.path.join(data_dir, file_name)
            print(f'Processing {file_path}...')
            process_csv(file_path)

if __name__ == "__main__":
    process_directory(data_dir)