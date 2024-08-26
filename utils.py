import os
from obspy import read
import numpy as np
import pandas as pd
from scipy.fft import fft
from scipy.stats import skew, kurtosis

# Function to extract features from a given trace
def extract_features(trace):
    data = trace.data
    
    # Statistical features
    mean_val = np.mean(data)
    std_val = np.std(data)
    skewness = skew(data)
    kurt = kurtosis(data)
    
    # Frequency domain features
    fft_vals = fft(data)
    fft_magnitude = np.abs(fft_vals)
    fft_mean = np.mean(fft_magnitude)
    fft_std = np.std(fft_magnitude)
    
    # Signal energy
    energy = np.sum(data ** 2)
    
    features = {
        'mean': mean_val,
        'std': std_val,
        'skewness': skewness,
        'kurtosis': kurt,
        'fft_mean': fft_mean,
        'fft_std': fft_std,
        'energy': energy
    }
    
    return features