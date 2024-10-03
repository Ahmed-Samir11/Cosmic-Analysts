'''
Author: Ahmed Samir

Purpose: A script that will automatically select the last modified, i.e. recorded, .mseed file and run the
seismic events detection on it.

Date: 3/10/2024
'''

import psutil
import time
import numpy as np
from obspy import read
from sklearn.ensemble import IsolationForest
import os

def get_last_modified_mseed_file():
    """Get the last modified .mseed file in the directory"""
    files = [f for f in os.listdir('.') if os.path.isfile(f) and f.endswith('.mseed')]
    if not files:
        print("No .mseed files found in the directory.")
        return None
    return max(files, key=os.path.getmtime)

start_time = time.time()
process = psutil.Process()
mem_before = process.memory_info().rss / (1024 * 1024)

file_path = get_last_modified_mseed_file()
if file_path is None:
    exit()

print(f"Processing file: {file_path}")

st = read(file_path)
tr = st[0]
tr_data = tr.data
tr_times = np.linspace(0, len(tr_data) / tr.stats.sampling_rate, num=len(tr_data))
tr_data_norm = (tr_data - np.mean(tr_data)) / np.std(tr_data)
data_points = tr_data_norm.reshape(-1, 1)
iso_forest = IsolationForest(contamination=0.001, random_state=42)
iso_forest.fit(data_points)
anomalies = iso_forest.predict(data_points)
anomalies_indices = np.where(anomalies == -1)[0]
event_times = tr_times[anomalies_indices]
clustered_points = [event_times[0]]
for i in range(1, len(event_times)):
    if event_times[i] - clustered_points[-1] >= 600:
        clustered_points.append(event_times[i])
print("Clustered seismic events:", clustered_points)
end_time = time.time()

mem_after = process.memory_info().rss / (1024 * 1024) 

print(f"Running time: {end_time - start_time:.2f} seconds")
print(f"Memory used: {mem_after - mem_before:.2f} MB")
