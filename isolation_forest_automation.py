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
    os.chdir(r'C:\Users\memoh\Desktop\Samir Here')
    print("Current working directory:", os.getcwd())
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

events = []
max_gap = 600
current_event_start = event_times[0]
current_event_end = event_times[0]

event_times = tr_times[anomalies_indices]

clustered_points = [event_times[0]]  
proximity_threshold = 600
min_points_required = 5
for i in range(1, len(event_times)):
    nearby_points = np.sum((event_times >= event_times[i] - proximity_threshold) & (event_times <= event_times[i] + proximity_threshold))
    if nearby_points >= min_points_required and event_times[i] - clustered_points[-1] >= 600:
        clustered_points.append(event_times[i])

for i in range(1, len(event_times)):
    time_diff = event_times[i] - event_times[i - 1]
    if time_diff <= max_gap:
        current_event_end = event_times[i]
    else:
        events.append((current_event_start, current_event_end))
        current_event_start = event_times[i]
        current_event_end = event_times[i]

events.append((current_event_start, current_event_end))

print("Detected seismic events:")
for idx, (start_time, end_time) in enumerate(events):
    print(f"Event {idx + 1}: Start = {start_time:.2f}s, End = {end_time:.2f}s")
end_time = time.time()

mem_after = process.memory_info().rss / (1024 * 1024) 
running_time = end_time - start_time
print(f"Running time: {running_time:.2f} seconds")
print(f"Memory used: {mem_after - mem_before:.2f} MB")
