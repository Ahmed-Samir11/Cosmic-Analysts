import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# Base URL of the directory
base_url = 'https://pds-geosciences.wustl.edu/insight/urn-nasa-pds-insight_seis/data/xb/continuous_waveform/'

# Stations to loop through
stations = ['elyh0', 'elyhk', 'elys0', 'elyse']

# Directory to save the downloaded files
download_dir = r'C:\Users\ahmed\OneDrive\Desktop\Cosmic Analysts\continuous_waveform'
if not os.path.exists(download_dir):
    os.makedirs(download_dir)

def download_mseed_files(base_url, station):
    # URL for the current station
    station_url = urljoin(base_url, station + '/')
    
    # Get the HTML content of the page for the station
    response = requests.get(station_url)
    if response.status_code != 200:
        print(f"Failed to retrieve the directory page for {station}. Status code: {response.status_code}")
        return
    
    # Parse the HTML content
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Find all links to directories (years)
    links = soup.find_all('a')
    year_links = [urljoin(station_url, link.get('href')) for link in links if link.get('href').endswith('/')]

    for year_link in year_links:
        # Get the HTML content of the year page
        response = requests.get(year_link)
        if response.status_code != 200:
            print(f"Failed to retrieve the directory page for year {year_link}. Status code: {response.status_code}")
            continue
        
        # Parse the HTML content
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find all links to directories (days)
        day_links = [urljoin(year_link, link.get('href')) for link in soup.find_all('a') if link.get('href').endswith('/')]

        for day_link in day_links:
            # Get the HTML content of the month page
            response = requests.get(day_link)
            if response.status_code != 200:
                print(f"Failed to retrieve the directory page for month {day_link}. Status code: {response.status_code}")
                continue
            
            # Parse the HTML content
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find all links to .mseed files
            mseed_links = [urljoin(day_link, link.get('href')) for link in soup.find_all('a') if link.get('href').endswith('.mseed')]
            
            for mseed_link in mseed_links:
                filename = os.path.join(download_dir, os.path.basename(mseed_link))
                
                # Check if the file already exists
                if os.path.exists(filename):
                    print(f"File {filename} already exists, skipping download.")
                    continue
                
                print(f"Downloading {mseed_link}...")
                with requests.get(mseed_link, stream=True) as r:
                    if r.status_code == 200:
                        with open(filename, 'wb') as file:
                            for chunk in r.iter_content(chunk_size=8192):
                                file.write(chunk)
                        print(f"Downloaded {filename}")
                    else:
                        print(f"Failed to download {mseed_link}. Status code: {r.status_code}")

# Loop through each station
for station in stations:
    print(f"Processing station: {station}")
    download_mseed_files(base_url, station)

print("All downloads completed.")
