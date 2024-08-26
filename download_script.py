import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# Base URL of the directory
base_url = 'https://pds-geosciences.wustl.edu/insight/urn-nasa-pds-insight_seis/data/xb/continuous_waveform/elyh0/2018/353/'

# Directory to save the downloaded files
download_dir = r'C:\Users\ahmed\Downloads\continous_waveform\elyh0\2018\353'
if not os.path.exists(download_dir):
    os.makedirs(download_dir)

# Get the HTML content of the page
response = requests.get(base_url)
if response.status_code != 200:
    print(f"Failed to retrieve the directory page. Status code: {response.status_code}")
    exit()

# Parse the HTML content
soup = BeautifulSoup(response.content, 'html.parser')

# Find all links to .mseed files
links = soup.find_all('a')
mseed_links = [urljoin(base_url, link.get('href')) for link in links if link.get('href').endswith('.mseed')]

# Download each .mseed file
for mseed_link in mseed_links:
    filename = os.path.join(download_dir, os.path.basename(mseed_link))
    print(f"Downloading {mseed_link}...")
    with requests.get(mseed_link, stream=True) as r:
        if r.status_code == 200:
            with open(filename, 'wb') as file:
                for chunk in r.iter_content(chunk_size=8192):
                    file.write(chunk)
            print(f"Downloaded {filename}")
        else:
            print(f"Failed to download {mseed_link}. Status code: {r.status_code}")

print("All downloads completed.")
