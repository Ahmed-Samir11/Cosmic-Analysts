# Cosmic-Analysts

## Data Preparation
Before the resources were published, we started working on the InSight Lander mission data. At first, downloading single data files one by one was inefficient, so we devised a download script to collect all data across the stations and years.

At first, we thought of engineering new features to work with, such as mean, standard deviation (std), Fast Fourier Transform (FFT) mean, FFT std, and energy, then, transform the files into pandas dataframe with these values. The issue with the mentioned approach is that it classifies a trace as an anomaly, or a seismic event, rather than identifies an interval within the trace. This approach could work in case the trace's interval was small, around an hour for example. However, most traces were a full-day record, which means every trace needs careful exploration.

Afterward, we realized another mistake. That is, the location code corresponded to different measurements. The records contained velocity, position, temperature, pressure, and even wind, with the difference being which data was a raw transmission from the lander and which was processed data on Earth. Learning from that mistake, we carefully examined the documents provided with the mission and started reprocessing our ideas.

The location code is made up of 3 letters: Band code, Instrument code, and Orientation code. Orientation codes are mostly our educated guesses, rather than direct translation from the documents like the Band codes and Instrument Codes.
```python
band_code = { 'SP' : { # Short Period
    'E': 100,
    'S': [10, 80], # List elements are range, not discrete values
    'M': [2, 5],
    'L': 1,
    'V': [0.1 , 0.5],
    'U': [0.01, 0.05],
    'R': [1/3600, 0.001]
}, 
'VBB': { # Very Broadband
    'H': 100,
    'B': [10, 80],
    'M': [2, 5],
    'L': 1,
    'V': [0.1 , 0.5],
    'U': [0.01, 0.05],
    'R': [1/3600, 0.001]
}
}
instrument_code = {'H': 'High Gain Seismometer',
                   'L': 'Low Gain Seismometer',
                   'M': 'Mass Position Seismometer',
                   'D': 'Pressure',
                   'F': 'Magnetometer',
                   'k': 'Temperature',
                   'W': 'Wind',
                   'Z': 'Sythetized Beam Data',
                   'Y': 'Non-specific Instruments',
                   'E': 'Electric Test Point'
                   }
orientation_code = {'U': 'Up',
                    'V': 'Vertical (non-orthogonal VBB axis)',
                    'W': 'West (non-orthogonal VBB axis)',
                    # U, V, W are chosen because VBB axes are non-orthogonal.
                    'I': 'Instrument reference frame',
                    'S': 'Surface horizontal',
                    'D': 'Down',
                    'O': 'Operational mode or orientation unknown'
}
```

After understanding the data, we realized that the data compiled by the team hosted by the Planetary Data System Geosciences Node at Washington University in St. Louis had only a single code, bhv, which is only 1 range of frequencies among others. Therefore, we decided that our solution should work on every trace, regardless of its frequency and orientation. Moreover, since the structures of planents aren't alike, saving a pretrained model would have the opposite effect on predictions. The developed algorithm should treat every trace as a standalone and find the interval of the seismic event after training on the data points of the trace.


## Requirements
matplotlib==3.9.2
numpy==2.1.1
obspy==1.4.1
scikit_learn==1.5.2
Python==3.12.7
## References
1-	https://pds-geosciences.wustl.edu/insight/urn-nasa-pds-insight_seis/readme.txt

2-	https://pds-geosciences.wustl.edu/insight/urn-nasa-pds-insight_documents/document_seis/

3-	https://www.researchgate.net/publication/224384174_Isolation_Forest

4-	https://www.science.vt.edu/research/around-the-college/marsquakes.html#:~:text=%E2%80%9CThis%20is%20an%20important%20result,the%20data%20collected%20by%20InSight

5-	https://gfzpublic.gfz-potsdam.de/rest/items/item_4097/component/file_4098/content#:~:text=Today%2C%20the%20'short%2Dtime,two%20consecutive%20moving%2Dtime%20windows

