# Cosmic-Analysts

## Data Preparation
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

## References
1-	https://pds-geosciences.wustl.edu/insight/urn-nasa-pds-insight_seis/readme.txt

2-	https://pds-geosciences.wustl.edu/insight/urn-nasa-pds-insight_documents/document_seis/

3-	https://www.researchgate.net/publication/224384174_Isolation_Forest

4-	https://www.science.vt.edu/research/around-the-college/marsquakes.html#:~:text=%E2%80%9CThis%20is%20an%20important%20result,the%20data%20collected%20by%20InSight

5-	https://gfzpublic.gfz-potsdam.de/rest/items/item_4097/component/file_4098/content#:~:text=Today%2C%20the%20'short%2Dtime,two%20consecutive%20moving%2Dtime%20windows

