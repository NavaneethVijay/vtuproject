#importing required lib 

from datetime import datetime, timedelta  
import time  
from collections import namedtuple  
import pandas as pd  
import requests  
import matplotlib.pyplot as plt  
import pickle

#api and baseurl for making request
API_KEY = '98912e10038ad211'  
BASE_URL = "http://api.wunderground.com/api/{0}/history_{1}/q/IN/chennai.json"

#date and time to begin
target_date = datetime(2015, 1, 1)  
features = ["date", "meantempm", "meandewptm", "meanpressurem", "maxhumidity", "minhumidity", "maxtempm",  
            "mintempm", "maxdewptm", "mindewptm", "maxpressurem", "minpressurem", "precipm"]
DailySummary = namedtuple("DailySummary", features) 

#function that extracts the data 

def extract_weather_data(url, api_key, target_date, days):  
    records = []
    for _ in range(days):
        request = BASE_URL.format(API_KEY, target_date.strftime('%Y%m%d'))
        response = requests.get(request)
        if response.status_code == 200:
            data = response.json()['history']['dailysummary'][0]
            records.append(DailySummary(
                date=target_date,
                meantempm=data['meantempm'],
                meandewptm=data['meandewptm'],
                meanpressurem=data['meanpressurem'],
                maxhumidity=data['maxhumidity'],
                minhumidity=data['minhumidity'],
                maxtempm=data['maxtempm'],
                mintempm=data['mintempm'],
                maxdewptm=data['maxdewptm'],
                mindewptm=data['mindewptm'],
                maxpressurem=data['maxpressurem'],
                minpressurem=data['minpressurem'],
                precipm=data['precipm']))
        time.sleep(6)
        target_date += timedelta(days=1)
    return records

#we will use pickle to store the tuple values into the file


records = extract_weather_data(BASE_URL, API_KEY, target_date, 500) 

with open('firstDay.pickle', 'wb') as handle:
    pickle.dump(records, handle, protocol=pickle.HIGHEST_PROTOCOL)

print(records)