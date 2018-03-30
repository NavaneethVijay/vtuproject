from datetime import datetime, timedelta
import time
from collections import namedtuple
import pandas as pd
import requests
import matplotlib.pyplot as plt
import pickle
import urllib.request, json
import csv


API_KEY = "dbe60ccafe99d108"
BASE_URL = "http://api.wunderground.com/api/{}/history_{}/q/IN/chennai.json"

features = ["date", "meantempm", "meandewptm", "meanpressurem", "maxhumidity", "minhumidity", "maxtempm",
            "mintempm", "maxdewptm", "mindewptm", "maxpressurem", "minpressurem", "precipm"]
target_date = datetime(2015, 1, 1)

def extract_weather_data(url, api_key, target_date, days):
        with open('f2016.csv', 'w', newline='') as f:
            thewriter = csv.DictWriter(f, fieldnames=features)
            thewriter.writeheader()
       
            for _ in range(days):
                url = BASE_URL.format(API_KEY, target_date.strftime('%Y%m%d'))
                response = urllib.request.urlopen(url)
                data = json.loads(response.read())
                thewriter.writerow({
                'date' : target_date,
                'meantempm' : data['history']['dailysummary'][0]['meantempm'],
                'meandewptm' : data['history']['dailysummary'][0]['meandewptm'],
                'meanpressurem' : data['history']['dailysummary'][0]['meanpressurem'],
                'maxhumidity' : data['history']['dailysummary'][0]['maxhumidity'],
                'minhumidity' : data['history']['dailysummary'][0]['minhumidity'],
                'maxtempm' : data['history']['dailysummary'][0]['maxtempm'],
                'mintempm' : data['history']['dailysummary'][0]['mintempm'],
                'maxdewptm' : data['history']['dailysummary'][0]['maxdewptm'],
                'mindewptm' : data['history']['dailysummary'][0]['mindewptm'],
                'maxpressurem' : data['history']['dailysummary'][0]['maxpressurem'],
                'minpressurem' : data['history']['dailysummary'][0]['minpressurem'],
                'precipm' : data['history']['dailysummary'][0]['precipm']})
                print (data['history']['dailysummary'][0]['meantempm'])
                time.sleep(6)
                target_date += timedelta(days=1)
            return 

extract_weather_data(BASE_URL, API_KEY, target_date,365)