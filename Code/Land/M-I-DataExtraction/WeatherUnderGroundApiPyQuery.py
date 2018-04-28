import requests
from pyquery import PyQuery as pq
import csv


i = 0
idx = 0
fname = [
    "2010.csv",
    "2011.csv",
    "2012.csv",
    "2013.csv",
    "2014.csv",
    "2015.csv",
    "2016.csv",
]

BaseUrl = "https://www.wunderground.com/history/airport/VOMM"
StartYear = [2010,2011,2012,2013,2014,2015,2016]
EndYear = [2010,2011,2012,2013,2014,2015,2016]
StartDayMonth = "/1/1/"
EndDate = "CustomHistory.html?dayend=31&monthend=12"
YearEnd = "&yearend="
UrlEnd = "&req_city=&req_state=&req_statename=&reqdb.zip=&reqdb.magic=&reqdb.wmo="



for value in StartYear:
    url = BaseUrl+str(StartYear[idx])+StartDayMonth+EndDate+YearEnd+str(EndYear[idx])+UrlEnd

    response = requests.get(url)
    fh = open(fname[idx],"w")
    doc = pq(response.content)
    tablehead = [th.text() for th in doc('#observations_details td:not(:nth-child(21))').items()]

    for item in tablehead:
        i = i+1
        fh.write("%s," % item)
        if(i%20 == 0):
            fh.write('\n')

    idx = idx + 1
    print("Done bro")
    fh.close()
