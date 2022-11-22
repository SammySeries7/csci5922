# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 13:37:24 2022

@author: samsu
"""
# Import libraries
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
import os
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline
# NLTK VADER for sentiment analysis
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from datetime import datetime as dt

finwiz_url = 'https://finviz.com/quote.ashx?t='

news_tables = {}
tickers = ['TSLA']

for ticker in tickers:
    url = finwiz_url + ticker
    req = Request(url=url,headers={'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:20.0) Gecko/20100101 Firefox/20.0'}) 
    response = urlopen(req)    
    # Read the contents of the file into 'html'
    html = BeautifulSoup(response)
    # Find 'news-table' in the Soup and load it into 'news_table'
    news_table = html.find(id='news-table')
    # Add the table to our dictionary
    news_tables[ticker] = news_table
    
# Read one single day of headlines for 'AMZN' 
amzn = news_tables['TSLA']
# Get all the table rows tagged in HTML with <tr> into 'amzn_tr'
amzn_tr = amzn.findAll('tr')

parsed_news = []

# Iterate through the news
for file_name, news_table in news_tables.items():
    # Iterate through all tr tags in 'news_table'
    for x in news_table.findAll('tr'):
        # read the text from each tr tag into text
        # get text from a only
        try:
            text = x.a.get_text()
            
        except:
            print("An exception occurred")
        # splite text in the td tag into a list 
        date_scrape = x.td.text.split()
        # if the length of 'date_scrape' is 1, load 'time' as the only element

        if len(date_scrape) == 1:
            time = date_scrape[0]
            
        # else load 'date' as the 1st element and 'time' as the second    
        else:
            date = date_scrape[0]
            time = date_scrape[1]
        # Extract the ticker from the file name, get the string up to the 1st '_'  
        ticker = file_name.split('_')[0]
        
        # Append ticker, date, time and headline as a list to the 'parsed_news' list
        parsed_news.append([ticker, date, time, text])
        
newstitle=[]
dates=[]
for i in range(len(parsed_news)):
    temp=parsed_news[i][3]
    date_temp=parsed_news[i][1]
    dates.append(date_temp)
    newstitle.append(temp)
    
df=pd.DataFrame({'date':dates,'newstitle':newstitle})
df['date']=pd.to_datetime(df['date'],format='%b-%d-%y')
today=dt.today()
d1 = today.strftime("%d-%m-%y")

name='flinviz'+d1+'.csv'
df.to_csv(name,index=False)