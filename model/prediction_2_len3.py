# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 11:41:35 2022

@author: samsu
"""

import pandas as pd
import numpy as np
import keras

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.layers import Bidirectional
from keras.preprocessing import sequence
from keras.layers import Dropout
import h5py
import codecs

from keras.models import model_from_json
from keras.models import load_model
from flask import Flask, url_for, request
import json
from nltk.tokenize import RegexpTokenizer
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime as dt

def live_test(trained_model, data, word_idx):

    #data = "Pass the salt"
    #data_sample_list = data.split()
    live_list = []
    live_list_np = np.zeros((30,1))
    # split the sentence into its words and remove any punctuations.
    tokenizer = RegexpTokenizer(r'\w+')
    data_sample_list = tokenizer.tokenize(data)

    labels = np.array(['1','2','3','4','5','6','7','8','9','10'], dtype = "int")
    #word_idx['I']
    # get index for the live stage
    data_index = np.array([word_idx[word.lower()] if word.lower() in word_idx else 0 for word in data_sample_list])
    data_index_np = np.array(data_index)
    #print(data_index_np)

    # padded with zeros of length 56 i.e maximum length
    padded_array = np.zeros(30) # use the def maxSeqLen(training_data) function to detemine the padding length for your data
    padded_array[:data_index_np.shape[0]] = data_index_np
    data_index_np_pad = padded_array.astype(int)
    live_list.append(data_index_np_pad)
    live_list_np = np.asarray(live_list)
    #type(live_list_np)
    #print('live_list \n:', live_list)
    
    # get score from the model
    score = trained_model.predict(live_list_np, batch_size=1, verbose=0)
    #print (score)

    single_score = np.round(np.argmax(score)/10, decimals=2) # maximum of the array i.e single band

    # weighted score of top 3 bands
    top_3_index = np.argsort(score)[0][-3:]
    top_3_scores = score[0][top_3_index]
    top_3_weights = top_3_scores/np.sum(top_3_scores)
    single_score_dot = np.round(np.dot(top_3_index, top_3_weights)/10, decimals = 2)

    #print (single_score)
    return single_score_dot
        
def load_embeddings(embedding_path):
  """Loads embedings, returns weight matrix and dict from words to indices."""
  #print('loading word embeddings from %s' % embedding_path)
  weight_vectors = []
  word_idx = {}
  with codecs.open(embedding_path, encoding='utf-8') as f:
    for line in f:
      word, vec = line.split(u' ', 1)
      word_idx[word] = len(weight_vectors)
      weight_vectors.append(np.array(vec.split(), dtype=np.float32))
  # Annoying implementation detail; '(' and ')' are replaced by '-LRB-' and
  # '-RRB-' respectively in the parse-trees.
  word_idx[u'-LRB-'] = word_idx.pop(u'(')
  word_idx[u'-RRB-'] = word_idx.pop(u')')
  # Random embedding vector for unknown words.
  weight_vectors.append(np.random.uniform(
      -0.05, 0.05, weight_vectors[0].shape).astype(np.float32))
  return np.stack(weight_vectors), word_idx


path = 'C:\\Users\\samsu\\Downloads\\deepsentiment-master\\'
data_dir = path+'/Data'
all_data_path = path+'/Data/'
pred_path = path+'/Data/output_model/test_pred.csv'
gloveFile = path+'/Data/glove/glove_6B_100d.txt'


weight_matrix, word_idx = load_embeddings(gloveFile)
weight_path = path +'model\\best_model_full.h5'
loaded_model = load_model(weight_path)
loaded_model.summary()

df_test=pd.read_csv('business_insider.csv')
score=[]
for i in range(df_test.shape[0]):
    
    #data_sample = "look like chinese company everything tsla try fly car fraction market cap xiaomi baidu xpeng byd xn l rqbx"
    data_sample=df_test.iloc[i,1]
    result = live_test(loaded_model,data_sample, word_idx)
    score.append(result)
    #print (result)

df_test['score']=score
df_test['0']=pd.to_datetime(df_test['0'],format='%m/%d/%Y')

avg_score=df_test.groupby('0').mean('score')




plt.plot(avg_score.index,avg_score.iloc[:,0])
plt.xticks(rotation = 90)

tsla = yf.Ticker("TSLA")

tsla_stock = tsla.history(
    start=df_test['0'].min().strftime('%Y-%m-%d'),
    end=df_test['0'].max().strftime('%Y-%m-%d'),
    interval='1d'
).reset_index()

tsla_stock1=tsla_stock[['Date','Close']]
tsla_stock_add=pd.DataFrame({'Date':['2022-09-17','2022-09-18','2022-10-01','2022-10-02','2022-10-08',
                                     '2022-10-09','2022-10-15','2022-10-16','2022-10-23','2022-10-29','2022-10-30',
                                     '2022-10-05','2022-10-06','2022-10-07'],'Close':[303.35,303.35,265.25,265.25,223.07,223.07,204.99,204.99,214.44,228.52,228.52,207.47,207.47,197.08]})
tsla_stock1=tsla_stock1.append(tsla_stock_add, ignore_index = True)
tsla_stock1['Date']=tsla_stock1['Date'].astype('str')
k=[] 
for i in range(len(tsla_stock1)):
    k.append(tsla_stock1.iloc[i,0].split(' ')[0])
tsla_stock1['Date']=k

tsla_stock1.Date=pd.to_datetime(tsla_stock1.Date,format='%Y-%m-%d')
tsla_stock1=tsla_stock1.sort_values(by='Date')
tsla_stock1['avgind']=list(avg_score['score'])

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax2 = ax1.twinx()
ax2.plot(tsla_stock1.Date,tsla_stock1.avgind,color='g')
ax1.plot(tsla_stock1.Date,tsla_stock1.Close,color='b')
ax1.set_xticks(tsla_stock1.Date)
ax1.set_xticklabels(tsla_stock1.Date,rotation = 90)
plt.title("Sentimental vs Real Stock price (tsla)")
plt.show()

tsla_stock2=tsla_stock1.iloc[19:,:]
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax2 = ax1.twinx()
ax2.plot(tsla_stock2.Date,tsla_stock2.avgind,color='g')
ax1.plot(tsla_stock2.Date,tsla_stock2.Close,color='b')
ax1.set_xticks(tsla_stock2.Date)
ax1.set_xticklabels(tsla_stock2.Date,rotation = 90)
plt.title("Sentimental vs Real Stock price (tsla)")
plt.show()

#"look like chinese company everything tsla try fly car fraction market cap xiaomi baidu xpeng byd xn l rqbx"



'''
2022-09-16  0.355000
2022-09-17  0.390000
2022-09-19  0.455000
2022-09-20  0.500000
2022-09-21  0.384000
2022-09-22  0.350000
2022-09-23  0.360000
2022-09-26  0.475000
2022-09-27  0.580000
2022-09-28  0.350000
2022-09-29  0.293333
2022-09-30  0.396667
2022-10-01  0.300000
2022-10-02  0.400000
2022-10-03  0.198000
2022-10-04  0.283333
2022-10-05  0.317143
2022-10-06  0.266000
2022-10-07  0.338333
2022-10-08  0.370000
2022-10-09  0.160000
2022-10-10  0.410000
2022-10-11  0.382857
2022-10-12  0.350000
2022-10-13  0.345000
2022-10-14  0.463333
2022-10-15  0.290000
2022-10-16  0.420000
'''

df_test2=pd.read_csv(path+'\\train_code\\dataset_twitter_tesla_len3.csv')
df_test2['1']=df_test2['1'].str.strip()

score=[]
for i in range(df_test2.shape[0]):
    
    #data_sample = "look like chinese company everything tsla try fly car fraction market cap xiaomi baidu xpeng byd xn l rqbx"
    data_sample=df_test2.iloc[i,1]
    try:
        result = live_test(loaded_model,data_sample, word_idx)
    except:
        print('error')
        result=0
        #score.append(result)
    score.append(result)
    print(i)
    print (result)


df_test2['score']=score
df_test2['0']=pd.to_datetime(df_test2['0'],format='%m/%d/%Y')

avg_score2=df_test2.groupby('0').mean('score')

plt.plot(avg_score2.index,avg_score2.iloc[:,1])
plt.xticks(rotation = 90)

df_test2.to_csv(path+'\\train_code\\dataset_twitter_tesla_result_correct_len3.csv')

####
score_test=score
score_test.remove(0)

score_test= [x for x in score_test if x!=0]

df_test2['score']=score_test
df_test2['score_follower']=df_test2['score']*df_test2['2']
avg_score_follower=df_test2.groupby('0').sum('score_follower')
follower=df_test2.groupby('0').sum('2')

g=avg_score_follower['score_follower']/follower['2']

plt.plot(g.index,g[:])
plt.xticks(rotation = 90)

#Normalize
#g.to_csv(path+'\\train_code\\normalize.csv')

tsla = yf.Ticker("TSLA")

tsla_stock = tsla.history(
    start=(g.index.min()).strftime('%Y-%m-%d'),
    end=g.index.max().strftime('%Y-%m-%d'),
    interval='1d'
).reset_index()


tsla_stock1=tsla_stock[['Date','Close']]
tsla_stock_add=pd.DataFrame({'Date':['2022-08-06','2022-08-07','2022-08-11'],'Close':[288.17,288.17,286.63]})
tsla_stock1=tsla_stock1.append(tsla_stock_add, ignore_index = True)
tsla_stock1['Date']=tsla_stock1['Date'].astype('str')
k=[] 
for i in range(len(tsla_stock1)):
    k.append(tsla_stock1.iloc[i,0].split(' ')[0])
tsla_stock1['Date']=k

tsla_stock1.Date=pd.to_datetime(tsla_stock1.Date,format='%Y-%m-%d')
tsla_stock1=tsla_stock1.sort_values(by='Date')
tsla_stock1['avgind']=list(avg_score2['score'])
'''
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax2 = ax1.twinx()
ax2.plot(g.index,g[:],color='g')
ax1.plot(tsla_stock1.Date,tsla_stock1.Close,color='b')
ax1.set_xticks(g.index)
ax1.set_xticklabels(g.index,rotation = 90)
plt.show()
'''

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax2 = ax1.twinx()
ax2.plot(tsla_stock1.Date,tsla_stock1.avgind,color='g')
ax1.plot(tsla_stock1.Date,tsla_stock1.Close,color='b')
ax1.set_xticks(tsla_stock1.Date)
ax1.set_xticklabels(tsla_stock1.Date,rotation = 90)
plt.title("Sentimental vs Real Stock price (tsla)")
plt.show()