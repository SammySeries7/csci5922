# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 14:14:41 2022

@author: samsu
"""

import requests  #to query the API 
import re  #regular expressions
import pandas as pd   # for dataframes

from sklearn.feature_extraction.text import CountVectorizer   
#for text vectorization
from datetime import datetime as dt
import re 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import sent_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from io import StringIO
from sklearn.feature_extraction.text import CountVectorizer
from datetime import datetime as dt
'''
####################################################
### WAY 2
##  This is simply another option for using requests 
##  in Python. I use Way 1 when I want to place variables
##  into the dictionary structure. I use Way 2 when I want a 
##  copy of the URL generated. 
##  You can use whichever way you need for your goals
## - The right tool for the job....
##########################################################
url = ('https://newsapi.org/v2/everything?'
       'q=tesla&'
       #'from=2022-09-15&' 
       #'to=2022-08-12&'
       #'category=business&'
       'sources=business-insider&'
       #'pageSize=100&'
       #'apiKey=8f4134f7d0de43b8b49f91e22100f22b'
       'apiKey=6f3cc4e843ae47cb81c61a464a179122'
       #'qInTitle=Georgetown&'
       #'country=us'
)

## Some of the parameters are commented out. They are options.
print(url)



## Here again, we are using requests.get
## We can just give requests.get the url we made.
response2 = requests.get(url)
jsontxt2 = response2.json()
print(jsontxt2, "\n")

print(jsontxt2['articles'][0]['title'])

title=[]
dates=[]
for i in jsontxt2['articles']:
    title.append(i['title'])
    dates.append(i['publishedAt'])

for i in range(len(dates)):
    temp=dates[i].split('T')[0]
    dates[i]=dt.strptime(temp,'%Y-%m-%d')
    
df=pd.DataFrame({'dates':dates,'title':title})

today=dt.today()
d1 = today.strftime("%d-%m-%y")

name='newsapi'+d1+'.csv'
df.to_csv(name,index=False)

'''   

#############
def penn2morphy(penntag):
    """ Converts Penn Treebank tags to WordNet. """
    morphy_tag = {'NN':'n', 'JJ':'a',
                  'VB':'v', 'RB':'r'}
    try:
        return morphy_tag[penntag[:2]]
    except:
        return 'n' # if mapping isn't found, fall back to Noun.

df=pd.read_csv('tsla_head_line.csv')



# Choose the important columns
df1=df.copy()

df1['clean_txt']=''
df1['tokenize']=''
### Clean Data ####
stop_words= set(stopwords.words("english"))
stop_words.update(['https','http','amp','CO','t','u','new',"I'm",'would','co','oxkoa','jzu'])

#list of word after process
lemma=[]

for _ in range(df1.shape[0]):
    text=df1.iloc[_,1] #text
    tweet=re.sub(r'[,.;@#?!&$\-\']+', ' ', text, flags=re.IGNORECASE)
    tweet=re.sub(' +', ' ', tweet, flags=re.IGNORECASE) # Remove space
    tweet=re.sub(r'\"', ' ', tweet, flags=re.IGNORECASE)
    tweet=re.sub(r'[^a-zA-Z]', " ", tweet, flags=re.VERBOSE)
    tweet=tweet.replace(',', '')
    tweet=' '.join(tweet.split())
    
    df1.iloc[_,2]=tweet
    
    #Tokenize
    word_tokens=word_tokenize(tweet)
    word_tokens=list(map(lambda x:x.lower(),word_tokens))
    
    #Add subject verb noun
    wnl = WordNetLemmatizer()
    postag = pos_tag(word_tokens)
    
    #print(postag)
    word_tokens=[wnl.lemmatize(word, pos=penn2morphy(tag)) for word, tag in postag]

    
    #Remove stop word
    filtered_sentence =[w for w in word_tokens if not w.lower() in stop_words]
    
    #Lemmatizer

    #merge text
    temp=''
    wnl = WordNetLemmatizer()

    for word in filtered_sentence:
        #print(wnl.lemmatize(word))
        if len(word)>=3:    
            temp=temp+" "+wnl.lemmatize(word)
        
    lemma.append(temp.strip())        
    #print(filtered_sentence)

dated=df1.dates.to_list()


df_final=pd.DataFrame(list(zip(dated,lemma)))
#df_final.to_csv('business_insider.csv',index=False)
