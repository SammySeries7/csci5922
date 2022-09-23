# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 09:59:20 2022

@author: samsu
"""

import pandas as pd
import re 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import sent_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from io import StringIO
from sklearn.feature_extraction.text import CountVectorizer

#https://www.kaggle.com/code/alvations/basic-nlp-with-nltk
#https://github.com/chandravenky/Stock-price-prediction-using-Twitter/blob/master/Stock_price_prediction_using_Twitter.ipynb

def penn2morphy(penntag):
    """ Converts Penn Treebank tags to WordNet. """
    morphy_tag = {'NN':'n', 'JJ':'a',
                  'VB':'v', 'RB':'r'}
    try:
        return morphy_tag[penntag[:2]]
    except:
        return 'n' # if mapping isn't found, fall back to Noun.

df=pd.read_csv('tweet.csv')

df['dates']=0
for _ in range(df.shape[0]):
    df.iloc[_,12]=df.iloc[_,7].split(" ")[0]
df['dates']=pd.to_datetime(df['dates'])

# Choose the important columns
df1=df[['dates','author_followers','text']].copy()

df1['clean_txt']=''
df1['tokenize']=''
### Clean Data ####
stop_words= set(stopwords.words("english"))
stop_words.update(['https','http','amp','CO','t','u','new',"I'm",'would','co','oxkoa','jzu'])

#list of word after process
lemma=[]

for _ in range(df1.shape[0]):
    text=df1.iloc[_,2] #text
    tweet=re.sub(r'[,.;@#?!&$\-\']+', ' ', text, flags=re.IGNORECASE)
    tweet=re.sub(' +', ' ', tweet, flags=re.IGNORECASE) # Remove space
    tweet=re.sub(r'\"', ' ', tweet, flags=re.IGNORECASE)
    tweet=re.sub(r'[^a-zA-Z]', " ", tweet, flags=re.VERBOSE)
    tweet=tweet.replace(',', '')
    tweet=' '.join(tweet.split())
    
    df1.iloc[_,3]=tweet
    
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
        temp=temp+" "+wnl.lemmatize(word)
        
    lemma.append(temp.strip())        
    #print(filtered_sentence)

dated=df1.dates.to_list()
follower=df1.author_followers.to_list()

df_final=pd.DataFrame(list(zip(dated,lemma,follower)))
#df_final.to_csv('dataset_twitter.csv',index=False)




