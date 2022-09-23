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

df=pd.read_table('C:\\Users\\samsu\\Downloads\\csci5922\\Project\\stanford_cleandata\\dictionary.txt')

df1=df.copy()

### Clean Data ####
stop_words= set(stopwords.words("english"))
stop_words.update(['https','http','amp','CO','t','u','new',"I'm",'would','co','oxkoa','jzu'])

#list of word after process
lemma=[]

df1['clean_text']=''
ids=[]
for _ in range(df1.shape[0]):
    text=df1.iloc[_,0].split('|')[0] #text
    ids_text=df1.iloc[_,0].split('|')[1]
    tweet=re.sub(r'[,.;@#?!&$\-\']+', ' ', text, flags=re.IGNORECASE)
    tweet=re.sub(' +', ' ', tweet, flags=re.IGNORECASE) # Remove space
    tweet=re.sub(r'\"', ' ', tweet, flags=re.IGNORECASE)
    tweet=re.sub(r'[^a-zA-Z]', " ", tweet, flags=re.VERBOSE)
    tweet=tweet.replace(',', '')
    tweet=' '.join(tweet.split())
    
    df1.iloc[_,1]=tweet
    
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
    ids.append(ids_text)        
    #print(filtered_sentence)

df_final=pd.DataFrame(list(zip(lemma,ids)))
df_final.drop(df_final[df_final[0]==''].index,axis=0,inplace=True)
df_final1=df_final.drop_duplicates(subset=0,keep='first')
#df_final1['Phrase|Index']=df_final1[0]+'|'+df_final1[1]
#df_final1['Phrase|Index'].to_csv('result.csv',index=False)
df_final1['Phrase|Index'].to_csv('dataset_stanford.csv',index=False)




