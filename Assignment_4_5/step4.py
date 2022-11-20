# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 09:32:17 2022

@author: samsu
"""


########################################
##  - Gates
##
## Topics: 
    # Data gathering via API
    #  - URLs and GET
    # Cleaning and preparing text DATA
    # DTM and Data Frames
    # Training and Testing at DT
        
#########################################    
    
    
## ATTENTION READER...
##
## First, you will need to go to 
## https://newsapi.org/
## https://newsapi.org/register
## and get an API key



################## —---------------------------------
## You will need a key
##
###################################################


### API KEY  - get a key!
##https://newsapi.org/

## Example URL
## https://newsapi.org/v2/everything?
## q=tesla&from=2021-05-20&sortBy=publishedAt&
## apiKey=YOUR KEY HERE


## What to import
import requests  ## for getting data from a server GET
import re   ## for regular expressions
import pandas as pd    ## for dataframes and related
from pandas import DataFrame

## To tokenize and vectorize text type data
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
## For word clouds
## conda install -c conda-forge wordcloud
## May also have to run conda update --all on cmd
#import PIL
#import Pillow
#import wordcloud
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
import random as rd
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
#from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import tree
## conda install python-graphviz
## restart kernel (click the little red x next to the Console)
import graphviz

from sklearn.decomposition import LatentDirichletAllocation 
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import silhouette_samples, silhouette_score
import sklearn
from sklearn.cluster import KMeans

from sklearn import preprocessing

import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from mpl_toolkits.mplot3d import Axes3D
from scipy.cluster.hierarchy import ward, dendrogram
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder

####################################
##
##  Step 1: Connect to the server
##          Send a query
##          Collect and clean the 
##          results
####################################

####################################################
##In the following loop, we will query the newsapi servers
##for all the topic names in the list
## We will then build a large csv file 
## where each article is a row
##
## From there, we will convert this data
## into a labeled dataframe
## so we can train and then test our DT
## model
####################################################

####################################################
## Build the URL and GET the results
## NOTE: At the bottom of this code
## commented out, you will find a second
## method for doing the following. This is FYI.
####################################################

## This is the endpoint - the server and 
## location on the server where your data 
## will be retrieved from

## TEST FIRST!
## We are about to build this URL:
## https://newsapi.org/v2/everything?apiKey=8f4134 your key here 0f22b&q=bitcoin



#topics=["politics", "analytics", "business", "sports"]

topics=["politics","sports"]


## topics needs to be a list of strings (words)
## Next, let's build the csv file
## first and add the column names
## Create a new csv file to save the headlines
filename="C:\\Users\\samsu\\Downloads\\csci5922\\assignment5\\NewHeadlines.csv"
MyFILE=open(filename,"w")  # "a"  for append   "r" for read
## with open
### Place the column names in - write to the first row
WriteThis="LABEL,Date,Source,Title,Headline\n"
MyFILE.write(WriteThis)
MyFILE.close()

## CHeck it! Can you find this file?
    
#### --------------------> GATHER - CLEAN - CREATE FILE    

## RE: documentation and options
## https://newsapi.org/docs/endpoints/everything

endpoint="https://newsapi.org/v2/everything"

################# enter for loop to collect
################# data on three topics
#######################################

for topic in topics:

    ## Dictionary Structure
    URLPost = {'apiKey':'6f3cc4e843ae47cb81c61a464a179122',
               'q':topic,
               'pagesize':100,
               'from':2022-11-19
    }

    response=requests.get(endpoint, URLPost)
    print(response)
    jsontxt = response.json()
    print(jsontxt)
    #####################################################
    
    
    ## Open the file for append
    MyFILE=open(filename, "a")
    LABEL=topic
    for items in jsontxt["articles"]:
        print(items, "\n\n\n")
                  
        #Author=items["author"]
        #Author=str(Author)
        #Author=Author.replace(',', '')
        
        Source=items["source"]["name"]
        print(Source)
        
        Date=items["publishedAt"]
        ##clean up the date
        NewDate=Date.split("T")
        Date=NewDate[0]
        print(Date)
        
        ## CLEAN the Title
        ##----------------------------------------------------------
        ##Replace punctuation with space
        # Accept one or more copies of punctuation         
        # plus zero or more copies of a space
        # and replace it with a single space
        Title=items["title"]
        Title=str(Title)
        #print(Title)
        Title=re.sub(r'[,.;@#?!&$\-\']+', ' ', str(Title), flags=re.IGNORECASE)
        Title=re.sub(' +', ' ', str(Title), flags=re.IGNORECASE)
        Title=re.sub(r'\"', ' ', str(Title), flags=re.IGNORECASE)
        
        # and replace it with a single space
        ## NOTE: Using the "^" on the inside of the [] means
        ## we want to look for any chars NOT a-z or A-Z and replace
        ## them with blank. This removes chars that should not be there.
        Title=re.sub(r'[^a-zA-Z]', " ", str(Title), flags=re.VERBOSE)
        Title=Title.replace(',', '')
        Title=' '.join(Title.split())
        Title=re.sub("\n|\r", "", Title)
        print(Title)
        ##----------------------------------------------------------
        
        Headline=items["description"]
        Headline=str(Headline)
        Headline=re.sub(r'[,.;@#?!&$\-\']+', ' ', Headline, flags=re.IGNORECASE)
        Headline=re.sub(' +', ' ', Headline, flags=re.IGNORECASE)
        Headline=re.sub(r'\"', ' ', Headline, flags=re.IGNORECASE)
        Headline=re.sub(r'[^a-zA-Z]', " ", Headline, flags=re.VERBOSE)
        ## Be sure there are no commas in the headlines or it will
        ## write poorly to a csv file....
        Headline=Headline.replace(',', '')
        Headline=' '.join(Headline.split())
        Headline=re.sub("\n|\r", "", Headline)
        
        ### AS AN OPTION - remove words of a given length............
        Headline = ' '.join([wd for wd in Headline.split() if len(wd)>3])
    
        #print("Author: ", Author, "\n")
        #print("Title: ", Title, "\n")
        #print("Headline News Item: ", Headline, "\n\n")
        
        #print(Author)
        print(Title)
        print(Headline)
        
        WriteThis=str(LABEL)+","+str(Date)+","+str(Source)+","+ str(Title) + "," + str(Headline) + "\n"
        print(WriteThis)
        
        MyFILE.write(WriteThis)
        
    ## CLOSE THE FILE
    MyFILE.close()

    
################## END for loop

####################################################
##
## Where are we now?
## 
## So far, we have created a csv file
## with labeled data. Each row is a news article
##
## - BUT - 
## We are not done. We need to choose which
## parts of this data to use to model our decision tree
## and we need to convert the data into a data frame.
##
########################################################



bBC_DF=pd.read_csv(filename, error_bad_lines=False)

'''
#Fix the line
bBC_DF.iloc[375,2]=bBC_DF.iloc[375,5]
bBC_DF.iloc[375,3]=bBC_DF.iloc[375,6]
bBC_DF.iloc[375,5]=np.nan
bBC_DF.iloc[375,6]=np.nan
'''

bBC_DF=bBC_DF[bBC_DF.loc[:,"Headline"]!="None"]



print(bBC_DF.head())

# iterating the columns 
for col in bBC_DF.columns: 
    print(col) 
    
print(bBC_DF["Headline"])

## REMOVE any rows with NaN in them
bBC_DF = bBC_DF.dropna()
print(bBC_DF["Headline"])

bBC_DF=bBC_DF.reset_index(drop=True)

### Tokenize and Vectorize the Headlines
## Create the list of headlines
## Keep the labels!

HeadlineLIST=[]
LabelLIST=[]

for nexthead, nextlabel in zip(bBC_DF["Headline"], bBC_DF["LABEL"]):
    HeadlineLIST.append(nexthead)
    LabelLIST.append(nextlabel)

print("The headline list is:\n")
print(HeadlineLIST)

print("The label list is:\n")
print(LabelLIST)



##########################################
## Remove all words that match the topics.
## For example, if the topics are food and covid
## remove these exact words.
##
## We will need to do this by hand. 
NewHeadlineLIST=[]

for element in HeadlineLIST:
    print(element)
    print(type(element))
    ## make into list
    AllWords=element.split(" ")
    print(AllWords)
    
    ## Now remove words that are in your topics
    NewWordsList=[]
    for word in AllWords:
        print(word)
        word=word.lower()
        if word in topics:
            print(word)
        else:
            NewWordsList.append(word)
            
    ##turn back to string
    NewWords=" ".join(NewWordsList)
    ## Place into NewHeadlineLIST
    NewHeadlineLIST.append(NewWords)


##
## Set the     HeadlineLIST to the new one
HeadlineLIST=NewHeadlineLIST
print(HeadlineLIST)     
#########################################
##
##  Build the labeled dataframe
##
######################################################

from sklearn.feature_extraction import text 

stop_words = text.ENGLISH_STOP_WORDS.union('https','href')


### Vectorize
## Instantiate your CV
MyCountV=CountVectorizer(
        input="content",  ## because we have a csv file
        lowercase=True, 
        stop_words = text.ENGLISH_STOP_WORDS.union(['https','href']),
        max_features=2000
        )

## Use your CV 
MyDTM = MyCountV.fit_transform(HeadlineLIST)  # create a sparse matrix
print(type(MyDTM))


ColumnNames=MyCountV.get_feature_names()
#print(type(ColumnNames))

'''
Check word in list
keyword_list = ['href']

if any(word in ColumnNames for word in keyword_list):
    print('found one of em')
'''

'''
# ## Create Vocab
counter = Counter([words for reviews in HeadlineLIST for words in reviews.split()])
df = pd.DataFrame()
df['key'] = counter.keys()
df['value'] = counter.values()
df.sort_values(by='value', ascending=False, inplace=True)
print(df.head(10))
'''

## Build the data frame
MyDTM_DF=pd.DataFrame(MyDTM.toarray(),columns=ColumnNames)

## Convert the labels from list to df
Labels_DF = DataFrame(LabelLIST,columns=['LABEL'])

def Logical_Numbers_Present(anyString):
    return any(char.isdigit() for char in anyString)
##----------------------------------------------------

for nextcol in MyDTM_DF.columns:
    #print(nextcol)
    ## Remove unwanted columns
    #Result=str.isdigit(nextcol) ## Fast way to check numbers
    #print(Result)
    
    ##-------------call the function -------
    LogResult=Logical_Numbers_Present(nextcol)
    #print(LogResult)
    ## The above returns a logical of True or False
    
    ## The following will remove all columns that contains numbers
    if(LogResult==True):
        #print(LogResult)
        #print(nextcol)
        MyDTM_DF=MyDTM_DF.drop([nextcol], axis=1)

    ## The following will remove any column with name
    ## of 3 or smaller - like "it" or "of" or "pre".
    ##print(len(nextcol))  ## check it first
    ## NOTE: You can also use this code to CONTROL
    ## the words in the columns. For example - you can
    ## have only words between lengths 5 and 9. 
    ## In this case, we remove columns with words <= 3.
    elif(len(str(nextcol))<3):
        print(nextcol)
        MyDTM_DF=MyDTM_DF.drop([nextcol], axis=1)

## Check your new DF and you new Labels df:
print("Labels\n")
print(Labels_DF)
print("News df\n")
print(MyDTM_DF.iloc[:,0:6])

##Save original DF - without the lables
My_Orig_DF=MyDTM_DF
print(My_Orig_DF)
######################
## AND - just to make sure our dataframe is fair
## let's remove columns called:
## food, bitcoin, and sports (as these are label names)
######################
#MyDTM_DF=MyDTM_DF.drop(topics, axis=1)


## Now - let's create a complete and labeled
## dataframe:
dfs = [Labels_DF, MyDTM_DF]
print(dfs)

Final_News_DF_Labeled = pd.concat(dfs,axis=1, join='inner')
## DF with labels
print(Final_News_DF_Labeled)





#############################################
##
## Create Training and Testing Data
##
## Then model and test the Decision Tree
##
################################################


## Before we start our modeling, let's visualize and
## explore.

##It might be very interesting to see the word clouds 
## for each  of the topics. 
##--------------------------------------------------------
List_of_WC=[]

for mytopic in topics:

    tempdf = Final_News_DF_Labeled[Final_News_DF_Labeled['LABEL'] == mytopic]
    print(tempdf)
    
    tempdf =tempdf.sum(axis=0,numeric_only=True)
    #print(tempdf)
    
    #Make var name
    NextVarName=str("wc"+str(mytopic))
    #print( NextVarName)
    
    ##In the same folder as this code, I have three images
    ## They are called: food.jpg, bitcoin.jpg, and sports.jpg
    #next_image=str(str(mytopic) + ".jpg")
    #print(next_image)
    
    ## https://amueller.github.io/word_cloud/generated/wordcloud.WordCloud.html
    
    ###########
    ## Create and store in a list the wordcloud OBJECTS
    #########
    NextVarName = WordCloud(width=1000, height=600, background_color="white",
                   min_word_length=4, #mask=next_image,
                   max_words=200).generate_from_frequencies(tempdf)
    
    ## Here, this list holds all three wordclouds I am building
    List_of_WC.append(NextVarName)
    

##------------------------------------------------------------------
print(List_of_WC)
##########
########## Create the wordclouds
##########
fig=plt.figure(figsize=(25, 25))
#figure, axes = plt.subplots(nrows=2, ncols=2)
NumTopics=len(topics)
for i in range(NumTopics):
    print(i)
    ax = fig.add_subplot(NumTopics,1,i+1)
    plt.imshow(List_of_WC[i], interpolation='bilinear')
    plt.axis("off")
    plt.savefig("C:\\Users\\samsu\\Downloads\\csci5922\\assignment5\\NewClouds.pdf")

#%%#  
## Create list of all words
print(Final_News_DF_Labeled.columns[0])
NumCols=Final_News_DF_Labeled.shape[1]-1
print(NumCols)
print(len(list(Final_News_DF_Labeled.columns)))

top_words=list(Final_News_DF_Labeled.columns[1:NumCols+1])
## Exclude the Label

print(top_words[0])
print(top_words[-1])


#print(type(top_words))
#print(top_words.index("aamir")) ## index 0 in top_words
#print(top_words.index("zucco")) #index NumCols - 2 in top_words

def find_maxword(review):
    count=0
    for i in review:
        len_word = len(i.split())
        if len_word>count:
            count=len_word
    return count

maxword=find_maxword(HeadlineLIST)


## Encoding the data
def Encode(review):
    words = review.split()
   # print(words)
    if len(words) > 31:
        words = words[:31]
        #print(words)
    encoding = []
    for word in words:
        try:
            index = top_words.index(word)
        except:
            index = (NumCols - 1)
        encoding.append(index)
    while len(encoding) < 31:
        encoding.append(NumCols)
    return encoding



encoded_data = np.array([Encode(review) for review in HeadlineLIST])
print(encoded_data[20])
print(encoded_data.shape)

le = preprocessing.LabelEncoder()
y=le.fit_transform(Final_News_DF_Labeled['LABEL'].values.tolist()).reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(encoded_data, y, test_size=0.33, random_state=42)

ohe = OneHotEncoder()
y_train_one_hot=ohe.fit_transform(y_train).toarray()

## ANN
input_dim=NumCols+1
import tensorflow
from tensorflow.keras.layers import Activation
 #https://www.tensorflow.org/api_docs/python/tf/keras/Input
input_data = tensorflow.keras.layers.Input(shape=(31))
 #https://www.tensorflow.org/api_docs/python/tf/keras/layers/Embedding

data = tensorflow.keras.layers.Embedding(input_dim=input_dim, output_dim=64, input_length=31)(input_data)

data = tensorflow.keras.layers.Dense(32)(data)
data = tensorflow.keras.layers.Activation('relu')(data)
#data = tensorflow.keras.layers.Dropout(0.5)(data)
 
data = tensorflow.keras.layers.Dense(16)(data)
data = tensorflow.keras.layers.Activation('relu')(data)

'''
#data = tensorflow.keras.layers.Dropout(0.5)(data)
 
data = tensorflow.keras.layers.Dense(4)(data)
data = tensorflow.keras.layers.Activation('sigmoid')(data)
#data = tensorflow.keras.layers.Dropout(0.5)(data)
'''

data=tensorflow.keras.layers.Flatten()(data)
 
data = tensorflow.keras.layers.Dense(2)(data)
output_data = tensorflow.keras.layers.Activation('softmax')(data)
 
model = tensorflow.keras.models.Model(inputs=input_data, outputs=output_data)
 
model.compile(loss=tensorflow.keras.losses.CategoricalCrossentropy(), optimizer='adam', metrics='accuracy')
model.summary()

print(X_train[0:3, 0:3])
print(X_train.shape)
model.fit(X_train, y_train_one_hot, epochs=100, batch_size=32, validation_split=0.1)

# Generate a prediction using model.predict() 
# and calculate it's shape:
print("Generate a prediction")
prediction = model.predict(X_test)
prediction = np.argmax(prediction,axis=1)
print(prediction)

from sklearn.metrics import confusion_matrix
print(confusion_matrix(prediction, y_test))

count=0
for i in range(len(y_test)):
    if prediction[i]==y_test[i]:
        count+=1

print(count/y_test.shape[0])

'''
[0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 1 0 1 0 0 0 0 0 0 1 0 0 0 1 0 0 0 0 0
 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 1 0]
[[27 22]
 [ 5 11]]
0.5846153846153846

[[24 26]
 [ 8  7]]
0.47692307692307695

31
[[26 19]
 [ 6 14]]
0.6153846153846154
'''

#ANN with binary cross entropy

input_dim=NumCols+1
import tensorflow
from tensorflow.keras.layers import Activation
 #https://www.tensorflow.org/api_docs/python/tf/keras/Input
input_data = tensorflow.keras.layers.Input(shape=(31))
 #https://www.tensorflow.org/api_docs/python/tf/keras/layers/Embedding

data = tensorflow.keras.layers.Embedding(input_dim=input_dim, output_dim=64, input_length=31)(input_data)

data = tensorflow.keras.layers.Dense(32)(data)
data = tensorflow.keras.layers.Activation('relu')(data)
#data = tensorflow.keras.layers.Dropout(0.5)(data)
 
data = tensorflow.keras.layers.Dense(16)(data)
data = tensorflow.keras.layers.Activation('relu')(data)

'''
#data = tensorflow.keras.layers.Dropout(0.5)(data)
 
data = tensorflow.keras.layers.Dense(4)(data)
data = tensorflow.keras.layers.Activation('sigmoid')(data)
#data = tensorflow.keras.layers.Dropout(0.5)(data)
'''

data=tensorflow.keras.layers.Flatten()(data)
 
data = tensorflow.keras.layers.Dense(1)(data)
output_data = tensorflow.keras.layers.Activation('sigmoid')(data)
 
model = tensorflow.keras.models.Model(inputs=input_data, outputs=output_data)
 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics='accuracy')
model.summary()

print(X_train[0:3, 0:3])
print(X_train.shape)
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1)

# Generate a prediction using model.predict() 
# and calculate it's shape:
print("Generate a prediction")
prediction = model.predict(X_test)
prediction[prediction > .5] = 1
prediction[prediction <= .5] = 0
print(prediction)

from sklearn.metrics import confusion_matrix
print(confusion_matrix(prediction, y_test))

count=0
for i in range(len(y_test)):
    if prediction[i]==y_test[i]:
        count+=1

print(count/y_test.shape[0])

'''
[[19 11]
 [13 22]]
0.6307692307692307

[[15 16]
 [17 17]]
0.49230769230769234

[[23 16]
 [ 9 17]]
0.6153846153846154

100
[[10  9]
 [22 24]]
0.5230769230769231
'''


#CNN
import tensorflow
 
input_data = tensorflow.keras.layers.Input(shape=(31))
 
data = tensorflow.keras.layers.Embedding(input_dim=input_dim, output_dim=64, input_length=31)(input_data)
 
data = tensorflow.keras.layers.Conv1D(50, kernel_size=3, activation='relu')(data)
data = tensorflow.keras.layers.MaxPool1D(pool_size=2)(data)
 
data = tensorflow.keras.layers.Conv1D(40, kernel_size=3, activation='relu')(data)
data = tensorflow.keras.layers.MaxPool1D(pool_size=2)(data)
 

 
data = tensorflow.keras.layers.Flatten()(data)
 
data = tensorflow.keras.layers.Dense(20)(data)
data = tensorflow.keras.layers.Activation('relu')(data)
data = tensorflow.keras.layers.Dropout(0.5)(data)
 
data = tensorflow.keras.layers.Dense(1)(data)
output_data = tensorflow.keras.layers.Activation('sigmoid')(data)
 
model = tensorflow.keras.models.Model(inputs=input_data, outputs=output_data)
 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics='accuracy')
model.summary()

model.fit(X_train, y_train, epochs=100, batch_size=32 , validation_split=0.1)

print("Evaluate model on test data")
results = model.evaluate(X_test, y_test, batch_size=32)
print("test loss, test acc:", results)

# Generate a prediction using model.predict() 
# and calculate it's shape:
print("Generate a prediction")
prediction = model.predict(X_test)
print(prediction)
print("prediction shape:", prediction.shape)
print(type(prediction))
prediction[prediction > .5] = 1
prediction[prediction <= .5] = 0
print(prediction)

from sklearn.metrics import confusion_matrix
print(confusion_matrix(prediction, y_test))

count=0
for i in range(len(y_test)):
    if prediction[i]==y_test[i]:
        count+=1

print(count/y_test.shape[0])

'''
[[30 17]
 [ 2 16]]
0.7076923076923077

[[30 17]
 [ 2 16]]
0.7076923076923077

[[29 11]
 [ 3 22]]
0.7846153846153846


100
[[28 20]
 [ 4 13]]
0.6307692307692307

[[31 22]
 [ 1 11]]
0.6461538461538462
'''

#LSTM
import tensorflow
from tensorflow.keras.layers import Activation
 
input_data = tensorflow.keras.layers.Input(shape=(31))
 
data = tensorflow.keras.layers.Embedding(input_dim=input_dim, output_dim=64, input_length=31)(input_data)
 
data = tensorflow.keras.layers.Bidirectional(tensorflow.keras.layers.LSTM(100))(data)


data = tensorflow.keras.layers.Dropout(0.5)(data) 

data = tensorflow.keras.layers.Dense(32)(data)
data = tensorflow.keras.layers.Activation('relu')(data)
data = tensorflow.keras.layers.Dropout(0.5)(data) 

data = tensorflow.keras.layers.Dense(16)(data)
data = tensorflow.keras.layers.Activation('relu')(data)


data = tensorflow.keras.layers.Dropout(0.5)(data) 
data = tensorflow.keras.layers.Dense(1)(data)
output_data = tensorflow.keras.layers.Activation('sigmoid')(data)

 
model = tensorflow.keras.models.Model(inputs=input_data, outputs=output_data)
 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics='accuracy')
model.summary()

model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1)


print("Evaluate model on test data")
results = model.evaluate(X_test, y_test, batch_size=256)
print("test loss, test acc:", results)

# Generate a prediction using model.predict() 
# and calculate it's shape:
print("Generate a prediction")
prediction = model.predict(X_test)
print(prediction)
print("prediction shape:", prediction.shape)
print(type(prediction))
prediction[prediction > .5] = 1
prediction[prediction <= .5] = 0
print(prediction)


print(confusion_matrix(prediction, y_test))

count=0
for i in range(len(y_test)):
    if prediction[i]==y_test[i]:
        count+=1

print(count/y_test.shape[0])

'''
[[23  5]
 [ 9 28]]
0.7846153846153846

[[30 14]
 [ 2 19]]
0.7538461538461538

[[16  3]
 [16 30]]
0.7076923076923077

100
[[20  7]
 [12 26]]
0.7076923076923077

[[23 11]
 [ 9 22]]
0.6923076923076923
'''