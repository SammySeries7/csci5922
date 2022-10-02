import pandas as pd
import numpy as np
import re
import codecs
import os
from nltk.tokenize import RegexpTokenizer

import os



#%%
################################### Paths to Data ########################################################################

path = 'C:/Users/samsu/Downloads/deepsentiment-master/Data/'
gloveFile = 'C:/Users/samsu/Downloads/deepsentiment-master/Data/glove/glove_6B_100d.txt' 
vocab_path = 'C:/Users/samsu/Downloads/deepsentiment-master/Data/glove/vocab_glove.csv'

#Split Data path
train_data_path ='C:/Users/samsu/Downloads/deepsentiment-master/Data/TrainingData/train.csv'
val_data_path ='C:/Users/samsu/Downloads/deepsentiment-master/Data/TrainingData/val.csv'
test_data_path ='C:/Users/samsu/Downloads/deepsentiment-master/Data/TrainingData/test.csv'

sent_matrix_path ='C:/Users/samsu/Downloads/deepsentiment-master/Data/inputs_model/sentence_matrix.csv'
sent_matrix_path_val ='C:/Users/samsu/Downloads/deepsentiment-master/Data/inputs_model/sentence_matrix_val.csv'
sent_matrix_path_test ='C:/Users/samsu/Downloads/deepsentiment-master/Data/inputs_model/sentence_matrix_test.csv'
sequence_len_path = 'C:/Users/samsu/Downloads/deepsentiment-master/Data/inputs_model/sequence_length.csv'
sequence_len_val_path = 'C:/Users/samsu/Downloads/deepsentiment-master/Data/inputs_model/sequence_length_val.csv'
sequence_len_test_path = 'C:/Users/samsu/Downloads/deepsentiment-master/Data/inputs_model/sequence_length_test.csv'
wordVectors_path = 'C:/Users/samsu/Downloads/deepsentiment-master/Data/inputs_model/wordVectors.csv'
#%%#

#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Filtered Vocabulary from Glove document >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def filter_glove(full_glove_path, data_dir):
  vocab = set()
  sentence_path = os.path.join(data_dir,'SOStr.txt')
  filtered_glove_path = os.path.join(data_dir, 'filtered_glove.txt')
  # Download the full set of unlabeled sentences separated by '|'.
  #sentence_path, = download_and_unzip(
    #'http://nlp.stanford.edu/~socherr/', 'stanfordSentimentTreebank.zip',
    #'stanfordSentimentTreebank/SOStr.txt')
  with codecs.open(sentence_path, encoding='utf-8') as f:
    for line in f:
      # Drop the trailing newline and strip backslashes. Split into words.
      vocab.update(line.strip().replace('\\', '').split('|'))
      
  nread = 0
  nwrote = 0
  with codecs.open(full_glove_path, encoding='utf-8') as f:
    with codecs.open(filtered_glove_path, 'w', encoding='utf-8') as out:
      for line in f:
        nread += 1
        line = line.strip()
        if not line: continue
        if line.split(u' ', 1)[0] in vocab:
          out.write(line + '\n')
          nwrote += 1
  print('read %s lines, wrote %s' % (nread, nwrote))

#%%#

#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Filtered Vocabulary from live cases >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>



# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< load embeddings >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

def load_embeddings(embedding_path):
  """Loads embedings, returns weight matrix and dict from words to indices."""
  print('loading word embeddings from %s' % embedding_path)
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


# Combine and split the data into train and test
def read_data(path):
    # read dictionary into df
    df_data_sentence = pd.read_csv(path + 'dictionary.csv')
    df_data_sentence_processed = df_data_sentence['Phrase|Index'].str.split('|', expand=True)
    df_data_sentence_processed = df_data_sentence_processed.rename(columns={0: 'Phrase', 1: 'phrase_ids'})

    # read sentiment labels into df
    df_data_sentiment = pd.read_table(path + 'sentiment_labels.txt')
    df_data_sentiment_processed = df_data_sentiment['phrase ids|sentiment values'].str.split('|', expand=True)
    df_data_sentiment_processed = df_data_sentiment_processed.rename(columns={0: 'phrase_ids', 1: 'sentiment_values'})


    #combine data frames containing sentence and sentiment
    df_processed_all = df_data_sentence_processed.merge(df_data_sentiment_processed, how='inner', on='phrase_ids')

    return df_processed_all

def training_data_split(all_data, spitPercent, data_dir):

    msk = np.random.rand(len(all_data)) < spitPercent
    train_only = all_data[msk]
    test_and_dev = all_data[~msk]


    msk_test = np.random.rand(len(test_and_dev)) <0.5
    test_only = test_and_dev[msk_test]
    dev_only = test_and_dev[~msk_test]

    dev_only.to_csv(os.path.join(data_dir, 'TrainingData/dev.csv'))
    test_only.to_csv(os.path.join(data_dir, 'TrainingData/test.csv'))
    train_only.to_csv(os.path.join(data_dir, 'TrainingData/train.csv'))

    return train_only, test_only, dev_only
#%%
################################### Glove Vector  ########################################################################
def loadGloveModel(gloveFile):
    print ("Loading Glove Model")
    f = open(gloveFile,'r',encoding='utf-8')
    model = {}
    for line in f:
        try:
            splitLine = line.split()
            word = splitLine[0]
            embedding = [float(val) for val in splitLine[1:]]
            model[word] = embedding
        except:
            print (word)
            continue

    print ("Done.",len(model)," words loaded!")
    return model
#%%


#%%
################################### Create Vocab subset GLove vectors ########################################################################

def word_vec_index(training_data, glove_model):

    sentences = training_data['Phrase'] # get the phrases as a df series
    #sentences = sentences[0:100]
    sentences_concat = sentences.str.cat(sep=' ')
    sentence_words = re.findall(r'\S+', sentences_concat)
    sentence_words_lwr = [x.lower() for x in sentence_words]
    subdict = {word: glove_model[word] for word in glove_model.keys() & sentence_words_lwr}
    #print(subdict)
    vocab_df = pd.DataFrame(subdict)
    vocab_df.to_csv(vocab_path)
    return vocab_df
#%%
################################### Convertdf to list ########################################################################
def word_list(vocab_df):

    wordVectors = vocab_df.values.T.tolist()
    wordVectors_np = np.array(wordVectors)
    wordList = list(vocab_df.columns.values)

    return wordList, wordVectors_np
 #%%
################################### tensorflow data pipeline ########################################################################


def maxSeqLen(training_data):

    total_words = 0
    sequence_length = []
    idx = 0
    for index, row in training_data.iterrows():

        sentence = (row['Phrase'])
        sentence_words = sentence.split(' ')
        len_sentence_words = len(sentence_words)
        total_words = total_words + len_sentence_words

        # get the length of the sequence of each training data
        sequence_length.append(len_sentence_words)

        if idx == 0:
            max_seq_len = len_sentence_words


        if len_sentence_words > max_seq_len:
            max_seq_len = len_sentence_words
        idx = idx + 1

    avg_words = total_words/index

    # convert to numpy array
    sequence_length_np = np.asarray(sequence_length)

    return max_seq_len, avg_words, sequence_length_np

  #%%
def tf_data_pipeline(data, word_idx, weight_matrix, max_seq_len):
    #(training_data, wordList, wordVectors, max_seq_len)
    #training_data = training_data[0:50]

    maxSeqLength = max_seq_len #Maximum length of sentence
    no_rows = len(data)
    ids = np.zeros((no_rows, maxSeqLength), dtype='int32')
    # conver keys in dict to lower case
    word_idx_lwr =  {k.lower(): v for k, v in word_idx.items()}
    #print(word_idx_lwr)
    idx = 0

    for index, row in data.iterrows():


        sentence = (row['Phrase'])
        sentence_words = sentence.split(' ')

        i = 0
        for word in sentence_words:
            #print(index)
            word_lwr = word.lower()
            try:
                #print ('inside',word_idx_lwr[word_lwr])
                ids[idx][i] =  word_idx_lwr[word_lwr]

            except Exception as e:
                #print (e)
                #print (word)
                if str(e) == word:
                    ids[idx][i] = 0
                continue
            i = i + 1
        idx = idx + 1
    return ids

  #%%
# create labels matrix for the rnn


def tf_data_pipeline_nltk(data, word_idx, weight_matrix, max_seq_len):
   

    #training_data = training_data[0:50]

    maxSeqLength = max_seq_len #Maximum length of sentence
    no_rows = len(data)
    #Include padding
    ids = np.zeros((no_rows, maxSeqLength), dtype='int32')
    #Convert keys in dict to lower case
    word_idx_lwr =  {k.lower(): v for k, v in word_idx.items()}
    idx = 0

    for index, row in data.iterrows():


        sentence = (row['Phrase'])
        #print (sentence)
        #search for word characters
        tokenizer = RegexpTokenizer(r'\w+')
        sentence_words = tokenizer.tokenize(sentence)
        #print (sentence_words)
        i = 0
        for word in sentence_words:
            #print(index)
            word_lwr = word.lower()
            #print (word_lwr)
            try:
                #print ('inside',word_lwr)
                ids[idx][i] =  word_idx_lwr[word_lwr]

            except Exception as e:
                #print (e)
                #print (word)
                if str(e) == word:
                    ids[idx][i] = 0
                continue
            i = i + 1
        idx = idx + 1

    return ids


def labels_matrix(data):

    labels = data['sentiment_values']

    lables_float = labels.astype(float)

    cats = ['0','1','2','3','4','5','6','7','8','9']
    labels_mult = (lables_float * 10).astype(int)
    dummies = pd.get_dummies(labels_mult, prefix='', prefix_sep='')
    dummies = dummies.T.reindex(cats).T.fillna(0)
    #print(dummies)
    #print(type(dummies))
    labels_matrix = dummies.values

    return labels_matrix


def labels_matrix_unmod(data):

    labels = data['sentiment_values']

    lables_float = labels.astype(float)

    labels_mult = (lables_float * 10).astype(int)
    labels_matrix = labels_mult.as_matrix()

    return labels_matrix

#%%

# load Training data matrix
#max_words = 56 # max no of words in your training data
max_words = 30 # max no of words in your training data
batch_size = 2000 # batch size for training
EMBEDDING_DIM = 100 # size of the word embeddings

path = 'C:\\Users\\samsu\\Downloads\\deepsentiment-master\\'
data_dir = path+'/Data'
all_data_path = path+'/Data/'
pred_path = path+'/Data/output_model/test_pred.csv'
gloveFile = path+'/Data/glove/glove_6B_100d.txt'

all_data = read_data(all_data_path)
weight_matrix, word_idx = load_embeddings(gloveFile)


maxSeqLength, avg_words, sequence_length = maxSeqLen(all_data)
numClasses = 10

train_data, test_data, dev_data = training_data_split(all_data, 0.8, data_dir)
train_x = tf_data_pipeline_nltk(train_data, word_idx, weight_matrix, maxSeqLength)
test_x = tf_data_pipeline_nltk(test_data, word_idx, weight_matrix, maxSeqLength)
val_x = tf_data_pipeline_nltk(dev_data, word_idx, weight_matrix, maxSeqLength)

# load labels data matrix
train_y = labels_matrix(train_data)
val_y = labels_matrix(dev_data)
test_y = labels_matrix(test_data)

# summarize size
print("Training data: ")
print(train_x.shape)
print(train_y.shape)


print("Classes: ")
print(np.unique(train_y.shape[1]))



#g=np.where(train_y[:,9]==1)
#q=np.where(train_y[:,0]==1)

num=5000
temp=np.empty(0)
temp_test=np.empty(0)
for i in range(num):
    for j in range(30):
        ind1=int((train_x[i,j]))
        temp=np.concatenate((temp,weight_matrix[ind1]))
        
num_test=1000
for i in range(num_test):
    for j in range(30):
        ind1=int((test_x[i,j]))
        temp_test=np.concatenate((temp_test,weight_matrix[ind1]))
        
        
##########################################################################

x=temp.reshape(-1,3000)
y=train_y[:num,:10]
x_test=temp_test.reshape(-1,3000)
y_test=test_y[:num_test,:10]
class myneuronnet():
    def __init__(self,n):
        #self.eta=eta
        #self.epoch=epoch
        self.InputNumColumns = 3000  ## columns
        self.OutputSize = 10
        self.HiddenUnits = 500  ## one layer with h units
        self.eta=0.1
        self.n = n  ## number of training examples, n
        self.w1=np.random.randn(self.InputNumColumns,self.HiddenUnits) #c by h
        self.w2 = np.random.randn(self.HiddenUnits, self.OutputSize) # h by o
        self.bs = np.random.randn(1, self.HiddenUnits) # o by h
        self.c=np.random.randn(1, self.OutputSize) # 1 by output
    
    def sigmoids(self,value,dS=False):
        if dS==True:
            return value*(1-value)
        else:
            return 1/(1+np.exp(-value))
        
    def softmax(self,value):
        #print("M is\n", M)
        expM = np.exp(value)
        #print("expM is\n", expM)
        SM=expM/np.sum(expM, axis=1)[:,None]
        #print("SM is\n",SM )
        return SM 

        
    def FeedForward(self,x):
        self.x=x
        #print(self.x.shape)
        #print(self.w1.shape)
        #print(self.bs.shape)
        self.y=y
        self.z1=np.dot(self.x,self.w1)+self.bs
        self.a_z1=self.sigmoids(self.z1,dS=False)
        self.z2=np.dot(self.a_z1,self.w2)+self.c
        y_hat=self.softmax(self.z2)
        print('y_hat (softmax) \n',y_hat)
        
        return y_hat
    

        
    def BackProp(self,x,y,y_hat):
            
            dl_dz2=y_hat-y
            print('dl_dz2 \n',dl_dz2)
            print('dl_dz2.shape \n',dl_dz2.shape)

            

            #dl/dh=w2*(y^)(1-y^)*(y^-y)
            dl_dh=np.dot(dl_dz2,self.w2.T)
            #dl/dh=(h1)(1-h1)*w2*(y^)(1-y^)*(y^-y)
            dl_dz1=self.sigmoids(self.a_z1,dS=True)*dl_dh
            #dl/dw2=h.T*(y^)(1-y^)*(y^-y)
            dl_dw2=np.dot(self.a_z1.T,dl_dz2)
            dl_dw1=np.dot(x.T,dl_dz1)
            
            #print('h',self.a_z1)
            #print('y_hat',y_hat)
            #print('error',error)
            #print('y_hat_d_error: ',y_hat_d_error)
            ##print('dl/dz2: \n',dl_dz2)
            #print('dl/dh: \n',dl_dh)
            #print('dl_dz1: \n',dl_dz1)
            #print('dl_dw2: \n',dl_dw2)
            #print('dl_dw1: \n',dl_dw1)
            
            
            #print('w2: ',self.w2)
            #print('w1: ',self.w1)
            #print('bs: ',self.bs)
            #print('c: ',self.c)
        

            self.w2=self.w2-self.eta*(dl_dw2/self.n)
            #print('w2_new: ',self.w2)
            self.w1=self.w1-self.eta*(dl_dw1/self.n)
            #print('w1_new: ',self.w1)
            #print('dl/db',np.sum(dl_dz1,axis=0))
            self.bs=self.bs-self.eta*np.mean(dl_dz1,axis=0)
            #print('bs_new: ',self.bs)
            #print('dl/dc',np.sum(y_hat_d_error,axis=0))
            self.c=self.c-self.eta*np.mean(dl_dz2,axis=0)
            #print('c_new: ',self.c)
            #print('y_hat: ',y_hat)
          
    def TrainNetwork(self, x, y):
        output = self.FeedForward(x)
        print('backprop \n')
        self.BackProp(x, y, output)
        return output
    
    def confusion_matrix_hand(self,y_true,y_pred):
        y_actu = pd.Series(y_true, name='Actual')
        y_pred = pd.Series(y_pred, name='Predicted')
        return pd.crosstab(y_actu, y_pred)
    
    def accuracy_hand(self,y_true,y_pred):
        count=0
        for i in range(len(y_true)):
            if y_true[i]==y_pred[i]:
                count+=1
        return count/len(y_true)*100
    
    def PredictNetwork(self, x, y):
        output = self.FeedForward(x)
        output_n=np.argmax(output,axis=1)
        y_test_n=np.argmax(y,axis=1)
        print('Accuracy score: ',self.accuracy_hand(y_test_n, output_n), ' %')
        print('Confusion matrix: ','\n',self.confusion_matrix_hand(y_test_n,output_n))
        
    
    def plot(self):
        fig1 = plt.figure()
        plt.plot(self.Aerrors)
        
        fig2 = plt.figure()
        plt.plot(self.Terrors,color='red')
            
            
N=20
mynn=myneuronnet(N)
TotalLoss=[]
AvgLoss=[]
Epochs=500


'''
for i in range(Epochs): 
    print("\nRUN:\n ", i)
    output=mynn.TrainNetwork(x, y)
   
    #print("The y is ...\n", y)
    
    ### old loss ###

    print("The output is: ", output)
    print("Total Loss:", .5*(np.sum(np.square(output-y))))
    TotalLoss.append( .5*(np.sum(np.square(output-y))))
    
    print("Average Loss:", .5*(np.mean(np.square((output-y)))))
    AvgLoss.append(.5*(np.mean(np.square((output-y)))))

    
    loss=np.mean(-y*np.log(output))
    TotalLoss.append(loss)
    AvgLoss.append(loss)
    

'''

for i in range(Epochs):
        
    no_of_batches = len(x) // N
        
    for j in range(no_of_batches):
        x_train = x[j*N:(j+1)*N, :]
        y_train = y[j*N:(j+1)*N, :] 
        print("\nRUN:\n ", i)
        output=mynn.TrainNetwork(x_train, y_train)
       
        '''
        print("The output is: ", output)
        print("Total Loss:", .5*(np.sum(np.square(output-y_train))))
        TotalLoss.append( .5*(np.sum(np.square(output-y_train))))
        
        print("Average Loss:", .5*(np.mean(np.square((output-y_train)))))
        AvgLoss.append(.5*(np.mean(np.square((output-y_train)))))
        '''
        loss=np.mean(-y_train*np.log(output))
        print('loss: \n:', loss)
        TotalLoss.append(loss)
        AvgLoss.append(loss)
        
mynn.PredictNetwork(x_test,y_test)


###################-output and vis----------------------    
#print("Total Loss List:", TotalLoss) 
import matplotlib.pyplot as plt

fig1 = plt.figure()
ax = plt.axes()
x1 = np.linspace(0, 10, Epochs*no_of_batches)
ax.plot(x1, TotalLoss)    

fig2 = plt.figure()
ax = plt.axes()
x1 = np.linspace(0, 10, Epochs*no_of_batches)
ax.plot(x1, AvgLoss)  

