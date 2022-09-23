import pandas as pd
import numpy as np
import tensorflow as tf
import re
import codecs
import os
from nltk.tokenize import RegexpTokenizer

import os
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
#import utility_functions as uf
from keras.models import model_from_json
from keras.models import load_model
from nltk.tokenize import RegexpTokenizer

from itertools import islice

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


