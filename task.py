#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 21:11:58 2018

@author: abdullah
"""



from __future__ import absolute_import, division, print_function
import numpy as np
import pandas as pd
import re
import nltk
import copy
import codecs
import glob
import multiprocessing
import os
import gensim.models.word2vec as w2v
import sklearn.manifold
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

# PLEASE CHANGE FILENAME AS REQUIRED

dataset = pd.read_csv('Sentence Completion System/taskData.csv', header=None, encoding='latin-1' )



X = dataset.iloc[:, 5:6].values




corpus = []
for i in range(0,len(X)):
    review = re.sub('[^a-zA-Z]', ' ', dataset[5][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    


a =copy.deepcopy(len(X))  

i=0
while i<a:
    if len(corpus[i].split()) < 5:
            del(corpus[i])
            a=a-1
    else:
        i=i+1
            
    
            
            
            
         
        
       

    
corpus2=[]

for i  in  range(0,len(corpus)):
    corpus2.append(corpus[i].split(' '))
    
    
    
    
df = pd.DataFrame(corpus2)
print (df)

X = df.iloc[:, 0:4].values
y = df.iloc[:, 4:5].values

ip = []

for i in range(0,len(corpus)):
    ip.append(X[i][0]+" "+X[i][1]+" "+X[i][2]+" "+X[i][3]) 
    
op = []

for i in range(0,len(corpus)):
    op.append(y[i]) 

    
    

    

    
for i in range(0,len(corpus)):
    leng = len(corpus2[i])
    for j in range(0,leng-5):
        corpus2[i].pop()
        


corpus3 = copy.deepcopy(corpus2)


for i in range(0,len(corpus)):
    leng = len(corpus2[i])
    for j in range(0,leng-4):
        corpus2[i].pop()

for i in range(0,len(corpus)):
    for j in range(0,4):
        del corpus3[i][0]
    
    

    



        
        

     

import csv
data = corpus

out = csv.writer(open("vocab.csv","w"), delimiter=',',quoting=csv.QUOTE_ALL)
out.writerow(data)

book_filenames = sorted(glob.glob("vocab.csv"))
print("Found books:")
book_filenames
corpus_raw = u""
for book_filename in book_filenames:
    print("Reading '{0}'...".format(book_filename))
    with codecs.open(book_filename, "r", "latin-1") as book_file:
        corpus_raw += book_file.read()
    print("Corpus is now {0} characters long".format(len(corpus_raw)))
    print()
    
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
raw_sentences = tokenizer.tokenize(corpus_raw)

def sentence_to_wordlist(raw):
    clean = re.sub("[^a-zA-Z]"," ", raw)
    words = clean.lower()
    words = clean.split()
    ps = PorterStemmer()
    words = [ps.stem(word) for word in words if not word in set(stopwords.words('english'))]
    
    
    return words

sentences = []
for raw_sentence in raw_sentences:
    if len(raw_sentence) > 0:
        sentences.append(sentence_to_wordlist(raw_sentence))



token_count = sum([len(sentence) for sentence in sentences])
print("The book corpus contains {0:,} tokens".format(token_count))

num_features = 1000

min_word_count = 3


num_workers = multiprocessing.cpu_count()


context_size = 77


downsampling = 1e-3


seed = 1
tweet2vec = w2v.Word2Vec(
    sg=1,
    seed=seed,
    workers=num_workers,
    size=num_features,
    min_count=min_word_count,
    window=context_size,
    sample=downsampling
)
tweet2vec.build_vocab(sentences)
print("Word2Vec vocabulary length:", len(tweet2vec.wv.vocab))
tweet2vec.train(sentences, total_examples=tweet2vec.corpus_count, epochs=1)
if not os.path.exists("trained"):
    os.makedirs("trained")
tweet2vec.save(os.path.join("trained", "tweet2vec.w2v"))
tweet2vec = w2v.Word2Vec.load(os.path.join("trained", "tweet2vec.w2v"))

tsne = sklearn.manifold.TSNE(n_components=2, random_state=0)
all_word_vectors_matrix = tweet2vec.wv.vectors

all_word_vectors_matrix_2d = tsne.fit_transform(all_word_vectors_matrix)
points = pd.DataFrame(
    [
        (word, coords[0], coords[1])
        for word, coords in [
            (word, all_word_vectors_matrix_2d[tweet2vec.wv.vocab[word].index])
            for word in tweet2vec.wv.vocab
        ]
    ],
    columns=["word", "x", "y"]
)
    
points.head(10)


    

    
            


corpusf=np.zeros((len(corpus),8))
corpusfo=np.zeros((len(corpus),2))

pxn = points.iloc[:, 0].values
px1 = points.iloc[:, 1 ].values
py1 = points.iloc[:, 2].values

for i in range(0,len(corpus)):
    for j in range(0,len(points)):
        if corpus2[i][0]==pxn[j]:
            corpusf[i,0]=px1[j]
            corpusf[i,1]=py1[j]
        if corpus2[i][1]==pxn[j]:
            corpusf[i,2]=px1[j]
            corpusf[i,3]=py1[j]
        if corpus2[i][2]==pxn[j]:
            corpusf[i,4]=px1[j]
            corpusf[i,5]=py1[j]
        if corpus2[i][3]==pxn[j]:
            corpusf[i,6]=px1[j]
            corpusf[i,7]=py1[j]
        if corpus3[i][0]==pxn[j]:
            corpusfo[i,0]=px1[j]
            corpusfo[i,1]=py1[j]

a=copy.deepcopy(len(corpus))

i=0            
while i<a:
    if corpusf[i][0]==0 or corpusf[i][1]==0 or corpusf[i][2]==0 or corpusf[i][3]==0 or corpusf[i][4]==0 or corpusf[i][5]==0 or corpusf[i][6]==0 or corpusf[i][7]==0 or corpusfo[i][0]==0 or corpusfo[i][1]==0:
        corpusf=np.delete(corpusf, np.s_[i], axis=0) 
        corpusfo=np.delete(corpusfo, np.s_[i], axis=0) 
        a=a-1
    else:
        i=i+1
            
       
        
# ------------------------------------
# The Neural Network Part 
            


# Splitting the dataset into the Training set and Test set
        
X_train, X_test, y_train, y_test = train_test_split(corpusf, corpusfo, test_size = 0.2, random_state = 0)


# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 5, init = 'uniform', activation = 'relu', input_dim = 8))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 5, init = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 2, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 100, nb_epoch = 100)


# Predicting the Test set results
y_pred = classifier.predict(X_test)




#----------------------------------END------------------
