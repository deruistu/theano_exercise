'''
Created on Mar 4, 2016

@author: lqy
'''

import csv
import itertools
import operator
import numpy as np
import nltk
import sys
import os
import time
import theano.tensor as T
from RNNNumpy import RNNNumpy

# in the data sources, one line represent one comment. one comment may has many sentences.
vocabulary_size = 8000
sentence_start = "sentence starts"
sentence_end = "sentence end"
unknown_token = "UNKNOWN_TOKEN"
#sentences = []
print "readling resources files"

# the first step to preprocess the corpus data
with open("/Users/lqy/Documents/rnn-tutorial-rnnlm-master/data/reddit-comments-2015-08.csv","rb") as f:
    reader = csv.reader(f, skipinitialspace=True)
    reader.next()
    sentences = itertools.chain(*[nltk.sent_tokenize(x[0].decode('utf-8').lower()) for x in reader])
    sentences = ["%s %s %s" % (sentence_start, x, sentence_end) for x in sentences] # sentence combined

print "parased %d sentences" %len(sentences)

#tokenized the sentences to words
tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]

#count word frequency, return the list of unique words
word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
print "found %d unique words" %len(word_freq)

#get the most common words and build index_to_words and words_to_index
# vocab is a two element tuple, [word,frequency]
vocab = word_freq.most_common(vocabulary_size-1) # the result is sorted by the frequency
print len(vocab)

#build index to words
index_to_word = [x[0] for x in vocab]
index_to_word.append(unknown_token)

# build a dictionary for word_to_index
word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])

print "the least frequecy word is %s, the frequecy is %d" %(vocab[-1][0], vocab[-1][1])
print "the most frequecy word is %s, the frequecy is %d" %(vocab[0][0],vocab[0][1])

#replace all words that not in our vocabulary as unknown_token
for i, sent in enumerate(tokenized_sentences):
    tokenized_sentences[i] = [w if w in index_to_word else unknown_token for w in sent]

print "Example sentence: %s" %sentences[0]
print "Example tokenized sentence: %s" %tokenized_sentences[0]

x_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
y_train = np.asarray([[word_to_index[w] for w in sent [1:]] for sent in tokenized_sentences])

print "x_train example for sentence %s, its vector is %s" %(tokenized_sentences[0], x_train[0])
print "y_train example for sentence %s, its vector is %s" %(tokenized_sentences[0], y_train[0])

print "start prediction"
np.random.seed(10)
model = RNNNumpy(vocabulary_size)
print "-------new start----------"
s = np.zeros(100)
word_dim = 8000
hidden_dim = 100
U = np.random.uniform(-np.sqrt(1./8000), np.sqrt(1./8000), (8000, 100))
V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim))
W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))
snew = np.tanh(U[:,5] + W.dot(s))
print "s's shape: ", snew.shape()
print "s : ", snew  




    

