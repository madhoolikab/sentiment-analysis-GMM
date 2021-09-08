import nltk
import heapq
import numpy as np

#%%
f=open('./data/movieReviews1000.txt',"r")
text=f.readlines()
  
labels = []
reviews = []
for line in text:
    review, label = line.split('\t')
    labels.append(int(label.strip()))
    reviews.append(review.strip())

#%%
# Finding total frequency of all words over all the reviews
wordfreq = {}
for review in reviews:
    tokens = nltk.word_tokenize(review)
    for token in tokens:
        if token not in wordfreq.keys():
            wordfreq[token] = 1
        else:
            wordfreq[token] += 1

#%%
most_freq = heapq.nlargest(342, wordfreq, key=wordfreq.get)
sentence_vectors = []
for sentence in reviews:
    sentence_tokens = nltk.word_tokenize(sentence)
    sent_vec = []
    for token in most_freq:
        if token in sentence_tokens:
            sent_vec.append(1)
        else:
            sent_vec.append(0)
    sentence_vectors.append(sent_vec)

sentence_vectors = np.asarray(sentence_vectors)

#%%





