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
# Extracting TF_IDF features for each review
N = 1000 # total number of reviews
feature = np.zeros([1000,200])
df = np.sum(sentence_vectors,axis=0) # number of reviews containing the respective frequent word
for i in range(1000):
    for j in range(200):
        feature[i, j] = (float)(sentence_vectors[i,j]*np.log(N/df[j]))

#%%
def pca(data, k):
    def eigen_sort(value, vector):
        idx = value.argsort()[::-1]
        eigenValues = value[idx]
        eigenVectors = vector[:,idx]
        return (eigenValues, eigenVectors)

    def final_projection(eigen_matrix, x, k):
        u = eigen_matrix[:, :k]
        y = np.matmul(x, u)
        return y

    cov_data = np.cov(np.transpose(data))
    eig_val, eig_vector = np.linalg.eig(cov_data)
    eig_vals,eig_vectors=eigen_sort(eig_val,eig_vector)

    return final_projection(eig_vectors,data,k)

# Reducing our data to have 10 Dimensions using PCA
new_feature = pca(feature, 10)

#%%

