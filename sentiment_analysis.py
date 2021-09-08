import nltk
import heapq
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

#%%
f = open('./data/movieReviews1000.txt',"r")
text = f.readlines()
  
orig_labels = []
reviews = []
for line in text:
    review, label = line.split('\t')
    orig_labels.append(int(label.strip()))
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
# Gaussian Mixture Models (2 centres)
#finding initial centres for the kernels 
kmeans = KMeans(n_clusters=2, random_state=0).fit(new_feature)
labels = kmeans.labels_
cluster_centers = kmeans.cluster_centers_

mean1 = cluster_centers[0, :]
mean2 = cluster_centers[1, :]
mean_list = [mean1, mean2]

def ClusterIndicesNumpy(clustNum, labels_array): #numpy 
    return np.where(labels_array == clustNum)[0]

data1 = np.transpose(new_feature[ClusterIndicesNumpy(0, kmeans.labels_)])
data2 = np.transpose(new_feature[ClusterIndicesNumpy(1, kmeans.labels_)])

tdata1 = np.transpose(new_feature[ClusterIndicesNumpy(0, np.array(orig_labels))])
tdata2 = np.transpose(new_feature[ClusterIndicesNumpy(1, np.array(orig_labels))])

tdata1=np.transpose(tdata1)
tdata2=np.transpose(tdata2)

plt.figure()
plt.scatter(data1[0,:],data1[1,:],c='c',label='Kmean',marker='o')
plt.scatter(data2[0,:],data2[1,:],c='m',label='Kmean',marker='o')
plt.grid(True)
plt.legend(loc='upper right')

cov1 = np.cov(data1)
cov2 = np.cov(data2)
cov_list = [cov1, cov2]

alpha1 = data1.shape[1]/new_feature.shape[0] # fraction of negative reviews
alpha2 = 1 - alpha1 # fraction of positive reviews
alpha_list = [alpha1, alpha2]

def likelihood(data_array_transposed, alpha_list, cov_list, mean_list):
    j = len(alpha_list)
    n = data_array_transposed.shape[1]
    weighted_normal=[]
    like = 0
    for i in range(n):
        for k in range(j):
            weighted_normal.append(alpha_list[k]*(multivariate_normal.pdf(data_array_transposed[:, i], mean=mean_list[k], cov=cov_list[k])))
        a = np.array(weighted_normal)
        l = np.log(np.sum(a))
        weighted_normal = []
        like = like+l
    return like

cov1d = np.diag(np.diag(cov1))
cov2d = np.diag(np.diag(cov2))
covd_list = [cov1d, cov2d]
L_initial = likelihood(np.transpose(new_feature), alpha_list, covd_list, mean_list)

def maximization(data_array_transposed, alpha_list, cov_list, mean_list, f):
    j = len(alpha_list)
    n = data_array_transposed.shape[1]
    m = data_array_transposed.shape[0]
    weighted_normal = np.zeros((n, j))
    gama = np.zeros((n,j))
    for i in range(n):
        for k in range(j):
            x = (alpha_list[k]*(multivariate_normal.pdf(data_array_transposed[:,i], mean=mean_list[k], cov=cov_list[k])))
            weighted_normal[i,k] = x
    
    denominator = np.sum(weighted_normal, axis=1) 
    for i in range(n):
        for k in range(j):
            gama[i,k] = weighted_normal[i,k]/denominator[i]
    mean_deviation_denomintor = np.sum(gama, axis=0)
    alpha = mean_deviation_denomintor/n
    #updated weights
    alpha_list_updated=alpha.tolist()
    
    #mean calculation
    #numerator
    sum_mean = []
    for k in range(j):
        e = np.zeros(m)
        for i in range(n):
            v = (float)(gama[i, k])
            if e.shape == data_array_transposed[0:m, i].shape:
                e = e + v*data_array_transposed[:, i]
            else:
                print('gfs')
        sum_mean.append(e)
        
    mean_sum = np.array(sum_mean)
    mean_new = []
    for k in range(j):
        mean_new.append(mean_sum[k]/mean_deviation_denomintor[k])
        
    #standard dev
    sum_mean1 = []
    for k in range(j):
        e = np.zeros([m, m])
        for i in range(n):
            v = (float)(gama[i, k])
            e = e + v * np.outer(data_array_transposed[:,i] - mean_new[k], data_array_transposed[:,i] - mean_new[k])
        sum_mean1.append(e)
    
    mean_sum1=np.array(sum_mean1)
    
    sd_new = []
    for k in range(j):
        if(f == 1):
            sd_new.append(np.diag(np.diag(mean_sum1[k]/mean_deviation_denomintor[k])))
        else:
            sd_new.append(mean_sum1[k]/mean_deviation_denomintor[k])

    return sd_new,mean_new,alpha_list_updated

#%%
def classification(sigma, alpha, mean_, data):
    n = data.shape[0]
    data_transposed = np.transpose(data)
    data0 = []
    data1 = []
    for i in range(n):
        w0 = multivariate_normal.pdf(data_transposed[:,i], mean=mean_[0], cov=sigma[0])
        w1 = multivariate_normal.pdf(data_transposed[:,i], mean=mean_[1], cov=sigma[1])
        if np.log(w0) >= np.log(w1):
            data0.append(data_transposed[:, i])
        else:
            data1.append(data_transposed[:, i])
    
    return data0, data1

#%%
def plot(sigma, alpha, mean_, data, j):
    data_transposed = np.transpose(data)
    data1 = []
    data2 = []
    for i in range(1000):
        w1 = multivariate_normal.pdf(data_transposed[:,i],mean=mean_[0],cov=sigma[0])
        w2 = multivariate_normal.pdf(data_transposed[:,i],mean=mean_[1],cov=sigma[1])
        if np.log(w1) >= np.log(w2):
            data1.append(data_transposed[:,i])
        else:
            data2.append(data_transposed[:,i])
    s = "EM Iteration =" + str(j)
    data1 = np.array(data1)
    data2 = np.array(data2)
    plt.figure()
    plt.scatter(data1[:,0],data1[:,1],c='c',marker='o',label=s)
    plt.scatter(data2[:,0],data2[:,1],c='m',marker='o',label=s)
    plt.scatter(mean_[0][0],mean_[0][1],c='k',marker='D',label="Mean(Blue Dots)",s=100)
    plt.scatter(mean_[1][0],mean_[1][1],c='y',marker='D',label="Mean(Purple Dots)",s=100)
    plt.grid(True)
    plt.legend(loc='upper right')    

#%%
i = 1
iteration_diagonal = [1]
likelihood_diagonal = [L_initial]
k = [1]
sd_updated, mean_updated, alpha_list_updated1 = maximization(np.transpose(new_feature),alpha_list,covd_list,mean_list,1)

sigma_diag = []
mean_diag = []
alpha_diag = []
while(1):
    check = 0
    sd_updated,mean_updated,alpha_list_updated1 = maximization(np.transpose(new_feature),alpha_list_updated1,sd_updated,mean_updated,1)
    L1 = likelihood(np.transpose(new_feature), alpha_list_updated1, sd_updated, mean_updated)
    plot(sd_updated, alpha_list_updated1, mean_updated, new_feature, i)
    i = i+1
    iteration_diagonal.append(i)
    likelihood_diagonal.append(L1)
    diff = (likelihood_diagonal[i-1]-likelihood_diagonal[i-2])
    if i >= 10:
        for x in range(i,i-10,-1):
            diff1 = (likelihood_diagonal[x-1]-likelihood_diagonal[x-2])
            if diff1 <= 1:
                check += 1
        if check == 9:
            sigma_diag = sd_updated
            mean_diag = mean_updated
            alpha_diag = alpha_list_updated1
            break
        else:
            check = 0
    if i >= 15:
        sigma_diag = sd_updated
        mean_diag = mean_updated
        alpha_diag = alpha_list_updated1
        break

#%%
s = "Number of Iteration ="+str(i-1)
plt.figure()
plt.plot(iteration_diagonal, likelihood_diagonal, marker='*', c='g', label=s)
plt.xlabel("Iteration")
plt.ylabel("Log Likelihood")
plt.legend(loc='upper right')
plt.title("Number Of Gaussian=2, Train Data, Diagonal Covariance")
plt.show()

#classification
c1, c2 = classification(sd_updated, alpha_list_updated1, mean_updated, tdata1)
k1, k2 = classification(sd_updated, alpha_list_updated1, mean_updated, tdata2)

accuracy=(len(c2)+len(k1))/1000
print("The classification Accuracy is {}".format (accuracy))

