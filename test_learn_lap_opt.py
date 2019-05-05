import numpy as np 
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import Normalizer, MinMaxScaler
from scipy.sparse import csgraph 
import scipy
import os 
from sklearn import datasets
#os.chdir('Documents/research/graph_bandit/code/')
from bandit_model import *
from adaptive_graph_bandit_model import *
from utils import *
from learn_lap_optimization_algorithm import *
path='../results/'

user_num=10
dimension=5
item_num=100
pool_size=10
iteration=1000
I_user_num=np.identity(user_num)
I_dimension=np.identity(dimension)

sigma=0.1 # noise
delta=0.1# high probability
alpha=0.1 # regularizer
beta=0.1 # regularizer
mu=0.1 ## step size
lambda_=0.1 ## step size

user_array=np.random.choice(range(user_num), size=iteration)
item_pool_array=np.random.choice(range(item_num), size=(iteration, pool_size))
item_feature_matrix=Normalizer().fit_transform(np.random.normal(size=(item_num, dimension)))

node_f, _=datasets.make_blobs(n_samples=user_num, n_features=dimension, centers=5, cluster_std=0.1, shuffle=False, random_state=2019)
node_f=Normalizer().fit_transform(node_f)
adj=rbf_kernel(node_f)
lap=np.diag(np.sum(adj, axis=1))-adj
cov=np.linalg.pinv(lap)
user_feature_matrix=np.random.multivariate_normal(np.zeros(user_num), cov, size=dimension).T
user_feature_matrix=Normalizer().fit_transform(user_feature_matrix)
true_payoffs=np.dot(user_feature_matrix, item_feature_matrix.T)
noisy_payoffs=true_payoffs+np.random.normal(scale=sigma, size=(user_num, item_num))

L, Y, lap_error, signal_error=learn_lap_and_denoise_signal(noisy_payoffs, true_payoffs, max_iteration, lap, alpha, beta)



plt.figure(figsize=(5,5))
plt.plot(signal_error)
plt.ylabel('signal_error', fontsize=12)
plt.xlabel('Time', fontsize=12)
plt.legend(loc=0, fontsize=12)
plt.tight_layout()
plt.show()

plt.figure(figsize=(5,5))
plt.plot(lap_error)
plt.ylabel('lap Error', fontsize=12)
plt.xlabel('Time', fontsize=12)
plt.legend(loc=0, fontsize=12)
plt.tight_layout()
plt.show()






