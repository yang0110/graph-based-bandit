import numpy as np 
import networkx as nx
import seaborn as sns
sns.set_style("white")
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import Normalizer, MinMaxScaler
from scipy.sparse import csgraph 
import scipy
import os 
os.chdir('C:/Kaige_Research/Code/graph_bandit/code/')
from sklearn import datasets
from linucb import LINUCB
from gob import GOB 
from colin import COLIN
from lapucb import LAPUCB
from lapucb_sim import LAPUCB_SIM
from sclub import SCLUB
from club import CLUB
from utils import *
from sklearn.decomposition import NMF
from Recommender.matrix_factor_model import ProductRecommender
input_path='../processed_data/movielens/'
path='../bandit_results/movielens/'

# rate_matrix=np.load(input_path+'rating_matrix_100_user_500_movies.npy')
# true_payoffs=rate_matrix/np.max(rate_matrix)
# true_payoffs[true_payoffs==0]=np.nan
# nmf_model=ProductRecommender()
# nmf_model.fit(true_payoffs,5)
# user_feature_matrix, item_feature_matrix=nmf_model.get_models()

# # nmf_model=NMF(n_components=5,random_state=2019)
# # user_feature_matrix=nmf_model.fit_transform(true_payoffs)
# # item_feature_matrix=nmf_model.components_.T

# np.save(input_path+'user_feature_matrix_100.npy', user_feature_matrix)
# np.save(input_path+'item_feature_matrix_500.npy', item_feature_matrix)

####
user_feature_matrix=np.load(input_path+'user_feature_matrix_100.npy')
item_feature_matrix=np.load(input_path+'item_feature_matrix_500.npy')
user_feature_matrix=user_feature_matrix[:30]
#user_feature_matrix=Normalizer().fit_transform(user_feature_matrix)
#item_feature_matrix=Normalizer().fit_transform(item_feature_matrix)
true_payoffs=np.dot(user_feature_matrix, item_feature_matrix.T)


user_num=true_payoffs.shape[0]
dimension=item_feature_matrix.shape[1]
item_num=item_feature_matrix.shape[0]
pool_size=10
iteration=1000
sigma=0.1# noise
delta=0.01# high probability
alpha=1# regularizer
alpha_2=0.2# edge delete CLUB
beta=0.01 # exploration for CLUB, SCLUB and GOB
thres=0.7
k=3
true_adj=rbf_kernel(user_feature_matrix, gamma=0.5)
true_adj[true_adj<=thres]=0.0
true_normed_adj=true_adj/true_adj.sum(axis=0,keepdims=1)
true_lap=csgraph.laplacian(true_adj)
true_normed_lap=csgraph.laplacian(true_adj, normed=False)
true_adj_binary=true_adj.copy()
true_adj_binary[true_adj_binary>0]=1
true_lap_binary=np.diag(np.sum(true_adj_binary, axis=1))-true_adj_binary

adj=np.ones((user_num, user_num))
normed_lap=csgraph.laplacian(adj, normed=True)

noise_matrix=np.zeros((user_num, item_num))
user_seq=np.random.choice(range(user_num), size=iteration)
item_pool_seq=np.random.choice(range(item_num), size=(iteration, pool_size))

linucb_model=LINUCB(dimension, user_num, item_num, pool_size, item_feature_matrix, user_feature_matrix, true_payoffs, 1, delta, sigma)
lapucb_model=LAPUCB(dimension, user_num, item_num, pool_size, item_feature_matrix, user_feature_matrix, true_payoffs, noise_matrix, normed_lap, 0.1, delta, sigma, beta, 0.0)
lapucb_sim_model=LAPUCB_SIM(dimension, user_num, item_num, pool_size, item_feature_matrix, user_feature_matrix, true_payoffs, noise_matrix, normed_lap, 0.1, delta, sigma, beta, 0.0)


linucb_regret, linucb_error, linucb_beta=linucb_model.run(user_seq, item_pool_seq, iteration)
lapucb_regret, lapucb_error, lapucb_beta=lapucb_model.run(user_seq, item_pool_seq, iteration)
lapucb_sim_regret, lapucb_sim_error, lapucb_sim_beta=lapucb_sim_model.run(user_seq, item_pool_seq, iteration)


plt.figure(figsize=(5,5))
plt.plot(linucb_regret,'-.', label='LinUCB')
plt.plot(lapucb_regret, '-*', markevery=0.1, label='G-UCB')
plt.plot(lapucb_sim_regret, '-*', markevery=0.1, label='G-UCB SIM')
plt.ylabel('Cumulative Regret', fontsize=12)
plt.xlabel('Time', fontsize=12)
plt.legend(loc=2, fontsize=10)
plt.tight_layout()
plt.show()

plt.figure(figsize=(5,5))
plt.plot(linucb_error,'-.', label='LinUCB')
plt.plot(lapucb_error, '-*', markevery=0.1, label='G-UCB')
plt.plot(lapucb_sim_error, '-*', markevery=0.1, label='G-UCB SIM')
plt.ylabel('Error', fontsize=12)
plt.xlabel('Time', fontsize=12)
plt.legend(loc=1, fontsize=10)
plt.tight_layout()
plt.show()

