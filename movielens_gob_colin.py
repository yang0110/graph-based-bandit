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
from lapucb_sim import LAPUCB_SIM
from utils import *
from sklearn.decomposition import NMF
from Recommender.matrix_factor_model import ProductRecommender
input_path='../processed_data/movielens/'
path='../bandit_results/movielens/'

####
user_feature_matrix=np.load(input_path+'user_feature_matrix_100.npy')
item_feature_matrix=np.load(input_path+'item_feature_matrix_500.npy')
user_feature_matrix=user_feature_matrix[:20]
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
thres=0.0
k=3
true_adj=rbf_kernel(user_feature_matrix, gamma=0.5)
true_adj[true_adj<=thres]=0.0
true_normed_adj=true_adj/true_adj.sum(axis=0,keepdims=1)
true_lap=csgraph.laplacian(true_adj)
true_normed_lap=csgraph.laplacian(true_adj, normed=True)
true_adj_binary=true_adj.copy()
true_adj_binary[true_adj_binary>0]=1
true_lap_binary=np.diag(np.sum(true_adj_binary, axis=1))-true_adj_binary

adj=np.ones((user_num, user_num))
normed_lap=csgraph.laplacian(adj, normed=True)

noise_matrix=np.zeros((user_num, item_num))
user_seq=np.random.choice(range(user_num), size=iteration)
item_pool_seq=np.random.choice(range(item_num), size=(iteration, pool_size))

linucb_model=LINUCB(dimension, user_num, item_num, pool_size, item_feature_matrix, user_feature_matrix, true_payoffs, alpha, delta, sigma)

gob_model=GOB(dimension, user_num, item_num, pool_size, item_feature_matrix, user_feature_matrix, true_payoffs, true_normed_lap, 0.1, delta, sigma, beta)

colin_model=COLIN(dimension, user_num, item_num, pool_size, item_feature_matrix, user_feature_matrix, true_payoffs, true_adj, 0.05, delta, sigma, beta)

lapucb_sim_model=LAPUCB_SIM(dimension, user_num, item_num, pool_size, item_feature_matrix, user_feature_matrix, true_payoffs, noise_matrix, normed_lap, 0.1, delta, sigma, beta, 0.0)


linucb_regret, linucb_error, linucb_beta=linucb_model.run(user_seq, item_pool_seq, iteration)
gob_regret, gob_error, gob_beta=gob_model.run(user_seq, item_pool_seq, iteration)
colin_regret, colin_error, colin_beta=colin_model.run(user_seq, item_pool_seq, iteration)
lapucb_sim_regret, lapucb_sim_error, lapucb_sim_beta=lapucb_sim_model.run(user_seq, item_pool_seq, iteration)


plt.figure(figsize=(5,5))
plt.plot(linucb_regret,'-.', label='LinUCB')
plt.plot(gob_regret, label='GOB')
plt.plot(colin_regret, label='CoLin')
plt.plot(lapucb_sim_regret, '-*', markevery=0.1, label='G-UCB SIM')
plt.ylabel('Cumulative Regret', fontsize=12)
plt.xlabel('Time', fontsize=12)
plt.legend(loc=2, fontsize=10)
plt.tight_layout()
plt.show()


plt.figure(figsize=(5,5))
plt.plot(linucb_error,'-.', label='LinUCB')
plt.plot(gob_error, label='GOB')
plt.plot(colin_error, label='CoLin')
plt.plot(lapucb_sim_error, '-*', markevery=0.1, label='G-UCB SIM')
plt.ylabel('Error', fontsize=12)
plt.xlabel('Time', fontsize=12)
plt.legend(loc=1, fontsize=10)
plt.tight_layout()
plt.show()


