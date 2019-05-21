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
#os.chdir('~/Documents/research/graph_bandit/code/')
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

user_feature_matrix=np.load(input_path+'user_feature_matrix_100.npy')
item_feature_matrix=np.load(input_path+'item_feature_matrix_500.npy')
rating_matrix=np.load(input_path+'normed_rating_matrix_100_user_500_movies.npy')
rating_matrix_mask=np.load(input_path+'rating_matrix_mask_100_user_500_movies.npy')
user_num=20
user_feature_matrix=user_feature_matrix[:user_num]
true_payoffs=np.dot(user_feature_matrix, item_feature_matrix.T)
# true_rating=rating_matrix[:user_num]
# mask=rating_matrix_mask[:user_num]
# true_rating_matrix=true_rating*mask
# true_payoffs=true_payoffs*(1-mask)
# true_payoffs=true_payoffs+true_rating_matrix
true_payoffs=np.round(true_payoffs, decimals=1)
# a=true_payoffs.ravel()
# plt.plot(a, '.')
# plt.show()

dimension=item_feature_matrix.shape[1]
item_num=item_feature_matrix.shape[0]
pool_size=10
iteration=3000
sigma=0.1# noise
delta=0.01# high probability
alpha=1# regularizer
alpha_2=0.1# edge delete CLUB
beta=0.1 # exploration for CLUB, SCLUB and GOB
thres=0.0
state=True # False for artificial dataset, True for real dataset

true_adj=rbf_kernel(user_feature_matrix, gamma=0.5)

noise_matrix=np.zeros((user_num, item_num))
user_seq=np.random.choice(range(user_num), size=iteration)
item_pool_seq=np.random.choice(range(item_num), size=(iteration, pool_size))

linucb_model=LINUCB(dimension, user_num, item_num, pool_size, item_feature_matrix, user_feature_matrix, true_payoffs, alpha, delta, sigma, state)
gob_model=GOB(dimension, user_num, item_num, pool_size, item_feature_matrix, user_feature_matrix, true_payoffs,true_adj, alpha, delta, sigma, beta, state)
lapucb_model=LAPUCB(dimension, user_num, item_num, pool_size, item_feature_matrix, user_feature_matrix, true_payoffs,true_adj, noise_matrix, alpha, delta, sigma, beta, thres, state)
lapucb_sim_model=LAPUCB_SIM(dimension, user_num, item_num, pool_size, item_feature_matrix, user_feature_matrix, true_payoffs,true_adj, noise_matrix,  alpha, delta, sigma, beta, thres, state)
club_model = CLUB(dimension, user_num, item_num, pool_size, item_feature_matrix, user_feature_matrix, true_payoffs, alpha, alpha_2, delta, sigma, beta, state)

linucb_regret, linucb_error, linucb_beta=linucb_model.run(user_seq, item_pool_seq, iteration)
gob_regret, gob_error, gob_beta, gob_graph=gob_model.run(user_seq, item_pool_seq, iteration)
lapucb_regret, lapucb_error, lapucb_beta, lapucb_graph=lapucb_model.run(user_seq, item_pool_seq, iteration)
lapucb_sim_regret, lapucb_sim_error, lapucb_sim_beta, lapucb_sim_graph=lapucb_sim_model.run(user_seq, item_pool_seq, iteration)
club_regret, club_error, club_cluster_num, club_beta=club_model.run(user_seq, item_pool_seq, iteration)


plt.figure(figsize=(5,5))
plt.plot(linucb_regret,'-.', label='LinUCB')
plt.plot(gob_regret, label='GOB')
plt.plot(lapucb_regret, '-*', markevery=0.1, label='G-UCB')
plt.plot(lapucb_sim_regret, '-s', markevery=0.1, label='G-UCB SIM')
plt.plot(club_regret, label='CLUB')
plt.ylabel('Cumulative Regret', fontsize=12)
plt.xlabel('Time', fontsize=12)
plt.legend(loc=2, fontsize=10)
plt.tight_layout()
plt.savefig(path+'regret_movielens_user_num_%s_item_num_%s'%(user_num, item_num)+'.png', dpi=300)
plt.savefig(path+'regret_movielens_user_num_%s_item_num_%s'%(user_num, item_num)+'.eps', dpi=300)
plt.show()


plt.figure(figsize=(5,5))
plt.plot(linucb_error,'-.', label='LinUCB')
plt.plot(gob_error, label='GOB')
plt.plot(lapucb_error, '-*', markevery=0.1, label='G-UCB')
plt.plot(lapucb_sim_error, '-s', markevery=0.1, label='G-UCB SIM')
plt.plot(club_error, label='CLUB')
plt.ylabel('Error', fontsize=12)
plt.xlabel('Time', fontsize=12)
plt.legend(loc=1, fontsize=10)
plt.tight_layout()
plt.savefig(path+'error_movielens_user_num_%s_item_num_%s'%(user_num, item_num)+'.png', dpi=300)
plt.savefig(path+'error_movielens_user_num_%s_item_num_%s'%(user_num, item_num)+'.eps', dpi=300)
plt.show()


