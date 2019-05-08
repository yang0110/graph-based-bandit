import numpy as np 
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import Normalizer, MinMaxScaler
from scipy.sparse import csgraph 
import scipy
import os 
os.chdir('/Users/KGYaNG/Documents/research/graph_bandit/code/')
from sklearn import datasets
from linucb import LINUCB
from gob import GOB 
from lapucb import LAPUCB
from lapucb_sim import LAPUCB_SIM
from sclub import SCLUB
from club import CLUB
from utils import *
input_path='../processed_data/movielens/'
path='../bandit_results/movielens/'

rate_matrix=np.load(input_path+'rating_matrix_30_user_100_movies.npy')
true_payoffs=rate_matrix/np.max(rate_matrix)
user_feature_matrix=np.load(input_path+'user_feature_matrix_30.npy')
item_feature_matrix=np.load(input_path+'item_feature_matrix_100.npy')
user_feature_matrix=Normalizer().fit_transform(user_feature_matrix)
item_feature_matrix=Normalizer().fit_transform(item_feature_matrix)


user_num=true_payoffs.shape[0]
dimension=item_feature_matrix.shape[1]
item_num=item_feature_matrix.shape[0]
pool_size=10
iteration=2000
sigma=0.1# noise
delta=0.1# high probability
alpha=1# regularizer
alpha_2=0.01# edge delete CLUB
beta=0.3 # exploration for CLUB, SCLUB and GOB

lap=np.identity(user_num)
normed_lap=lap

noise_matrix=np.zeros((user_num, item_num))
user_seq=np.random.choice(range(user_num), size=iteration)
item_pool_seq=np.random.choice(range(item_num), size=(iteration, pool_size))

linucb_model=LINUCB(dimension, user_num, item_num, pool_size, item_feature_matrix, user_feature_matrix, true_payoffs, alpha, delta, sigma)

gob_model=GOB(dimension, user_num, item_num, pool_size, item_feature_matrix, user_feature_matrix, true_payoffs, lap, alpha, delta, sigma, beta)
lapucb_model=LAPUCB(dimension, user_num, item_num, pool_size, item_feature_matrix, user_feature_matrix, true_payoffs, noise_matrix, normed_lap, alpha, delta, sigma)
lapucb_sim_model=LAPUCB_SIM(dimension, user_num, item_num, pool_size, item_feature_matrix, user_feature_matrix, true_payoffs, noise_matrix, normed_lap, alpha, delta, sigma)

club_model = CLUB(dimension, user_num, item_num, pool_size, item_feature_matrix, user_feature_matrix, true_payoffs,normed_lap, alpha, alpha_2, delta, sigma, beta)
sclub_model = SCLUB(dimension, user_num, item_num, pool_size, item_feature_matrix, user_feature_matrix, true_payoffs,normed_lap, alpha, delta, sigma, beta)

linucb_regret, linucb_error, linucb_beta=linucb_model.run(user_seq, item_pool_seq, iteration)
gob_regret, gob_error, gob_beta=gob_model.run(user_seq, item_pool_seq, iteration)
lapucb_regret, lapucb_error, lapucb_beta=lapucb_model.run(user_seq, item_pool_seq, iteration)
lapucb_sim_regret, lapucb_sim_error, lapucb_sim_beta=lapucb_sim_model.run(user_seq, item_pool_seq, iteration)
club_regret, club_error,club_graph_error, club_cluster_num, club_beta=club_model.run(user_seq, item_pool_seq, iteration)
sclub_regret, sclub_error,sclub_graph_error, sclub_cluster_num, sclub_beta=sclub_model.run(user_seq, item_pool_seq, iteration)


plt.figure(figsize=(5,5))
plt.plot(linucb_regret, label='LINUCB')
plt.plot(gob_regret, label='GOB')
plt.plot(lapucb_regret, label='LAPUCB')
plt.plot(lapucb_sim_regret, label='LAPUCB SIM')
plt.plot(club_regret, label='CLUB')
plt.plot(sclub_regret, label='SCLUB')
plt.ylabel('Cumulative Regret', fontsize=12)
plt.xlabel('Time', fontsize=12)
plt.legend(loc=2, fontsize=10)
plt.tight_layout()
plt.show()
 



