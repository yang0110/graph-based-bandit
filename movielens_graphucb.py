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
from sklearn import datasets
# os.chdir('/Kaige_Research/Code/graph_bandit/code/')
from linucb import LINUCB
from t_sampling import TS
from gob import GOB 
from lapucb import LAPUCB
from lapucb_sim import LAPUCB_SIM
from club import CLUB
from utils import *
path='../bandit_results/simulated/'
np.random.seed(2018)

user_num=20
item_num=300
dimension=10
pool_size=20
iteration=1000
loop=1
sigma=0.01# noise
delta=0.1# high probability
alpha=0.1# regularizer
alpha_2=0.05# edge delete CLUB
beta=0.1 # exploration for CLUB, SCLUB and GOB
state=False # False for artificial dataset, True for real dataset
thres=0

input_path='../processed_data/movielens/'
user_feature_matrix=np.load(input_path+'user_feature_matrix_100.npy')
item_feature_matrix=np.load(input_path+'item_feature_matrix_500.npy')

user_feature_matrix=user_feature_matrix[:user_num]
item_feature_matrix=item_feature_matrix[:item_num]

noise_matrix=np.random.normal(scale=sigma, size=(user_num, item_num))
user_seq=np.random.choice(range(user_num), size=iteration)
item_pool_seq=np.random.choice(range(item_num), size=(iteration, pool_size))

true_adj=rbf_kernel(user_feature_matrix, gamma=0.5)
np.fill_diagonal(true_adj,0)
edges=true_adj.ravel()
plt.figure(figsize=(5,5))
plt.hist(edges)
plt.ylabel('Counts', fontsize=12)
plt.xlabel('Edge weights', fontsize=12)
plt.tight_layout()
plt.savefig(path+'hist_edge_weights_movielens'+'.png', dpi=100)
plt.show()


D=np.diag(np.sum(true_adj, axis=1))
lap=D-true_adj
true_lap=np.dot(np.linalg.inv(D), lap)
true_payoffs=np.dot(user_feature_matrix, item_feature_matrix.T)+noise_matrix

payoffs=true_payoffs.ravel()
plt.figure(figsize=(5,5))
plt.hist(payoffs)
plt.ylabel('Counts', fontsize=12)
plt.xlabel('Payoffs', fontsize=12)
plt.tight_layout()
plt.savefig(path+'hist_payoffs_movielens'+'.png', dpi=100)
plt.show()

alpha_list=np.linspace(0.1,1,5)
loop=5
lapucb_regret_matrix=np.zeros((loop, iteration))
lapucb_error_matrix=np.zeros((loop, iteration))
lapucb_sim_regret_matrix=np.zeros((loop, iteration))
lapucb_sim_error_matrix=np.zeros((loop, iteration))


for index, alpha in enumerate(alpha_list):
	print('alpha', alpha)
	lapucb_model=LAPUCB(dimension, user_num, item_num, pool_size, item_feature_matrix, user_feature_matrix, true_payoffs, true_adj,true_lap, noise_matrix, alpha, delta, sigma, beta, thres, state)
	lapucb_sim_model=LAPUCB_SIM(dimension, user_num, item_num, pool_size, item_feature_matrix, user_feature_matrix, true_payoffs, true_adj,true_lap, noise_matrix, alpha, delta, sigma, beta, thres, state)

	lapucb_regret, lapucb_error, lapucb_beta, lapucb_x_norm, lapucb_inst_regret, lapucb_ucb, lapucb_sum_x_norm, lapucb_real_beta=lapucb_model.run(user_seq, item_pool_seq, iteration)
	lapucb_sim_regret, lapucb_sim_error, lapucb_sim_beta, lapucb_sim_x_norm, lapucb_sim_avg_norm, lapucb_sim_inst_regret, lapucb_sim_ucb, lapucb_sim_sum_x_norm=lapucb_sim_model.run( user_seq, item_pool_seq, iteration)

	lapucb_regret_matrix[index], lapucb_error_matrix[index]=lapucb_regret, lapucb_error
	lapucb_sim_regret_matrix[index], lapucb_sim_error_matrix[index]=lapucb_sim_regret, lapucb_sim_error


plt.figure(figsize=(5,5))
p=plt.plot(lapucb_regret_matrix.T)
plt.legend(p, alpha_list)
plt.show()


plt.figure(figsize=(5,5))
p=plt.plot(lapucb_sim_regret_matrix.T)
plt.legend(p, alpha_list)
plt.show()



