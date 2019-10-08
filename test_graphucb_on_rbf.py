import numpy as np 
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import Normalizer, MinMaxScaler
from scipy.sparse import csgraph  
import scipy
import os 
from sklearn import datasets
# os.chdir('Users/kaigeyang/Documents/research/bandit/code/graph_based_bandit/')
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
item_num=500
dimension=5
pool_size=20
iteration=1000
loop=5
sigma=0.01# noise
delta=0.1# high probability
alpha=1 # regularizer
alpha_2=0.15# edge delete CLUB
epsilon=8 # Ts
beta=0.1 # exploration for CLUB, SCLUB and GOB
thres=0.0
state=False # False for artificial dataset, True for real dataset
lambda_list=[4]

user_seq=np.random.choice(range(user_num), size=iteration)
item_pool_seq=np.random.choice(range(item_num), size=(iteration, pool_size))
item_feature_matrix=Normalizer().fit_transform(np.random.normal(size=(item_num, dimension)))
noise_matrix=np.random.normal(scale=sigma, size=(user_num, item_num))


old_adj=np.load(path+'random_graph_weights.npy')
rbf_adj=np.load(path+'rbf_sparse_graph.npy')
rbf_adj[rbf_adj>0]=1
# rbf_adj=rbf_adj*old_adj
true_adj=rbf_adj.copy()
sparsity=np.sum(true_adj>0)/(user_num*(user_num-1))
print('sparsity', sparsity)

D=np.diag(np.sum(true_adj, axis=1))
true_lap=np.zeros((user_num, user_num))
for i in range(user_num):
	for j in range(user_num):
		if D[i,i]==0:
			true_lap[i,j]=0
		else:
			true_lap[i,j]=-true_adj[i,j]/D[i,i]

np.fill_diagonal(true_lap, 1)

length=10
smooth_list=np.linspace(0,20,length)
smoothness_list=np.zeros(length)

graphucb_regret=np.zeros(length)
local_regret=np.zeros(length)


for index, smooth in enumerate(smooth_list):

	lapucb_regret_matrix=np.zeros((loop, iteration))
	lapucb_error_matrix=np.zeros((loop, iteration))
	lapucb_sim_regret_matrix=np.zeros((loop, iteration))
	lapucb_sim_error_matrix=np.zeros((loop, iteration))
	for l in range(loop):
		user_feature_matrix=dictionary_matrix_generator(user_num, dimension, true_lap, smooth)
		smoothness=np.trace(np.dot(np.dot(user_feature_matrix.T, true_lap), user_feature_matrix))
		print('smoothness', smoothness)
		smoothness_list[index]=smoothness
		true_payoffs=np.dot(user_feature_matrix, item_feature_matrix.T)+noise_matrix

		lapucb_model=LAPUCB(dimension, user_num, item_num, pool_size, item_feature_matrix, user_feature_matrix, true_payoffs, true_adj, true_lap, noise_matrix, alpha, delta, sigma, beta, thres, state)
		lapucb_sim_model=LAPUCB_SIM(dimension, user_num, item_num, pool_size, item_feature_matrix, user_feature_matrix, true_payoffs, true_adj, true_lap, noise_matrix, alpha, delta, sigma, beta, thres, state)

		lapucb_regret, lapucb_error, lapucb_beta, lapucb_x_norm, lapucb_inst_regret, lapucb_ucb, lapucb_sum_x_norm, lapucb_real_beta=lapucb_model.run(user_seq, item_pool_seq, iteration)
		lapucb_sim_regret, lapucb_sim_error, lapucb_sim_beta, lapucb_sim_x_norm, lapucb_sim_avg_norm, lapucb_sim_inst_regret, lapucb_sim_ucb, lapucb_sim_sum_x_norm=lapucb_sim_model.run( user_seq, item_pool_seq, iteration)

		lapucb_regret_matrix[l], lapucb_error_matrix[l]=lapucb_regret, lapucb_error
		lapucb_sim_regret_matrix[l], lapucb_sim_error_matrix[l]=lapucb_sim_regret, lapucb_sim_error

	lapucb_regret=np.mean(lapucb_regret_matrix, axis=0)
	lapucb_error=np.mean(lapucb_error_matrix, axis=0)

	lapucb_sim_regret=np.mean(lapucb_sim_regret_matrix, axis=0)
	lapucb_sim_error=np.mean(lapucb_sim_error_matrix, axis=0)

	graphucb_regret[index]=lapucb_regret[-1]
	local_regret[index]=lapucb_sim_regret[-1]

plt.figure(figsize=(5,5))
plt.plot(smoothness_list, local_regret, '-s', color='g',markevery=0.1, label='GraphUCB-Local')
plt.plot(smoothness_list, graphucb_regret, '-o', color='r',markevery=0.1, label='GraphUCB')
plt.xlabel('Smoothness', fontsize=16)
plt.ylabel('Cumulative Regret', fontsize=16)
plt.legend(loc=0, fontsize=14)
plt.tight_layout()
plt.savefig(path+'graphucb_vs_smooth_rbf_binary'+'.png', dpi=100)
plt.show()


np.save(path+'graphucb_smooth_rbf_binary.npy', graphucb_regret)
np.save(path+'graphucb_local_smooth_rbf_binary.npy', local_regret)


