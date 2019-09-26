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
os.chdir('/Kaige_Research/Code/graph_bandit/code/')
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
iteration=500
loop=1
sigma=0.01# noise
delta=0.1# high probability
alpha=1 # regularizer
alpha_2=0.1# edge delete CLUB
epsilon=8 # Ts
beta=0.1 # exploration for CLUB, SCLUB and GOB
thres=0.0
state=False # False for artificial dataset, True for real dataset
lambda_list=[4]

user_seq=np.random.choice(range(user_num), size=iteration)
item_pool_seq=np.random.choice(range(item_num), size=(iteration, pool_size))
item_feature_matrix=Normalizer().fit_transform(np.random.normal(size=(item_num, dimension)))
noise_matrix=np.random.normal(scale=sigma, size=(user_num, item_num))

old_adj=rbf_kernel(np.random.normal(size=(user_num, dimension)), gamma=0.25/dimension)
np.fill_diagonal(old_adj, 0)
D=np.diag(np.sum(old_adj, axis=1))
lap=D-old_adj
true_lap=np.dot(np.linalg.inv(D), lap)



# edges=true_adj.ravel()
# plt.figure(figsize=(5,5))
# plt.hist(edges)
# plt.ylabel('Counts', fontsize=12)
# plt.xlabel('Edge weights', fontsize=12)
# plt.tight_layout()
# plt.savefig(path+'hist_edge_weights'+'.png', dpi=100)
# plt.show()



er_adj=ER_graph(user_num, 0.2)
er_adj=er_adj*old_adj

ba_adj=BA_graph(user_num, 1)
ba_adj=ba_adj*old_adj

true_adj=old_adj.copy()
true_adj[true_adj<0]=0
sparsity=np.round(np.sum(true_adj>0)/(user_num*(user_num-1)), decimals=2)
print('sparsity', sparsity)

D=np.diag(np.sum(true_adj, axis=1))
lap=D-true_adj
true_lap=np.zeros((user_num, user_num))
for i in range(user_num):
	for j in range(user_num):
		if D[i,i]==0:
			true_lap[i,j]=0
		else:
			true_lap[i,j]=-true_adj[i,j]/D[i,i]

np.fill_diagonal(true_lap, 1)

smooth_list=np.linspace(0,10,5)
sparsity_list=np.zeros(5)

linucb_regret_matrix=np.zeros((5, iteration))
linucb_error_matrix=np.zeros((5, iteration))
lapucb_regret_matrix=np.zeros((5, iteration))
lapucb_error_matrix=np.zeros((5, iteration))
lapucb_sim_regret_matrix=np.zeros((5, iteration))
lapucb_sim_error_matrix=np.zeros((5, iteration))

for index, smooth in enumerate(smooth_list):
	user_feature_matrix=dictionary_matrix_generator(user_num, dimension, true_lap, smooth)
	smoothness=np.trace(np.dot(np.dot(user_feature_matrix.T, true_lap), user_feature_matrix))
	print('smoothness', smoothness)
	true_payoffs=np.dot(user_feature_matrix, item_feature_matrix.T)
	linucb_model=LINUCB(dimension, user_num, item_num, pool_size, item_feature_matrix, user_feature_matrix, true_payoffs, alpha, delta, sigma, state)
	lapucb_model=LAPUCB(dimension, user_num, item_num, pool_size, item_feature_matrix, user_feature_matrix, true_payoffs, true_adj, true_lap, noise_matrix, alpha, delta, sigma, beta, thres, state)
	lapucb_sim_model=LAPUCB_SIM(dimension, user_num, item_num, pool_size, item_feature_matrix, user_feature_matrix, true_payoffs, true_adj, true_lap, noise_matrix, alpha, delta, sigma, beta, thres, state)

	linucb_regret, linucb_error, linucb_beta, linucb_x_norm, linucb_inst_regret, linucb_ucb, linucb_sum_x_norm, linucb_real_beta=linucb_model.run(user_seq, item_pool_seq, iteration)
	lapucb_regret, lapucb_error, lapucb_beta, lapucb_x_norm, lapucb_inst_regret, lapucb_ucb, lapucb_sum_x_norm, lapucb_real_beta=lapucb_model.run(user_seq, item_pool_seq, iteration)
	lapucb_sim_regret, lapucb_sim_error, lapucb_sim_beta, lapucb_sim_x_norm, lapucb_sim_avg_norm, lapucb_sim_inst_regret, lapucb_sim_ucb, lapucb_sim_sum_x_norm=lapucb_sim_model.run( user_seq, item_pool_seq, iteration)

	linucb_regret_matrix[index], linucb_error_matrix[index]=linucb_regret, linucb_error
	lapucb_regret_matrix[index], lapucb_error_matrix[index]=lapucb_regret, lapucb_error
	lapucb_sim_regret_matrix[index], lapucb_sim_error_matrix[index]=lapucb_sim_regret, lapucb_sim_error




plt.figure(figsize=(5,5))
plt.plot(linucb_regret_matrix[0], label='LinUCB')
for ind, sm in enumerate(smooth_list):
	plt.plot(lapucb_regret_matrix[ind], label='GraphUCB (%s)'%(np.round(smooth_list[ind], decimals=2)))
plt.legend(loc=0, fontsize=14)
plt.xlabel('Time', fontsize=16)
plt.ylabel('Cumulative Regret', fontsize=16)
plt.title('Sparsity = %s'%(sparsity), fontsize=16)
plt.tight_layout()
plt.savefig(path+'graphucb_smooth_rbf_even_sparse'+'.png', dpi=100)
plt.show()

plt.figure(figsize=(5,5))
plt.plot(linucb_regret_matrix[0], label='LinUCB')
for ind2, sm in enumerate(smooth_list):
	plt.plot(lapucb_sim_regret_matrix[ind2], '-*', markevery=0.1, label='Local %s'%(np.round(smooth_list[ind2], decimals=2)))
plt.legend(loc=0, fontsize=14)
plt.xlabel('Time', fontsize=16)
plt.ylabel('Cumulative Regret', fontsize=16)
plt.title('Sparsity = %s'%(sparsity), fontsize=16)
plt.tight_layout()
plt.savefig(path+'graphucb_local_smooth_rbf_even_sparse'+'.png', dpi=100)
plt.show()



plt.figure(figsize=(5,5))
for ind, sm in enumerate(smooth_list):
	plt.plot(lapucb_regret_matrix[ind], label='G-UCB %s'%(np.round(smooth_list[ind], decimals=2)))
for ind2, sm in enumerate(smooth_list):
	plt.plot(lapucb_sim_regret_matrix[ind2], '-*', markevery=0.1, label='Local %s'%(np.round(smooth_list[ind2], decimals=2)))
plt.legend(loc=0, fontsize=12)
plt.show()


