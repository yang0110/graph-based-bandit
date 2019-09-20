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
#np.random.seed(2018)

user_num=20
item_num=500
dimension=5
pool_size=20
iteration=500
loop=1
sigma=0.01# noise
delta=0.01# high probability
alpha=1# regularizer
alpha_2=0.1# edge delete CLUB
beta=0.1# exploration for CLUB, SCLUB and GOB
thres=0.0
state=False # False for artificial dataset, True for real dataset
lambda_list=[4]

item_feature_matrix=Normalizer().fit_transform(np.random.normal(size=(item_num, dimension)))
user_seq=np.random.choice(range(user_num), size=iteration)
item_pool_seq=np.random.choice(range(item_num), size=(iteration, pool_size))
noise_matrix=np.random.normal(scale=sigma, size=(user_num, item_num))

true_adj=np.random.uniform(size=(user_num, user_num))
true_adj=(true_adj.T+true_adj)/2.0
edge_weights=true_adj.copy()
np.fill_diagonal(true_adj, 0)
lap=csgraph.laplacian(true_adj, normed=False)
D=np.diag(np.sum(true_adj, axis=1))
true_lap=np.dot(np.linalg.inv(D), lap)
user_feature_matrix=dictionary_matrix_generator(user_num, dimension, true_lap, 4)

cum_matrix=np.zeros((5, 10))
m_list=np.linspace(1, user_num-1, 10).astype(int)
for index, m in enumerate(m_list):
	true_adj=BA_graph(user_num, m)
	true_adj=true_adj*edge_weights
	np.fill_diagonal(true_adj,1)
	lap=csgraph.laplacian(true_adj, normed=False)
	D=np.diag(np.sum(true_adj, axis=1))
	true_lap=np.dot(np.linalg.inv(D), lap)

	true_payoffs=np.dot(user_feature_matrix, item_feature_matrix.T)+noise_matrix
	linucb_regret_matrix=np.zeros((loop, iteration))
	linucb_error_matrix=np.zeros((loop, iteration))
	gob_regret_matrix=np.zeros((loop, iteration))
	gob_error_matrix=np.zeros((loop, iteration))
	lapucb_regret_matrix=np.zeros((loop, iteration))
	lapucb_error_matrix=np.zeros((loop, iteration))
	lapucb_sim_regret_matrix=np.zeros((loop, iteration))
	lapucb_sim_error_matrix=np.zeros((loop, iteration))
	club_regret_matrix=np.zeros((loop, iteration))
	club_error_matrix=np.zeros((loop, iteration))

	for l in range(loop):
		print('loop/total_loop', l, loop)
		linucb_model=LINUCB(dimension, user_num, item_num, pool_size, item_feature_matrix, user_feature_matrix, true_payoffs, alpha, delta, sigma, state)
		gob_model=GOB(dimension, user_num, item_num, pool_size, item_feature_matrix, user_feature_matrix, true_payoffs, true_adj,true_lap, alpha, delta, sigma, beta, state)
		lapucb_model=LAPUCB(dimension, user_num, item_num, pool_size, item_feature_matrix, user_feature_matrix, true_payoffs, true_adj,true_lap, noise_matrix, alpha, delta, sigma, beta, thres, state)
		lapucb_sim_model=LAPUCB_SIM(dimension, user_num, item_num, pool_size, item_feature_matrix, user_feature_matrix, true_payoffs, true_adj,true_lap, noise_matrix, alpha, delta, sigma, beta, thres, state)
		club_model = CLUB(dimension, user_num, item_num, pool_size, item_feature_matrix, user_feature_matrix, true_payoffs, alpha, alpha_2, delta, sigma, beta, state)

		linucb_regret, linucb_error, linucb_beta, linucb_x_norm, linucb_inst_regret, linucb_ucb, linucb_sum_x_norm, linucb_real_beta=linucb_model.run(user_seq, item_pool_seq, iteration)
		gob_regret, gob_error, gob_beta,  gob_x_norm, gob_ucb, gob_sum_x_norm, gob_real_beta=gob_model.run(user_seq, item_pool_seq, iteration)
		lapucb_regret, lapucb_error, lapucb_beta, lapucb_x_norm, lapucb_inst_regret, lapucb_ucb, lapucb_sum_x_norm, lapucb_real_beta=lapucb_model.run(user_seq, item_pool_seq, iteration)
		lapucb_sim_regret, lapucb_sim_error, lapucb_sim_beta, lapucb_sim_x_norm, lapucb_sim_avg_norm, lapucb_sim_inst_regret, lapucb_sim_ucb, lapucb_sim_sum_x_norm=lapucb_sim_model.run( user_seq, item_pool_seq, iteration)
		club_regret, club_error,club_cluster_num, club_beta, club_x_norm=club_model.run(user_seq, item_pool_seq, iteration)

		linucb_regret_matrix[l], linucb_error_matrix[l]=linucb_regret, linucb_error
		gob_regret_matrix[l], gob_error_matrix[l]=gob_regret, gob_error
		lapucb_regret_matrix[l], lapucb_error_matrix[l]=lapucb_regret, lapucb_error
		lapucb_sim_regret_matrix[l], lapucb_sim_error_matrix[l]=lapucb_sim_regret, lapucb_sim_error
		club_regret_matrix[l], club_error_matrix[l]=club_regret, club_error


	linucb_regret=np.mean(linucb_regret_matrix, axis=0)
	linucb_error=np.mean(linucb_error_matrix, axis=0)
	gob_regret=np.mean(gob_regret_matrix, axis=0)
	gob_error=np.mean(gob_error_matrix, axis=0)

	lapucb_regret=np.mean(lapucb_regret_matrix, axis=0)
	lapucb_error=np.mean(lapucb_error_matrix, axis=0)

	lapucb_sim_regret=np.mean(lapucb_sim_regret_matrix, axis=0)
	lapucb_sim_error=np.mean(lapucb_sim_error_matrix, axis=0)

	club_regret=np.mean(club_regret_matrix, axis=0)
	club_error=np.mean(club_error_matrix, axis=0)



	# plt.figure(figsize=(5,5))
	# #plt.plot(linucb_regret,'-.', markevery=0.1, label='LinUCB')
	# #plt.plot(gob_regret, '-o', color='orange', markevery=0.1, label='GOB')
	# plt.plot(lapucb_sim_regret, '-s', markevery=0.1, label='GraphUCB-Local')
	# plt.plot(lapucb_regret, '-*', markevery=0.1, label='GraphUCB')
	# #plt.plot(club_regret,'-p', markevery=0.1, label='CLUB')
	# plt.ylabel('Cumulative Regret', fontsize=14)
	# plt.xlabel('Time', fontsize=14)
	# plt.legend(loc=4, fontsize=14)
	# plt.tight_layout()
	# plt.savefig(path+'rbf'+'.png', dpi=100)
	# plt.show()


	cum_matrix[0,index]=linucb_regret[-1]
	cum_matrix[1,index]=gob_regret[-1]
	cum_matrix[2,index]=club_regret[-1]
	cum_matrix[3,index]=lapucb_regret[-1]
	cum_matrix[4,index]=lapucb_sim_regret[-1]


plt.figure(figsize=(5,5))
plt.plot(m_list, cum_matrix[0], '-', label='LinUCB')
plt.plot(m_list, cum_matrix[1], '-p', label='Gob.Lin')
plt.plot(m_list, cum_matrix[4], '-s', label='GraphUCB-Local')
plt.plot(m_list, cum_matrix[3], '-o', label='GraphUCB')
plt.plot(m_list, cum_matrix[2], '-*', label='CLUB')
plt.legend(loc=1, fontsize=12)
plt.xlabel('Neighbor Num', fontsize=16)
plt.ylabel('Cumulative Regret', fontsize=16)
plt.tight_layout()
plt.savefig(path+'m_ba'+'.png', dpi=100)
plt.show()