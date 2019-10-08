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
# os.chdir('/Kaige_Research/Code/graph_bandit/code/')
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
iteration=1000
loop=1
sigma=0.01# noise
delta=0.1# high probability
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

true_adj=rbf_kernel(np.random.normal(size=(user_num, dimension)), gamma=0.5/dimension)
np.fill_diagonal(true_adj, 0)
D=np.diag(np.sum(true_adj, axis=1))
lap=D-true_adj
true_lap=np.dot(np.linalg.inv(D), lap)
user_feature_matrix=dictionary_matrix_generator(user_num, dimension, true_lap, 10)
smoothness=np.trace(np.dot(np.dot(user_feature_matrix.T, true_lap), user_feature_matrix))
print('smoothness', smoothness)
true_payoffs=np.dot(user_feature_matrix, item_feature_matrix.T)+noise_matrix

cum_matrix=np.zeros((5, 10))
thres_list=np.round(np.linspace(0,1,10), decimals=2)
for index, thres in enumerate(thres_list):
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
		true_adj[true_adj<thres]=0
		D=np.diag(np.sum(true_adj, axis=1))
		true_lap=np.zeros((user_num, user_num))
		for i in range(user_num):
			for j in range(user_num):
				if D[i,i]==0:
					true_lap[i,j]=0
				else:
					true_lap[i,j]=-true_adj[i,j]/D[i,i]

		np.fill_diagonal(true_lap, 1)

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
plt.plot(thres_list, cum_matrix[0], '-.', markevery=0.1, linewidth=2, markersize=8,label='LinUCB')
plt.plot(thres_list, cum_matrix[1], '-p',color='orange', markevery=0.1,linewidth=2, markersize=8, label='Gob.Lin')
plt.plot(thres_list, cum_matrix[4], '-s',color='g', markevery=0.1,linewidth=2, markersize=8, label='GraphUCB-Local')
plt.plot(thres_list, cum_matrix[3], '-o',color='r', markevery=0.1,linewidth=2, markersize=8, label='GraphUCB')
plt.plot(thres_list, cum_matrix[2], '-*',color='k', markevery=0.1,linewidth=2, markersize=8, label='CLUB')
plt.legend(loc=1, fontsize=14)
plt.xlabel('s', fontsize=16)
plt.ylim([0,80])
plt.ylabel('Cumulative Regret', fontsize=16)
plt.tight_layout()
plt.savefig(path+'threshold_rbf'+'.png', dpi=100)
plt.show()

np.save(path+'cum_regret_sparsity_rbf_all_algorithms.npy', cum_matrix)