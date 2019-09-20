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
os.chdir('/Kaige_Research/Code/graph_bandit/code/')
from linucb import LINUCB
from gob import GOB 
from lapucb_sim import LAPUCB_SIM
from lapucb_sim2 import LAPUCB_SIM2

from utils import *
path='../bandit_results/simulated/'

user_num=10
dimension=5
item_num=100
pool_size=10
iteration=1000
loop=2
sigma=0.1# noise
delta=0.01# high probability
alpha=1# regularizer
alpha_2=0.01# edge delete CLUB
beta=0.01 # exploration for CLUB, SCLUB and GOB
thres=0.0
state=False # False for artificial dataset, True for real dataset
lambda_list=[1]

user_seq=np.random.choice(range(user_num), size=iteration)
item_pool_seq=np.random.choice(range(item_num), size=(iteration, pool_size))
item_feature_matrix=Normalizer().fit_transform(np.random.normal(size=(item_num, dimension)))
true_adj=RBF_graph(user_num, dimension, thres=thres, gamma=0.5)
true_lap=csgraph.laplacian(true_adj, normed=False)
noise_matrix=np.random.normal(scale=sigma, size=(user_num, item_num))

# np.fill_diagonal(true_adj,0)
# true_adj=np.round(true_adj,decimals=2)
# graph, edge_num=create_networkx_graph(user_num, true_adj)
# labels = nx.get_edge_attributes(graph,'weight')
# edge_weight=true_adj[np.triu_indices(user_num,1)]
# edge_color=edge_weight[edge_weight>0]
# pos = nx.spring_layout(graph)
# plt.figure(figsize=(5,5))
# nodes=nx.draw_networkx_nodes(graph, pos, node_size=10, node_color='y')
# edges=nx.draw_networkx_edges(graph, pos, width=0.05, alpha=1, edge_color='k')
# edge_labels=nx.draw_networkx_edge_labels(graph,pos, edge_labels=labels, font_size=5)
# plt.axis('off')
# plt.savefig(path+'network_rbf'+'.png', dpi=300)
# plt.savefig(path+'network_rbf'+'.eps', dpi=300)
# plt.show()

for lambda_ in lambda_list:

	user_feature_matrix=dictionary_matrix_generator(user_num, dimension, true_lap, lambda_)
	true_payoffs=np.dot(user_feature_matrix, item_feature_matrix.T)+noise_matrix
	smoothness=np.trace(np.dot(np.dot(user_feature_matrix.T, true_lap), user_feature_matrix))
	linucb_regret_matrix=np.zeros((loop, iteration))
	gob_regret_matrix=np.zeros((loop, iteration))
	lapucb_sim_regret_matrix=np.zeros((loop, iteration))
	lapucb_sim2_regret_matrix=np.zeros((loop, iteration))

	for l in range(loop):
		print('loop/total_loop', l, loop)
		linucb_model=LINUCB(dimension, user_num, item_num, pool_size, item_feature_matrix, user_feature_matrix, true_payoffs, alpha, delta, sigma, state)
		gob_model=GOB(dimension, user_num, item_num, pool_size, item_feature_matrix, user_feature_matrix, true_payoffs, true_adj, alpha, delta, sigma, beta, state)
		lapucb_sim_model=LAPUCB_SIM(dimension, user_num, item_num, pool_size, item_feature_matrix, user_feature_matrix, true_payoffs, true_adj, noise_matrix, alpha, delta, sigma, beta, thres, state)
		lapucb_sim2_model=LAPUCB_SIM2(dimension, user_num, item_num, pool_size, item_feature_matrix, user_feature_matrix, true_payoffs, true_adj, noise_matrix, alpha, delta, sigma, beta, thres, state)

		linucb_regret, linucb_error, linucb_beta=linucb_model.run(user_seq, item_pool_seq, iteration)
		gob_regret, gob_error, gob_beta, gob_graph=gob_model.run(user_seq, item_pool_seq, iteration)
		lapucb_sim_regret, lapucb_sim_error, lapucb_sim_beta, lapucb_sim_graph=lapucb_sim_model.run( user_seq, item_pool_seq, iteration)
		lapucb_sim2_regret, lapucb_sim2_error, lapucb_sim2_beta, lapucb_sim2_graph, lapucb_sim2_diff_list=lapucb_sim2_model.run( user_seq, item_pool_seq, iteration)

		linucb_regret_matrix[l], _=linucb_regret, linucb_error
		gob_regret_matrix[l], _, _=gob_regret, gob_error, gob_graph
		lapucb_sim_regret_matrix[l], _, _=lapucb_sim_regret, lapucb_sim_error, lapucb_sim_graph
		lapucb_sim2_regret_matrix[l], _, _=lapucb_sim2_regret, lapucb_sim2_error, lapucb_sim2_graph


	linucb_regret=np.mean(linucb_regret_matrix, axis=0)
	gob_regret=np.mean(gob_regret_matrix, axis=0)
	lapucb_sim_regret=np.mean(lapucb_sim_regret_matrix, axis=0)
	lapucb_sim2_regret=np.mean(lapucb_sim2_regret_matrix, axis=0)


	plt.figure(figsize=(5,5))
	plt.plot(linucb_regret,'-.', label='LinUCB')
	plt.plot(gob_regret, 'orange', label='GOB')
	plt.plot(lapucb_sim_regret, '-*', markevery=0.1, label='G-UCB SIM')
	plt.plot(lapucb_sim2_regret, '-*', markevery=0.1, label='G-UCB SIM2')
	plt.ylabel('Cumulative Regret', fontsize=12)
	plt.xlabel('Time', fontsize=12)
	plt.legend(loc=2, fontsize=10)
	plt.tight_layout()
	plt.show()

	 


plt.figure(figsize=(5,5))
plt.plot(linucb_beta,'-.', label='LinUCB')
plt.plot(gob_beta, 'orange', label='GOB')
plt.plot(lapucb_sim_beta, '-*', markevery=0.1, label='G-UCB SIM')
plt.plot(lapucb_sim2_beta, '-*', markevery=0.1, label='G-UCB SIM2')
plt.ylabel('Beta', fontsize=12)
plt.xlabel('Time', fontsize=12)
plt.legend(loc=1, fontsize=10)
plt.tight_layout()
plt.show()

plt.figure(figsize=(5,5))
plt.plot(linucb_error,'-.', label='LinUCB')
plt.plot(gob_error, 'orange', label='GOB')
plt.plot(lapucb_sim_error, '-*', markevery=0.1, label='G-UCB SIM')
plt.plot(lapucb_sim2_error[100:], '-*', markevery=0.1, label='G-UCB SIM2')
plt.ylabel('error', fontsize=12)
plt.xlabel('Time', fontsize=12)
plt.legend(loc=1, fontsize=10)
plt.tight_layout()
plt.show()


plt.figure(figsize=(5,5))
plt.plot(lapucb_sim2_diff_list[100:], '-*', markevery=0.1, label='G-UCB SIM2')
plt.ylabel('diff_list', fontsize=12)
plt.xlabel('Time', fontsize=12)
plt.legend(loc=1, fontsize=10)
plt.tight_layout()
plt.show()


