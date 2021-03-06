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
from colin import COLIN
from lapucb import LAPUCB
from lapucb_sim import LAPUCB_SIM
from sclub import SCLUB
from club import CLUB
from utils import *
path='../bandit_results/simulated/'

user_num=10
dimension=10
item_num=500
pool_size=10
iteration=1000
loop=5
sigma=0.0# noise
delta=0.01# high probability
alpha=1# regularizer
alpha_2=0.03# edge delete CLUB
beta=0.01 # exploration for CLUB, SCLUB and GOB
thres=0.0
state=False # False for artificial dataset, True for real dataset
lambda_=1
sigma_list=[0.01, 0.1, 0.2]

user_seq=np.random.choice(range(user_num), size=iteration)
item_pool_seq=np.random.choice(range(item_num), size=(iteration, pool_size))
item_feature_matrix=Normalizer().fit_transform(np.random.normal(size=(item_num, dimension)))
true_adj=RBF_graph(user_num, dimension, thres=thres, gamma=0.5)
true_lap=csgraph.laplacian(true_adj, normed=False)

np.fill_diagonal(true_adj,0)
true_adj=np.round(true_adj,decimals=2)
graph, edge_num=create_networkx_graph(user_num, true_adj)
labels = nx.get_edge_attributes(graph,'weight')
edge_weight=true_adj[np.triu_indices(user_num,1)]
edge_color=edge_weight[edge_weight>0]
pos = nx.spring_layout(graph)
plt.figure(figsize=(5,5))
nodes=nx.draw_networkx_nodes(graph, pos, node_size=10, node_color='y')
edges=nx.draw_networkx_edges(graph, pos, width=0.05, alpha=1, edge_color='k')
edge_labels=nx.draw_networkx_edge_labels(graph,pos, edge_labels=labels, font_size=5)
plt.axis('off')
plt.savefig(path+'network_rbf_noise'+'.png', dpi=300)
plt.savefig(path+'network_rbf_noise'+'.eps', dpi=300)

for sigma in sigma_list:
	noise_matrix=np.random.normal(scale=sigma, size=(user_num, item_num))
	true_payoffs=np.dot(user_feature_matrix, item_feature_matrix.T)+noise_matrix
	smoothness=np.trace(np.dot(np.dot(user_feature_matrix.T, true_lap), user_feature_matrix))
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
	gob_graph_matrix=np.zeros((loop, iteration))
	lapucb_graph_matrix=np.zeros((loop, iteration))
	lapucb_sim_graph_matrix=np.zeros((loop, iteration))

	for l in range(loop):
		print('loop/total_loop', l, loop)
		linucb_model=LINUCB(dimension, user_num, item_num, pool_size, item_feature_matrix, user_feature_matrix, true_payoffs, alpha, delta, sigma, state)
		gob_model=GOB(dimension, user_num, item_num, pool_size, item_feature_matrix, user_feature_matrix, true_payoffs, true_adj, alpha, delta, sigma, beta, state)
		lapucb_model=LAPUCB(dimension, user_num, item_num, pool_size, item_feature_matrix, user_feature_matrix, true_payoffs, true_adj, noise_matrix, alpha, delta, sigma, beta, thres, state)
		lapucb_sim_model=LAPUCB_SIM(dimension, user_num, item_num, pool_size, item_feature_matrix, user_feature_matrix, true_payoffs, true_adj, noise_matrix, alpha, delta, sigma, beta, thres, state)
		club_model = CLUB(dimension, user_num, item_num, pool_size, item_feature_matrix, user_feature_matrix, true_payoffs, alpha, alpha_2, delta, sigma, beta, state)

		linucb_regret, linucb_error, linucb_beta=linucb_model.run(user_seq, item_pool_seq, iteration)
		gob_regret, gob_error, gob_beta, gob_graph=gob_model.run(user_seq, item_pool_seq, iteration)
		lapucb_regret, lapucb_error, lapucb_beta, lapucb_graph=lapucb_model.run(user_seq, item_pool_seq, iteration)
		lapucb_sim_regret, lapucb_sim_error, lapucb_sim_beta, lapucb_sim_graph=lapucb_sim_model.run( user_seq, item_pool_seq, iteration)
		club_regret, club_error,club_cluster_num, club_beta=club_model.run(user_seq, item_pool_seq, iteration)

		linucb_regret_matrix[l], linucb_error_matrix[l]=linucb_regret, linucb_error
		gob_regret_matrix[l], gob_error_matrix[l], gob_graph_matrix[l]=gob_regret, gob_error, gob_graph
		lapucb_regret_matrix[l], lapucb_error_matrix[l], lapucb_graph_matrix[l]=lapucb_regret, lapucb_error, lapucb_graph
		lapucb_sim_regret_matrix[l], lapucb_sim_error_matrix[l], lapucb_sim_graph_matrix[l]=lapucb_sim_regret, lapucb_sim_error, lapucb_sim_graph
		club_regret_matrix[l], club_error_matrix[l]=club_regret, club_error


	linucb_regret=np.mean(linucb_regret_matrix, axis=0)
	linucb_error=np.mean(linucb_error_matrix, axis=0)
	gob_regret=np.mean(gob_regret_matrix, axis=0)
	gob_error=np.mean(gob_error_matrix, axis=0)
	gob_graph=np.mean(gob_graph_matrix, axis=0)

	lapucb_regret=np.mean(lapucb_regret_matrix, axis=0)
	lapucb_error=np.mean(lapucb_error_matrix, axis=0)
	lapucb_graph=np.mean(lapucb_graph_matrix, axis=0)

	lapucb_sim_regret=np.mean(lapucb_sim_regret_matrix, axis=0)
	lapucb_sim_error=np.mean(lapucb_sim_error_matrix, axis=0)
	lapucb_sim_graph=np.mean(lapucb_sim_graph_matrix, axis=0)

	club_regret=np.mean(club_regret_matrix, axis=0)
	club_error=np.mean(club_error_matrix, axis=0)


	plt.figure(figsize=(5,5))
	plt.plot(linucb_regret,'-.', label='LinUCB')
	plt.plot(gob_regret, label='GOB')
	plt.plot(lapucb_regret, '-*', markevery=0.1, label='G-UCB')
	plt.plot(lapucb_sim_regret, '-*', markevery=0.1, label='G-UCB SIM')
	plt.plot(club_regret, label='CLUB')
	plt.ylabel('Cumulative Regret', fontsize=12)
	plt.xlabel('Time', fontsize=12)
	plt.legend(loc=2, fontsize=10)
	plt.tight_layout()
	plt.savefig(path+'cum_regret_rbf_noise_%s'%(int(sigma*10))+'.png', dpi=300)
	plt.savefig(path+'cum_regret_rbf_noise_%s'%(int(sigma*10))+'.eps', dpi=300)

	plt.figure(figsize=(5,5))
	plt.plot(linucb_error,'-.', label='LinUCB')
	plt.plot(gob_error, label='GOB')
	plt.plot(lapucb_error, '-*', markevery=0.1, label='G-UCB')
	plt.plot(lapucb_sim_error, '-*', markevery=0.1, label='G-UCB SIM')
	plt.plot(club_error, label='CLUB')
	plt.ylabel('Error', fontsize=12)
	plt.xlabel('Time', fontsize=12)
	plt.legend(loc=1, fontsize=10)
	plt.tight_layout()
	plt.savefig(path+'bandit_learning_error_rbf_noise_%s'%(int(sigma*10))+'.png', dpi=300)
	plt.savefig(path+'bandit_learning_error_rbf_noise_%s'%(int(sigma*10))+'.eps', dpi=300)


	plt.figure(figsize=(5,5))
	plt.plot(linucb_beta,'-.', label='LinUCB')
	plt.plot(gob_beta, label='GOB')
	plt.plot(lapucb_beta, '-*', markevery=0.1, label='G-UCB')
	plt.plot(lapucb_sim_beta, '-*', markevery=0.1, label='G-UCB SIM')
	plt.plot(club_beta, label='CLUB')
	plt.ylabel('Error', fontsize=12)
	plt.xlabel('Time', fontsize=12)
	plt.legend(loc=1, fontsize=10)
	plt.tight_layout()
	plt.savefig(path+'bandit_beta_rbf_noise_%s'%(int(sigma*10))+'.png', dpi=300)
	plt.savefig(path+'bandit_beta_rbf_noise_%s'%(int(sigma*10))+'.eps', dpi=300)





