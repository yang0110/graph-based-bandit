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
os.chdir('C:/Kaige_Research/Code/graph_bandit/code/')
from linucb import LINUCB
from gob import GOB 
from colin import COLIN
from lapucb import LAPUCB
from lapucb_sim import LAPUCB_SIM
from sclub import SCLUB
from club import CLUB
from utils import *
path='../bandit_results/simulated/'

user_num=20
dimension=10
item_num=500
pool_size=10
iteration=1000
loop=1
sigma=0.1# noise
delta=0.1# high probability
alpha=1# regularizer
alpha_2=0.1# edge delete CLUB
beta=0.01 # exploration for CLUB, SCLUB and GOB
thres=0.0
k=3 # edge number each node SCLUB to control the sparsity
state=False # False for artificial dataset, True for real dataset

user_seq=np.random.choice(range(user_num), size=iteration)
item_pool_seq=np.random.choice(range(item_num), size=(iteration, pool_size))
item_feature_matrix=Normalizer().fit_transform(np.random.normal(size=(item_num, dimension)))

true_adj=RBF_graph(user_num, dimension, thres=thres, gamma=0.5)
#true_adj=BA_graph(user_num, 3)
#true_adj=ER_graph(user_num, 0.5)
true_normed_adj=true_adj/true_adj.sum(axis=0, keepdims=1)
true_lap=csgraph.laplacian(true_adj, normed=False)
true_normed_lap=csgraph.laplacian(true_adj, normed=True)
true_adj_binary=true_adj.copy()
true_adj_binary[true_adj_binary>0]=1
true_lap_binary=np.diag(np.sum(true_adj_binary, axis=1))-true_adj_binary

user_feature_matrix=dictionary_matrix_generator(user_num, dimension, true_lap, 1)
noise_matrix=np.random.normal(scale=sigma, size=(user_num, item_num))
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
	adj=np.ones((user_num, user_num))
	normed_lap=csgraph.laplacian(adj, normed=True)
	linucb_model=LINUCB(dimension, user_num, item_num, pool_size, item_feature_matrix, user_feature_matrix, true_payoffs, alpha, delta, sigma, state)
	gob_model=GOB(dimension, user_num, item_num, pool_size, item_feature_matrix, user_feature_matrix, true_payoffs, true_lap_binary, alpha/10.0, delta, sigma, beta, state)
	colin_model=COLIN(dimension, user_num, item_num, pool_size, item_feature_matrix, user_feature_matrix, true_payoffs, true_normed_adj, alpha/10.0, delta, sigma, beta, state)
	lapucb_model=LAPUCB(dimension, user_num, item_num, pool_size, item_feature_matrix, user_feature_matrix, true_payoffs, noise_matrix, normed_lap, alpha, delta, sigma, beta, thres, state)
	lapucb_sim_model=LAPUCB_SIM(dimension, user_num, item_num, pool_size, item_feature_matrix, user_feature_matrix, true_payoffs, noise_matrix, normed_lap, alpha, delta, sigma, beta, thres, state)
	club_model = CLUB(dimension, user_num, item_num, pool_size, item_feature_matrix, user_feature_matrix, true_payoffs,true_normed_lap, alpha, alpha_2, delta, sigma, beta, state)

	linucb_regret, linucb_error, linucb_beta=linucb_model.run(user_seq, item_pool_seq, iteration)
	gob_regret, gob_error, gob_beta=gob_model.run(user_seq, item_pool_seq, iteration)
	colin_regret, colin_error, colin_beta=colin_model.run(user_seq, item_pool_seq, iteration)
	lapucb_regret, lapucb_error, lapucb_beta=lapucb_model.run(user_seq, item_pool_seq, iteration)
	lapucb_sim_regret, lapucb_sim_error, lapucb_sim_beta=lapucb_sim_model.run( user_seq, item_pool_seq, iteration)
	club_regret, club_error,club_cluster_num, club_beta=club_model.run(user_seq, item_pool_seq, iteration)

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

np.fill_diagonal(true_adj,0)
graph, edge_num=create_networkx_graph(user_num, true_adj)
labels = nx.get_edge_attributes(graph,'weight')
edge_weight=adj[np.triu_indices(user_num,1)]
edge_color=edge_weight[edge_weight>0]
pos = nx.spring_layout(graph)
plt.figure(figsize=(5,5))
nodes=nx.draw_networkx_nodes(graph, pos, node_size=10, node_color='y')
edges=nx.draw_networkx_edges(graph, pos, width=0.1, alpha=1, edge_color='k')
#nx.draw_networkx_labels(graph, pos, font_color='k')
edge_labels=nx.draw_networkx_edge_labels(graph,pos, edge_labels=labels, font_size=5)
plt.axis('off')
plt.savefig(path+'network_rbf'+'.png', dpi=300)
plt.savefig(path+'network_rbf'+'.eps', dpi=300)
plt.show()

plt.figure(figsize=(5,5))
plt.plot(linucb_regret,'-.', label='LinUCB')
plt.plot(gob_regret, label='GOB')
plt.plot(colin_regret, label='CoLin')
plt.plot(lapucb_regret, '-*', markevery=0.1, label='G-UCB')
plt.plot(lapucb_sim_regret, '-*', markevery=0.1, label='G-UCB SIM')
plt.plot(club_regret, label='CLUB')
plt.ylabel('Cumulative Regret', fontsize=12)
plt.xlabel('Time', fontsize=12)
plt.legend(loc=2, fontsize=10)
plt.tight_layout()
plt.savefig(path+'cum_regret_rbf'+'.png', dpi=300)
plt.savefig(path+'cum_regret_rbf'+'.eps', dpi=300)
plt.show()
 
plt.figure(figsize=(5,5))
plt.plot(linucb_error,'-.', label='LinUCB')
plt.plot(gob_error, label='GOB')
plt.plot(colin_error, label='CoLin')
plt.plot(lapucb_error, '-*', markevery=0.1, label='G-UCB')
plt.plot(lapucb_sim_error, '-*', markevery=0.1, label='G-UCB SIM')
plt.plot(club_error, label='CLUB')
plt.ylabel('Error', fontsize=12)
plt.xlabel('Time', fontsize=12)
plt.legend(loc=1, fontsize=10)
plt.tight_layout()
plt.savefig(path+'bandit_learning_error_rbf'+'.png', dpi=300)
plt.savefig(path+'bandit_learning_error_rbf'+'.eps', dpi=300)
plt.show()


plt.figure(figsize=(5,5))
plt.plot(linucb_beta, '-.', label='LINUCB')
plt.plot(gob_beta, label='GOB')
plt.plot(colin_beta, label='CoLin')
plt.plot(lapucb_beta, '-*', markevery=0.1, label='LAPUCB')
plt.plot(lapucb_sim_beta, '-*', markevery=0.1, label='LAPUCB SIM')
plt.plot(club_beta, label='CLUB')
plt.ylabel('beta', fontsize=12)
plt.xlabel('Time', fontsize=12)
plt.legend(loc=1, fontsize=10)
plt.tight_layout()
plt.savefig(path+'bandit_beta_rbf'+'.png', dpi=300)
plt.savefig(path+'bandit_beta_rbf'+'.eps', dpi=300)
plt.show()



