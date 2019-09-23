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
iteration=1000
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

old_adj=rbf_kernel(np.random.normal(size=(user_num, dimension)), gamma=0.3/dimension)
np.fill_diagonal(old_adj, 0)
# lap=csgraph.laplacian(true_adj, normed=False)
# D=np.diag(np.sum(true_adj, axis=1))
# true_lap=np.dot(np.linalg.inv(D), lap)

er_adj=ER_graph(user_num, 0.2)
ba_adj=BA_graph(user_num, 1)
er_adj=er_adj*old_adj
ba_adj=ba_adj*old_adj

true_adj=old_adj
true_adj[true_adj<0.5]=0
true_adj=er_adj
#true_adj=ba_adj
sparsity=np.sum(true_adj>0)/(user_num*(user_num-1))
print('sparsity', sparsity)

D=np.diag(np.sum(true_adj, axis=1))
lap=D-true_adj
true_lap=np.dot(np.linalg.inv(D), lap)

user_feature_matrix=dictionary_matrix_generator(user_num, dimension, true_lap, 5.5)
smoothness=np.trace(np.dot(np.dot(user_feature_matrix.T, true_lap), user_feature_matrix))
print('smoothness', smoothness)

true_payoffs=np.dot(user_feature_matrix, item_feature_matrix.T)+noise_matrix

edges=true_adj.ravel()
plt.figure(figsize=(5,5))
plt.hist(edges)
plt.ylabel('Counts', fontsize=12)
plt.xlabel('Edge weights', fontsize=12)
plt.tight_layout()
plt.savefig(path+'hist_edge_weights'+'.png', dpi=100)
plt.show()


plot_adj=np.round(true_adj, decimals=2)
graph, edge_num=create_networkx_graph(user_num, plot_adj)
labels = nx.get_edge_attributes(graph,'weight')
edge_weight=plot_adj[np.triu_indices(user_num, 1)]
edge_color=edge_weight[edge_weight>0]
pos = nx.spring_layout(graph)

plt.figure(figsize=(5,5))
nodes=nx.draw_networkx_nodes(graph, pos, node_size=10, node_color='y')
edges=nx.draw_networkx_edges(graph, pos, width=0.1, alpha=1, edge_color='k')
edge_labels=nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels, font_size=5)
plt.axis('off')
plt.savefig(path+'network_rbf_sparse'+'.png', dpi=100)
plt.show()


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
	gob_model=GOB(dimension, user_num, item_num, pool_size, item_feature_matrix, user_feature_matrix, true_payoffs, true_adj, true_lap, alpha, delta, sigma, beta, state)
	lapucb_model=LAPUCB(dimension, user_num, item_num, pool_size, item_feature_matrix, user_feature_matrix, true_payoffs, true_adj, true_lap, noise_matrix, alpha, delta, sigma, beta, thres, state)
	lapucb_sim_model=LAPUCB_SIM(dimension, user_num, item_num, pool_size, item_feature_matrix, user_feature_matrix, true_payoffs, true_adj, true_lap, noise_matrix, alpha, delta, sigma, beta, thres, state)
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

plt.figure(figsize=(5,5))
plt.plot(linucb_regret,'-.', markevery=0.1, label='LinUCB')
plt.plot(gob_regret, '-p', color='orange', markevery=0.1, label='GOB.Lin')
plt.plot(lapucb_sim_regret, '-s', markevery=0.1, label='GraphUCB-Local')
plt.plot(lapucb_regret, '-o', markevery=0.1, label='GraphUCB')
plt.plot(club_regret,'-*', markevery=0.1, label='CLUB')
plt.ylabel('Cumulative Regret', fontsize=16)
plt.xlabel('Time', fontsize=16)
plt.title('Smoothness=%s, Sparsity=%s'%(np.int(smoothness), np.round(sparsity, decimals=1)), fontsize=16)
plt.legend(loc=4, fontsize=14)
plt.tight_layout()
plt.savefig(path+'rbf_sparse'+'.png', dpi=100)
plt.show()


plt.figure(figsize=(5,5))
plt.plot(linucb_error,'-.', markevery=0.1, label='LinUCB')
plt.plot(gob_error, '-p', color='orange', markevery=0.1, label='GOB.Lin')
plt.plot(lapucb_sim_error, '-s', markevery=0.1, label='GraphUCB-Local')
plt.plot(lapucb_error, '-o', markevery=0.1, label='GraphUCB')
plt.plot(club_error, '-*', markevery=0.1, label='CLUB')
plt.ylabel('Error', fontsize=14)
plt.xlabel('Time', fontsize=14)
plt.legend(loc=1, fontsize=14)
plt.tight_layout()
#plt.savefig(path+'error'+'.png', dpi=100)
plt.show()


plt.figure(figsize=(5,5))
plt.plot(linucb_beta,'-.', markevery=0.1, label='LinUCB')
plt.plot(gob_beta, '-p', color='orange', markevery=0.1, label='GOB.Lin')
plt.plot(lapucb_sim_beta, '-s', markevery=0.1, label='GraphUCB-Local')
plt.plot(lapucb_beta, '-o', color='red', markevery=0.1, label='GraphUCB')
#plt.plot(club_beta,'-p', markevery=0.1, label='CLUB')
plt.ylabel('Beta', fontsize=16)
plt.xlabel('Time', fontsize=16)
plt.legend(loc=0, fontsize=16)
plt.tight_layout()
#plt.savefig(path+'beta'+'.png', dpi=100)
plt.show()



plt.figure(figsize=(5,5))
plt.plot(linucb_x_norm,'-.', markevery=0.1, label='LinUCB')
plt.plot(gob_x_norm, '-p', color='orange', markevery=0.1, label='GOB.Lin')
plt.plot(lapucb_sim_x_norm, '-s', markevery=0.1, label='GraphUCB-Local')
plt.plot(lapucb_x_norm, '-o',color='red', markevery=0.1, label='GraphUCB')
#plt.plot(club_x_norm, label='CLUB')
plt.ylabel('x norm', fontsize=16)
plt.xlabel('Time', fontsize=16)
plt.legend(loc=1, fontsize=16)
plt.tight_layout()
#plt.savefig(path+'x_norm'+'.png', dpi=100)
plt.show()




plt.figure(figsize=(5,5))
plt.plot(linucb_ucb,'-.', markevery=0.1, label='LinUCB')
plt.plot(gob_ucb, '-p', color='orange', markevery=0.1, label='GOB.Lin')
plt.plot(lapucb_sim_ucb, '-s', markevery=0.1, label='GraphUCB-Local')
plt.plot(lapucb_ucb, '-o',color='red', markevery=0.1, label='GraphUCB')
#plt.plot(club_x_norm, label='CLUB')
plt.ylabel('UCB', fontsize=16)
plt.xlabel('Time', fontsize=16)
plt.legend(loc=1, fontsize=16)
plt.tight_layout()
#plt.savefig(path+'UCB'+'.png', dpi=100)
plt.show()

# payoffs=true_payoffs.ravel()
# plt.figure(figsize=(5,5))
# plt.hist(payoffs)
# plt.ylabel('Counts', fontsize=12)
# plt.xlabel('Payoffs', fontsize=12)
# plt.tight_layout()
# plt.savefig(path+'hist_payoffs'+'.png', dpi=100)
# plt.clf()