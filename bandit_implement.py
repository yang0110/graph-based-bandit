import numpy as np 
import networkx as nx
import seaborn as sns
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
path='../bandit_results/'

user_num=10
dimension=5
item_num=100
pool_size=10
iteration=500
sigma=0.1# noise
delta=0.1# high probability
alpha=1# regularizer
alpha_2=0.01# edge delete CLUB
beta=0.5 # exploration for CLUB, SCLUB and GOB

user_seq=np.random.choice(range(user_num), size=iteration)
item_pool_seq=np.random.choice(range(item_num), size=(iteration, pool_size))
item_feature_matrix=Normalizer().fit_transform(np.random.normal(size=(item_num, dimension)))

adj=RBF_graph(user_num, dimension, thres=0.0)
adj=BA_graph(user_num, 3)
#adj=ER_graph(user_num, 0.5)
normed_adj=adj/adj.sum(axis=0,keepdims=1)
lap=csgraph.laplacian(adj, normed=False)
normed_lap=csgraph.laplacian(adj, normed=True)
adj_binary=adj.copy()
adj_binary[adj_binary>0]=1
lap_binary=np.diag(np.sum(adj_binary, axis=1))-adj_binary
user_feature_matrix=dictionary_matrix_generator(user_num, dimension, lap, 5)
noise_matrix=np.random.normal(scale=sigma, size=(user_num, item_num))
true_payoffs=np.dot(user_feature_matrix, item_feature_matrix.T)+noise_matrix

linucb_model=LINUCB(dimension, user_num, item_num, pool_size, item_feature_matrix, user_feature_matrix, true_payoffs, alpha, delta, sigma)
gob_model=GOB(dimension, user_num, item_num, pool_size, item_feature_matrix, user_feature_matrix, true_payoffs, lap_binary, alpha, delta, sigma, beta)
colin_model=COLIN(dimension, user_num, item_num, pool_size, item_feature_matrix, user_feature_matrix, true_payoffs, normed_adj, alpha, delta, sigma, beta)
lapucb_model=LAPUCB(dimension, user_num, item_num, pool_size, item_feature_matrix, user_feature_matrix, true_payoffs, noise_matrix, normed_lap, alpha, delta, sigma)
lapucb_sim_model=LAPUCB_SIM(dimension, user_num, item_num, pool_size, item_feature_matrix, user_feature_matrix, true_payoffs, noise_matrix, normed_lap, alpha, delta, sigma)
club_model = CLUB(dimension, user_num, item_num, pool_size, item_feature_matrix, user_feature_matrix, true_payoffs,normed_lap, alpha, alpha_2, delta, sigma, beta)
sclub_model = SCLUB(dimension, user_num, item_num, pool_size, item_feature_matrix, user_feature_matrix, true_payoffs,normed_lap, alpha, delta, sigma, beta)

linucb_regret, linucb_error, linucb_beta=linucb_model.run(user_seq, item_pool_seq, iteration)
gob_regret, gob_error, gob_beta=gob_model.run(user_seq, item_pool_seq, iteration)
colin_regret, colin_error, colin_beta=colin_model.run(user_seq, item_pool_seq, iteration)
lapucb_regret, lapucb_error, lapucb_beta=lapucb_model.run(user_seq, item_pool_seq, iteration)
lapucb_sim_regret, lapucb_sim_error, lapucb_sim_beta=lapucb_sim_model.run(user_seq, item_pool_seq, iteration)
club_regret, club_error,club_graph_error, club_cluster_num, club_beta=club_model.run(user_seq, item_pool_seq, iteration)
sclub_regret, sclub_error,sclub_graph_error, sclub_cluster_num, sclub_beta=sclub_model.run(user_seq, item_pool_seq, iteration)


np.fill_diagonal(adj,0)
graph, edge_num=create_networkx_graph(user_num, adj)
labels = nx.get_edge_attributes(graph,'weight')
edge_weight=adj[np.triu_indices(user_num,1)]
edge_color=edge_weight[edge_weight>0]
pos = nx.spring_layout(graph)
plt.figure(figsize=(5,5))
nodes=nx.draw_networkx_nodes(graph, pos, node_size=300, node_color='y')
edges=nx.draw_networkx_edges(graph, pos, width=1.0, alpha=1, edge_color='k')
nx.draw_networkx_labels(graph, pos, font_color='k')
edge_labels=nx.draw_networkx_edge_labels(graph,pos, edge_labels=labels)
#nx.draw_networkx_edge_labels(graph, pos,font_color='k')
plt.axis('off')
plt.savefig(path+'network_rbf'+'.png', dpi=300)
plt.savefig(path+'network_rbf'+'.eps', dpi=300)
plt.show()

plt.figure(figsize=(5,5))
plt.plot(linucb_regret, label='LINUCB')
plt.plot(gob_regret, label='GOB')
plt.plot(colin_regret, label='CoLin')
plt.plot(lapucb_regret, label='LAPUCB')
plt.plot(lapucb_sim_regret, label='LAPUCB SIM')
plt.plot(club_regret, label='CLUB')
plt.plot(sclub_regret, label='SCLUB')
plt.ylabel('Cumulative Regret', fontsize=12)
plt.xlabel('Time', fontsize=12)
plt.legend(loc=2, fontsize=10)
plt.tight_layout()
plt.savefig(path+'cum_regret_rbf'+'.png')
plt.savefig(path+'cum_regret_rbf'+'.eps')
plt.show()
 
plt.figure(figsize=(5,5))
plt.plot(linucb_error, label='LINUCB')
plt.plot(gob_error, label='GOB')
plt.plot(colin_error, label='CoLin')
plt.plot(lapucb_error, label='LAPUCB')
plt.plot(lapucb_sim_error, label='LAPUCB SIM')
plt.plot(club_error, label='CLUB')
plt.plot(sclub_error, label='SCLUB')
plt.ylabel('Error', fontsize=12)
plt.xlabel('Time', fontsize=12)
plt.legend(loc=1, fontsize=10)
plt.tight_layout()
plt.savefig(path+'bandit_learning_error_rbf'+'.png')
plt.savefig(path+'bandit_learning_error_rbf'+'.eps')
plt.show()


plt.figure(figsize=(5,5))
plt.plot(linucb_beta, label='LINUCB')
plt.plot(gob_beta, label='GOB')
plt.plot(colin_beta, label='CoLin')
plt.plot(lapucb_beta, label='LAPUCB')
plt.plot(lapucb_sim_beta, label='LAPUCB SIM')
plt.plot(club_beta, label='CLUB')
plt.plot(sclub_beta, label='SCLUB')
plt.ylabel('beta', fontsize=12)
plt.xlabel('Time', fontsize=12)
plt.legend(loc=1, fontsize=10)
plt.tight_layout()
plt.savefig(path+'bandit_beta_rbf'+'.png', dpi=300)
plt.savefig(path+'bandit_beta_rbf'+'.eps', dpi=300)
plt.show()


