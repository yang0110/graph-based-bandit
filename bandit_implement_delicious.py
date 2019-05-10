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
os.chdir('C:/Kaige_Research/Code/graph_bandit/code/')
from sklearn import datasets
from linucb import LINUCB
from gob import GOB 
from colin import COLIN
from lapucb import LAPUCB
from lapucb_sim import LAPUCB_SIM
from sclub import SCLUB
from club import CLUB
from utils import *
from sklearn.decomposition import NMF
from Recommender.matrix_factor_model import ProductRecommender
input_path='../processed_data/delicious/'
path='../bandit_results/delicious/'

# rate_matrix=np.load(input_path+'rating_matrix_30_user_100_bookmark.npy')
# true_payoffs=rate_matrix/np.max(rate_matrix)
# true_payoffs[true_payoffs==0]=np.nan
# nmf_model=ProductRecommender()
# nmf_model.fit(true_payoffs,5)
# user_feature_matrix, item_feature_matrix=nmf_model.get_models()

# # nmf_model=NMF(n_components=5,random_state=2019)
# # user_feature_matrix=nmf_model.fit_transform(true_payoffs)
# # item_feature_matrix=nmf_model.components_.T
# np.save(input_path+'user_feature_matrix_30.npy', user_feature_matrix)
# np.save(input_path+'item_feature_matrix_100.npy', item_feature_matrix)
# true_payoffs=np.dot(user_feature_matrix, item_feature_matrix.T)
# np.save(input_path+'true_payoffs_30_100.npy', true_payoffs)

####
user_feature_matrix=np.load(input_path+'user_feature_matrix_30.npy')
item_feature_matrix=np.load(input_path+'item_feature_matrix_100.npy')
true_payoffs=np.load(input_path+'true_payoffs_30_100.npy')
user_feature_matrix=Normalizer().fit_transform(user_feature_matrix)
item_feature_matrix=Normalizer().fit_transform(item_feature_matrix)
true_payoffs=np.dot(user_feature_matrix, item_feature_matrix.T)


user_num=true_payoffs.shape[0]
dimension=item_feature_matrix.shape[1]
item_num=item_feature_matrix.shape[0]
pool_size=10
iteration=5000
sigma=0.1# noise
delta=0.1# high probability
alpha=1# regularizer
alpha_2=0.01# edge delete CLUB
beta=0.3 # exploration for CLUB, SCLUB and GOB
thres=0.5
true_adj=rbf_kernel(user_feature_matrix)
true_adj[true_adj<=thres]=0.0
true_normed_adj=true_adj/true_adj.sum(axis=0,keepdims=1)
true_lap=csgraph.laplacian(true_adj)
true_adj_binary=true_adj.copy()
true_adj_binary[true_adj_binary>0]=1
true_lap_binary=np.diag(np.sum(true_adj_binary, axis=1))-true_adj_binary

lap=np.identity(user_num)
normed_lap=lap

noise_matrix=np.zeros((user_num, item_num))
user_seq=np.random.choice(range(user_num), size=iteration)
item_pool_seq=np.random.choice(range(item_num), size=(iteration, pool_size))

linucb_model=LINUCB(dimension, user_num, item_num, pool_size, item_feature_matrix, user_feature_matrix, true_payoffs, alpha, delta, sigma)
gob_model=GOB(dimension, user_num, item_num, pool_size, item_feature_matrix, user_feature_matrix, true_payoffs, true_lap_binary, alpha, delta, sigma, beta)
colin_model=COLIN(dimension, user_num, item_num, pool_size, item_feature_matrix, user_feature_matrix, true_payoffs, true_normed_adj, alpha, delta, sigma, beta)
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


np.fill_diagonal(true_adj,0)
true_adj=np.round(true_adj, decimals=1)
graph, edge_num=create_networkx_graph(user_num, true_adj)
labels = nx.get_edge_attributes(graph,'weight')
edge_weight=true_adj[np.triu_indices(user_num,1)]
edge_color=edge_weight[edge_weight>0]
pos = nx.random_layout(graph)
plt.figure(figsize=(5,5))
nodes=nx.draw_networkx_nodes(graph, pos, node_size=300, node_color='y')
edges=nx.draw_networkx_edges(graph, pos, width=1.0, alpha=1, edge_color='b')
nx.draw_networkx_labels(graph, pos, font_color='k')
edge_labels=nx.draw_networkx_edge_labels(graph,pos, edge_labels=labels)
plt.axis('off')
plt.show()



plt.figure(figsize=(5,5))
plt.plot(linucb_regret,'-.', label='LINUCB')
plt.plot(gob_regret, label='GOB')
plt.plot(colin_regret, label='CoLin')
plt.plot(lapucb_regret, '-*', markevery=0.1, label='LAPUCB')
plt.plot(lapucb_sim_regret, '-*', markevery=0.1, label='LAPUCB SIM')
plt.plot(club_regret, label='CLUB')
plt.plot(sclub_regret, label='SCLUB')
plt.ylabel('Cumulative Regret', fontsize=12)
plt.xlabel('Time', fontsize=12)
plt.legend(loc=2, fontsize=10)
plt.tight_layout()
plt.show()


plt.figure(figsize=(5,5))
plt.plot(linucb_error,'-.', label='LINUCB')
plt.plot(gob_error, label='GOB')
plt.plot(colin_error, label='CoLin')
plt.plot(lapucb_error, '-*', markevery=0.1, label='LAPUCB')
plt.plot(lapucb_sim_error, '-*', markevery=0.1, label='LAPUCB SIM')
plt.plot(club_error, label='CLUB')
plt.plot(sclub_error, label='SCLUB')
plt.ylabel('Error', fontsize=12)
plt.xlabel('Time', fontsize=12)
plt.legend(loc=1, fontsize=10)
plt.tight_layout()
plt.show()


plt.figure(figsize=(5,5))
plt.plot(linucb_beta, '-.', label='LINUCB')
plt.plot(gob_beta, label='GOB')
plt.plot(colin_beta, label='CoLin')
plt.plot(lapucb_beta, '-*', markevery=0.1, label='LAPUCB')
plt.plot(lapucb_sim_beta, '-*', markevery=0.1, label='LAPUCB SIM')
plt.plot(club_beta, label='CLUB')
plt.plot(sclub_beta, label='SCLUB')
plt.ylabel('beta', fontsize=12)
plt.xlabel('Time', fontsize=12)
plt.legend(loc=1, fontsize=10)
plt.tight_layout()
plt.show()
