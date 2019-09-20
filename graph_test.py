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
dimension=10
pool_size=20
iteration=2000
loop=1
sigma=0.01# noise
delta=0.1# high probability
alpha=1# regularizer
alpha_2=0.1# edge delete CLUB
epsilon=8 # Ts
beta=0.15# exploration for CLUB, SCLUB and GOB
thres=0.0
state=False # False for artificial dataset, True for real dataset
lambda_list=[4]


item_feature_matrix=Normalizer().fit_transform(np.random.normal(size=(item_num, dimension)))

neighbor_num=3
ws_adj=WS_graph(user_num, neighbor_num, 0.1)
er_adj=ER_graph(user_num, 0.2)
ba_adj=BA_graph(user_num, 3)
random_weights=np.round(np.random.uniform(size=(user_num, user_num)), decimals=2)
random_weights=(random_weights.T+random_weights)/2
ws_adj=ws_adj*random_weights
er_adj=er_adj*random_weights
ba_adj=ba_adj*random_weights



true_adj=rbf_kernel(np.random.normal(size=(user_num, dimension)), gamma=0.25/dimension)
#true_adj=ws_adj
#true_adj=er_adj
#true_adj=ba_adj
threshold_list=np.linspace(0, 0.75, 5)

average_local=np.zeros((5, user_num))
theta_norm=np.zeros(user_num)
for index_thres, thres in enumerate(threshold_list):
	true_adj[true_adj<=thres]=0
	#np.fill_diagonal(true_adj,0)

# edges=true_adj.ravel()
# plt.figure(figsize=(5,5))
# plt.hist(edges)
# plt.ylabel('Counts', fontsize=12)
# plt.xlabel('Edge weights', fontsize=12)
# plt.tight_layout()
# plt.savefig(path+'hist_edge_weights'+'.png', dpi=100)
# plt.clf()


	lap=csgraph.laplacian(true_adj, normed=False)
	D=np.diag(np.sum(true_adj, axis=1))
	true_lap=np.dot(np.linalg.inv(D), lap)
	noise_matrix=np.random.normal(scale=sigma, size=(user_num, item_num))


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
# plt.savefig(path+'network_ba'+'.eps', dpi=300)
# plt.clf()


	list_length=20
	lambda_list=np.round(np.linspace(0,100,list_length), decimals=2)
	local=np.zeros((user_num, list_length))


	for index, lambda_ in enumerate(lambda_list):
		user_feature_matrix=dictionary_matrix_generator(user_num, dimension, true_lap, lambda_)
		for user in range(user_num):
			lap_row=true_lap[user]
			average=np.dot(user_feature_matrix.T, lap_row)
			local[user, index]=np.linalg.norm(average)
			theta_norm[user]=np.linalg.norm(user_feature_matrix[user])

	average_local[index_thres]=np.mean(local, axis=0)


plt.figure(figsize=(5,5))
for ind, th in enumerate(threshold_list):
	plt.plot(lambda_list, average_local[ind], '-.', label='Sparsity='+np.str(np.round(th, decimals=2)))
plt.plot(lambda_list, theta_norm, label='||theta||_2')
plt.ylabel('Delta', fontsize=16)
plt.xlabel('Smoothness (Lambda)', fontsize=16)
plt.legend(loc=0, fontsize=14)
plt.savefig(path+'local_smoothness'+'.png', dpi=100)
plt.show()