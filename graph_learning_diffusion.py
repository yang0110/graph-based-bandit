import numpy as np 
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import Normalizer, MinMaxScaler
from scipy.sparse import csgraph 
import scipy
import os 
from sklearn import datasets
os.chdir('Documents/research/graph_bandit/code/')
from bandit_model import *
from utils import *
os.chdir('Documents/research/graph_bandit/results/')

node_num=10
dimension=5
step_size=0.001
iteration=100
pos=np.random.normal(size=(node_num, 2))
graph=nx.barabasi_albert_graph(node_num, 5)
adj=np.array(nx.to_numpy_matrix(graph))*np.random.uniform(size=(node_num, node_num))
lap=csgraph.laplacian(adj, normed=False)

user_f=np.random.random(size=(node_num, dimension))
adj=rbf_kernel(user_f)
np.fill_diagonal(adj,0)
lap=csgraph.laplacian(adj, normed=False)

def generate_signal(lap, s_t_1):
	p_t=np.random.normal(size=1, scale=1)
	s_t=np.dot(np.exp(-lap), s_t_1)+p_t
	return s_t

s_0=np.random.uniform(low=0, high=1, size=node_num)
for i in range(20):
	if i==0:
		s_t_1=s_0
	else:
		pass 
	print(s_t_1)
	s_t=generate_signal(lap, s_t_1)
	s_t_1=s_t.copy()


def projection(w_t_bar):
	w_t_bar[w_t_bar<(-1/node_num)]=-1/node_num
	w_t_bar=(w_t_bar+w_t_bar.T)/2
	w_t_bar=w_t_bar-(1/node_num)*w_t_bar
	return w_t_bar

s_0=np.random.normal(size=node_num)
s_0_bar=s_0-s_0*(1/node_num)
w_0_bar=np.zeros((node_num, node_num))

for i in range(iteration):
	print('i=',i)
	if i==0:
		s_t_1=s_0
		s_t_1_bar=s_0_bar
		w_t_1_bar=w_0_bar
	else:
		pass 
	s_t=generate_signal(lap, s_t_1)
	s_t_1=s_t.copy()
	s_t_bar=s_t-(1/node_num)*s_t
	s_t_1_bar=s_t_1-(1/node_num)*s_t_1
	
	w_t_bar=w_t_1_bar+step_size*np.outer(s_t_bar-np.dot(w_t_1_bar, s_t_1_bar), s_t_1_bar)
	w_t_bar=projection(w_t_bar)
	w_t_1_bar=w_t_bar
	print(s_t)
	print(w_t_bar)

w_t=w_t_bar+(1/node_num)*np.ones((node_num, node_num))
lap_learnt=-np.log(w_t)
adj_learnt=-lap_learnt
np.fill_diagonal(adj_learnt,0)

graph=create_networkx_graph(node_num, adj)
edge_num=graph.number_of_edges()
edge_weights=adj[np.triu_indices(node_num,1)]
edge_color=edge_weights[edge_weights>0]
edge_alpha=edge_color
nodes=nx.draw_networkx_nodes(graph, pos, node_color=s_0, node_size=100, cmap=plt.cm.Reds)
edges=nx.draw_networkx_edges(graph, pos, width=1.0, alpha=1.0, edge_color=edge_color, edge_cmap=plt.cm.Blues)
plt.axis('off')
plt.show()

graph=create_networkx_graph(node_num, adj_learnt)
edge_num=graph.number_of_edges()
edge_weights=adj_learnt[np.triu_indices(node_num,1)]
edge_color=edge_weights[edge_weights>0]
edge_alpha=edge_color
nodes=nx.draw_networkx_nodes(graph, pos, node_color=s_0, node_size=100, cmap=plt.cm.Reds)
edges=nx.draw_networkx_edges(graph, pos, width=1.0, alpha=1.0, edge_color=edge_color, edge_cmap=plt.cm.Blues)
plt.axis('off')
plt.show()


