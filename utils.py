import numpy as np 
import networkx as nx
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import Normalizer, MinMaxScaler
from scipy.sparse import csgraph 
import scipy
import os 
from sklearn import datasets

def create_networkx_graph(node_num, adj_matrix):
	G=nx.Graph()
	G.add_nodes_from(list(range(node_num)))
	for i in range(node_num):
		for j in range(node_num):
			if adj_matrix[i,j]!=0.0:
				G.add_edge(i,j,weight=adj_matrix[i,j])
			else:
				pass
	return G, G.number_of_edges()


def RBF_graph(node_num, dimension, gamma=None, thres=None, clusters=False): ##
	if clusters==False:
		node_f=np.random.uniform(low=-0.5, high=0.5, size=(node_num, dimension))
	else:
		node_f, _=datasets.make_blobs(n_samples=node_num, n_features=dimension, centers=5, cluster_std=0.1, center_box=(-1,1),  shuffle=False, random_state=2019)
	if gamma==None:
		gamma=0.5
	else:
		pass 
	adj=rbf_kernel(node_f, gamma=gamma)
	adj=np.round(adj, decimals=2)
	if thres!=None:
		adj[adj<=thres]=0.0
	else:
		pass
	np.fill_diagonal(adj,1)
	return adj 

def ER_graph(node_num, prob):## unweighted
	G=nx.erdos_renyi_graph(node_num, prob)
	adj=nx.to_numpy_matrix(G)
	adj=np.asarray(adj)
	adj=adj+np.identity(node_num)
	return adj 

def BA_graph(node_num, edge_num): #unweighted 
	G=nx.barabasi_albert_graph(node_num, edge_num)
	adj=nx.to_numpy_matrix(G)
	adj=np.asarray(adj)
	adj=adj+np.identity(node_num)
	return adj 
 
def graph_signal_samples_from_laplacian(laplacian, node_num, dimension):
	cov=np.linalg.pinv(laplacian)
	samples=np.random.multivariate_normal(np.zeros(node_num),cov, size=dimension).T
	return samples 

def dictionary_matrix_generator(node_num, dimension, laplacian, lambda_):
	D0=np.random.normal(size=(node_num, dimension))
	D=np.dot(np.linalg.pinv(np.identity(node_num)+lambda_*laplacian), D0)
	D=Normalizer().fit_transform(D)
	return D


def normalized_trace(matrix, target_trace):
	normed_matrix=target_trace*matrix/np.trace(matrix)
	return normed_matrix


def calculate_graph_approximation(i, dimension, user_num, user_index, alpha, normed_lap, user_v_i, user_f_matrix_ls, user_f_matrix_ridge):
	ridge=user_f_matrix_ridge[user_index]
	if i>=10*user_num:
		avg=np.zeros(dimension)
		for u in range(user_num):
			if u==user_index:
				pass 
			else:
				avg+=-normed_lap[user_index, u]*user_f_matrix_ls[u]
		graph=ridge+alpha*np.dot(np.linalg.pinv(user_v_i), avg)
	else:
		graph=ridge
	return graph 

def modify_normed_lap(normed_lap):
	for i in range(normed_lap.shape[0]):
		if normed_lap[i,i]==0.0:
			normed_lap[i,i]=1
		else:
			pass
	return normed_lap
