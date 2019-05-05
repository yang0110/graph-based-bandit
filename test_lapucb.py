import numpy as np 
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import Normalizer, MinMaxScaler
from scipy.sparse import csgraph 
import scipy
import os 
from sklearn import datasets
os.chdir('C:/Kaige_Research/Code/graph_bandit/code/')
from lapucb import LAPUCB
from neighbor_lapucb import Neighbor_LAPUCB 
from utils import *
path='../results/'

user_num=20
dimension=5
item_num=100
iteration=1000
sigma=0.1# nosie 
alpha=0.1# regularizer 


adj=RBF_graph(user_num, dimension, thres=0.75)
#adj=BA_graph(user_num, 3)
#adj=ER_graph(user_num, 0.5)
lap=csgraph.laplacian(adj, normed=False)
A=np.kron(lap, np.identity(dimension))
user_feature_matrix=graph_signal_samples_from_laplacian(lap, user_num, dimension)
user_feature_matrix=Normalizer().fit_transform(user_feature_matrix)
user_feature_matrix=dictionary_matrix_generator(user_num, dimension, lap, 5)

item_feature_matrix=np.random.normal(size=(item_num,dimension))
item_feature_matrix=Normalizer().fit_transform(item_feature_matrix)
true_payoffs=np.dot(user_feature_matrix, item_feature_matrix.T)+np.random.normal(scale=sigma, size=(user_num, item_num))

user_cov={}
user_bias={}
neighbor_cov_inv=np.zeros((len(neighbors)*dimension))
neighbor_bias=np.zeros(len(neighbors)*dimension)
cov=alpha*A+alpha*np.identity(user_num*dimension)
bias=np.zeros(user_num*dimension)
neighbor_dict={}
user_neighbor_index={}

for u in range(user_num):
	print('initial user', u)
	adj_row=adj[u]
	neighbors=np.where(adj_row>0)[0].tolist()
	neighbor_dict[u]=neighbors
	index=np.where(np.array(neighbors)==u)[0].tolist()[0]
	user_neighbor_index[u]=index
	lap=csgraph.laplacian(adj, normed=False)
	user_cov[u]=alpha*np.identity(dimension)
	user_bias[u]=np.zeros(dimension)


user_seq=np.random.choice(range(user_num), size=iteration)
item_seq=np.random.choice(range(item_num), size=iteration)
learnt_user_feature_matrix=np.zeros((user_num, dimension))
learnt_user_feature_vector=np.zeros((user_num,dimension))
learnt_neighbor_feature_matrix=np.zeros((user_num, dimension))
error_1=[]
error_2=[]
error_3=[]
error_4=[]
for i in range(iteration):
	print('i/iteration', i, iteration)
	user_index=user_seq[i]
	item_index=item_seq[i]
	y=true_payoffs[user_index, item_index]
	x=item_feature_matrix[item_index]
	x_long=np.zeros(user_num*dimension)
	x_long[user_index*dimension:(user_index+1)*dimension]=x
	neighbors=neighbor_dict[user_index]
	index=user_neighbor_index[user_index]
	x_short=np.zeros(len(neighbors)*dimension)
	x_short[index*dimension:(index+1)*dimension]=x
	cov+=np.outer(x_long, x_long)
	bias+=y*x_long
	user_cov[user_index]+=np.outer(x, x)
	user_bias[user_index]+=y*x 
	cov_inv=np.linalg.pinv(cov)
	neighbor_index=np.zeros((len(neighbors), dimension))
	for j, nei in enumerate(neighbors):
		neighbor_index[j]=np.arange(nei*dimension,(nei+1)*dimension)
	neighbor_index=neighbor_index.flatten().astype(int)
	sub_cov_inv=cov_inv[user_index*dimension:(user_index+1)*dimension]
	neighbor_cov_inv=sub_cov_inv
	neighbor_bias=bias
	learnt_user_feature_matrix[user_index]=np.dot(np.linalg.pinv(user_cov[user_index]), user_bias[user_index])
	learnt_user_feature_vector[user_index]=np.dot(cov_inv, bias).reshape(user_num, dimension)[user_index]
	learnt_neighbor_feature_matrix[user_index]=np.dot(neighbor_cov_inv, neighbor_bias)
	error_1.extend([np.linalg.norm(learnt_user_feature_matrix-user_feature_matrix)])
	error_2.extend([np.linalg.norm(learnt_user_feature_vector-user_feature_matrix)])
	error_3.extend([np.linalg.norm(learnt_neighbor_feature_matrix-user_feature_matrix)])


plt.figure(figsize=(5,5))
plt.plot(error_1, label='LINUCB')
plt.plot(error_2, label='LAPUCB')
plt.plot(error_3, label='LAPUCB Neighbor')
plt.legend(loc=0)
plt.show()

