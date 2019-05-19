import numpy as np 
import seaborn as sns
sns.set_style("white")
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import Normalizer, MinMaxScaler
from scipy.sparse import csgraph 
import scipy
import os 
os.chdir('C:/Kaige_Research/Code/graph_bandit/code/')
from utils import *
path='../est_error_bound_ucb_results/'

user_num=20
item_num=500
dimension=10
alpha=1
sigma=0.1
iteration=50*user_num
delta=0.1
beta=0.01

adj=RBF_graph(user_num, dimension)
graph, edge_num=create_networkx_graph(user_num, adj)
pos = nx.spring_layout(graph)


lap=csgraph.laplacian(adj, normed=False)
normed_lap=csgraph.laplacian(adj, normed=True)
user_f=dictionary_matrix_generator(user_num, dimension, lap, 1)
item_f=Normalizer().fit_transform((np.random.normal(size=(item_num, dimension), scale=0.5)))
payoffs=np.dot(user_f, item_f.T)
noise=np.random.normal(size=(user_num, item_num), scale=sigma)
noisy_payoffs=payoffs+noise
user_seq=np.random.choice(range(user_num), size=iteration)
item_seq=np.random.choice(range(item_num), size=iteration)
user_seq=np.ones((user_num, 100))
for a in range(user_num):
	user_seq[a]=user_seq[a]*a
user_seq=user_seq.flatten().astype(int)

user_f_matrix_ls=np.zeros((user_num, dimension))
user_f_matrix_ridge=np.zeros((user_num, dimension))
user_f_matrix_graph=np.zeros((user_num, dimension))
user_f_vector_graph=np.zeros(user_num*dimension)
user_f_matrix_appro=np.zeros((user_num, dimension))
user_f_matrix_avg=np.zeros((user_num, dimension))

A=np.kron(normed_lap+beta*np.identity(user_num), np.identity(dimension))
M=alpha*A 
B=np.zeros(user_num*dimension)
user_bias={}
user_xnoise={}
user_phi={}
user_v={}
user_m={}
user_xx={}

for u in range(user_num):
	user_bias[u]=np.zeros(dimension)
	user_xnoise[u]=np.zeros(dimension)
	user_phi[u]=np.zeros((dimension, dimension))
	user_v[u]=alpha*np.identity(dimension)
	user_m[u]=alpha*np.identity(dimension)
	user_xx[u]=np.zeros((dimension, dimension))

total_error_ls=np.zeros(iteration)
total_error_ridge=np.zeros(iteration)
total_error_graph=np.zeros(iteration)
total_error_appro=np.zeros(iteration)

user_error_ls=np.zeros((user_num, iteration))
user_error_ridge=np.zeros((user_num, iteration))
user_error_graph=np.zeros((user_num, iteration))
user_error_appro=np.zeros((user_num, iteration))

user_error_bound_ridge=np.zeros((user_num, iteration))
user_error_bound_graph=np.zeros((user_num, iteration))

user_confidence_graph=np.zeros((user_num, iteration))
user_confidence_ridge=np.zeros((user_num, iteration))
user_confidence_ridge_old=np.zeros((user_num, iteration))

user_confidence_bound_graph=np.zeros((user_num, iteration))
user_confidence_bound_ridge=np.zeros((user_num, iteration))
user_confidence_bound_ridge_old=np.zeros((user_num, iteration))


for i in range(iteration):
	print('i/iteration', i, iteration)
	user_index=user_seq[i]
	item_index=item_seq[i]
	x=item_f[item_index]
	x_long=np.zeros(user_num*dimension)
	x_long[user_index*dimension:(user_index+1)*dimension]=x
	y=noisy_payoffs[user_index, item_index]
	B+=y*x_long
	M+=np.outer(x_long, x_long)
	user_bias[user_index]+=y*x 
	user_v[user_index]+=np.outer(x,x)
	user_xx[user_index]+=np.outer(x,x)
	user_m[user_index]+=np.outer(x,x)
	user_xnoise[user_index]+=x*noise[user_index, item_index]
	xx_inv=np.linalg.pinv(user_xx[user_index])
	v_inv=np.linalg.pinv(user_v[user_index])
	if np.linalg.norm(xx_inv)>=np.linalg.norm(v_inv):
		xx_inv=v_inv
	else:
		pass
	user_f_matrix_ls[user_index]=np.dot(xx_inv, user_bias[user_index])
	user_f_matrix_ridge[user_index]=np.dot(v_inv, user_bias[user_index])
	user_f_vector_graph=np.dot(np.linalg.pinv(M), B)
	user_f_matrix_graph=user_f_vector_graph.reshape((user_num, dimension))
	user_f_matrix_avg[user_index]=np.dot(user_f_matrix_ls.T, -normed_lap[user_index])+normed_lap[user_index, user_index]*user_f_matrix_ls[user_index]
	user_f_matrix_appro[user_index]=user_f_matrix_ridge[user_index]+alpha*np.dot(xx_inv, user_f_matrix_avg[user_index])

	total_error_ls[i]=np.linalg.norm(user_f_matrix_ls-user_f)
	total_error_ridge[i]=np.linalg.norm(user_f_matrix_ridge-user_f)
	total_error_graph[i]=np.linalg.norm(user_f_matrix_graph-user_f)
	total_error_appro[i]=np.linalg.norm(user_f_matrix_appro-user_f)

	user_error_ls[user_index, i]=np.linalg.norm(user_f_matrix_ls[user_index]-user_f[user_index])
	user_error_ridge[user_index, i]=np.linalg.norm(user_f_matrix_ridge[user_index]-user_f[user_index])
	user_error_graph[user_index, i]=np.linalg.norm(user_f_matrix_graph[user_index]-user_f[user_index])
	user_error_appro[user_index, i]=np.linalg.norm(user_f_matrix_appro[user_index]-user_f[user_index])

	user_error_bound_ridge[user_index,i]=ridge_error_bound(user_v[user_index], user_f_matrix_ridge[user_index], alpha, user_xnoise[user_index])
	user_error_bound_graph[user_index,i]=graph_error_bound(user_index, user_v[user_index], normed_lap, user_f_matrix_ls, user_f_matrix_graph[user_index], alpha, user_xnoise[user_index])

	user_confidence_ridge[user_index,i]=ridge_true_UCB(user_v[user_index], user_f_matrix_ridge[user_index], alpha, user_xnoise[user_index])
	user_confidence_graph[user_index,i]=graph_true_UCB(user_index, user_v[user_index], normed_lap, user_f_matrix_ls, user_f_matrix_graph[user_index], alpha, user_xnoise[user_index])

	user_confidence_bound_ridge[user_index,i]=ridge_UCB(user_v[user_index], user_f_matrix_ridge[user_index], alpha, user_xnoise[user_index])
	user_confidence_bound_ridge_old[user_index,i]=ridge_UCB_old(user_v[user_index], user_f_matrix_ridge[user_index], alpha, user_xnoise[user_index])
	user_confidence_bound_graph[user_index,i]=graph_UCB(user_index, user_v[user_index], normed_lap, user_f_matrix_ls, user_f_matrix_graph[user_index], alpha, user_xnoise[user_index])


np.fill_diagonal(adj,0)
graph, edge_num=create_networkx_graph(user_num, adj)
labels = nx.get_edge_attributes(graph,'weight')
edge_weight=adj[np.triu_indices(user_num,1)]
edge_color=edge_weight[edge_weight>0]
plt.figure(figsize=(5,5))
nodes=nx.draw_networkx_nodes(graph, pos, node_size=10, node_color='y')
edges=nx.draw_networkx_edges(graph, pos, width=0.1, alpha=1, edge_color='k')
edge_labels=nx.draw_networkx_edge_labels(graph,pos, edge_labels=labels, font_size=5)
plt.axis('off')
plt.savefig(path+'graph_of_error_bound_tighness_node_num_edge_num_%s_%s'%(user_num, edge_num)+'.png', dpi=300)
plt.savefig(path+'graph_of_error_bound_tighness_node_num_edge_num_%s_%s'%(user_num, edge_num)+'.eps', dpi=300)
plt.show()

plt.figure(figsize=(5,5))
plt.plot(total_error_ls, label='least-square')
plt.plot(total_error_ridge, label='ridge')
plt.plot(total_error_graph, label='graph')
plt.plot(total_error_appro, label='approx')
plt.legend(loc=0, fontsize=12)
plt.show()