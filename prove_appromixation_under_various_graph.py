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
path='../approximation_results/'

user_num=10
item_num=100
dimension=5
alpha=1
sigma=0.1
iteration=500
delta=0.1
beta=0.01
thres_list=[0.5, 0.7, 0.9]

adj=RBF_graph(user_num, dimension)

for ind, thres in enumerate(thres_list):
	print('thres/thres_list', thres, thres_list)
	adj[adj<=thres]=0.0
	graph, edge_num=create_networkx_graph(user_num, adj)
	if ind==0:
		pos = nx.spring_layout(graph)
	else:
		pass

	lap=csgraph.laplacian(adj, normed=False)
	normed_lap=csgraph.laplacian(adj, normed=True)
	normed_lap=modify_normed_lap(normed_lap)
	user_f=dictionary_matrix_generator(user_num, dimension, lap, 5)
	item_f=Normalizer().fit_transform((np.random.normal(size=(item_num, dimension), scale=0.5)))
	payoffs=np.dot(user_f, item_f.T)
	noise=np.random.normal(size=(user_num, item_num), scale=sigma)
	noisy_payoffs=payoffs+noise
	user_seq=np.random.choice(range(user_num), size=iteration)
	item_seq=np.random.choice(range(item_num), size=iteration)

	user_f_vector_ls=np.zeros((user_num*dimension))
	user_f_vector_ridge=np.zeros(user_num*dimension)
	user_f_vector_graph=np.zeros(user_num*dimension)

	user_f_matrix_ls=np.zeros((user_num, dimension))
	user_f_matrix_ridge=np.zeros((user_num, dimension))
	user_f_matrix_graph=np.zeros((user_num, dimension))
	user_f_matrix_appro=np.zeros((user_num, dimension))

	A=np.kron(normed_lap+beta*np.identity(user_num), np.identity(dimension))
	M=alpha*A 
	B=np.zeros(user_num*dimension)
	user_bias={}
	user_xnoise={}
	user_phi={}
	user_v={}
	user_m={}
	user_xx={}
	user_error_ls={}
	user_error_ridge={}
	user_error_graph={}
	user_error_appro={}

	for u in range(user_num):
		user_bias[u]=np.zeros(dimension)
		user_xnoise[u]=np.zeros(dimension)
		user_phi[u]=np.zeros((dimension, dimension))
		user_v[u]=alpha*np.identity(dimension)
		user_m[u]=alpha*np.identity(dimension)
		user_xx[u]=np.zeros((dimension, dimension))
		user_error_ls[u]=[]
		user_error_ridge[u]=[]
		user_error_graph[u]=[]
		user_error_appro[u]=[]


	total_error_ls=np.zeros(iteration)
	total_error_ridge=np.zeros(iteration)
	total_error_graph=np.zeros(iteration)
	total_error_appro=np.zeros(iteration)


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

		user_f_matrix_ls[user_index]=np.dot(np.linalg.pinv(user_xx[user_index]), user_bias[user_index])
		user_f_matrix_ridge[user_index]=np.dot(np.linalg.pinv(user_v[user_index]), user_bias[user_index])
		user_f_vector_graph=np.dot(np.linalg.pinv(M), B)
		user_f_matrix_graph=user_f_vector_graph.reshape((user_num, dimension))
		user_f_matrix_appro[user_index]=calculate_graph_approximation(i, dimension, user_num, user_index, alpha, normed_lap,user_v[user_index], user_f_matrix_ls, user_f_matrix_ridge)

		total_error_ls[i]=np.linalg.norm(user_f_matrix_ls-user_f)
		total_error_ridge[i]=np.linalg.norm(user_f_matrix_ridge-user_f)
		total_error_graph[i]=np.linalg.norm(user_f_matrix_graph-user_f)
		total_error_appro[i]=np.linalg.norm(user_f_matrix_appro-user_f)

		user_error_ls[user_index].extend([np.linalg.norm(user_f_matrix_ls[user_index]-user_f[user_index])])
		user_error_ridge[user_index].extend([np.linalg.norm(user_f_matrix_ridge[user_index]-user_f[user_index])])
		user_error_graph[user_index].extend([np.linalg.norm(user_f_matrix_graph[user_index]-user_f[user_index])])
		user_error_appro[user_index].extend([np.linalg.norm(user_f_matrix_appro[user_index]-user_f[user_index])])

	plt.figure(figsize=(5,5))
	#plt.plot(total_error_ls, label='least-square')
	plt.plot(total_error_ridge, label='Ridge')
	plt.plot(total_error_graph, label='Graph')
	plt.plot(total_error_appro, label='Graph-Approx')
	plt.ylabel('Estimation Error', fontsize=12)
	plt.xlabel('Time', fontsize=12)
	plt.tight_layout()
	plt.legend(loc=0, fontsize=10)
	plt.savefig(path+'approximation_tighness_node_num_edge_num_thres_%s_%s_%s'%(user_num, edge_num, int(thres*10))+'.png', dpi=300)
	plt.savefig(path+'approximation_tighness_node_num_edge_num_thres_%s_%s_%s'%(user_num, edge_num, int(thres*10))+'.eps', dpi=300)
	plt.show()

	# for u in range(user_num):
	# 	plt.figure(figsize=(5,5))
	# 	plt.plot(user_error_ridge[u], label='Ridge')
	# 	plt.plot(user_error_graph[u], label='Graph')
	# 	plt.plot(user_error_appro[u], label='Graph-Appro')
	# 	plt.ylabel('Estimation Error', fontsize=12)
	# 	plt.xlabel('Time', fontsize=12)
	# 	plt.title('node, edge, T = %s, %s, %s'%(user_num, edge_num, thres), fontsize=12)
	# 	plt.tight_layout()
	# 	plt.legend(loc=0, fontsize=10)
	# 	plt.show()


	np.fill_diagonal(adj,0)
	graph, edge_num=create_networkx_graph(user_num, adj)
	labels = nx.get_edge_attributes(graph,'weight')
	edge_weight=adj[np.triu_indices(user_num,1)]
	edge_color=edge_weight[edge_weight>0]
	plt.figure(figsize=(5,5))
	nodes=nx.draw_networkx_nodes(graph, pos, node_size=300, node_color='y')
	edges=nx.draw_networkx_edges(graph, pos, width=1.0, alpha=1, edge_color='k')
	nx.draw_networkx_labels(graph, pos, font_color='k')
	edge_labels=nx.draw_networkx_edge_labels(graph,pos, edge_labels=labels)
	plt.axis('off')
	plt.savefig(path+'graph_of_approximation_tighness_node_num_edge_num_thres_%s_%s_%s'%(user_num, edge_num, int(thres*10))+'.png', dpi=300)
	plt.savefig(path+'graph_of_approximation_tighness_node_num_edge_num_thres_%s_%s_%s'%(user_num, edge_num, int(thres*10))+'.eps', dpi=300)
	plt.show()

