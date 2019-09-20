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

user_num=10
item_num=500
dimension=10
alpha=1
sigma=0.5
iteration=5000
delta=0.01
beta=0.01

adj=RBF_graph(user_num, dimension)
graph, edge_num=create_networkx_graph(user_num, adj)
pos = nx.spring_layout(graph)

lap=csgraph.laplacian(adj, normed=False)
normed_lap=csgraph.laplacian(adj, normed=True)
user_f=dictionary_matrix_generator(user_num, dimension, lap, 1)
item_f=Normalizer().fit_transform((np.random.normal(size=(item_num, dimension))))
payoffs=np.dot(user_f, item_f.T)
noise=np.random.normal(size=(user_num, item_num), scale=sigma)
noisy_payoffs=payoffs+noise

user_seq=np.random.choice(range(user_num), size=iteration)
item_seq=np.random.choice(range(item_num), size=iteration)

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
user_v={}
user_xx={}
user_graph_error={}
user_approx_error={}
user_ridge_error={}
total_graph_error=[]
total_approx_error=[]
total_ls_error=[]
total_ridge_error=[]
graph_error_bound_list=[]
ridge_error_bound_list=[]
user_xnoise=np.zeros((user_num, dimension))

user_graph_error_bound={}
user_ridge_error_bound={}

total_new_ridge_ucb_list=[]
total_old_ridge_ucb_list=[]
total_graph_ucb_list=[]
a_graph_list=[]
a_ridge_list=[]
for u in range(user_num):
	user_bias[u]=np.zeros(dimension)
	user_v[u]=alpha*np.identity(dimension)
	user_xx[u]=np.zeros((dimension, dimension))
	user_graph_error[u]=[]
	user_approx_error[u]=[]
	user_graph_error_bound[u]=[]
	user_ridge_error[u]=[]
	user_ridge_error_bound[u]=[]

for i in range(iteration):
	print('time/iteration', i, iteration)
	user_index=user_seq[i]
	item_index=item_seq[i]
	x=item_f[item_index]
	y=payoffs[user_index, item_index]
	n=noise[user_index, user_index]
	user_xnoise[user_index]+=x*n
	x_long=np.zeros(user_num*dimension)
	x_long[user_index*dimension:(user_index+1)*dimension]=x 
	M+=np.outer(x_long, x_long)
	B+=y*x_long
	user_xx[user_index]+=np.outer(x,x)
	user_v[user_index]+=np.outer(x,x)
	user_bias[user_index]+=y*x 
	M_inv=np.linalg.pinv(M)
	user_f_vector_graph=np.dot(M_inv, B)
	user_f_matrix_graph=user_f_vector_graph.reshape((user_num, dimension))
	xx_inv=np.linalg.pinv(user_xx[user_index])
	v_inv=np.linalg.pinv(user_v[user_index])
	if np.linalg.norm(xx_inv)>2*np.linalg.norm(v_inv):
		xx_inv=v_inv 
	else:
		pass
	user_f_matrix_ls[user_index]=np.dot(v_inv, user_bias[user_index])
	user_f_matrix_ridge[user_index]=np.dot(v_inv, user_bias[user_index])
	user_f_matrix_avg[user_index]=np.dot(user_f_matrix_ls.T, -normed_lap[user_index])+user_f_matrix_ls[user_index]
	user_f_matrix_appro[user_index]=user_f_matrix_ridge[user_index]+alpha*np.dot(v_inv, user_f_matrix_avg[user_index])

	total_graph_error.append(np.linalg.norm(user_f_matrix_graph-user_f))
	total_approx_error.append(np.linalg.norm(user_f_matrix_appro-user_f))
	total_ls_error.append(np.linalg.norm(user_f_matrix_ls-user_f))
	total_ridge_error.append(np.linalg.norm(user_f_matrix_ridge-user_f))

	graph_ucb, a_graph=graph_UCB(dimension,sigma, delta, user_index, user_v[user_index], normed_lap, user_f_matrix_ls, user_f_matrix_graph[user_index], alpha, user_xnoise[user_index])
	ridge_ucb, a_ridge=ridge_UCB(dimension, sigma, delta, user_v[user_index], user_f_matrix_ridge[user_index], alpha, user_xnoise[user_index])

	total_new_ridge_ucb_list.append(ridge_ucb)
	total_old_ridge_ucb_list.append(ridge_UCB_old(dimension, sigma, delta, user_v[user_index], user_f_matrix_ridge[user_index], alpha, user_xnoise[user_index]))
	total_graph_ucb_list.append(graph_ucb)

	graph_error_bound_list.append(graph_error_bound(user_index, user_v[user_index], normed_lap, user_f_matrix_ls, user_f_matrix_graph[user_index], alpha, user_xnoise[user_index]))
	ridge_error_bound_list.append(ridge_error_bound(user_v[user_index], user_f_matrix_ridge[user_index], alpha, user_xnoise[user_index]))

u=1
index_list=list(np.where(user_seq==u)[0])
# plt.figure(figsize=(5,5))
# plt.plot(np.array(total_graph_error)[index_list], 'r-*', markevery=0.1, label='Lap-reg')
# plt.plot(np.array(total_approx_error)[index_list], 'g-s',markevery=0.1, label='Approx')
# plt.plot(np.array(total_ridge_error)[index_list],'-.',markevery=0.1, label='Ridge')
# plt.legend(loc=1, fontsize=12)
# plt.ylabel('Estimation Error', fontsize=12)
# plt.xlabel('Time', fontsize=12)
# plt.tight_layout()
# plt.savefig(path+'approx_tightness'+'.png', dpi=300)
# plt.savefig(path+'approx_tightness'+'.eps', dpi=300)
# plt.show()


plt.figure(figsize=(5,5))
plt.plot(np.array(total_graph_ucb_list)[index_list], 'r-*', markevery=0.1, label='Lap-reg beta')
plt.plot(np.array(total_new_ridge_ucb_list)[index_list],'g-s',markevery=0.1, label='Ridge beta (new)')
plt.plot(np.array(total_old_ridge_ucb_list)[index_list], '-.', label='Ridge beta (old)')
plt.legend(loc=4, fontsize=12)
plt.ylabel('beta', fontsize=12)
plt.xlabel('Time', fontsize=12)
plt.tight_layout()
plt.savefig(path+'beta'+'.png', dpi=300)
plt.savefig(path+'beta'+'.eps', dpi=300)
plt.show()


plt.figure(figsize=(5,5))
plt.plot(np.array(total_graph_error)[index_list][100:],label='Lap-reg error')
plt.plot(np.array(total_ridge_error)[index_list][100:], label='Ridge error')
plt.plot(np.array(graph_error_bound_list)[index_list][100:], label='Lap-reg error bound')
plt.plot(np.array(ridge_error_bound_list)[index_list][100:], label='Ridge error bound')
plt.legend(loc=1, fontsize=12)
plt.ylabel('Estimation Error and Bound', fontsize=12)
plt.xlabel('Time', fontsize=12)
plt.tight_layout()
plt.savefig(path+'error_bound'+'.png', dpi=300)
plt.savefig(path+'error_bound'+'.eps', dpi=300)
plt.show()