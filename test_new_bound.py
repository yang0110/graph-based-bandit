import numpy as np 
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import Normalizer, MinMaxScaler
from scipy.sparse import csgraph 
import scipy
import os 
os.chdir('C:/Kaige_Research/Code/graph_bandit/code/')
from utils import *
path='../results/'

def ridge_conf_true(v_ii, xnoise, true_theta, alpha):
	a=-alpha*true_theta
	b=a+xnoise
	v_inv_ii=np.linalg.pinv(v_ii)
	conf=np.sqrt(np.dot(np.dot(b, v_inv_ii), b))
	return conf 

def graph_conf_true(user_index, M_ii, M_sub_inv, B, M_inv_ii, xnoise, true_theta, phi, user_num, dimension):
	a=-np.dot(phi, true_theta)
	b=xnoise
	c=np.zeros(dimension)
	for u in range(user_num):
		if u==user_index:
			pass 
		else:
			M_inv_ij=M_inv_sub[:, u*dimension:(u+1)*dimension]
			B_ij=B[u*dimension:(u+1)*dimension]
			c+=np.dot(np.dot(M_ii, M_inv_ij), B_ij)
	d=a+b+c 
	conf=np.sqrt(np.dot(np.dot(d, M_inv_ii), d))
	return conf 

def ridge_bound(v_ii, alpha, true_theta, xnoise):
	v_inv_ii=np.linalg.pinv(v_ii)
	a=alpha*np.sqrt(np.trace(v_inv_ii))*np.linalg.norm(true_theta)
	b=np.sqrt(np.dot(np.dot(xnoise, v_inv_ii), xnoise))
	bound=a+b 
	return bound 

def graph_bound(v_ii, alpha, true_theta, avg_theta, xnoise):
	v_inv_ii=np.linalg.pinv(v_ii)
	a=alpha*np.sqrt(np.trace(v_inv_ii))*np.linalg.norm(true_theta-avg_theta)
	b=np.sqrt(np.dot(np.dot(xnoise, v_inv_ii), xnoise))
	bound=a+b 
	return bound 


user_num=10
item_num=100
dimension=5
alpha=1
sigma=0.1
iteration=1000
delta=0.1
beta=0.01

adj=RBF_graph(user_num, dimension, thres=0.5)
lap=csgraph.laplacian(adj, normed=False)
normed_lap=csgraph.laplacian(adj, normed=True)

user_f=dictionary_matrix_generator(user_num, dimension, lap, 5)
item_f=np.random.normal(size=(item_num, dimension), scale=0.5)
item_f=Normalizer().fit_transform(item_f)
payoffs=np.dot(user_f, item_f.T)
noise=np.random.normal(size=(user_num, item_num), scale=sigma)
noisy_payoffs=payoffs+noise
user_seq=np.random.choice(range(user_num), size=iteration)
# user_seq=np.zeros((user_num, int(iteration/user_num)))
# for u in range(user_num):
# 	user_seq[u]=u 
# user_seq=user_seq.ravel().astype(int)
item_seq=np.random.choice(range(item_num), size=iteration)

user_f_vector_ridge=np.zeros(user_num*dimension)
user_f_vector_graph=np.zeros(user_num*dimension)
user_f_matrix_ridge=np.zeros((user_num, dimension))
user_f_matrix_graph=np.zeros((user_num, dimension))
user_f_vector_ls=np.zeros((user_num*dimension))
user_f_matrix_ls=np.zeros((user_num, dimension))
user_f_matrix_appro=np.zeros((user_num, dimension))

A=np.kron(normed_lap+beta*np.identity(user_num), np.identity(dimension))
M=alpha*A 
B=np.zeros(user_num*dimension)
V=alpha*np.identity(user_num*dimension)
XX=beta*np.identity(user_num*dimension)
M_inv_diag=np.zeros((user_num*dimension, user_num*dimension))
M_inv_offdiag=np.zeros((user_num*dimension, user_num*dimension))
target_offdiag=np.zeros((user_num*dimension, user_num*dimension))
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
	user_m[u]=1*np.identity(dimension)
	user_xx[u]=np.zeros((dimension, dimension))

total_error_ridge=np.zeros(iteration)
total_error_graph=np.zeros(iteration)
new_error=np.zeros(iteration)
new_user_error=np.zeros((user_num, iteration))
user_error_ridge=np.zeros((user_num, iteration))
user_error_graph=np.zeros((user_num, iteration))

conf_true_ridge=np.zeros((user_num, iteration))
conf_true_graph=np.zeros((user_num, iteration))
bound_ridge=np.zeros((user_num, iteration))
bound_graph=np.zeros((user_num, iteration))
bound2_ridge=np.zeros((user_num, iteration))
bound2_graph=np.zeros((user_num, iteration))
m_inv_norm=np.zeros((user_num, iteration))
avg_theta_norm=np.zeros((user_num, iteration))
new_user_error_bound=np.zeros((user_num, iteration))
new_ridge_user_error_bound=np.zeros((user_num, iteration))

for i in range(iteration):
	print('time/iteration', i, iteration)
	user_index=user_seq[i]
	item_index=item_seq[i] 
	y=noisy_payoffs[user_index, item_index]
	x=item_f[item_index]
	noi=noise[user_index, item_index]
	true_theta=user_f[user_index]
	x_long=np.zeros(user_num*dimension)
	x_long[user_index*dimension:(user_index+1)*dimension]=x
	M+=np.outer(x_long, x_long)
	V+=np.outer(x_long, x_long)
	XX+=np.outer(x_long, x_long)
	B+=y*x_long
	M_inv=np.linalg.pinv(M)
	V_inv=np.linalg.pinv(V)
	XX_inv=np.linalg.pinv(XX)
	user_f_vector_ls=np.dot(XX_inv, B)
	user_f_matrix_ls=user_f_vector_ls.reshape((user_num,dimension))
	user_f_vector_ridge=np.dot(V_inv, B)
	user_f_vector_graph=np.dot(M_inv,B)
	user_f_matrix_ridge=user_f_vector_ridge.reshape((user_num,dimension))
	user_f_matrix_graph=user_f_vector_graph.reshape((user_num, dimension))
	user_bias[user_index]+=y*x 
	user_xnoise[user_index]+=x*noi 
	user_v[user_index]+=np.outer(x,x)
	user_m[user_index]+=np.outer(x,x)
	user_xx[user_index]+=np.outer(x,x)
	total_error_ridge[i]=np.linalg.norm(user_f_matrix_ridge-user_f)
	total_error_graph[i]=np.linalg.norm(user_f_matrix_graph-user_f)
	for user in range(user_num):
		user_error_ridge[user, i]=np.linalg.norm(user_f_matrix_ridge[user]-user_f[user])
		user_error_graph[user, i]=np.linalg.norm(user_f_matrix_graph[user]-user_f[user])
		conf_true_ridge[user,i]=ridge_conf_true(user_v[user], user_xnoise[user], user_f[user], alpha)
		M_inv_sub=M_inv[user*dimension:(user+1)*dimension]
		M_inv_ii=M_inv_sub[:, user*dimension:(user+1)*dimension]
		M_ii=np.linalg.pinv(M_inv_ii)
		user_phi[user]=M_ii-user_xx[user]
		conf_true_graph[user,i]=graph_conf_true(user, M_ii, M_inv_sub, B, M_inv_ii, user_xnoise[user], user_f[user], user_phi[user], user_num, dimension)
		bound_ridge[user, i]=ridge_bound(user_v[user], alpha, user_f[user], user_xnoise[user])
		L_i=normed_lap[user]
		avg_theta=np.dot(user_f_matrix_ls.T, -L_i)+user_f_matrix_ls[user] # only neighbors
		bound_graph[user, i]=graph_bound(user_v[user], alpha, user_f[user], avg_theta, user_xnoise[user])
		m_inv=np.linalg.pinv(user_m[user])
		m_inv_norm[user,i]=np.linalg.norm(m_inv)
		avg_theta_norm[user,i]=np.linalg.norm(np.dot(alpha*m_inv, avg_theta))
		user_f_matrix_appro[user]=user_f_matrix_ridge[user]+np.dot(alpha*m_inv, avg_theta)
		new_user_error[user,i]=np.linalg.norm(user_f_matrix_appro[user]-user_f[user])
		new_user_error_bound[user,i]=np.trace(m_inv)*(alpha*np.linalg.norm(user_f[user]-avg_theta)+np.linalg.norm(user_xnoise[user]))
		new_ridge_user_error_bound[user,i]=np.trace(m_inv)*(alpha*np.linalg.norm(user_f[user])+np.linalg.norm(user_xnoise[user]))
	new_error[i]=np.linalg.norm(user_f_matrix_appro-user_f)


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
plt.savefig(path+'network'+'.png', dpi=300)
plt.savefig(path+'network'+'.eps', dpi=300)
plt.show()


plt.figure(figsize=(5,5))
plt.plot(total_error_ridge, label='ridge')
plt.plot(total_error_graph, label='graph-based')
plt.plot(new_error, label='approx-graph-based')
plt.xlabel('Time', fontsize=12)
plt.ylabel('Estimation Error', fontsize=12)
plt.legend(loc=0, fontsize=8)
plt.tight_layout()
plt.savefig(path+'est_error'+'.png', dpi=300)
plt.savefig(path+'est_error'+'.eps', dpi=300)
plt.show()



plt.figure(figsize=(5,5))
plt.plot(np.mean(conf_true_ridge,axis=0), label='ridge (Truth)')
plt.plot(np.mean(conf_true_graph,axis=0), label='graph-based (Truth)')
plt.plot(np.mean(bound_ridge,axis=0), label='ridge (upper bound)')
plt.plot(np.mean(bound_graph,axis=0), label='graph-based (upper bound)')
plt.legend(loc=0, fontsize=12)
plt.ylabel('Confidence Upper Bound', fontsize=12)
plt.xlabel('Time', fontsize=12)
plt.tight_layout()
plt.savefig(path+'Confidence_set_per_user'+'.png', dpi=300)
plt.savefig(path+'Confidence_set_per_user'+'.eps', dpi=300)
plt.legend(loc=0)
plt.show()

plt.figure(figsize=(5,5))
plt.plot(np.mean(user_error_ridge, axis=0), label='ridge')
plt.plot(np.mean(user_error_graph, axis=0), label='graph-based')
plt.plot(np.mean(new_user_error, axis=0), label='approx-graph-based')
plt.plot(np.mean(new_user_error_bound, axis=0), label='user error bound')
plt.plot(np.mean(new_ridge_user_error_bound, axis=0), label='user error bound')
plt.xlabel('Time', fontsize=12)
plt.ylabel('Estimation Error', fontsize=12)
plt.legend(loc=0, fontsize=8)
plt.tight_layout()
plt.savefig(path+'est_error_and_bound'+'.png', dpi=300)
plt.savefig(path+'est_error_and_bound'%(user_index)+'.eps', dpi=300)
plt.show()

# for user_index in range(user_num):
# 	plt.figure(figsize=(5,5))
# 	plt.plot(user_error_ridge[user_index], label='ridge')
# 	plt.plot(user_error_graph[user_index], label='graph-based')
# 	plt.plot(new_user_error[user_index], label='approx-graph-based')
# 	plt.plot(new_user_error_bound[user_index], label='user error bound')
# 	plt.xlabel('Time', fontsize=12)
# 	plt.ylabel('Estimation Error', fontsize=12)
# 	plt.legend(loc=0, fontsize=8)
# 	plt.tight_layout()
# 	plt.savefig(path+'est_error_per_user_%s'%(user_index)+'.png', dpi=300)
# 	plt.savefig(path+'est_error_per_user_%s'%(user_index)+'.eps', dpi=300)
# 	plt.show()



# for user_index in range(user_num):
# 	plt.figure(figsize=(5,5))
# 	plt.plot(conf_true_ridge[user_index], label='ridge')
# 	plt.plot(conf_true_graph[user_index], label='graph-based')
# 	plt.legend(loc=0, fontsize=12)
# 	plt.ylabel('Confidence Set', fontsize=12)
# 	plt.xlabel('Time', fontsize=12)
# 	plt.tight_layout()
# 	plt.savefig(path+'Confidence_set_per_user_%s'%(user_index)+'.png', dpi=300)
# 	plt.savefig(path+'Confidence_set_per_user_%s'%(user_index)+'.eps', dpi=300)
# 	plt.show()


# for user_index in range(user_num):
# 	plt.figure(figsize=(5,5))
# 	plt.plot(conf_true_ridge[user_index], label='ridge (Truth)')
# 	plt.plot(conf_true_graph[user_index], label='graph-based (Truth)')
# 	plt.plot(bound_ridge[user_index], label='ridge (upper bound)')
# 	plt.plot(bound_graph[user_index], label='graph-based (upper bound)')
# 	plt.legend(loc=0, fontsize=12)
# 	plt.ylabel('Confidence Upper Bound', fontsize=12)
# 	plt.xlabel('Time', fontsize=12)
# 	plt.tight_layout()
# 	plt.savefig(path+'Confidence_set_per_user_%s'%(user_index)+'.png', dpi=300)
# 	plt.savefig(path+'Confidence_set_per_user_%s'%(user_index)+'.eps', dpi=300)
# 	plt.legend(loc=0)
# 	plt.show()




