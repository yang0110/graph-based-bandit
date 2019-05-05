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


def ridge_bound(dimension, sigma, delta, alpha, true_theta, v_ii, z):
	v_inv=np.linalg.pinv(v_ii)
	a=np.sqrt(np.dot(np.dot(alpha*true_theta, v_inv), alpha*true_theta))
	b=np.sqrt(np.dot(np.dot(z, v_inv),z))
	#a=np.sqrt(alpha)*np.linalg.norm(true_theta)
	aa = np.linalg.det(v_ii)**(1/2)
	bb = np.linalg.det(alpha*np.identity(dimension))**(-1/2)
	b_bound=sigma*np.sqrt(2*np.log(aa*bb/delta))
	bound=a+b_bound
	return bound, a, b, b_bound

def graph_bound(dimension, sigma, delta, user_index, user_num, phi_i, es_theta, true_theta, z, M_ii, M_inv_ii, M_inv_sub,B):
	phi_theta=np.dot(phi_i, true_theta)
	a=np.sqrt(np.dot(np.dot(phi_theta, M_inv_ii), phi_theta))
	b=np.sqrt(np.dot(np.dot(z, M_inv_ii), z))
	aa = np.linalg.det(M_ii)**(1/2)
	bb = np.linalg.det(phi_i)**(-1/2)
	b_bound=sigma*np.sqrt(2*np.log(aa*bb/delta))
	dot=np.zeros(dimension)
	for u in range(user_num):
		if u==user_index:
			pass 
		else:
			M_inv_ij=M_inv_sub[:,u*dimension:(u+1)*dimension].copy()
			bias=B[u*dimension:(u+1)*dimension].copy()
			dot+=np.dot(np.dot(M_ii, M_inv_ij), bias)

	c=np.sqrt(np.dot(np.dot(-phi_theta+dot, M_inv_ii), -phi_theta+dot))
	bound=b_bound+c
	return bound, a, b, c, np.linalg.norm(dot), b_bound

def ridge_error(alpha, v_ii, true_theta, z):
	v_inv=np.linalg.pinv(v_ii)
	a=-alpha*np.dot(v_inv, true_theta)
	b=np.dot(v_inv, z)
	diff=a+b 
	error=np.linalg.norm(diff)
	return error

def graph_error(user_index, user_num, dimension, M_inv_ii, phi_ii, true_theta, z, M_inv_sub,B):
	a=-np.dot(np.dot(M_inv_ii, phi_ii), true_theta)
	b=np.dot(M_inv_ii, z)
	c=np.zeros(dimension)
	for u in range(user_num):
		if u==user_index:
			pass 
		else:
			M_inv_ij=M_inv_sub[:,u*dimension:(u+1)*dimension].copy()
			bias=B[u*dimension:(u+1)*dimension].copy()
			c+=np.dot(M_inv_ij, bias)
	diff=a+b+c 
	error=np.linalg.norm(diff)
	return error

def ridge_new_bound(alpha, true_theta, z, v_ii):
	v_inv=np.linalg.pinv(v_ii)
	diff=z-alpha*true_theta
	bound=np.sqrt(np.dot(np.dot(diff, v_inv), diff))
	return bound 

def graph_new_bound(user_index, user_num, alpha, phi_ii, true_theta, z, M_ii, M_inv_ii, M_inv_sub,B):
	diff=z-np.dot(phi_ii, true_theta)
	a=np.sqrt(np.dot(np.dot(diff, M_inv_ii), diff))
	dot=np.zeros(dimension)
	dot2=np.zeros(dimension)
	for u in range(user_num):
		if u==user_index:
			pass 
		else:
			M_inv_ij=M_inv_sub[:, u*dimension:(u+1)*dimension].copy()
			bias=B[u*dimension:(u+1)*dimension].copy()
			dot+=np.dot(np.dot(M_ii, M_inv_ij), bias)
			dot2+=np.dot(M_inv_ij, bias)

	b=np.sqrt(np.dot(np.dot(dot2, M_ii), dot2))
	bound2=a+b 
	D=diff+dot
	bound=np.sqrt(np.dot(np.dot(D, M_inv_ii),D))
	return bound,bound2, a, b 

def graph_new_bound_z(user_index, user_num, sigma, delta, alpha, phi_ii, true_theta, z, M_ii, M_inv_ii, M_inv_sub,B):
	diff=-np.dot(phi_ii, true_theta)
	dot=np.zeros(dimension)
	for u in range(user_num):
		if u==user_index:
			pass 
		else:
			M_inv_ij=M_inv_sub[:, u*dimension:(u+1)*dimension].copy()
			bias=B[u*dimension:(u+1)*dimension].copy()
			dot+=np.dot(np.dot(M_ii, M_inv_ij), bias)

	D=diff+dot
	a = np.linalg.det(M_ii)**(1/2)
	b = np.linalg.det(phi_ii)**(-1/2)
	beta=sigma*np.sqrt(2*np.log(a*b/delta))
	c=np.sqrt(np.dot(np.dot(D, M_inv_ii),D))
	bound=beta+c
	return bound,beta, c

def ridge_second_term(user_index, true_theta, v_ii):
	bound=np.sqrt(np.dot(np.dot(true_theta, v_ii), true_theta))
	return bound 

def graph_second_term(user_index, dimension, user_num, true_theta, M_ii, M_inv_sub,B, v_ii):
	sum_=np.zeros(dimension)
	for u in range(user_num):
		if u==user_index:
			pass 
		else:
			M_ij=M_inv_sub[:, u*dimension:(u+1)*dimension]
			B_j=B[u*dimension:(u+1)*dimension]
			sum_+=np.dot(M_ij, B_j)
	diff=true_theta-sum_
	bound=np.sqrt(np.dot(np.dot(diff, M_ii), diff))
	a=np.sqrt(np.dot(np.dot(true_theta, M_ii), true_theta))
	norm=np.linalg.norm(M_ii-v_ii)
	return bound, a, norm



user_num=5
item_num=100
dimension=5
alpha=1
sigma=0.1
iteration=1000
delta=0.1
beta=0.01

adj=RBF_graph(user_num, dimension, thres=0.0)
lap=csgraph.laplacian(adj, normed=False)
normed_lap=csgraph.laplacian(adj, normed=True)
I_d=np.identity(dimension)
user_f=dictionary_matrix_generator(user_num, dimension, lap, 5)
item_f=np.random.normal(size=(item_num, dimension), scale=0.1)
item_f=Normalizer().fit_transform(item_f)
payoffs=np.dot(user_f, item_f.T)
noise=np.random.normal(size=(user_num, item_num), scale=sigma)
noisy_payoffs=payoffs+noise
user_seq=np.random.choice(range(user_num), size=iteration)
user_seq=np.zeros((user_num, int(iteration/user_num)))
for u in range(user_num):
	user_seq[u]=u 
user_seq=user_seq.ravel().astype(int)

item_seq=np.random.choice(range(item_num), size=iteration)

user_f_matrix=np.zeros((user_num, dimension))
user_f_vector=np.zeros(user_num*dimension)

A=np.kron(normed_lap+beta*np.identity(user_num), I_d)
M=alpha*A 
B=np.zeros(user_num*dimension)
V=alpha*np.identity(user_num*dimension)
Plain_M=np.zeros((user_num*dimension, user_num*dimension))

v={}
m={}
phi={}
z={}
for u in range(user_num):
	v[u]=alpha*np.identity(dimension)
	m[u]=np.zeros((dimension, dimension))
	phi[u]=np.zeros((dimension, dimension))
	z[u]=np.zeros(dimension)

error=np.zeros((user_num, iteration))
graph_error_matrix=np.zeros((user_num, iteration))
ridge_error_matrix=np.zeros((user_num, iteration))

ridge_bound_list=np.zeros((user_num, iteration))
ridge_a=np.zeros((user_num, iteration))
ridge_b=np.zeros((user_num, iteration))
graph_bound_list=np.zeros((user_num, iteration))
graph_a=np.zeros((user_num, iteration))
graph_b=np.zeros((user_num, iteration))
graph_c=np.zeros((user_num, iteration))
ridge_b_bound=np.zeros((user_num, iteration))
graph_b_bound=np.zeros((user_num, iteration))


ridge_error_list=np.zeros((user_num, iteration))
graph_error_list=np.zeros((user_num, iteration))
ridge_error_list_emp=np.zeros((user_num, iteration))
graph_error_list_emp=np.zeros((user_num, iteration))

ridge_new_bound_list=np.zeros((user_num, iteration))
graph_new_bound_list=np.zeros((user_num, iteration))
graph_new_bound_list2=np.zeros((user_num, iteration))
graph_new_a=np.zeros((user_num, iteration))
graph_new_b=np.zeros((user_num, iteration))
ridge_all_bound=np.zeros(iteration)
graph_all_bound=np.zeros(iteration)
graph_new_bound_z_list=np.zeros((user_num, iteration))
graph_new_bound_z_beta=np.zeros((user_num, iteration))
graph_new_bound_z_c=np.zeros((user_num, iteration))

dot_matrix=np.zeros((user_num, iteration))
ridge_term2=np.zeros((user_num, iteration))
graph_term2=np.zeros((user_num, iteration))
a=np.zeros((user_num, iteration))
norm_list=np.zeros((user_num, iteration))
for i in range(iteration):
	print('time/iteration', i, iteration)
	user_index=user_seq[i]
	item_index=item_seq[i]
	x=item_f[item_index]
	y=noisy_payoffs[user_index, item_index]
	x_long=np.zeros(user_num*dimension)
	x_long[user_index*dimension:(user_index+1)*dimension]=x
	M+=np.outer(x_long, x_long)
	B+=y*x_long 
	V+=np.outer(x_long, x_long)
	ridge_user_f_matrix=np.dot(np.linalg.pinv(V), B).reshape((user_num, dimension))
	v[user_index]+=np.outer(x,x)
	m[user_index]+=np.outer(x,x)
	M_inv=np.linalg.pinv(M)
	M_inv_sub=M_inv[user_index*dimension:(user_index+1)*dimension].copy()
	M_inv_ii=M_inv[user_index*dimension:(user_index+1)*dimension, user_index*dimension:(user_index+1)*dimension].copy()
	M_ii=np.linalg.pinv(M_inv_ii)
	phi[user_index]=M_ii-m[user_index]
	user_f_vector=np.dot(M_inv, B)
	user_f_matrix=user_f_vector.reshape((user_num, dimension))
	graph_error_matrix[:, i]=np.linalg.norm(user_f_matrix-user_f, axis=1)
	ridge_error_matrix[:, i]=np.linalg.norm(ridge_user_f_matrix-user_f, axis=1)
	average_theta=np.dot(user_f_matrix.T, adj[user_index]/np.sum(adj[user_index]))
	es_theta=user_f_matrix[user_index]
	true_theta=user_f[user_index]
	error[user_index, i]=np.linalg.norm(true_theta-es_theta)
	n=noise[user_index, item_index]
	z[user_index]+=x*n
	ridge_bound_list[user_index, i], ridge_a[user_index, i], ridge_b[user_index, i], ridge_b_bound[user_index, i]=ridge_bound(dimension, sigma, delta, alpha, es_theta, v[user_index], z[user_index])
	graph_bound_list[user_index, i], graph_a[user_index, i], graph_b[user_index, i], graph_c[user_index, i], dot_matrix[user_index, i], graph_b_bound[user_index, i]=graph_bound(dimension,sigma, delta, user_index, user_num, phi[user_index], es_theta, es_theta, z[user_index], M_ii, M_inv_ii, M_inv_sub, B)
	ridge_error_list[user_index, i]=ridge_error(alpha, v[user_index], true_theta, z[user_index])
	graph_error_list[user_index,i]=graph_error(user_index, user_num, dimension, M_inv_ii, phi[user_index], true_theta, z[user_index], M_inv_sub,B)
	ridge_new_bound_list[user_index, i]=ridge_new_bound(alpha, true_theta, z[user_index], v[user_index])
	graph_new_bound_list[user_index,i],graph_new_bound_list2[user_index,i], graph_new_a[user_index,i], graph_new_b[user_index,i]=graph_new_bound(user_index, user_num, alpha, phi[user_index], true_theta, z[user_index], M_ii, M_inv_ii, M_inv_sub, B)
	ridge_all_bound[i]=ridge_new_bound_list[user_index,i]
	graph_all_bound[i]=graph_new_bound_list[user_index,i]
	graph_new_bound_z_list[user_index, i], graph_new_bound_z_beta[user_index, i], graph_new_bound_z_c[user_index,i]=graph_new_bound_z(user_index, user_num, sigma, delta, alpha, phi[user_index], true_theta, z[user_index], M_ii, M_inv_ii, M_inv_sub, B)
	ridge_term2[user_index, i]=ridge_second_term(user_index, true_theta, v[user_index])
	graph_term2[user_index, i],	a[user_index, i], norm_list[user_index, i]=graph_second_term(user_index,dimension, user_num, true_theta, M_ii, M_inv_sub,B,v[user_index])


for u in range(user_num):
	plt.figure()
	plt.plot(ridge_term2[u][ridge_term2[u]>0], label='ridge')
	plt.plot(graph_term2[u][graph_term2[u]>0], label='graph')
	plt.plot(a[user_index][a[u]>0], label='a')
	plt.legend(loc=1)
	plt.show()

for u in range(user_num):
	plt.figure()
	plt.plot(norm_list[u][norm_list[u]>0], label='norm')
	plt.legend(loc=1)
	plt.show()


user_index=2
plt.figure(figsize=(5,5))
plt.plot(ridge_new_bound_list[user_index][ridge_new_bound_list[user_index]>0], label='ridge true')
plt.plot(graph_new_bound_list[user_index][graph_new_bound_list[user_index]>0],label='graph true')
plt.plot(ridge_bound_list[user_index][ridge_bound_list[user_index]>0], label='ridge bound')
plt.plot(graph_bound_list[user_index][graph_bound_list[user_index]>0], label='graph bound')
plt.xlabel('Time')
plt.ylabel('Bound')
plt.legend(loc=1)
plt.tight_layout()
#plt.savefig(path+'true_and_bound_and_bound_decoupled'+'.png')
plt.show()


plt.figure(figsize=(5,5))
plt.plot(ridge_new_bound_list[user_index][ridge_new_bound_list[user_index]>0], label='ridge true')
plt.plot(ridge_bound_list[user_index][ridge_bound_list[user_index]>0], label='ridge bound=a+b')
plt.xlabel('Time')
plt.ylabel('Bound')
plt.legend(loc=1)
plt.tight_layout()
#plt.savefig(path+'ridge_true_and_bound_and_bound_decoupled_noise'+'.png')
plt.show()


np.fill_diagonal(adj,0)
graph, edge_num=create_networkx_graph(user_num, adj)
edge_weight=adj[np.triu_indices(user_num,1)]
edge_color=edge_weight[edge_weight>0]
#pos=np.random.uniform(size=(user_num, 2))
pos = nx.spring_layout(graph)
plt.figure(figsize=(5,5))
nodes=nx.draw_networkx_nodes(graph, pos, node_size=300, cmap=plt.cm.jet)
edges=nx.draw_networkx_edges(graph, pos, width=1.0, alpha=1)
#edges=nx.draw_networkx_edges(graph, pos, width=1.0, alpha=1, edge_color=edge_color, edge_cmap=plt.cm.Blues)
nx.draw_networkx_labels(graph, pos)
plt.axis('off')
#plt.savefig(path+'graph_5_users'+'.png')
plt.show()


