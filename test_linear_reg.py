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

user_num =5
item_num = 100
dimension = 5 
alpha = 0.1
sigma = 0.1
iteration = 100
adj=RBF_graph(user_num, dimension, thres=0.7)
D=np.diag(np.sum(adj, axis=1))
P=np.dot(np.linalg.pinv(D),adj)
un_normed_lap=csgraph.laplacian(adj, normed=False)
lap=csgraph.laplacian(adj, normed=True)
user_f=dictionary_matrix_generator(user_num, dimension, un_normed_lap, 5)
item_f = np.random.normal(size=(item_num, dimension))
item_f = Normalizer().fit_transform(item_f)
payoffs = np.dot(user_f, item_f.T)
noise = np.random.normal(scale=sigma, size=(user_num,item_num))
noisy_payoffs = payoffs+noise 

A = np.kron(lap+0.1*np.identity(user_num), np.identity(dimension))

M = alpha*A
B = np.zeros(user_num*dimension)
m = {}
m_inv = {}
m_norm = {}
m_inv_norm = np.zeros((user_num, iteration))
m_inv_diag_norm = np.zeros((user_num, iteration))

cov = {}
bias = {}
cov_inv = {}
cov_norm = {}
cov_inv_norm = np.zeros((user_num, iteration))
cov_diag_norm = {}
cov_inv_diag_norm = np.zeros((user_num, iteration))

error = np.zeros((user_num, iteration))
graph_error = np.zeros((user_num, iteration))
for u in range(user_num):
	cov[u] = alpha*np.identity(dimension)
	bias[u] = np.zeros(dimension)
	cov_inv[u]=[]
	cov_norm[u] = []
	cov_diag_norm[u] = []
	m_norm[u] = []


user_matrix = np.zeros((user_num, dimension))
user_vector = np.zeros(user_num*dimension)
user_seq = np.random.choice(range(user_num), size=iteration)
user_seq=np.zeros((user_num, int(iteration/user_num)))
for u in range(user_num):
	user_seq[u]=u 
user_seq=user_seq.ravel().astype(int)
item_seq = np.random.choice(range(item_num), size=iteration)
total_error = np.zeros(iteration)
total_graph_error = np.zeros(iteration)
for i in range(iteration):
	user_index = user_seq[i]
	item_index = item_seq[i]
	x = item_f[item_index]
	y = noisy_payoffs[user_index, item_index]
	cov[user_index] +=np.outer(x,x)
	bias[user_index] +=y*x 
	cov_inv[user_index] = np.linalg.pinv(cov[user_index])
	user_matrix[user_index] = np.dot(cov_inv[user_index], bias[user_index])
	cov_norm[user_index].extend([np.linalg.norm(cov[user_index])])
	cov_diag_norm[user_index].extend([np.linalg.norm(np.diag(cov[user_index]))])
	x_long = np.zeros(user_num*dimension)
	x_long[user_index*dimension:(user_index+1)*dimension]=x 
	M +=np.outer(x_long, x_long)
	B +=y*x_long
	M_inv = np.linalg.pinv(M)
	user_vector = np.dot(M_inv, B)
	for uu in range(user_num):
		m[uu] = M[uu*dimension:(uu+1)*dimension, uu*dimension:(uu+1)*dimension]
		m_inv[uu] = M_inv[uu*dimension:(uu+1)*dimension, uu*dimension:(uu+1)*dimension]
		m_norm[uu].extend([np.linalg.norm(m[uu])])
		m_inv_norm[uu,i] = np.linalg.norm(m_inv[uu])
		m_inv_diag_norm[uu,i] = np.linalg.norm(np.diag(m_inv[uu]))
		cov_inv_norm[uu,i] = np.linalg.norm(cov_inv[uu])
		cov_inv_diag_norm[uu,i] = np.linalg.norm(np.diag(cov_inv[uu]))
		error[uu,i]=np.linalg.norm(user_matrix[uu]-user_f[uu])
		graph_error[uu,i]=np.linalg.norm(user_vector.reshape((user_num, dimension))[uu]-user_f[uu])

	total_error[i] = np.linalg.norm(user_matrix-user_f)
	total_graph_error[i] = np.linalg.norm(user_vector.reshape((user_num, dimension))-user_f)

plt.figure()
plt.plot(total_error, label='error')
plt.plot(total_graph_error, label='graph error')
plt.legend(loc=0)
plt.show()

plt.figure()
for user_index in range(user_num):
	plt.plot(error[user_index], '.-', label=user_index)	
plt.legend(loc=0)
plt.show()

plt.figure()
for user_index in range(user_num):
	plt.plot(graph_error[user_index], '.-', label=user_index)	
plt.legend(loc=0)
plt.show()

plt.figure()
for user_index in range(user_num):
	plt.plot(cov_inv_norm[user_index], '.-', label=user_index)	
plt.legend(loc=0)
plt.show()


plt.figure()
for user_index in range(user_num):
	plt.plot(m_inv_norm[user_index], '.-', label=user_index)	
plt.legend(loc=0)
plt.show()





# plt.figure()
# for user_index in range(user_num):
# 	plt.plot(m_inv_diag_norm[user_index],'.', label=user_index)	
# plt.legend(loc=0)
# plt.show()

# plt.figure()
# for user_index in range(user_num):
# 	plt.plot(cov_inv_diag_norm[user_index],'.', label=user_index)	
# plt.legend(loc=0)
# plt.show()

# plt.figure()
# for user_index in range(user_num):
# 	plt.plot(error[user_index], label=user_index)
# 	plt.plot(graph_error[user_index], '.',label=user_index)
# plt.legend(loc=0)
# plt.show()


# plt.figure()
# for user_index in range(user_num):
# 	plt.plot(cov_inv_norm[user_index], label=user_index)
# 	plt.plot(m_inv_norm[user_index],'.', label=user_index)	
# plt.legend(loc=0)
# plt.show()




# plt.figure()
# for user_index in range(user_num):
# 	plt.plot(cov_inv_diag_norm[user_index], label=user_index)
# 	plt.plot(m_inv_diag_norm[user_index], '.',label=user_index)	
# plt.legend(loc=0)
# plt.show()
