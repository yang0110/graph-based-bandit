import numpy as np 
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers
import networkx as nx
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import Normalizer, MinMaxScaler
from scipy.sparse import csgraph 
import scipy
import os 
from sklearn import datasets

def gradient_decent_update_L(L_old, theta_vector, dimension, lambda_, beta):## lambda_ step size, beta regularizer
	theta_matrix=theta_vector.reshape((L_old.shape[0], dimension))
	L=L_old-lambda_*(np.dot(theta_matrix, theta_matrix.T)+2*beta*L_old)
	## negative off diagonal
	L_diag=np.diag(np.diag(L))
	L_off_diag=L-L_diag
	L_off_diag[L_off_diag>0]=0
	L_diag[L_diag<=0]=0
	L=L_off_diag+L_diag
	## row sum to 1
	n=L_old.shape[0]
	L=L-(1/n)*np.dot(L, np.ones((n,n)))
	## sysmmetric 
	L=(L+L.T)/2
	## positive semidefinite 
	eigen_values, eigen_vectors=np.linalg.eig(L)
	eigen_values[eigen_values<0]=0
	eigen_values[eigen_values>1]=1
	L=np.dot(np.dot(eigen_vectors, np.diag(eigen_values)), eigen_vectors.T)
	A=np.kron(L, np.identity(dimension))
	# assert np.allclose(L.trace(), n)
	# assert np.all(L-np.diag(np.diag(L))<=0)
	# assert np.allclose(np.dot(L, np.ones(n)), np.zeros(n))
	# print('All constraints satisfied')
	return L,A

def gradient_decent_update_Theta(Theta_old_vector, x, y, A, mu, alpha): # mu  step_size, alpha regularizer
	Theta_vector=Theta_old_vector+2*mu*(x*y-np.dot(np.outer(x,x), Theta_old_vector)-alpha*np.dot(A, Theta_old_vector))
	return Theta_vector


user_num=20
item_num=100
dimension=5
alpha=0.1 # regularizer
beta=0.1  #regularizer

mu=0.001 #step size
lambda_=0.1 #step size

user_feature=np.random.normal(size=(user_num, dimension))
user_feature=Normalizer().fit_transform(user_feature)
user_feature_vector=user_feature.flatten()
adj=rbf_kernel(user_feature)
lap=csgraph.laplacian(adj, normed=False)

item_feature=np.random.normal(size=(item_num, dimension))
item_feature=Normalizer().fit_transform(item_feature)

Y=np.dot(user_feature, item_feature.T)+np.random.normal(size=(user_num, item_num), scale=0.1)

A_true=np.kron(lap, np.identity(dimension))
A=np.identity(user_num*dimension)
Theta_matrix=np.zeros((user_num, dimension))
Theta_vector=Theta_matrix.flatten()
L=np.identity(user_num)

lap_error=[]
learning_error=[]
for i in range(Y.shape[0]):
	for j in range(Y.shape[1]):
		print('i,j', i, j)
		x=item_feature[j]
		y=Y[i,j]
		x_vector=np.zeros(user_num*dimension)
		x_vector[i*dimension:(i+1)*dimension]=x 
		Theta_vector=gradient_decent_update_Theta(Theta_vector, x_vector, y, A, mu, alpha)
		L, A=gradient_decent_update_L(L, Theta_vector, dimension, lambda_, beta)
		lap_error.extend([np.linalg.norm(L-lap)])
		learning_error.extend([np.linalg.norm(Theta_vector-user_feature_vector)])


plt.figure(figsize=(5,5))
plt.plot(lap_error, label='lap error')
plt.legend(loc=0, fontsize=12)
plt.show()

plt.figure(figsize=(5,5))
plt.plot(learning_error, label='learning error')
plt.legend(loc=0, fontsize=12)
plt.show()

