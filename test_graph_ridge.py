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

user_num=20
dimension=10
item_num=100
iteration=1000
alpha=0.1
sigma=0.5

adj=RBF_graph(user_num, dimension, thres=0)
lap1=csgraph.laplacian(adj, normed=True)
A1=np.kron(lap1, np.identity(dimension))

lap2=csgraph.laplacian(adj, normed=False)
A2=np.kron(lap2, np.identity(dimension))

user_feature_matrix=dictionary_matrix_generator(user_num, dimension, lap1, 5)
user_feature_vector=user_feature_matrix.flatten()
item_f=np.random.normal(size=(item_num, dimension))
x_matrix=Normalizer().fit_transform(item_f)
true_payoffs=np.dot(user_feature_matrix, x_matrix.T)+np.random.normal(scale=sigma, size=(user_num, item_num))
y=true_payoffs.ravel()

X_matrix=np.zeros((user_num*item_num, user_num*dimension))
for u in range(user_num):
	for j in range(item_num):
		X_matrix[u*item_num+j][u*dimension:(u+1)*dimension]=x_matrix[j]


alpha_list=list(np.round(np.arange(0.1,20, 0.1), decimals=4))

error=np.zeros(len(alpha_list))
error1=np.zeros(len(alpha_list))
error2=np.zeros(len(alpha_list))
for i, alpha in enumerate(alpha_list):
	cov=alpha*np.identity(user_num*dimension)
	bias=np.zeros(user_num*dimension)
	cov+=np.dot(X_matrix.T, X_matrix)
	bias+=np.dot(X_matrix.T,y)
	Theta=np.dot(np.linalg.pinv(cov), bias)
	error[i]=np.linalg.norm(Theta-user_feature_vector)

	cov1=alpha*A1
	bias1=np.zeros(user_num*dimension)
	cov1+=np.dot(X_matrix.T, X_matrix)
	bias1+=np.dot(X_matrix.T,y)
	Theta1=np.dot(np.linalg.pinv(cov1), bias1)
	error1[i]=np.linalg.norm(Theta1-user_feature_vector)

	cov2=alpha*A2
	bias2=np.zeros(user_num*dimension)
	cov2+=np.dot(X_matrix.T, X_matrix)
	bias2+=np.dot(X_matrix.T,y)
	Theta2=np.dot(np.linalg.pinv(cov2), bias2)
	error2[i]=np.linalg.norm(Theta2-user_feature_vector)


ind=np.arange(len(alpha_list))
plt.figure(figsize=(5,5))
plt.plot(error, label='Reg: I')
plt.plot(error1, label='Reg: Normed L')
plt.plot(error2, label='Reg: L')
plt.xticks(ind, [])
plt.ylabel('RMSE')
plt.title('Precision: Normed L')
plt.legend(loc=0)
plt.tight_layout()
plt.savefig(path+'Precision_normed_L_RMSE_vs_alpha'+'.png')
plt.show()

