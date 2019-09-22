import numpy as np 
import networkx as nx
import seaborn as sns
sns.set_style("white")
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import Normalizer, MinMaxScaler
from scipy.sparse import csgraph
from scipy.linalg import sqrtm   
import scipy
import os 
os.chdir('/Kaige_Research/Code/graph_bandit/code/')
from sklearn import datasets
from utils import *
np.random.seed(2018)
path='../bandit_results/simulated/'


iteration=500
user_num=20
item_num=1000
dimension=5
item_f=np.random.normal(size=(item_num, dimension))
item_f=Normalizer().fit_transform(item_f)
sigma=0.01
alpha=1

adj=rbf_kernel(np.random.normal(size=(user_num, dimension)))
adj[adj<0]=0
#np.fill_diagonal(adj,0)
D=np.diag(np.sum(adj, axis=1))
lap=csgraph.laplacian(adj, normed=False)
lap=np.dot(np.linalg.inv(D), lap) 
lambda_list=np.linspace(0,50,5)

user_A={}
user_V={}
user_lambda={}
for user in range(user_num):
	user_A[user]=0.01*np.identity(dimension)
	user_V[user]=alpha*np.identity(dimension)
	user_lambda[user]=2*alpha*np.identity(dimension)

user_seq=np.random.choice(range(user_num), size=iteration)
item_seq=np.random.choice(range(item_num), size=iteration)

thres_num=5
thres_list=np.linspace(0,1,5)
norm1=np.zeros((thres_num, iteration))
norm2=np.zeros((thres_num, iteration))
cum_norm1=np.zeros((thres_num, iteration))
cum_norm2=np.zeros((thres_num, iteration))

# norm1=np.zeros(iteration)
# norm2=np.zeros(iteration)
# cum_norm1=np.zeros(iteration)
# cum_norm2=np.zeros(iteration)
for thres_index, thres in enumerate(thres_list):
	adj[adj<=thres]=0
	#np.fill_diagonal(adj,1)
	if thres==1.0:
		lap=np.zeros((user_num, user_num))
	else:
		lap=csgraph.laplacian(adj, normed=False)
		D=np.diag(np.sum(adj, axis=1))
		lap=np.dot(np.linalg.inv(D), lap) 
	#lambda_list=np.linspace(0,50,5)
	for i in range(iteration):
		print('index', i)
		item_index=item_seq[i]
		user_index=user_seq[i]
		x=item_f[item_index]
		user_A[user_index]+=np.outer(x,x)
		user_V[user_index]+=np.outer(x,x)
		user_lambda[user_index]+=np.outer(x,x)
		sum_A=np.zeros((dimension, dimension))
		for u in range(user_num):
			sum_A+=(lap[user_index,u]**2)*np.linalg.pinv(user_A[u])

		user_lambda[user_index]+=alpha**2*sum_A
		#if user_index==1:
		v_inv=np.linalg.pinv(user_V[user_index])
		Lambda_inv=np.linalg.pinv(user_lambda[user_index])
		norm1[thres_index, i]=np.dot(np.dot(x, v_inv),x)
		norm2[thres_index, i]=np.dot(np.dot(x, Lambda_inv),x)
		cum_norm1[thres_index, i]=np.sum(norm1[thres_index, :i])
		cum_norm2[thres_index, i]=np.sum(norm2[thres_index, :i])



plt.figure()
plt.plot(cum_norm1.T, label='v')
plt.plot(cum_norm2.T, label='Lambda')
plt.legend(loc=0, fontsize=12)
plt.show()

a=np.array(cum_norm2)/np.array(cum_norm1)
plt.figure()
p=plt.plot(a.T)
plt.legend(p, thres_list)
plt.show()