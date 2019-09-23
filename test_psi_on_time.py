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

user_num=20
item_num=1000
dimension=5
iteration=500
item_f=np.random.normal(size=(item_num, dimension))
item_f=Normalizer().fit_transform(item_f)
sigma=0.01
alpha=1

true_adj=rbf_kernel(np.random.normal(size=(user_num, dimension)), gamma=0.25/dimension)
np.fill_diagonal(true_adj, 0)
edge_list=true_adj.ravel()
plt.hist(edge_list)
plt.show()


user_seq=np.random.choice(range(user_num), size=iteration)
item_seq=np.random.choice(range(item_num), size=iteration)

thres_num=4
thres_list=np.round(np.linspace(0,0.75,thres_num), decimals=2)
thres_list=[0,0.4,0.6,0.75]
norm1=np.zeros((thres_num, iteration))
norm2=np.zeros((thres_num, iteration))
cum_norm1=np.zeros((thres_num, iteration))
cum_norm2=np.zeros((thres_num, iteration))
ratio=np.zeros((thres_num, iteration))
edge_num_list=np.zeros(thres_num)


single_norm1={}
single_norm2={}
single_cum_norm1={}
single_cum_norm2={}
for thres_index, thres in enumerate(thres_list):
	adj=true_adj.copy()
	adj[adj<thres]=0
	D=np.diag(np.sum(adj, axis=1))
	lap=D-adj
	norm_lap=np.dot(np.linalg.inv(D), lap) 
	edge_num_list[thres_index]=np.int(np.sum(adj>0))/(user_num*user_num)

	single_norm1[thres_index]=[]
	single_norm2[thres_index]=[]
	single_cum_norm1[thres_index]=[]
	single_cum_norm2[thres_index]=[]

	user_A={}
	user_V={}
	user_lambda={}
	for user in range(user_num):
		user_A[user]=0.01*np.identity(dimension)
		user_V[user]=alpha*np.identity(dimension)
		user_lambda[user]=2*alpha*np.identity(dimension)

	for i in range(iteration):
		print('index', i)
		item_index=item_seq[i]
		user_index=user_seq[i]
		x=item_f[item_index]
		user_A[user_index]+=np.outer(x,x)
		user_V[user_index]+=np.outer(x,x)
		sum_A=np.zeros((dimension, dimension))
		for u in range(user_num):
			sum_A+=(norm_lap[user_index,u]**2)*np.linalg.inv(user_A[u])

		user_lambda[user_index]=user_A[user_index]+2*alpha*norm_lap[user_index, user_index]*np.identity(dimension)+alpha**2*sum_A
		v_inv=np.linalg.inv(user_V[user_index])
		Lambda_inv=np.linalg.inv(user_lambda[user_index])
		a=np.dot(np.dot(x, v_inv),x)
		b=np.dot(np.dot(x, Lambda_inv),x)
		norm1[thres_index, i]=a
		norm2[thres_index, i]=b
		cum_norm1[thres_index, i]=np.sum(norm1[thres_index, :i])
		cum_norm2[thres_index, i]=np.sum(norm2[thres_index, :i])
		ratio[thres_index, i]=b/a
		if user_index==1:
			single_norm1[thres_index].extend([a])
			single_norm2[thres_index].extend([b])
			single_cum_norm1[thres_index].extend([np.sum(single_norm1[thres_index])])
			single_cum_norm2[thres_index].extend([np.sum(single_norm2[thres_index])])


plt.figure()
for thres_index, thres in enumerate(thres_list):
	plt.plot(np.array(single_cum_norm2[thres_index])/np.array(single_cum_norm1[thres_index]), label='%s, %s'%(thres, edge_num_list[thres_index]))
plt.legend(loc=0)
plt.show()


plt.figure()
p=plt.plot(ratio.T)
plt.legend(p, thres_list)
plt.show()


# plt.figure()
# plt.plot(cum_norm1.T, label='v')
# plt.plot(cum_norm2.T, label='Lambda')
# plt.legend(loc=0, fontsize=12)
# plt.show()

a=np.array(cum_norm2)/np.array(cum_norm1)
plt.figure(figsize=(5,5))
plt.plot(a[0])
#plt.legend(p, edge_num_list, fontsize=16)
plt.xlabel('Time', fontsize=16)
plt.ylabel('Psi', fontsize=16)
plt.tight_layout()
plt.savefig(path+'psi_time'+'.png', dpi=100)
plt.show()

plt.figure()
plt.plot(edge_num_list)
plt.show()