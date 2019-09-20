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

iteration=1000
user_num=100
item_num=500
dimension=5
item_f=np.random.normal(size=(item_num, dimension))
item_f=Normalizer().fit_transform(item_f)
sigma=0.1
alpha=1
noise=np.random.normal(size=(user_num, item_num), scale=sigma)

user_f=np.random.normal(size=(user_num, dimension))
user_f=Normalizer().fit_transform(user_f)
adj=rbf_kernel(user_f)
#adj=rbf_kernel(np.random.normal(size=(user_num, dimension)))
adj[adj<0.1]=0
np.fill_diagonal(adj,0)
D=np.diag(np.sum(adj, axis=1))
lap=csgraph.laplacian(adj, normed=False)
lap=np.dot(np.linalg.inv(D), lap) 
#user_f=dictionary_matrix_generator(user_num, dimension, lap, 1)
L=np.kron(lap+np.identity(user_num), np.identity(dimension))

average_list=[]
for user_index in range(user_num):
	average=np.linalg.norm(np.dot(user_f.T, lap[user_index]))
	average_list.extend([average])


plt.figure()
plt.plot(average_list, label='average')
plt.legend(loc=1)
plt.show()