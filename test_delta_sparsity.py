import numpy as np 
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import Normalizer, MinMaxScaler
from scipy.sparse import csgraph  
import scipy
import os 
from sklearn import datasets
os.chdir('/Kaige_Research/Code/graph_bandit/code/')
from linucb import LINUCB
from t_sampling import TS
from gob import GOB 
from lapucb import LAPUCB
from lapucb_sim import LAPUCB_SIM
from club import CLUB
from utils import *
path='../bandit_results/simulated/'
#np.random.seed(2018)

user_num=20
item_num=500
dimension=10
pool_size=20
iteration=2000
loop=1
sigma=0.01# noise
delta=0.1# high probability
alpha=1# regularizer
alpha_2=0.1# edge delete CLUB
epsilon=8 # Ts
beta=0.15# exploration for CLUB, SCLUB and GOB
thres=0.0
state=False # False for artificial dataset, True for real dataset
lambda_list=[4]

thres_list=np.round(np.linspace(0,0.99,5), decimals=2)
local=np.zeros((user_num, 5))
true_adj=rbf_kernel(np.random.normal(size=(user_num, dimension)))
lap=csgraph.laplacian(true_adj, normed=False)
D=np.diag(np.sum(true_adj, axis=1))
true_lap=np.dot(np.linalg.inv(D), lap)
user_feature_matrix=dictionary_matrix_generator(user_num, dimension, true_lap, 4)

for index, thres in enumerate(thres_list):
	true_adj[true_adj<thres]=0
	lap=csgraph.laplacian(true_adj, normed=False)
	D=np.diag(np.sum(true_adj, axis=1))
	true_lap=np.dot(np.linalg.inv(D), lap)
	for user in range(user_num):
		lap_row=true_lap[user]
		if lap[user, user]==0.0:
			lap_row[user]=1
		average=np.dot(user_feature_matrix.T, lap_row)
		local[user, index]=np.linalg.norm(average)
	average_local=np.mean(local,axis=0)

plt.figure(figsize=(5, 5))
plt.plot(thres_list, average_local)
plt.xlabel('Sparsity', fontsize=16)
plt.ylabel('Norm of Delta', fontsize=16)
plt.tight_layout()
plt.savefig(path+'delta_sparsity'+'.png', dpi=100)
plt.show()


