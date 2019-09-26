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

lambda_list=np.round(np.linspace(0,20,20), decimals=2)
smoothness_list=np.zeros(20)
local=np.zeros((user_num, 20))
true_adj=rbf_kernel(np.random.normal(size=(user_num, dimension)), gamma=0.5/dimension)
np.fill_diagonal(true_adj,0)
lap=csgraph.laplacian(true_adj, normed=False)
D=np.diag(np.sum(true_adj, axis=1))
true_lap=np.dot(np.linalg.inv(D), lap)
degree=np.round(np.diag(D)-1, decimals=2)
for index, lambda_ in enumerate(lambda_list):
	user_feature_matrix=dictionary_matrix_generator(user_num, dimension, true_lap, lambda_)
	smoothness_list[index]=np.trace(np.dot(np.dot(user_feature_matrix.T, true_lap), user_feature_matrix))
	for user in range(user_num):
		lap_row=true_lap[user]
		average=np.dot(user_feature_matrix.T, lap_row)
		local[user, index]=np.linalg.norm(average)
		average_local=np.mean(local, axis=0)

smoothness_list=np.round(smoothness_list, decimals=2)
plt.figure(figsize=(5,5))
p=plt.plot(lambda_list, average_local)
plt.xlabel('Smooth regularizer', fontsize=14)
plt.ylabel('Norm of Delta', fontsize=16)
#plt.legend(p, degree[:5])
plt.tight_layout()
plt.savefig(path+'delta_smoothness'+'.png', dpi=100)
plt.show()


