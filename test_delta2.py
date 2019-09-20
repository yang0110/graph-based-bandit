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

thres_list=np.linspace(0,1,5)

lambda_list=np.round(np.linspace(0,100,20), decimals=2)
delta_matrix=np.zeros((user_num, 20))
average_delta_matrix=np.zeros((loop, user_num))
for l in range(loop):
	ws_adj=WS_graph(user_num, 3, 0.2)
	er_adj=ER_graph(user_num, 0.2)
	ba_adj=BA_graph(user_num, 1)
	random_weights=np.round(np.random.uniform(size=(user_num, user_num)), decimals=2)
	random_weights=(random_weights.T+random_weights)/2
	ws_adj=ws_adj*random_weights
	er_adj=er_adj*random_weights
	ba_adj=ba_adj*random_weights
	true_adj=rbf_kernel(np.random.normal(size=(user_num, dimension)), gamma=0.25/dimension)
	#true_adj=ws_adj
	#true_adj=er_adj
	#true_adj=ba_adj
	true_adj[true_adj<=thres]=0
	lap=csgraph.laplacian(true_adj, normed=False)
	D=np.diag(np.sum(true_adj, axis=1))
	true_lap=np.dot(np.linalg.inv(D), lap)

	for index, lambda_ in enumerate(lambda_list):
		user_feature_matrix=dictionary_matrix_generator(user_num, dimension, true_lap, lambda_)
		for user in range(user_num):
			lap_row=true_lap[user]
			average=np.dot(user_feature_matrix.T, lap_row)
			local[user, index]=np.linalg.norm(average)
			theta_norm[user]=np.linalg.norm(user_feature_matrix[user])

	average_local[index_thres]=np.mean(local, axis=0)
	user_feature=