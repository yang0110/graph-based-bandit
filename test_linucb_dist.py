import numpy as np 
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import Normalizer, MinMaxScaler
from scipy.sparse import csgraph 
import scipy
import os 
from sklearn import datasets
os.chdir('C:/Kaige_Research/Code/graph_bandit/code/')
from linucb import LINUCB_IND
from linucb_dist import LINUCB_DIST
from linucb_single import LINUCB_SIN
from lapucb import LAPUCB_SIN
from gob import GOB 
from individual_lapucb import LAPUCB_IND
from neighbor_lapucb import LAPUCB_NEI
from adaptive_lapucb import Adaptive_LAPUCB
from sclub import SCLUB
from utils import *
path='../results/'

user_num=50
dimension=5
item_num=100
pool_size=10
iteration=1000
I_user_num=np.identity(user_num)
I_dimension=np.identity(dimension)

sigma=0.1# noise
delta=0.1# high probability
alpha=0.1# regularizer
beta=0.1 # regularizer
mu=0.1 ## step size
lambda_=0.1 ## step size


user_array=np.random.choice(range(user_num), size=iteration)
item_pool_array=np.random.choice(range(item_num), size=(iteration, pool_size))
item_feature_matrix=Normalizer().fit_transform(np.random.normal(size=(item_num, dimension)))

adj=RBF_graph(user_num, dimension, thres=0.75)
#adj=BA_graph(user_num, 3)
#adj=ER_graph(user_num, 0.5)
lap=csgraph.laplacian(adj, normed=False)
user_feature_matrix=dictionary_matrix_generator(user_num, dimension, lap, 5)
true_payoffs=np.dot(user_feature_matrix, item_feature_matrix.T)+np.random.normal(scale=sigma, size=(user_num, item_num))

linucb_model=LINUCB_IND(dimension, user_num, item_num, pool_size, item_feature_matrix, user_feature_matrix, true_payoffs, alpha, delta, sigma)
linucb_dist_model=LINUCB_DIST(dimension, user_num, item_num, pool_size, item_feature_matrix, user_feature_matrix, true_payoffs, lap, alpha, delta, sigma)

linucb_regret, linucb_error, linucb_beta_list, linucb_x_norm, linucb_bound=linucb_model.run(user_array, item_pool_array, iteration)
dist_regret, dist_error, dist_lap_error=linucb_dist_model.run(user_array, item_pool_array, iteration)



plt.figure(figsize=(5,5))
plt.plot(linucb_regret,label='LinUCB')
plt.plot(dist_regret, label='LINUCB DIST')
plt.ylabel('Cumulative Regret', fontsize=12)
plt.xlabel('Time', fontsize=12)
plt.legend(loc=0, fontsize=10)
plt.tight_layout()
plt.show()


plt.figure(figsize=(5,5))
plt.plot(linucb_error, label='LinUCB')
plt.plot(dist_error, label='DIST')
plt.ylabel('Learning Error', fontsize=12)
plt.xlabel('Time', fontsize=12)
plt.legend(loc=0, fontsize=10)
plt.tight_layout()
plt.show()


plt.figure(figsize=(5,5))
plt.plot(dist_lap_error, label='lap error')
plt.legend(loc=0)
plt.show()