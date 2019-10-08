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
# os.chdir('Users/kaigeyang/Documents/research/bandit/code/graph_based_bandit/')
from linucb import LINUCB
from t_sampling import TS
from gob import GOB 
from lapucb import LAPUCB
from lapucb_sim import LAPUCB_SIM
from club import CLUB
from utils import *
path='../bandit_results/simulated/'
np.random.seed(2018)

smoothness_list=np.load(path+'smoothness_list_binary.npy')
graphucb_rbf=np.load(path+'graphucb_smooth_rbf_binary.npy')
graphucb_er=np.load(path+'graphucb_smooth_er_binary.npy')
graphucb_ba=np.load(path+'graphucb_smooth_ba_binary.npy')
graphucb_ws=np.load(path+'graphucb_smooth_ws_binary.npy')

graphucb_local_rbf=np.load(path+'graphucb_local_smooth_rbf_binary.npy')
graphucb_local_er=np.load(path+'graphucb_local_smooth_er_binary.npy')
graphucb_local_ba=np.load(path+'graphucb_local_smooth_ba_binary.npy')
graphucb_local_ws=np.load(path+'graphucb_local_smooth_ws_binary.npy')

plt.figure(figsize=(5,5))
plt.plot(smoothness_list, graphucb_rbf, '-*',color='b', markevery=0.1, label='RBF')
plt.plot(smoothness_list, graphucb_er, '-o',color='g', markevery=0.1, label='ER')
plt.plot(smoothness_list, graphucb_ba, '-p',color='r', markevery=0.1, label='BA')
plt.plot(smoothness_list, graphucb_ws, '-s',color='k', markevery=0.1, label='WS')
plt.xlabel('Smoothness', fontsize=16)
plt.ylabel('Cumulative Regret', fontsize=16)
plt.title('sp=0.4', fontsize=16)
plt.legend(loc=0, fontsize=14)
plt.ylim([0,80])
plt.tight_layout()
plt.savefig(path+'graphucb_smooth_graphs_binary'+'.png', dpi=100)
plt.show()

plt.figure(figsize=(5,5))
plt.plot(smoothness_list, graphucb_local_rbf, '-*',color='b', markevery=0.1, label='RBF')
plt.plot(smoothness_list, graphucb_local_er, '-o',color='g', markevery=0.1, label='ER')
plt.plot(smoothness_list, graphucb_local_ba, '-p',color='r', markevery=0.1, label='BA')
plt.plot(smoothness_list, graphucb_local_ws, '-s',color='k', markevery=0.1, label='WS')
plt.xlabel('Smoothness', fontsize=16)
plt.ylabel('Cumulative Regret', fontsize=16)
plt.title('sp=0.4', fontsize=16)
plt.legend(loc=0, fontsize=14)
plt.ylim([0,80])
plt.tight_layout()
plt.savefig(path+'graphucb_local_smooth_graphs_binary'+'.png', dpi=100)
plt.show()