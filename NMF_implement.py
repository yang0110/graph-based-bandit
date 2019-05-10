import numpy as np 
from NMF_miss_value import NMF 
import os 
os.chdir('C:/Kaige_Research/Code/graph_bandit/code/')
input_path='../processed_data/movielens/'
path='../bandit_results/movielens/'
rate_matrix=np.load(input_path+'rating_matrix_30_user_100_movies.npy')
true_payoffs=rate_matrix/np.max(rate_matrix)
true_payoffs[true_payoffs==0]=np.nan
W, H, error=NMF(true_payoffs, 5)

print(W.shape)
print(H.shape)
print(W)
print(H)
print(error)

