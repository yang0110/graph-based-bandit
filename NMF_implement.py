import numpy as np 
from NMF_miss_value import NMF 
import os 
from sklearn.preprocessing import Normalizer
os.chdir('C:/Kaige_Research/Code/graph_bandit/code/')
input_path='../processed_data/movielens/'
rate_matrix=np.load(input_path+'rating_matrix_100_user_500_movies.npy')
true_payoffs=rate_matrix/np.max(rate_matrix)
mask=true_payoffs!=0
np.save(input_path+'normed_rating_matrix_100_user_500_movies.npy', true_payoffs)
np.save(input_path+'rating_matrix_mask_100_user_500_movies.npy', mask)
true_payoffs[true_payoffs==0]=np.nan
W, H, error=NMF(true_payoffs, 5)
print(W.shape)
print(H.shape)
W=Normalizer().fit_transform(W)
H=Normalizer().fit_transform(H.T)
np.save(input_path+'user_feature_matrix_100.npy',  W)
np.save(input_path+'item_feature_matrix_500.npy', H)



input_path='../processed_data/lastfm/'
rate_matrix=np.load(input_path+'rating_matrix_100_user_500_artist.npy')
true_payoffs=rate_matrix/np.max(rate_matrix)
np.save(input_path+'normed_rating_matrix_100_user_500_artist.npy', true_payoffs)
mask=true_payoffs!=0
np.save(input_path+'rating_matrix_mask_100_user_500_artist.npy', mask)
true_payoffs[true_payoffs==0]=np.nan
W, H, error=NMF(true_payoffs, 5)
print(W.shape)
print(H.shape)
W=Normalizer().fit_transform(W)
H=Normalizer().fit_transform(H.T)
np.save(input_path+'user_feature_matrix_100.npy',  W)
np.save(input_path+'item_feature_matrix_500.npy', H)

input_path='../processed_data/delicious/'
rate_matrix=np.load(input_path+'rating_matrix_100_user_500_bookmark.npy')
true_payoffs=rate_matrix/np.max(rate_matrix)
np.save(input_path+'normed_rating_matrix_100_user_500_bookmark.npy', true_payoffs)
mask=true_payoffs!=0
np.save(input_path+'rating_matrix_mask_100_user_500_bookmark.npy', mask)
true_payoffs[true_payoffs==0]=np.nan
W, H, error=NMF(true_payoffs, 5)
print(W.shape)
print(H.shape)
W=Normalizer().fit_transform(W)
H=Normalizer().fit_transform(H.T)
np.save(input_path+'user_feature_matrix_100.npy',  W)
np.save(input_path+'item_feature_matrix_500.npy', H)


input_path='../processed_data/netflix/'
rate_matrix=np.load(input_path+'rating_matrix_100_user_500_movies.npy')
true_payoffs=rate_matrix/np.max(rate_matrix)
np.save(input_path+'normed_rating_matrix_100_user_500_movies.npy', true_payoffs)
mask=true_payoffs!=0
np.save(input_path+'rating_matrix_mask_100_user_500_movies.npy', mask)
true_payoffs[true_payoffs==0]=np.nan
W, H, error=NMF(true_payoffs, 5)
print(W.shape)
print(H.shape)
W=Normalizer().fit_transform(W)
H=Normalizer().fit_transform(H.T)
np.save(input_path+'user_feature_matrix_100.npy',  W)
np.save(input_path+'item_feature_matrix_500.npy', H)


#
input_path='../processed_data/lastfm/'
rate_matrix=np.load(input_path+'rating_matrix_100_user_500_artist.npy')
true_payoffs=rate_matrix/np.max(rate_matrix)
true_payoffs[true_payoffs!=0]=1.0
np.save(input_path+'binary_rating_matrix_100_user_500_artist.npy', true_payoffs)
W, H, error=NMF(true_payoffs, 5)
print(W.shape)
print(H.shape)
W=Normalizer().fit_transform(W)
H=Normalizer().fit_transform(H.T)
np.save(input_path+'binary_payoff_user_feature_matrix_100.npy',  W)
np.save(input_path+'binary_payoff_item_feature_matrix_500.npy', H)

input_path='../processed_data/delicious/'
rate_matrix=np.load(input_path+'rating_matrix_100_user_500_bookmark.npy')
true_payoffs=rate_matrix/np.max(rate_matrix)
true_payoffs[true_payoffs!=0]=1.0
np.save(input_path+'binary_rating_matrix_100_user_500_bookmark.npy', true_payoffs)
W, H, error=NMF(true_payoffs, 5)
print(W.shape)
print(H.shape)
W=Normalizer().fit_transform(W)
H=Normalizer().fit_transform(H.T)
np.save(input_path+'binary_payoff_user_feature_matrix_100.npy',  W)
np.save(input_path+'binary_payoff_item_feature_matrix_500.npy', H)