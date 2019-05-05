import numpy as np 
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import Normalizer, MinMaxScaler
from scipy.sparse import csgraph 
import scipy
import os 

class LAPUCB_SIN_F(): ## together update feature and confidence bound
	def __init__(self, dimension, user_num, item_num, pool_size, item_feature_matrix, true_user_feature_matrix, true_payoffs, lap, alpha, delta, sigma):
		self.dimension=dimension
		self.user_num=user_num
		self.item_num=item_num
		self.pool_size=pool_size
		self.item_feature_matrix=item_feature_matrix
		self.true_user_feature_matrix=true_user_feature_matrix
		self.true_payoffs=true_payoffs
		self.true_user_feature_vector=true_user_feature_matrix.flatten()
		self.user_feature=np.zeros(self.user_num*self.dimension)
		self.L=lap
		self.eigs,_ = np.linalg.eig(self.L)
		self.eigs=np.sort(self.eigs)
		self.A=np.kron(self.L+np.identity(self.user_num), np.identity(self.dimension))
		self.I=np.identity(self.A.shape[0])
		self.alpha=alpha
		self.delta=delta
		self.sigma=sigma
		self.beta=0
		self.covariance=self.alpha*(self.A+self.I)
		self.bias=np.zeros(self.user_num*self.dimension)
		self.beta_list=[]
		self.x_norm_list=[]
		self.true_confidence_bound=[]
		self.true_confidence_bound_small=[]

	def update_beta(self):
		r=np.linalg.matrix_rank(self.true_user_feature_matrix)
		norm=np.linalg.norm(np.dot(self.L, self.true_user_feature_matrix))
		k=1
		lam2=self.eigs[1]
		self.beta=self.alpha*(np.sqrt(r)+2*norm)/(k+self.alpha*lam2)
		self.beta_list.extend([self.beta])

	def select_item(self, item_pool, user_index):
		item_fs=self.item_feature_matrix[item_pool]
		item_feature_array=np.zeros((self.pool_size, self.user_num*self.dimension))
		item_feature_array[:,user_index*self.dimension:(user_index+1)*self.dimension]=item_fs
		estimated_payoffs=np.zeros(self.pool_size)
		self.update_beta()
		cov_inv=np.linalg.pinv(self.covariance)
		for j in range(self.pool_size):
			x=item_feature_array[j]
			x_norm=np.sqrt(np.dot(np.dot(x, cov_inv),x))
			est_y=np.dot(x, self.user_feature)+self.beta*x_norm
			estimated_payoffs[j]=est_y
		self.x_norm_list.extend([x_norm])

		max_index=np.argmax(estimated_payoffs)
		selected_item_index=item_pool[max_index]
		selected_item_feature=item_feature_array[max_index]
		true_payoff=self.true_payoffs[user_index, selected_item_index]
		max_ideal_payoff=np.max(self.true_payoffs[user_index][item_pool])
		regret=max_ideal_payoff-true_payoff
		return true_payoff, selected_item_feature, regret

	def update_user_feature(self, true_payoff, selected_item_feature, user_index):
		self.covariance+=np.outer(selected_item_feature, selected_item_feature)
		self.bias+=true_payoff*selected_item_feature
		cov_inv=np.linalg.pinv(self.covariance)
		self.user_feature=np.dot(cov_inv, self.bias)
		delta=self.user_feature-self.true_user_feature_vector
		bound=np.dot(np.dot(delta, self.covariance), delta)
		self.true_confidence_bound.extend([bound])
		delta=self.user_feature.reshape((self.user_num, self.dimension))[user_index]-self.true_user_feature_vector.reshape((self.user_num, self.dimension))[user_index]
		cov=self.covariance[user_index*self.dimension:(user_index+1)*self.dimension,user_index*self.dimension:(user_index+1)*self.dimension]
		bound=np.dot(np.dot(delta, cov), delta)
		self.true_confidence_bound_small.extend([bound])


	def run(self,  user_array, item_pool_array, iteration):
		cumulative_regret=[0]
		learning_error_list=np.zeros(iteration)
		for time in range(iteration):	
			print('time/iteration', time, iteration,'~~~LAPUCB SIN Fro')
			user_index=user_array[time]
			item_pool=item_pool_array[time]
			true_payoff, selected_item_feature, regret=self.select_item(item_pool, user_index)
			self.update_user_feature(true_payoff, selected_item_feature, user_index)
			error=np.linalg.norm(self.user_feature-self.true_user_feature_vector)
			cumulative_regret.extend([cumulative_regret[-1]+regret])
			learning_error_list[time]=error 

		return cumulative_regret, learning_error_list, np.array(self.beta_list)