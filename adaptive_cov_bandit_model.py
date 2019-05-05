import numpy as np 
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import Normalizer, MinMaxScaler
from scipy.sparse import csgraph 
import scipy
import os 
os.chdir('../code/')
from learn_lap_optimization_algorithm import * 

class Adaptive_COVUCB():
	def __init__(self, dimension, user_num, item_num, pool_size, item_feature_matrix, true_user_feature_matrix, true_payoffs, alpha, beta, delta, sigma, mu, lambda_):
		self.dimension=dimension
		self.user_num=user_num
		self.item_num=item_num
		self.pool_size=pool_size
		self.item_feature_matrix=item_feature_matrix
		self.true_user_features_matrix=true_user_feature_matrix
		self.true_payoffs=true_payoffs
		self.true_user_feature_vector=true_user_feature_matrix.flatten()
		self.user_feature=np.zeros(self.user_num*self.dimension)
		self.user_feature_matrix=np.zeros((self.user_num, self.dimension))
		self.I=np.identity(self.user_num*self.dimension)
		self.A=np.identity(self.user_num*self.dimension)
		self.L=np.identity(self.user_num)
		self.alpha=alpha # regularizer
		self.beta=beta # regularizer
		self.delta=delta # high probability
		self.sigma=sigma # noise
		self.mu=mu # step size
		self.lambda_=lambda_ ## step size
		self.c=0 # confidence bound
		self.covariance=self.alpha*self.A
		self.bias=np.zeros(self.user_num*self.dimension)


	def update_c(self):
		self.c=self.sigma*np.sqrt(2*np.log(np.linalg.det(self.covariance)**(1/2)*np.linalg.det(self.alpha*self.A)**(-1/2)/self.delta))+self.alpha*np.linalg.norm(self.true_user_feature_vector)

	def select_item(self, item_pool, user_index):
		item_fs=self.item_feature_matrix[item_pool]
		item_feature_array=np.zeros((self.pool_size, self.user_num*self.dimension))
		item_feature_array[:,user_index*self.dimension:(user_index+1)*self.dimension]=item_fs
		estimated_payoffs=np.zeros(self.pool_size)
		self.update_c()
		for j in range(self.pool_size):
			x=item_feature_array[j]
			x_norm=np.sqrt(np.dot(np.dot(x, np.linalg.pinv(self.covariance)),x))
			est_y=np.dot(x, self.user_feature)+self.c*x_norm
			estimated_payoffs[j]=est_y

		max_index=np.argmax(estimated_payoffs)
		selected_item_index=item_pool[max_index]
		selected_item_feature=item_feature_array[max_index]
		true_payoff=self.true_payoffs[user_index, selected_item_index]
		max_ideal_payoff=np.max(self.true_payoffs[user_index][item_pool])
		regret=max_ideal_payoff-true_payoff
		return true_payoff, selected_item_feature, regret

	def update_user_feature(self, true_payoff, selected_item_feature):
		self.covariance+=np.outer(selected_item_feature, selected_item_feature)
		self.bias+=true_payoff*selected_item_feature
		self.user_feature=np.dot(np.linalg.pinv(self.covariance), self.bias)
		self.user_feature_matrix=self.user_feature.reshape(self.user_num, self.dimension)

	def update_A(self):
		A_t_1=self.A.copy()
		#self.L=learn_lap(self.user_feature_matrix, self.alpha, self.beta)
		#self.L=csgraph.laplacian(rbf_kernel(self.user_feature_matrix), normed=False)
		self.L=np.cov(self.user_feature_matrix)+np.identity(self.user_num)
		self.L[self.L<=1e-4]=0
		self.A=np.kron(self.L, np.identity(self.dimension))
		self.covariance+=self.alpha*(self.A-A_t_1)

	def run(self,  user_array, item_pool_array, iteration, true_lap):
		cumulative_regret=[0]
		learning_error_list=np.zeros(iteration)
		lap_error=np.zeros(iteration)
		for time in range(iteration):	
			print('time/iteration', time, iteration,'~~~Adaptive COVUCB')
			user_index=user_array[time]
			item_pool=item_pool_array[time]
			true_payoff, selected_item_feature, regret=self.select_item(item_pool, user_index)
			self.update_user_feature(true_payoff, selected_item_feature)
			self.update_A()
			cumulative_regret.extend([cumulative_regret[-1]+regret])
			error=np.linalg.norm(self.user_feature-self.true_user_feature_vector)
			learning_error_list[time]=error 
			lap_error[time]=np.linalg.norm(self.L-true_lap)

		return cumulative_regret, learning_error_list, lap_error
