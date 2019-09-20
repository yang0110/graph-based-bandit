import numpy as np 
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import Normalizer, MinMaxScaler
from scipy.sparse import csgraph 
import scipy
import os 


class TS():
	def __init__(self, dimension, user_num, item_num, pool_size, item_feature_matrix, true_user_feature_matrix, true_payoffs, alpha, delta, sigma, epsilon, state):
		self.state=state
		self.dimension=dimension
		self.user_num=user_num
		self.item_num=item_num
		self.pool_size=pool_size
		self.item_feature_matrix=item_feature_matrix
		self.true_user_feature_matrix=true_user_feature_matrix
		self.true_payoffs=true_payoffs
		self.user_feature=np.zeros((self.user_num, self.dimension))
		self.I=np.identity(self.dimension)
		self.alpha=alpha
		self.delta=delta
		self.sigma=sigma
		self.epsilon=epsilon
		self.v=self.sigma*np.sqrt(24/self.epsilon*self.dimension*np.log(1/self.delta))
		self.beta=0
		self.user_cov={}
		self.user_xx={}
		self.user_bias={}
		self.beta_list=[]
		self.real_beta_list=[]

	def initial_user_parameter(self):
		for u in range(self.user_num):
			self.user_cov[u]=self.alpha*np.identity(self.dimension)
			self.user_xx[u]=0.01*np.identity(self.dimension)
			self.user_bias[u]=np.zeros(self.dimension)

	def select_item(self, item_pool, user_index, time):
		item_fs=self.item_feature_matrix[item_pool]
		estimated_payoffs=np.zeros(self.pool_size)
		sample_user_f=np.random.multivariate_normal(self.user_feature[user_index], np.linalg.pinv(self.user_cov[user_index]))
		for j in range(self.pool_size):
			x=item_fs[j]
			est_y=np.dot(x, sample_user_f)
			estimated_payoffs[j]=est_y

		max_index=np.argmax(estimated_payoffs)
		selected_item_index=item_pool[max_index]
		selected_item_feature=item_fs[max_index]
		true_payoff=self.true_payoffs[user_index, selected_item_index]
		max_ideal_payoff=np.max(self.true_payoffs[user_index][item_pool])
		regret=max_ideal_payoff-true_payoff
		return true_payoff, selected_item_feature, regret

	def update_user_feature(self, true_payoff, selected_item_feature, user_index):
		self.user_cov[user_index]+=np.outer(selected_item_feature, selected_item_feature)
		self.user_xx[user_index]+=np.outer(selected_item_feature, selected_item_feature)
		self.user_bias[user_index]+=true_payoff*selected_item_feature
		self.user_feature[user_index]=np.dot(np.linalg.pinv(self.user_cov[user_index]), self.user_bias[user_index])

	def run(self,user_array, item_pool_array, iteration):
		self.initial_user_parameter()
		cumulative_regret=[0]
		learning_error_list=[]
		inst_regret=[]
		for time in range(iteration):	
			print('time/iteration', time, iteration,'~~~TS')
			user_index=user_array[time]
			item_pool=item_pool_array[time]
			true_payoff, selected_item_feature, regret=self.select_item(item_pool, user_index, time)
			self.update_user_feature(true_payoff, selected_item_feature, user_index)
			error=np.linalg.norm(self.user_feature-self.true_user_feature_matrix)
			cumulative_regret.extend([cumulative_regret[-1]+regret])
			learning_error_list.extend([error])
			inst_regret.extend([regret])

		return cumulative_regret[1:], learning_error_list
