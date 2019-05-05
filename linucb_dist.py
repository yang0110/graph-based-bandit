import numpy as np 
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import Normalizer, MinMaxScaler
from scipy.sparse import csgraph 
import scipy
import os 

class LINUCB_DIST():
	def __init__(self, dimension, user_num, item_num, pool_size, item_feature_matrix, true_user_feature_matrix, true_payoffs, lap, alpha, delta, sigma):
		self.dimension=dimension
		self.user_num=user_num
		self.item_num=item_num
		self.pool_size=pool_size
		self.item_feature_matrix=item_feature_matrix
		self.true_user_feature_matrix=true_user_feature_matrix
		self.true_payoffs=true_payoffs
		self.user_feature=np.zeros((self.user_num, self.dimension))
		self.I=np.identity(self.dimension)
		self.lap=np.identity(self.user_num)
		self.true_lap=lap
		self.A=np.identity(self.user_num*self.dimension)
		self.A_inv=np.identity(self.user_num*self.dimension)
		self.alpha=alpha
		self.delta=delta
		self.sigma=sigma
		self.beta=0
		self.user_cov={}
		self.user_bias={}
		self.beta_list=[]
		self.x_norm_list=[]
		self.true_confidence_bound=[]

	def initial_user_parameter(self):
		for u in range(self.user_num):
			self.user_cov[u]=self.alpha*np.identity(self.dimension)
			self.user_bias[u]=np.zeros(self.dimension)

	def update_beta(self, user_index):
		self.beta=self.sigma*np.sqrt(2*np.log(np.linalg.det(self.user_cov[user_index])**(1/2)*np.linalg.det(self.alpha*self.I)**(-1/2)/self.delta))+self.alpha*np.linalg.norm(self.true_user_feature_matrix[user_index])
		self.beta_list.extend([self.beta])

	def select_item(self, item_pool, user_index):
		item_fs=self.item_feature_matrix[item_pool]
		estimated_payoffs=np.zeros(self.pool_size)
		self.update_beta(user_index)
		cov_inv=np.linalg.pinv(self.user_cov[user_index])
		for j in range(self.pool_size):
			x=item_fs[j]
			x_norm=np.sqrt(np.dot(np.dot(x, cov_inv),x))
			est_y=np.dot(x, self.user_feature[user_index])+self.beta*x_norm
			estimated_payoffs[j]=est_y
		self.x_norm_list.extend([x_norm])

		max_index=np.argmax(estimated_payoffs)
		selected_item_index=item_pool[max_index]
		selected_item_feature=item_fs[max_index]
		true_payoff=self.true_payoffs[user_index, selected_item_index]
		max_ideal_payoff=np.max(self.true_payoffs[user_index][item_pool])
		regret=max_ideal_payoff-true_payoff
		return true_payoff, selected_item_feature, regret

	def update_user_feature(self, true_payoff, selected_item_feature, user_index):
		self.user_cov[user_index]+=np.outer(selected_item_feature, selected_item_feature)
		self.user_bias[user_index]+=true_payoff*selected_item_feature
		self.user_feature[user_index]=np.dot(np.linalg.pinv(self.user_cov[user_index]), self.user_bias[user_index])
		x=selected_item_feature.copy()
		x_long=np.zeros(self.user_num*self.dimension)
		x_long[user_index*self.dimension:(user_index+1)*self.dimension]=x
		self.A+=np.outer(x_long,x_long)
		self.A_inv=np.linalg.pinv(self.A)
		for us in range(self.user_num):
			if us==user_index:
				pass 
			else:
				self.user_feature[us]+=np.dot(self.A_inv[us*self.dimension:(us+1)*self.dimension][:, user_index*self.dimension:(user_index+1)*self.dimension], true_payoff*x)

		delta=self.user_feature[user_index]-self.true_user_feature_matrix[user_index]
		bound=np.dot(np.dot(delta, self.user_cov[user_index]), delta)
		self.true_confidence_bound.extend([bound])

	def update_lap(self):
		adj=rbf_kernel(self.user_feature)
		lap_t_1=self.lap.copy()
		self.lap=self.true_lap.copy()
		#self.lap=csgraph.laplacian(adj, normed=False)
		self.A=self.A-np.kron(lap_t_1, np.identity(self.dimension))
		self.A+=np.kron(self.lap, np.identity(self.dimension))+np.identity(self.user_num*self.dimension)

	def run(self,  user_array, item_pool_array, iteration):
		self.initial_user_parameter()
		cumulative_regret=[0]
		learning_error_list=[]
		lap_error_list=[]
		for time in range(iteration):	
			print('time/iteration', time, iteration,'~~~LINUCB DIST')
			user_index=user_array[time]
			item_pool=item_pool_array[time]
			true_payoff, selected_item_feature, regret=self.select_item(item_pool, user_index)
			self.update_user_feature(true_payoff, selected_item_feature, user_index)
			self.update_lap()
			error=np.linalg.norm(self.user_feature-self.true_user_feature_matrix)
			cumulative_regret.extend([cumulative_regret[-1]+regret])
			learning_error_list.extend([error])
			lap_error_list.extend([np.linalg.norm(self.lap-self.true_lap)])

		return np.array(cumulative_regret), np.array(learning_error_list), np.array(lap_error_list)
