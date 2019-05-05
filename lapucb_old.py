import numpy as np 
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import Normalizer, MinMaxScaler
from scipy.sparse import csgraph 
import scipy
import os 

class LAPUCB_OLD(): ## together update feature and confidence bound
	def __init__(self, dimension, user_num, item_num, pool_size, item_feature_matrix, true_user_feature_matrix, true_payoffs,noise_matrix,  normed_lap, alpha,  delta, sigma):
		self.dimension=dimension
		self.user_num=user_num
		self.item_num=item_num
		self.pool_size=pool_size
		self.item_feature_matrix=item_feature_matrix
		self.true_user_feature_matrix=true_user_feature_matrix
		self.true_payoffs=true_payoffs
		self.noise_matrix=noise_matrix
		self.true_user_feature_vector=true_user_feature_matrix.flatten()
		self.user_feature=np.zeros(self.user_num*self.dimension)
		self.user_feature_matrix=np.zeros((self.user_num, self.dimension))
		self.L=normed_lap+0.01*np.identity(self.user_num)
		self.A=np.kron(self.L, np.identity(self.dimension))
		self.alpha=alpha
		self.delta=delta
		self.sigma=sigma
		self.beta=0
		self.covariance=self.alpha*self.A
		self.bias=np.zeros(self.user_num*self.dimension)
		self.user_cov={}
		self.beta_list=[]
		self.a_list=[]
		self.b_list=[]
		self.true_c_list=[]
		self.z={}

	def initialized_neighbor_parameter(self):
		for u in range(self.user_num):
			self.user_cov[u]=self.alpha*np.identity(self.dimension)
			self.z[u]=np.zeros(self.dimension)

	# def update_beta(self, user_index):
	# 	a=np.linalg.det(self.user_cov[user_index])**(1/2)
	# 	b=np.linalg.det(self.alpha_2*np.identity(self.dimension))**(-1/2)
	# 	c=np.sqrt(self.alpha_2)*np.linalg.norm(self.true_user_feature_matrix[user_index])
	# 	d=self.sigma*np.sqrt(2*np.log(a*b/self.delta))
	# 	theta=self.true_user_feature_matrix[user_index]
	# 	cov_inv=np.linalg.pinv(self.user_cov[user_index])
	# 	norm=self.alpha_2*np.sqrt(np.dot(np.dot(theta, cov_inv), theta))
	# 	self.beta=c+d
	# 	self.beta_list.extend([self.beta])
	# 	self.d_list.extend([d])
	# 	self.c_list.extend([c])
	# 	self.true_c_list.extend([norm])

	def true_beta(self, user_index):
		true_theta=self.true_user_feature_matrix[user_index].copy()
		est_theta=self.user_feature_matrix[user_index].copy()
		M_ii=self.user_cov[user_index].copy()
		M_inv_ii=np.linalg.pinv(M_ii)
		a=-self.alpha*est_theta
		b=self.z[user_index].copy()
		total=a+b 
		self.beta=np.sqrt(np.dot(np.dot(total, M_inv_ii), total))
		self.beta_list.extend([self.beta])
		self.a_list.extend([np.linalg.norm(a)])
		self.b_list.extend([np.linalg.norm(b)])

	def select_item(self, item_pool, user_index, time):
		item_fs=self.item_feature_matrix[item_pool]
		item_feature_array=np.zeros((self.pool_size, self.user_num*self.dimension))
		item_feature_array[:,user_index*self.dimension:(user_index+1)*self.dimension]=item_fs
		estimated_payoffs=np.zeros(self.pool_size)
		self.true_beta(user_index)
		cov_inv=np.linalg.pinv(self.covariance)
		for j in range(self.pool_size):
			x=item_feature_array[j]
			x_norm=np.sqrt(np.dot(np.dot(x, cov_inv),x))
			est_y=np.dot(x, self.user_feature)+self.beta*x_norm
			estimated_payoffs[j]=est_y

		max_index=np.argmax(estimated_payoffs)
		selected_item_index=item_pool[max_index]
		selected_item_feature=item_feature_array[max_index]
		true_payoff=self.true_payoffs[user_index, selected_item_index]
		max_ideal_payoff=np.max(self.true_payoffs[user_index][item_pool])
		regret=max_ideal_payoff-true_payoff
		x=selected_item_feature[user_index*self.dimension:(user_index+1)*self.dimension]
		self.z[user_index]+=x*self.noise_matrix[user_index, selected_item_index]
		return true_payoff, selected_item_feature, regret

	def update_user_feature(self, true_payoff, selected_item_feature, user_index):
		self.covariance+=np.outer(selected_item_feature, selected_item_feature)
		self.bias+=true_payoff*selected_item_feature
		cov_inv=np.linalg.pinv(self.covariance)
		self.user_feature=np.dot(cov_inv, self.bias)
		self.user_feature_matrix=self.user_feature.reshape((self.user_num, self.dimension))
		x=selected_item_feature[user_index*self.dimension:(user_index+1)*self.dimension]
		self.user_cov[user_index]+=np.outer(x,x)

	def run(self,  user_array, item_pool_array, iteration):
		self.initialized_neighbor_parameter()
		cumulative_regret=[0]
		learning_error_list=np.zeros(iteration)
		for time in range(iteration):	
			print('time/iteration', time, iteration,'~~~LAPUCB OLD')
			user_index=user_array[time]
			item_pool=item_pool_array[time]
			true_payoff, selected_item_feature, regret=self.select_item(item_pool, user_index,time)
			self.update_user_feature(true_payoff, selected_item_feature, user_index)
			error=np.linalg.norm(self.user_feature-self.true_user_feature_vector)
			cumulative_regret.extend([cumulative_regret[-1]+regret])
			learning_error_list[time]=error 

		return cumulative_regret, learning_error_list, self.beta_list, self.a_list, self.b_list