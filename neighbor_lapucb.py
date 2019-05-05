'''
Approximate both M_T and B_T and estimation 
'''
import numpy as np 
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import Normalizer, MinMaxScaler
from scipy.sparse import csgraph 
import scipy
import os 

class LAPUCB_NEI(): 
	def __init__(self, dimension, user_num, item_num, pool_size, item_feature_matrix, true_user_feature_matrix, true_payoffs, lap, alpha, delta, sigma):
		self.dimension=dimension
		self.user_num=user_num
		self.item_num=item_num
		self.pool_size=pool_size
		self.item_feature_matrix=item_feature_matrix
		self.true_user_feature_matrix=true_user_feature_matrix
		self.true_payoffs=true_payoffs
		self.user_feature_matrix=np.zeros((self.user_num, self.dimension))
		self.L=lap
		self.A=np.kron(self.L, np.identity(self.dimension))
		self.I=np.identity(self.A.shape[0])
		self.adj=-self.L
		np.fill_diagonal(self.adj, 1)
		self.alpha=alpha
		self.delta=delta
		self.sigma=sigma
		self.beta=0
		self.served_user_list=[]
		self.neighbor_dict={}
		self.user_cov={}
		self.user_bias={}
		self.user_neighbor_index={}
		self.cov=self.alpha*(self.A+self.I)
		self.bias=np.zeros((self.user_num*self.dimension))
		self.beta_list=[]
		self.x_norm_list=[]
		self.true_confidence_bound=[]

	def initialized_neighbor_parameter(self):
		for u in range(self.user_num):
			adj_row=self.adj[u]
			neighbors=np.where(adj_row>0)[0].tolist()
			self.user_neighbor_index[u]=np.where(np.array(neighbors)==u)[0].tolist()[0]
			self.neighbor_dict[u]=neighbors
			self.user_cov[u]=np.zeros((self.dimension, self.dimension))
			self.user_bias[u]=np.zeros(self.dimension)

	def update_beta(self, user_index):
		self.beta=self.sigma*np.sqrt(2*np.log(np.linalg.det(self.user_cov[user_index])**(1/2)/self.delta))+self.alpha*np.linalg.norm(self.true_user_feature_matrix[user_index])
		self.beta_list.extend([self.beta])
			
	def select_item(self, item_pool, user_index):
		neighbors=self.neighbor_dict[user_index]
		index=self.user_neighbor_index[user_index]
		item_fs=self.item_feature_matrix[item_pool]
		estimated_payoffs=np.zeros(self.pool_size)
		self.update_beta(user_index)
		cov_inv=np.linalg.pinv(self.user_cov[user_index])
		for j in range(self.pool_size):
			x=item_fs[j]
			x_norm=np.sqrt(np.dot(np.dot(x,cov_inv) ,x))
			est_y=np.dot(x, self.user_feature_matrix[user_index])+self.beta*x_norm
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
		x_long=np.zeros((self.user_num*self.dimension))
		x_long[user_index*self.dimension:(user_index+1)*self.dimension]=selected_item_feature
		self.cov+=np.outer(x_long, x_long)
		self.bias+=true_payoff*x_long

		neighbors=self.neighbor_dict[user_index]
		index=self.user_neighbor_index[user_index]
		neighbor_index=np.zeros((len(neighbors), self.dimension))
		for j, nei in enumerate(neighbors):
			neighbor_index[j]=np.arange(nei*self.dimension, (nei+1)*self.dimension)
		neighbor_index=neighbor_index.flatten().astype(int)
		neighbor_cov=self.cov[neighbor_index][:, neighbor_index].copy()
		neighbor_bias=self.bias[neighbor_index].copy()
		neighbor_cov_inv=np.linalg.pinv(neighbor_cov)
		self.user_feature_matrix[neighbors]=np.dot(neighbor_cov_inv, neighbor_bias).reshape((len(neighbors), self.dimension))

		sub_cov_inv=neighbor_cov_inv[index*self.dimension:(index+1)*self.dimension,index*self.dimension:(index+1)*self.dimension].copy()
		self.user_cov[user_index]=np.linalg.pinv(sub_cov_inv)
		delta=self.user_feature_matrix[user_index]-self.true_user_feature_matrix[user_index]
		bound=np.dot(np.dot(delta, self.user_cov[user_index]), delta)
		self.true_confidence_bound.extend([bound])

	def run(self,  user_array, item_pool_array, iteration):
		self.initialized_neighbor_parameter()
		cumulative_regret=[0]
		learning_error_list=np.zeros(iteration)
		learning_error_list_2=np.zeros(iteration)
		for time in range(iteration):	
			print('time/iteration', time, iteration,'~~~LAPUCB NEI')
			user_index=user_array[time]
			item_pool=item_pool_array[time]
			true_payoff, selected_item_feature, regret=self.select_item(item_pool,user_index)
			self.update_user_feature(true_payoff, selected_item_feature, user_index)
			error=np.linalg.norm(self.user_feature_matrix-self.true_user_feature_matrix)
			cumulative_regret.extend([cumulative_regret[-1]+regret])
			learning_error_list[time]=error 

		return np.array(cumulative_regret), learning_error_list, np.array(self.beta_list), np.array(self.x_norm_list), np.array(self.true_confidence_bound)