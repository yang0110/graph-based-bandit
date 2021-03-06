import numpy as np 
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import Normalizer, MinMaxScaler
from scipy.sparse import csgraph 
import scipy
import os 

class LAPUCB_SIM2(): 
	def __init__(self, dimension, user_num, item_num, pool_size, item_feature_matrix, true_user_feature_matrix, true_payoffs, true_adj,true_lap, noise_matrix, alpha, delta, sigma, beta, thres, state):
		self.true_adj=true_adj
		self.state=state
		self.dimension=dimension
		self.user_num=user_num
		self.item_num=item_num
		self.pool_size=pool_size
		self.item_feature_matrix=item_feature_matrix
		self.true_user_feature_matrix=true_user_feature_matrix
		self.true_payoffs=true_payoffs
		self.noise_matrix=noise_matrix
		self.user_feature_matrix=np.zeros((self.user_num, self.dimension))
		self.thres=thres
		self.adj=true_adj
		self.lap=true_lap
		self.L=self.lap+0.01*np.identity(self.user_num)
		self.alpha=alpha
		self.delta=delta
		self.sigma=sigma
		self.beta=beta
		self.user_bias={}
		self.user_v={}
		self.user_xx={}
		self.user_avg={}
		self.user_ridge=np.zeros((self.user_num, self.dimension))
		self.user_ls=np.zeros((self.user_num, self.dimension))
		self.beta_list=[]
		self.user_counter={}
		self.graph_error=[]
		self.user_h={}


	def initialized_parameter(self):
		for u in range(self.user_num):
			self.user_v[u]=self.alpha*np.identity(self.dimension)
			self.user_avg[u]=np.zeros(self.dimension)
			self.user_xx[u]=0.01*np.identity(self.dimension)
			self.user_bias[u]=np.zeros(self.dimension)
			self.user_counter[u]=0
			self.user_h[u]=np.zeros((self.dimension, self.dimension))

	def update_beta(self, user_index):
		a=np.linalg.det(self.user_v[user_index])**(1/2)
		b=np.linalg.det(self.alpha*np.identity(self.dimension))**(-1/2)
		d=self.sigma*np.sqrt(2*np.log(a*b/self.delta))
		#c=self.alpha*np.sqrt(np.trace(np.linalg.pinv(self.user_v[user_index])))*np.linalg.norm(self.user_feature_matrix[user_index]-self.user_avg[user_index])
		c=self.alpha*np.sqrt(np.dot(np.dot(self.user_avg[user_index],np.linalg.pinv(self.user_v[user_index])), self.user_avg[user_index]))
		self.beta=d+c
		self.beta_list.extend([self.beta])

	def select_item(self, item_pool, user_index, time):
		item_fs=self.item_feature_matrix[item_pool]
		estimated_payoffs=np.zeros(self.pool_size)
		v_inv=np.linalg.pinv(self.user_v[user_index])
		if self.state==False:
			self.update_beta(user_index)
			for j in range(self.pool_size):
				x=item_fs[j]
				x_norm=np.sqrt(np.dot(np.dot(x, v_inv),x))
				mean=np.dot(x, self.user_feature_matrix[user_index])
				est_y=mean+self.beta*x_norm
				estimated_payoffs[j]=est_y
				x_norm_bound=2*np.log(np.linalg.det(self.user_v[user_index])/np.linalg.det(self.alpha*np.identity(self.dimension)))

		else:
			for j in range(self.pool_size):
				x=item_fs[j]
				x_norm=np.sqrt(np.dot(np.dot(x, v_inv),x))
				mean=np.dot(x, self.user_feature_matrix[user_index])
				est_y=mean+self.beta*x_norm*np.sqrt(np.log(time+1))
				estimated_payoffs[j]=est_y

		max_index=np.argmax(estimated_payoffs)
		selected_item_index=item_pool[max_index]
		selected_item_feature=item_fs[max_index]
		true_payoff=self.true_payoffs[user_index, selected_item_index]
		max_ideal_payoff=np.max(self.true_payoffs[user_index][item_pool])
		regret=max_ideal_payoff-true_payoff
		return true_payoff, selected_item_feature, regret, x_norm,x_norm_bound

	def update_user_feature_upon_ridge(self, true_payoff, selected_item_feature, user_index):
		x=selected_item_feature
		self.user_xx[user_index]+=np.outer(x, x)
		self.user_v[user_index]+=np.outer(x, x)
		self.user_bias[user_index]+=true_payoff*x
		xx_inv=np.linalg.pinv(self.user_xx[user_index])
		v_inv=np.linalg.pinv(self.user_v[user_index])
		self.user_ls[user_index]=np.dot(xx_inv, self.user_bias[user_index])
		self.user_ridge[user_index]=np.dot(v_inv, self.user_bias[user_index])
		for u in range(self.user_num):
			v_inv=np.linalg.pinv(self.user_v[u])
			self.user_avg[u]=np.dot(self.user_ls.T, self.L[u])
			self.user_feature_matrix[u]=self.user_ridge[u]-self.alpha*np.dot(v_inv, self.user_avg[u]-self.user_ls[u])


	def run(self, user_array, item_pool_array, iteration):
		self.initialized_parameter()
		cumulative_regret=[0]
		learning_error_list=np.zeros(iteration)
		x_norm_list=[]
		x_norm_bound_list=[]
		inst_regret=[]
		for time in range(iteration):
			print('time/iteration', time, iteration,'~~~G-UCB SIM2')
			user_index=user_array[time]
			item_pool=item_pool_array[time]
			true_payoff, selected_item_feature, regret, x_norm, x_norm_bound=self.select_item(item_pool,user_index, time)
			x_norm_list.extend([x_norm])
			self.user_counter[user_index]+=1
			self.update_user_feature_upon_ridge(true_payoff, selected_item_feature, user_index)
			error=np.linalg.norm(self.user_feature_matrix-self.true_user_feature_matrix)
			cumulative_regret.extend([cumulative_regret[-1]+regret])
			learning_error_list[time]=error 
			inst_regret.extend([2*self.beta*x_norm])
			x_norm_bound_list.extend([x_norm_bound])
		return np.array(cumulative_regret[1:]), learning_error_list, self.beta_list, x_norm_list, inst_regret,x_norm_bound_list
