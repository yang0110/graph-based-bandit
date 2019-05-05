'''
Only approximate M_T for items selection
keep the exact estimation
'''
import numpy as np 
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import Normalizer, MinMaxScaler
from scipy.sparse import csgraph 
import scipy
import os 

class LAPUCB_ADAP(): 
	def __init__(self, dimension, user_num, item_num, pool_size, item_feature_matrix, true_user_feature_matrix, true_payoffs, noise_matrix, normed_lap, alpha, delta, sigma):
		self.dimension=dimension
		self.user_num=user_num
		self.item_num=item_num
		self.pool_size=pool_size
		self.item_feature_matrix=item_feature_matrix
		self.true_user_feature_matrix=true_user_feature_matrix
		self.true_payoffs=true_payoffs
		self.noise_matrix=noise_matrix
		self.user_feature_matrix=np.zeros((self.user_num, self.dimension))
		self.L=normed_lap+0.01*np.identity(self.user_num)
		self.true_L=normed_lap
		self.A=np.kron(self.L, np.identity(self.dimension))
		self.A_inv=np.linalg.pinv(self.A)
		self.alpha=alpha
		self.delta=delta
		self.sigma=sigma
		self.beta=0
		self.user_cov={}
		self.user_old_cov={}
		self.cov=self.alpha*self.A
		self.cov_inv=np.linalg.pinv(self.cov)
		self.user_phi={}
		self.bias=np.zeros((self.user_num*self.dimension))
		self.beta_list=[]
		self.a_list=[]
		self.b_list=[]
		self.c_list=[]
		self.M_ii_list=[]
		self.M_inv_ij_list=[]
		self.bias_ij_list=[]
		self.z={}
		self.cov_inv_sub={}

	def initialized_parameter(self):
		for u in range(self.user_num):
			self.user_cov[u]=np.zeros((self.dimension, self.dimension))
			self.user_phi[u]=np.zeros((self.dimension, self.dimension))
			self.user_old_cov[u]=np.zeros((self.dimension, self.dimension))
			self.z[u]=np.zeros(self.dimension)
			self.cov_inv_sub[u]=np.zeros((self.dimension, self.user_num*self.dimension))

	# def update_beta(self, user_index):
	# 	a=np.linalg.det(self.user_cov[user_index])**(1/2)
	# 	b=np.linalg.det(self.user_phi[user_index])**(-1/2)
	# 	d=self.sigma*np.sqrt(2*np.log(a*b/self.delta))
	# 	theta=self.true_user_feature_matrix[user_index]
	# 	dot=np.dot(self.user_phi[user_index], theta)
	# 	dot_norm=np.linalg.norm(dot)
	# 	c=np.sqrt(1/(self.alpha))*dot_norm
	# 	cov_inv=np.linalg.pinv(self.user_cov[user_index])
	# 	norm=np.sqrt(np.dot(np.dot(dot, cov_inv), dot))
	# 	self.beta=c+d
	# 	self.beta_list.extend([self.beta])
	# 	self.d_list.extend([d])
	# 	self.c_list.extend([c])

	def true_beta(self, user_index):
		M_inv_sub=self.cov_inv[user_index*self.dimension:(user_index+1)*self.dimension].copy()
		M_ii=self.user_cov[user_index]
		M_inv_ii=np.linalg.pinv(M_ii)
		phi_ii=self.user_phi[user_index]
		z=self.z[user_index]
		true_theta=self.true_user_feature_matrix[user_index]
		est_theta=self.user_feature_matrix[user_index]
		a=-np.dot(phi_ii, est_theta)
		b=z 
		c=np.zeros(self.dimension)
		for u in range(self.user_num):
			if u==user_index:
				pass 
			else:
				M_inv_ij=M_inv_sub[:,u*self.dimension:(u+1)*self.dimension]
				bias_ij=self.bias[u*self.dimension:(u+1)*self.dimension]
				c+=np.dot(np.dot(M_ii, M_inv_ij), bias_ij)

		total=a+b+c 
		self.beta=np.sqrt(np.dot(np.dot(total, M_inv_ii), total))
		self.beta_list.extend([self.beta])
		self.a_list.extend([np.linalg.norm(a)])
		self.b_list.extend([np.linalg.norm(b)])
		self.c_list.extend([np.linalg.norm(c)])
		self.M_ii_list.extend([np.linalg.norm(M_ii)])
		self.M_inv_ij_list.extend([np.linalg.norm(M_inv_ij)])
		self.bias_ij_list.extend([np.linalg.norm(bias_ij)])


	def select_item(self, item_pool, user_index, time):
		item_fs=self.item_feature_matrix[item_pool]
		estimated_payoffs=np.zeros(self.pool_size)
		self.true_beta(user_index)
		cov_inv=np.linalg.pinv(self.user_cov[user_index])
		for j in range(self.pool_size):
			x=item_fs[j]
			x_norm=np.sqrt(np.dot(np.dot(x, cov_inv),x))
			est_y=np.dot(x, self.user_feature_matrix[user_index])+self.beta*x_norm
			estimated_payoffs[j]=est_y

		max_index=np.argmax(estimated_payoffs)
		selected_item_index=item_pool[max_index]
		selected_item_feature=item_fs[max_index]
		true_payoff=self.true_payoffs[user_index, selected_item_index]
		max_ideal_payoff=np.max(self.true_payoffs[user_index][item_pool])
		regret=max_ideal_payoff-true_payoff
		self.z[user_index]+=selected_item_feature*self.noise_matrix[user_index,selected_item_index]
		return true_payoff, selected_item_feature, regret

	def update_user_feature(self, true_payoff, selected_item_feature, user_index):
		self.user_old_cov[user_index]+=np.outer(selected_item_feature, selected_item_feature)
		x_long=np.zeros((self.user_num*self.dimension))
		x_long[user_index*self.dimension:(user_index+1)*self.dimension]=selected_item_feature
		self.cov+=np.outer(x_long, x_long)
		self.bias+=true_payoff*x_long
		self.cov_inv=np.linalg.pinv(self.cov)
		self.user_feature_matrix=np.dot(self.cov_inv, self.bias).reshape((self.user_num, self.dimension))
		self.user_cov[user_index]=np.linalg.pinv(self.cov_inv[user_index*self.dimension:(user_index+1)*self.dimension,user_index*self.dimension:(user_index+1)*self.dimension])
		self.user_phi[user_index]=self.user_cov[user_index]-self.user_old_cov[user_index]

	def update_graph(self):
		adj=rbf_kernel(self.user_feature_matrix)
		self.L=csgraph.laplacian(adj, normed=True)
		A_t_1=self.A.copy()
		self.A=np.kron(self.L+0.01*np.identity(self.user_num), np.identity(self.dimension))
		self.cov+=self.alpha*(self.A-A_t_1)

	def run(self,  user_array, item_pool_array, iteration):
		self.initialized_parameter()
		cumulative_regret=[0]
		learning_error_list=np.zeros(iteration)
		graph_learning_error=np.zeros(iteration)
		for time in range(iteration):	
			print('time/iteration', time, iteration,'~~~LAPUCB ADAP')
			user_index=user_array[time]
			item_pool=item_pool_array[time]
			true_payoff, selected_item_feature, regret=self.select_item(item_pool,user_index, time)
			self.update_user_feature(true_payoff, selected_item_feature, user_index)
			self.update_graph()
			error=np.linalg.norm(self.user_feature_matrix-self.true_user_feature_matrix)
			cumulative_regret.extend([cumulative_regret[-1]+regret])
			learning_error_list[time]=error
			graph_learning_error[time]=np.linalg.norm(self.L-self.true_L)

		return np.array(cumulative_regret), learning_error_list, graph_learning_error, self.beta_list, self.a_list, self.b_list, self.c_list,	self.M_ii_list,self.M_inv_ij_list,self.bias_ij_list
