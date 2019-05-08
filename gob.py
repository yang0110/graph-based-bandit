import numpy as np 
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import Normalizer, MinMaxScaler
from scipy.sparse import csgraph 
import scipy


class GOB():
	def __init__(self, dimension, user_num, item_num, pool_size, item_feature_matrix, true_user_feature_matrix, true_payoffs, lap, alpha, delta, sigma, beta):
		self.dimension=dimension
		self.user_num=user_num
		self.item_num=item_num
		self.pool_size=pool_size
		self.item_feature_matrix=item_feature_matrix
		self.true_user_feature_matrix=true_user_feature_matrix
		self.true_payoffs=true_payoffs
		self.true_user_feature_vector=true_user_feature_matrix.flatten()
		self.user_feature=np.zeros(self.user_num*self.dimension)
		self.I=np.identity(self.user_num*self.dimension)
		self.L=lap+np.identity(self.user_num)
		self.A=np.kron(self.L, np.identity(self.dimension))
		self.A_inv=np.linalg.pinv(self.A)
		self.A_sqrt=scipy.linalg.sqrtm(self.A)
		self.A_inv_sqrt=scipy.linalg.sqrtm(self.A_inv)
		self.alpha=alpha
		self.delta=delta
		self.sigma=sigma
		self.beta=beta
		self.covariance=self.alpha*self.I
		self.bias=np.zeros(self.user_num*self.dimension)
		self.beta_list=[]

	def update_beta(self): #not used
		a=np.linalg.det(self.covariance)**(1/2)
		b=np.linalg.det(self.alpha*self.I)**(-1/2)
		self.beta=self.sigma*np.sqrt(2*np.log(a*b/self.delta))+np.sqrt(self.alpha)*np.linalg.norm(np.dot(self.A_sqrt,self.true_user_feature_vector))
		self.beta_list.extend([self.beta])
		
	def select_item(self, item_pool, user_index, time):
		item_fs=self.item_feature_matrix[item_pool]
		item_feature_array=np.zeros((self.pool_size, self.user_num*self.dimension))
		item_feature_array[:,user_index*self.dimension:(user_index+1)*self.dimension]=item_fs
		estimated_payoffs=np.zeros(self.pool_size)
		#self.update_beta()
		cov_inv=np.linalg.pinv(self.covariance)
		for j in range(self.pool_size):
			x=item_feature_array[j]
			x=np.dot(self.A_inv_sqrt, x)
			x_norm=np.sqrt(np.dot(np.dot(x, cov_inv),x))
			est_y=np.dot(x, self.user_feature)+self.beta*x_norm*np.sqrt(np.log(time+1))
			estimated_payoffs[j]=np.real(est_y)

		max_index=np.argmax(estimated_payoffs)
		selected_item_index=item_pool[max_index]
		selected_item_feature=item_feature_array[max_index]
		true_payoff=self.true_payoffs[user_index, selected_item_index]
		max_ideal_payoff=np.max(self.true_payoffs[user_index][item_pool])
		regret=max_ideal_payoff-true_payoff
		return true_payoff, selected_item_feature, regret

	def update_user_feature(self, true_payoff, selected_item_feature, user_index):
		x_share=np.real(np.dot(self.A_inv_sqrt, selected_item_feature))
		self.covariance+=np.outer(x_share, x_share)
		self.bias+=true_payoff*x_share
		cov_inv=np.linalg.pinv(self.covariance)
		self.user_feature=np.dot(cov_inv, self.bias)

	def run(self,  user_array, item_pool_array, iteration):
		cumulative_regret=[0]
		learning_error_list=np.zeros(iteration)
		learning_error_list_2=np.zeros(iteration)
		for time in range(iteration):	
			print('time/iteration', time, iteration, '~~~GOB')
			user_index=user_array[time]
			item_pool=item_pool_array[time]
			true_payoff, selected_item_feature, regret=self.select_item(item_pool, user_index, time)
			self.update_user_feature(true_payoff, selected_item_feature, user_index)
			error=np.linalg.norm(self.user_feature-self.true_user_feature_vector)
			cumulative_regret.extend([cumulative_regret[-1]+regret])
			learning_error_list[time]=error 
			error_2=np.linalg.norm(self.user_feature-np.dot(self.A_sqrt, self.true_user_feature_vector))
			learning_error_list_2[time]=error_2

		return np.array(cumulative_regret), learning_error_list, self.beta_list