import numpy as np 
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import Normalizer, MinMaxScaler
from scipy.sparse import csgraph 
import scipy

class COLIN():
	def __init__(self, dimension, user_num, item_num, pool_size, item_feature_matrix, true_user_feature_matrix, true_payoffs, w, alpha, delta, sigma, beta):
		self.w=w 
		self.user_num=user_num
		self.dimension=dimension
		self.item_num=item_num
		self.pool_size=pool_size
		self.alpha=alpha
		self.V=self.alpha*np.identity(self.user_num*self.dimension)
		self.B=np.zeros(self.user_num*self.dimension)
		self.V_inv=np.linalg.pinv(self.V)
		self.true_user_feature_matrix=true_user_feature_matrix.T
		self.true_payoffs=true_payoffs
		self.item_feature_matrix=item_feature_matrix
		self.user_f_matrix=np.zeros((self.dimension, self.user_num))
		self.co_user_f_matrix=np.zeros((self.dimension, self.user_num))
		self.user_f_vector=np.zeros((self.user_num*self.dimension))
		self.big_w=np.kron(self.w.T, np.identity(self.dimension))
		self.cca=np.dot(np.dot(self.big_w, self.V_inv), self.big_w.T)
		self.beta=0.0
		self.sigma=sigma 
		self.delta=delta
		self.beta_list=[]

	def select_item(self, item_pool, user_index):
		self.beta=0.01*np.sqrt(np.log(np.linalg.det(self.V)/float(self.delta*self.alpha)))+np.sqrt(self.alpha)
		self.beta_list.extend([self.beta])
		pta_list=np.zeros(self.pool_size)
		for ind, item_index in enumerate(item_pool):
			item_f=self.item_feature_matrix[item_index]
			item_f_matrix=np.zeros((self.dimension, self.user_num))
			item_f_matrix[:,user_index]=item_f 
			item_f_vector=item_f_matrix.flatten()
			mean=np.dot(self.co_user_f_matrix[:,user_index], item_f)
			var=np.sqrt(np.dot(np.dot(item_f_vector, self.cca), item_f_vector))
			pta=mean+self.beta*var 
			pta_list[ind]=pta 

		max_index=np.argmax(pta_list)
		item_index=item_pool[max_index]
		selected_item_feature=self.item_feature_matrix[item_index]
		true_payoff=self.true_payoffs[user_index, item_index]
		max_payoff=np.max(self.true_payoffs[user_index, item_pool])
		regret=max_payoff-true_payoff
		return true_payoff, selected_item_feature, regret

	def update_user_feature(self, true_payoff, selected_item_feature, user_index):
		y=true_payoff
		x_matrix=np.zeros((self.dimension, self.user_num))
		x_matrix[:,user_index]=selected_item_feature
		xw=np.dot(x_matrix, self.w.T)
		self.V+=np.outer(xw.flatten(), xw.flatten())
		self.B+=y*xw.flatten()
		self.V_inv=np.linalg.pinv(self.V)
		self.user_f_matrix=np.dot(self.V_inv, self.B).reshape((self.dimension, self.user_num))
		self.co_user_f_matrix=np.dot(self.user_f_matrix, self.w)
		self.cca=np.dot(np.dot(self.big_w, self.V_inv), self.big_w.T)

	def run(self,  user_array, item_pool_array, iteration):
		cumulative_regret=[0]
		learning_error_list=np.zeros(iteration)
		for time in range(iteration):	
			print('time/iteration', time, iteration, '~~~CoLin')
			user_index=user_array[time]
			item_pool=item_pool_array[time]
			true_payoff, selected_item_feature, regret=self.select_item(item_pool, user_index)
			self.update_user_feature(true_payoff, selected_item_feature, user_index)
			error=np.linalg.norm(self.user_f_matrix-self.true_user_feature_matrix)
			cumulative_regret.extend([cumulative_regret[-1]+regret])
			learning_error_list[time]=error 

		return np.array(cumulative_regret), learning_error_list, self.beta_list





		