import numpy as np 
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import Normalizer, MinMaxScaler
from scipy.sparse import csgraph 
import scipy

class COLIN():
	def __init__(self, dimension, user_num, item_num, pool_size, item_feature_matrix, true_user_feature_matrix, true_payoffs, true_adj, alpha, delta, sigma, beta, state):
		self.true_adj=true_adj
		self.state=state
		self.user_num=user_num
		self.w=np.identity(self.user_num)
		self.lap=csgraph.laplacian(self.w, normed=True)
		self.dimension=dimension
		self.item_num=item_num
		self.pool_size=pool_size
		self.alpha=alpha
		self.V=self.alpha*np.identity(self.user_num*self.dimension)
		self.B=np.zeros(self.user_num*self.dimension)
		self.V_inv=np.linalg.pinv(self.V)
		self.true_user_feature_matrix=true_user_feature_matrix
		self.true_user_feature_vector=self.true_user_feature_matrix.flatten()
		self.true_payoffs=true_payoffs
		self.item_feature_matrix=item_feature_matrix
		self.user_f_matrix=np.zeros((self.dimension, self.user_num))
		self.co_user_f_matrix=np.zeros((self.dimension, self.user_num))
		self.true_co_user_f_vector=np.dot(self.true_user_feature_matrix.T, self.w).flatten()
		self.user_f_vector=np.zeros((self.user_num*self.dimension))
		self.user_ridge=np.zeros((self.dimension, self.user_num))
		self.user_ls=np.zeros((self.dimension, self.user_num))
		self.big_w=np.kron(self.w.T, np.identity(self.dimension))
		self.beta=beta
		self.sigma=sigma 
		self.delta=delta
		self.beta_list=[]
		self.user_xx={}
		self.user_v={}
		self.user_bias={}
		self.user_counter={}
		self.graph_error=[]

	def initial(self):
		for u in range(self.user_num):
			self.user_xx[u]=np.zeros((self.dimension, self.dimension))
			self.user_v[u]=self.alpha*np.identity(self.dimension)
			self.user_bias[u]=np.zeros(self.dimension)
			self.user_counter[u]=0


	def update_beta(self): 
		a=np.linalg.det(self.V)**(1/2)
		b=np.linalg.det(self.alpha*np.identity(self.user_num*self.dimension))**(-1/2)
		beta=self.sigma*np.sqrt(2*np.log(a*b/self.delta))+np.sqrt(self.alpha)*np.linalg.norm(self.co_user_f_matrix.flatten())
		self.beta_list.extend([beta])

	def select_item(self, item_pool, user_index, time):
		est_payoffs=np.zeros(self.pool_size)
		self.V_inv=np.linalg.pinv(self.V)
		if self.state==False:
			self.update_beta()
			for j in range(self.pool_size):
				item_index=item_pool[j]
				item_f=self.item_feature_matrix[item_index]
				item_f_matrix=np.zeros((self.dimension, self.user_num))
				item_f_matrix[:, user_index]=item_f 
				item_f_vector=item_f_matrix.flatten('F')
				co_item_f_vector=np.dot(item_f_matrix, self.w.T).flatten('F')
				mean=np.dot(self.co_user_f_matrix.flatten('F'), item_f_vector)
				var=np.sqrt(np.dot(np.dot(co_item_f_vector, self.V_inv), co_item_f_vector))
				est_payoff=mean+self.beta*var
				est_payoffs[j]=est_payoff
		else:
			for j in range(self.pool_size):
				item_index=item_pool[j]
				item_f=self.item_feature_matrix[item_index]
				item_f_matrix=np.zeros((self.dimension, self.user_num))
				item_f_matrix[:, user_index]=item_f 
				item_f_vector=item_f_matrix.flatten('F')
				co_item_f_vector=np.dot(item_f_matrix, self.w.T).flatten('F')
				mean=np.dot(self.co_user_f_matrix.flatten('F'), item_f_vector)
				var=np.sqrt(np.dot(np.dot(co_item_f_vector, self.V_inv), co_item_f_vector))
				est_payoff=mean+self.beta*var*np.sqrt(np.log(time+1))
				est_payoffs[j]=est_payoff

		max_index=np.argmax(est_payoffs)
		item_index=item_pool[max_index]
		selected_item_feature=self.item_feature_matrix[item_index]
		true_payoff=self.true_payoffs[user_index, item_index]
		max_payoff=np.max(self.true_payoffs[user_index][item_pool])
		regret=max_payoff-true_payoff
		return true_payoff, selected_item_feature, regret

	def update_user_feature(self, true_payoff, selected_item_feature, user_index):
		y=true_payoff
		x_matrix=np.zeros((self.dimension, self.user_num))
		x_matrix[:,user_index]=selected_item_feature
		co_x=np.dot(x_matrix, self.w.T).flatten('F')
		self.V+=np.outer(co_x,co_x)
		self.B+=y*co_x
		self.V_inv=np.linalg.pinv(self.V)
		self.user_f_vector=np.dot(self.V_inv, self.B)
		self.user_f_matrix=self.user_f_vector.reshape((self.user_num, self.dimension)).T
		self.co_user_f_matrix=np.dot(self.user_f_matrix, self.w)
		self.user_xx[user_index]+=np.outer(selected_item_feature, selected_item_feature)
		self.user_v[user_index]+=np.outer(selected_item_feature, selected_item_feature)
		self.user_bias[user_index]+=y*selected_item_feature
		xx_inv=np.linalg.pinv(self.user_xx[user_index])
		v_inv=np.linalg.pinv(self.user_v[user_index])
		if (self.user_counter[user_index]<10) or (np.linalg.norm(xx_inv)>2*np.linalg.norm(v_inv)):
			xx_inv=v_inv 
		else:
			pass
		self.user_ridge[:, user_index]=np.dot(v_inv, self.user_bias[user_index])
		self.user_ls[:, user_index]=np.dot(xx_inv, self.user_bias[user_index])

	def update_graph(self, user_index):
		# w_row=rbf_kernel(self.user_f_matrix.T[user_index].reshape(1,-1), self.user_f_matrix.T, gamma=0.5)
		# self.w[user_index]=w_row
		# self.w[:,user_index]=w_row
		self.w=rbf_kernel(self.user_f_matrix.T, gamma=0.5)
		self.lap=csgraph.laplacian(self.w, normed=True)
		graph_error=np.linalg.norm(self.w-self.true_adj)
		self.graph_error.extend([graph_error])

	def run(self, user_array, item_pool_array, iteration):
		self.initial()
		cumulative_regret=[0]
		learning_error_list=np.zeros(iteration)
		learning_error_list_2=np.zeros(iteration)
		for time in range(iteration):	
			print('time/iteration', time, iteration, '~~~CoLin')
			user_index=user_array[time]
			item_pool=item_pool_array[time]
			self.user_counter[user_index]+=1
			true_payoff, selected_item_feature, regret=self.select_item(item_pool, user_index, time)
			self.update_user_feature(true_payoff, selected_item_feature, user_index)
			self.update_graph(user_index)
			error=np.linalg.norm(self.co_user_f_matrix-self.true_user_feature_matrix.T)
			cumulative_regret.extend([cumulative_regret[-1]+regret])
			learning_error_list[time]=error 
		return np.array(cumulative_regret[1:]), learning_error_list, self.beta_list, self.graph_error





		