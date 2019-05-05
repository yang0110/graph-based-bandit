from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import Normalizer, MinMaxScaler
from scipy.sparse import csgraph 
import scipy
import os 
from community import community_louvain
from utils import *


class SPECTRAL_UCB():
	def __init__(self,dimension, user_num, item_num, pool_size,item_feature_matrix, true_user_feature_matrix, true_payoffs, lap, alpha, delta, sigma):
		self.dimension = dimension
		self.user_num = user_num
		self.item_num = item_num 
		self.pool_size = pool_size 
		self.true_item_f = item_feature_matrix
		self.true_payoffs = true_payoffs
		self.lap = lap 
		self.eig_values, self.eig_vectors = np.linalg.eig(self.lap)
		self.eig_matrix = np.diag(self.eig_values)
		self.alpha = alpha 
		self.delta = delta 
		self.sigma = sigma 
		self.beta = 0
		self.C = 0
		self.d = self.dimension
		self.item_cov = {}
		self.item_bais = {}
		self.item_feature_matrix = np.zeros((self.item_num, self.user_num))

	def initialize(self):
		self.true_item_feature_matrix = np.zeros((self.item_num, self.user_num))
		self.true_item_feature_matrix[:,:self.dimension] = self.true_item_f
		for u in range(self.item_num):
			self.item_cov[u] = self.eig_matrix+self.alpha*np.identity(self.user_num)
			self.item_bias[u] = np.zeros(self.user_num)
		uf = self.true_item_feature_matrix[0]
		self.C = np.sqrt(np.dot(np.dot(uf, self.item_cov[0]), uf))

	def update_beta(self, time):
		self.beta = 2*self.sigma*np.sqrt(self.d*np.log(1+time/self.alpha)+2*np.log(1/self.delta))+self.C

	def select_item(self, item_index):
		est_list=np.zeros(self.user_num)
		for user_index in range(self.user_num)
			x = self.eig_vectors[user_index]
			mean = np.dot(x, self.item_feature_matrix[item_index])
			cov_inv = np.linalg.pinv(self.item_cov[item_index])
			norm = np.sqrt(np.dot(np.dot(x, cov_inv), x))
			est = mean+self.beta*norm 
			est_list[user_index] = est
		selected_user = np.argmax(est_list)
		selected_user = self.eig_vectors[selected_user]
		payoff = true_payoffs[selected_user, item_index]
		best_payoff = np.max(true_payoffs[:,item_index])
		regret = best_payoff-payoff
		return selected_user, regret, payoff

	def update_user_feature(self, item_index, selected_user, payoff):
		self.item_cov[item_index] += np.outer(selected_user, selected_user)
		self.item_bias[item_index] += payoff*selected_user
		self.item_feature_matrix[item_index] += np.dot(np.linalg.pinv(self.item_cov[item_index]), self.item_bias[item_index])

	def run(self, iteration, item_seq):
		self.initialize()
		regret_list=np.zeros(iteration)
		for time in range(iteration):
			print('time/iteration', time, iteration, '~~~~ Spectral UCB')
			item_index = item_seq[time]
			self.update_beta(time)
			selected_user, regret, payoff= self.select_item(item_index)
			self.update_user_feature(item_index, select_user, payoff)
			regret_list[time]=regret

		return regret_list





