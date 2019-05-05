import pandas as pd  
import numpy as np
import json
import matplotlib.pyplot as plt
import os
os.chdir('/Users/KGYaNG/Documents/research/graph_bandit/code/')
input_path='../processed_data/movielens/'

rate_matrix=np.load(input_path+'rating_matrix_30_user_100_movies.npy')
rate_matrix=rate_matrix/np.max(rate_matrix)


def matrix_factorization(R, P, Q, K, steps=1000, alpha=0.002, beta=0.02):
	Q = Q.T
	for step in range(steps):
		for i in range(len(R)):
			for j in range(len(R[i])):
				if R[i][j] > 0:
					eij = R[i][j] - np.dot(P[i,:],Q[:,j])
					for k in range(K):
						P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
						Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
		eR = np.dot(P,Q)
		e = 0
		for i in range(len(R)):
			for j in range(len(R[i])):
				if R[i][j] > 0:
					e = e + pow(R[i][j] - np.dot(P[i,:],Q[:,j]), 2)
					for k in range(K):
						e = e + (beta/2) * ( pow(P[i][k],2) + pow(Q[k][j],2) )
						print('step', step)
						print('error', e)
			if e < 0.001:
				break
	return P, Q.T


N=rate_matrix.shape[0]
M=rate_matrix.shape[1]
K=5
P = np.random.rand(N,K)
Q = np.random.rand(M,K)
R=rate_matrix

nP, nQ = matrix_factorization(R, P, Q, K)
nR = np.dot(nP, nQ.T)

np.save(input_path+'user_feature_matrix_30.npy', nP)
np.save(input_path+'item_feature_matrix_100.npy', nQ)


