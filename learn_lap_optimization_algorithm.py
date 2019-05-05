import numpy as np 
from cvxopt import matrix, solvers
import networkx as nx
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import Normalizer, MinMaxScaler
from scipy.sparse import csgraph 
import scipy
import os 
from sklearn import datasets

def learn_lap_and_denoise_signal(signal_matrix, true_signal, L_true, max_iteration, alpha, beta):
    Y=signal_matrix ## m times n
    m=Y.shape[0]
    M_mat, P_mat, A_mat, b_mat, G_mat, h_mat=create_static_matrices_for_L_opt(m, beta)
    P_c=matrix(P_mat)
    A_c=matrix(A_mat)
    b_c=matrix(b_mat)
    G_c=matrix(G_mat)
    h_c=matrix(h_mat)
    q_mat=alpha*np.dot(np.dot(Y, Y.T).flatten(), M_mat)
    lap_error=[]
    signal_error=[]
    for i in range(max_iteration):
        print('i/iteration', i, max_iteration)
        q_c=matrix(q_mat)
        sol=solvers.qp(P_c, q_c, G_c, h_c, A_c, b_c)
        solvers.options['show_progress']=False
        l_vech=np.array(sol['x'])
        l_vec=np.dot(M_mat, l_vech)
        L=l_vec.reshape(m, m)
        assert np.allclose(L.trace(), m)
        assert np.all(L-np.diag(np.diag(L))<=0)
        assert np.allclose(np.dot(L, np.ones(m)), np.zeros(m))
        print('All constraints satisfied')
        Y=np.dot(np.linalh.onv(np.eye(m)+alpha*L), Y)
        q_mat=alpha*np.dot(np.ravel(np.dot(Y, Y.T)), M_mat)
        lap_error.extend([np.linalg.norm(L-L_true)])
        signal_error.extend([np.linalg.norm(Y-true_signal)])

    return L, Y, lap_error, signal_error




def learn_lap(user_feature_matrix, alpha, beta):
	Y=user_feature_matrix ## m times d 
	user_num=Y.shape[0]
	M_mat, P_mat, A_mat, b_mat, G_mat, h_mat=create_static_matrices_for_L_opt(user_num, beta)
	P_c=matrix(P_mat)
	A_c=matrix(A_mat)
	b_c=matrix(b_mat)
	G_c=matrix(G_mat)
	h_c=matrix(h_mat)
	q_mat=alpha*np.dot(np.dot(Y, Y.T).flatten(), M_mat)
	q_c=matrix(q_mat)
	sol=solvers.qp(P_c, q_c, G_c, h_c, A_c, b_c)
	solvers.options['show_progress']=False
	l_vech=np.array(sol['x'])
	l_vec=np.dot(M_mat, l_vech)
	L=l_vec.reshape(user_num, user_num)
	assert np.allclose(L.trace(), user_num)
	assert np.all(L-np.diag(np.diag(L))<=0)
	assert np.allclose(np.dot(L, np.ones(user_num)), np.zeros(user_num))
	print('All constraints satisfied')
	return L 

def create_static_matrices_for_L_opt(user_num, beta):
	M_mat=create_dup_matrix(user_num)
	P_mat=2*beta*np.dot(M_mat.T, M_mat)
	A_mat=create_A_mat(user_num)
	b_mat=create_b_mat(user_num)
	G_mat=create_G_mat(user_num)
	h_mat=np.zeros(G_mat.shape[0])
	return M_mat, P_mat, A_mat, b_mat, G_mat, h_mat 

def create_dup_matrix(n):
    M_mat = np.zeros((n**2, n*(n + 1)//2))
    for j in range(1, n+1):
        for i in range(j, n+1):
            u_vec = get_u_vec(i, j, n)
            Tij = get_T_mat(i, j, n)
            M_mat += np.outer(u_vec, Tij).T

    return M_mat

def create_A_mat(n):
    A_mat = np.zeros((n+1, n*(n+1)//2))
    for i in range(0, A_mat.shape[0] - 1):
        A_mat[i, :] = get_a_vec(i, n)
    A_mat[n, 0] = 1
    A_mat[n, np.cumsum(np.arange(n, 1, -1))] = 1
    return A_mat

def create_b_mat(n):
    b_mat = np.zeros(n+1)
    b_mat[n] = n
    return b_mat


def create_G_mat(n):
    G_mat = np.zeros((n*(n-1)//2, n*(n+1)//2))
    tmp_vec = np.cumsum(np.arange(n, 1, -1))
    tmp2_vec = np.append([0], tmp_vec)
    tmp3_vec = np.delete(np.arange(n*(n+1)//2), tmp2_vec)
    for i in range(G_mat.shape[0]):
        G_mat[i, tmp3_vec[i]] = 1

    return G_mat


def get_u_vec(i, j, n):
    u_vec = np.zeros(n*(n+1)//2)
    pos = (j-1) * n + i - j*(j-1)//2
    u_vec[pos-1] = 1
    return u_vec


def get_T_mat(i, j, n):
    Tij_mat = np.zeros((n, n))
    Tij_mat[i-1, j-1] = Tij_mat[j-1, i-1] = 1
    return np.ravel(Tij_mat)


def get_a_vec(i, n):
    a_vec = np.zeros(n*(n+1)//2)
    if i == 0:
        a_vec[np.arange(n)] = 1
    else:
        tmp_vec = np.arange(n-1, n-i-1, -1)
        tmp2_vec = np.append([i], tmp_vec)
        tmp3_vec = np.cumsum(tmp2_vec)
        a_vec[tmp3_vec] = 1
        end_pt = tmp3_vec[-1]
        a_vec[np.arange(end_pt, end_pt + n-i)] = 1

    return a_vec















