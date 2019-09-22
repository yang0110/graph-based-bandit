import numpy as np 
import networkx as nx
import seaborn as sns
sns.set_style("white")
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import Normalizer, MinMaxScaler
from scipy.sparse import csgraph
from scipy.linalg import sqrtm   
import scipy
import os 
os.chdir('/Kaige_Research/Code/graph_bandit/code/')
from sklearn import datasets
from utils import *
np.random.seed(2018)
path='../bandit_results/simulated/'

def moving_average(a, n=30):
	ret=np.cumsum(a, dtype=float)
	ret[n:]=ret[n:]-ret[:-n]
	return ret[n-1:]/n

iteration=500
user_num=10
item_num=1000
dimension=5
item_f=np.random.normal(size=(item_num, dimension))
item_f=Normalizer().fit_transform(item_f)
sigma=0.01
alpha=1
noise=np.random.normal(size=(user_num, item_num), scale=sigma)

#user_f=np.random.normal(size=(user_num, dimension))
#user_f=Normalizer().fit_transform(user_f)
adj=rbf_kernel(np.random.normal(size=(user_num, dimension)))
adj[adj<0]=0
np.fill_diagonal(adj,0)
D=np.diag(np.sum(adj, axis=1))
D_inv=np.sqrt(np.linalg.pinv(D))
lap=csgraph.laplacian(adj, normed=False)
lap=np.dot(np.linalg.inv(D), lap) 
lambda_list=np.linspace(0,50,5)
ratio_matrix=np.zeros((10, iteration-39))
for index, smooth in enumerate(lambda_list):
	user_f=dictionary_matrix_generator(user_num, dimension, lap, smooth)
	L=np.kron(lap+np.identity(user_num), np.identity(dimension))
	L_half_inv=sqrtm(np.linalg.inv(L))
	U=user_f.ravel()

	user_A={}
	user_V={}
	user_xn={}
	user_axn={}
	user_H={}
	M=np.identity(user_num*dimension)
	Cov=alpha*np.kron(lap, np.identity(dimension))
	Xn=np.zeros(user_num*dimension)
	for u in range(user_num):
		user_A[u]=0.01*np.identity(dimension)
		user_V[u]=alpha*np.identity(dimension)
		user_H[u]=np.zeros((dimension,dimension))
		user_xn[u]=np.zeros(dimension)
		user_axn[u]=np.zeros(dimension)

	random_user_list=np.random.choice(range(user_num), size=iteration)
	random_item_list=np.random.choice(range(item_num), size=iteration)

	# degrees=list(np.diag(lap))
	# user_range=list(range(user_num))
	# user_list=[x for _,x in sorted(zip(degrees, user_range))]
	# random_user_list=np.ones((user_num, int(iteration/user_num)))
	# for ind, u in enumerate(user_list):
	# 	random_user_list[ind]=random_user_list[ind]*u 
	# random_user_list=random_user_list.ravel().astype(int)

	theta_avg_list=[]
	beta1=[]
	beta2=[]
	beta3=[]
	R1=[0]
	R2=[0]
	R3=[0]
	r1_list=[]
	r2_list=[]
	r3_list=[]
	gob_bound=[]
	old_norm_list=[]
	old_noise_list=[]
	new_norm_list=[]
	new_noise_list=[]
	u_norm3_list=[]
	noise_norm3_list=[]
	x_norm1_list=[]
	x_norm2_list=[]
	x_long_norm_list=[]
	A_det_list=[]
	A_det2_list=[]
	M_det_list=[]
	eig_list1=[]
	eig_list2=[]
	for i in range(iteration):
		print('time=',i)
		item_index=random_item_list[i]
		user_index=random_user_list[i]
		x=item_f[item_index].copy()
		x_long=np.zeros(user_num*dimension)
		x_long[user_index*dimension:(user_index+1)*dimension]=x.copy()
		phi=np.dot(L_half_inv, x_long)
		M+=np.outer(x_long, x_long)
		#M+=np.outer(phi, phi)
		M_inv=np.linalg.pinv(M)
		Cov+=np.outer(x_long, x_long)
		Cov_inv=np.linalg.pinv(Cov)
		theta=user_f[user_index]
		user_A[user_index]+=np.outer(x, x)
		user_V[user_index]+=np.outer(x, x)
		V_inv=np.linalg.inv(user_V[user_index])
		A_inv=np.linalg.inv(user_A[user_index])
		n=noise[user_index, item_index]
		user_xn[user_index]+=np.dot(x,n)
		user_axn[user_index]=np.dot(V_inv, user_xn[user_index])
		Xn+=np.dot(x_long, n)
		sum_A=np.zeros((dimension, dimension))
		sum_axn=np.zeros(dimension)
		for uu in range(user_num):
			sum_A+=((lap[user_index,uu])**2)*np.linalg.inv(user_A[uu])
			sum_axn+=(lap[user_index,uu])*user_axn[uu]
		A_inv_2=np.dot(A_inv, A_inv)
		H=user_V[user_index]+alpha*np.identity(dimension)+alpha**2*sum_A
		H_inv=np.linalg.pinv(H)
		xn=user_xn[user_index]
		old_norm=alpha*np.sqrt(np.dot(np.dot(theta, V_inv),theta))
		old_noise=np.sqrt(np.dot(np.dot(xn, V_inv), xn))
		x_norm1=np.sqrt(np.dot(np.dot(x, V_inv), x))
		#old_noise=np.sqrt(np.log(np.linalg.det(user_V[user_index])**(1/2)/0.1))
		b1=old_norm+old_noise
		theta_avg=np.dot(user_f.T, lap[user_index])
		theta_avg_list.extend([np.linalg.norm(theta_avg)])
		new_norm=alpha*np.sqrt(np.dot(np.dot(theta_avg, V_inv),theta_avg))
		new_noise=np.sqrt(np.dot(np.dot(xn-alpha*sum_axn, V_inv), xn-alpha*sum_axn))
		x_norm2=np.sqrt(np.dot(np.dot(x, H_inv),x))
		#new_noise=np.sqrt(np.log(np.linalg.det(np.linalg.pinv(np.dot(H, A_inv_2)))**(1/2)/0.1))
		if new_noise>0:
			pass
		else:
			new_noise=0
		b2=new_norm+new_noise
		U_til_norm=np.sqrt(np.dot(np.dot(U, M_inv),U))
		noise_norm3=alpha*np.sqrt(np.dot(np.dot(Xn, M_inv), Xn))
		x_long_norm=np.sqrt(np.dot(np.dot(phi, M_inv), phi))
		#noise_norm3=np.sqrt(np.log(np.linalg.det(M)**(1/2)/0.1))
		u_norm3=U_til_norm
		b3=u_norm3+noise_norm3
		beta1.extend([b1])
		beta2.extend([b2])
		beta3.extend([b3])
		old_norm_list.extend([old_norm])
		old_noise_list.extend([old_noise])
		new_norm_list.extend([new_norm])
		new_noise_list.extend([new_noise])
		u_norm3_list.extend([u_norm3])
		noise_norm3_list.extend([noise_norm3])
		eigs,_=np.linalg.eig(V_inv)
		eigs.sort()
		eig_list1.extend([eigs[-1]])
		eigs,_=np.linalg.eig(H_inv)
		eigs.sort()
		eig_list2.extend([eigs[-1]])
		x_norm1_list.extend([x_norm1])
		x_norm2_list.extend([x_norm2])
		x_long_norm_list.extend([x_long_norm])

		A_det=2*np.log(np.linalg.det(user_V[user_index])/np.linalg.det(alpha*np.identity(dimension)))
		A_det2=2*np.log(np.linalg.det(H)/np.linalg.det(H-user_A[user_index]))
		M_det=np.log(np.linalg.det(M))

		A_det_list.extend([A_det])
		A_det2_list.extend([A_det2])
		M_det_list.extend([M_det])

		r1=2*b1*x_norm1
		r2=2*b2*x_norm2
		r3=2*b3*x_long_norm

		# r1=2*b1*A_det
		# r2=2*b2*A_det2
		# r3=2*b3*M_det

		r1_list.extend([r1])
		r2_list.extend([r2])
		r3_list.extend([r3])

		R1.extend([R1[-1]+r1])
		R2.extend([R2[-1]+r2])
		R3.extend([R3[-1]+r3])
	ratio=np.array(x_norm2_list[10:])/np.array(x_norm1_list[10:])
	moving=moving_average(ratio)
	ratio_matrix[index]=moving

# plt.figure()
# plt.plot(old_norm_list[10:], '.-', label='u_norm LinUCB')
# plt.plot(u_norm3_list[10:], '.-', label='u_norm GOB')
# plt.plot(new_norm_list[10:], '.-', label='u_norm G-UCB')
# plt.legend(loc=0, fontsize=12)
# plt.show()


# plt.figure(figsize=(6,4))
# plt.plot(np.array(old_noise_list)[list(np.where(random_user_list==2)[0])][:120], '-', label='RHS')
# #plt.plot(noise_norm3_list[10:], '.-', label='noise norm GOB')
# plt.plot(np.array(new_noise_list)[list(np.where(random_user_list==2)[0])][:120], '.-', label='LHS')
# plt.xlabel('Time', fontsize=14)
# plt.ylabel('Noise Norm', fontsize=14)
# plt.legend(loc=0, fontsize=14)
# plt.tight_layout()
# plt.savefig(path+'noise_norm_approximation'+'.png', dpi=100)
# plt.show()

# plt.figure()
# plt.plot(beta1[10:], label='beta LinUCB')
# plt.plot(beta3[10:], label='beta GOB')
# plt.plot(beta2[10:], label='beta G-UCB')
# plt.legend(loc=0, fontsize=12)
# plt.show()


# plt.figure()
# plt.plot(x_norm1_list[10:], label='x_norm LinUCB')
# #plt.plot(x_long_norm_list[10:], label='x_norm GOB')
# plt.plot(x_norm2_list[10:], label='x_norm G-UCB')
# plt.legend(loc=0, fontsize=12)
# plt.show()

plt.figure(figsize=(5,5))
for ind, smooth in enumerate(lambda_list):
	plt.plot(ratio_matrix[ind], label='Smoothness= '+np.str(np.round(smooth, decimals=2)))
plt.legend(loc=0, fontsize=14)
plt.ylabel('Psi', fontsize=16)
plt.xlabel('Time', fontsize=16)
plt.tight_layout()
plt.savefig(path+'Psi_smooth'+'.png', dpi=100)
plt.show()

# plt.figure()
# plt.plot(r1_list[10:], label='r_t LinUCB')
# plt.plot(r3_list[10:], label='r_t GOB')
# plt.plot(r2_list[10:], label='r_t G-UCB')
# plt.legend(loc=0, fontsize=12)
# plt.show()

# plt.figure()
# plt.plot(R1[10:], label='R_T LinUCB')
# #plt.plot(R3[10:], label='R_T GOB')
# plt.plot(R2[10:], label='R_T G-UCB')
# plt.legend(loc=0, fontsize=12)
# plt.show()

# plt.figure()
# plt.plot(A_det_list[10:], label='A det LinUCB')
# plt.plot(np.array(M_det_list[10:])/user_num, label='M det GOB')
# plt.plot(A_det2_list[10:], label='A det2 G-UCB')
# plt.legend(loc=0, fontsize=12)
# plt.show()

# plt.figure()
# plt.plot(eig_list1[10:], label='eig LinUCB')
# plt.plot(eig_list2[10:], label='eig G-UCB')
# plt.legend(loc=0, fontsize=12)
# plt.show()

# plt.figure()
# plt.plot(np.diag(lap), label='node degree')
# plt.legend(loc=0, fontsize=12)
# plt.show()

# plt.figure()
# plt.plot(theta_avg_list[10:], label='theta_avg_list')
# plt.legend(loc=0, fontsize=12)
# plt.show()
