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

iteration=1000
user_num=20
item_num=1000
dimension=5
item_f=np.random.normal(size=(item_num, dimension))
item_f=Normalizer().fit_transform(item_f)
sigma=0.1
alpha=1
noise=np.random.normal(size=(user_num, item_num), scale=sigma)

#user_f=np.random.normal(size=(user_num, dimension))
#user_f=Normalizer().fit_transform(user_f)
adj=rbf_kernel(np.random.normal(size=(user_num, dimension)))
adj[adj<0.5]=0
#np.fill_diagonal(adj,0)
lap=csgraph.laplacian(adj, normed=True)
normed_lap=normalized_trace(lap, user_num)
print('trace(lap)=', np.trace(lap))
print('trace(lap)=', np.trace(normed_lap))
user_f=dictionary_matrix_generator(user_num, dimension, lap, 2)
L=np.kron(lap+np.identity(user_num), np.identity(dimension))
L_half_inv=sqrtm(np.linalg.inv(L))
U=user_f.ravel()

user_A={}
user_V={}
user_xn={}
user_axn={}
user_H={}
M=np.identity(user_num*dimension)
Xn=np.zeros(user_num*dimension)
for u in range(user_num):
	user_A[u]=0.1*np.identity(dimension)
	user_V[u]=alpha*np.identity(dimension)
	user_H[u]=np.zeros((dimension,dimension))
	user_xn[u]=np.zeros(dimension)
	user_axn[u]=np.zeros(dimension)

random_user_list=np.random.choice(range(user_num), size=iteration)
random_item_list=np.random.choice(range(item_num), size=iteration)

# random_user_list=np.ones((user_num, int(iteration/user_num)))
# for u in range(user_num):
# 	random_user_list[u]=random_user_list[u]*u 
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
for i in range(iteration):
	print('time=',i)
	item_index=random_item_list[i]
	user_index=random_user_list[i]
	x=item_f[item_index].copy()
	x_long=np.zeros(user_num*dimension)
	x_long[user_index*dimension:(user_index+1)*dimension]=x.copy()
	phi=np.dot(L_half_inv, x_long)
	M+=np.outer(phi, phi)
	#M+=np.outer(x_long, x_long)
	M_inv=np.linalg.pinv(M)
	theta=user_f[user_index]
	user_A[user_index]+=np.outer(x, x)
	user_V[user_index]+=np.outer(x, x)
	V_inv=np.linalg.inv(user_V[user_index])
	A_inv=np.linalg.inv(user_A[user_index])
	n=noise[user_index, item_index]
	user_xn[user_index]+=np.dot(x,n)
	user_axn[user_index]=np.dot(A_inv, user_xn[user_index])
	Xn+=np.dot(x_long, n)
	sum_A=np.zeros((dimension, dimension))
	sum_axn=np.zeros(dimension)
	for uu in range(user_num):
		sum_A+=((lap[user_index,uu])**2)*np.linalg.inv(user_A[u])
		sum_axn+=(lap[user_index,uu])*user_axn[uu]
	A_inv_2=np.dot(A_inv, A_inv)
	H=user_V[user_index]+alpha*np.identity(dimension)+alpha**2*sum_A
	H_inv=np.linalg.pinv(H)
	xn=user_xn[user_index]
	old_norm=alpha*np.sqrt(np.dot(np.dot(theta, V_inv),theta))
	old_noise=np.sqrt(np.dot(np.dot(xn, V_inv), xn))
	#old_noise=np.sqrt(np.log(np.linalg.det(user_V[user_index])**(1/2)/0.1))
	b1=old_norm+old_noise
	theta_avg=np.dot(user_f.T, lap[user_index])
	theta_avg_list.extend([np.linalg.norm(theta_avg)])
	new_norm=alpha*np.sqrt(np.dot(np.dot(theta_avg, H_inv),theta_avg))
	new_noise=np.sqrt(np.dot(np.dot(xn-alpha*sum_axn, H_inv),xn-alpha*sum_axn))
	#new_noise=np.sqrt(np.log(np.linalg.det(np.linalg.pinv(np.dot(H, A_inv_2)))**(1/2)/0.1))
	if new_noise>0:
		pass
	else:
		new_noise=0
	b2=new_norm+new_noise
	U_til_norm=np.sqrt(np.dot(np.dot(U, M_inv),U))
	noise_norm3=alpha*np.sqrt(np.dot(np.dot(Xn, M_inv), Xn))
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
	x_norm1=np.sqrt(np.dot(np.dot(x, V_inv), x))
	A_det=2*np.log(np.linalg.det(user_V[user_index])/np.linalg.det(alpha*np.identity(dimension)))
	A_det2=2*np.log(np.linalg.det(H)/np.linalg.det(2*alpha*np.identity(dimension)+alpha**2*sum_A))
	A_det_list.extend([A_det])
	A_det2_list.extend([A_det2])
	x_norm2=np.sqrt(np.dot(np.dot(x, H_inv),x))
	x_long_norm=np.sqrt(np.dot(np.dot(phi, M_inv), phi))
	M_det=np.log(np.linalg.det(M)/np.linalg.det(alpha*L))
	M_det_list.extend([M_det])
	r1=2*b1*x_norm1
	r2=2*b2*x_norm2
	r3=2*b3*x_long_norm
	r1_list.extend([r1])
	r2_list.extend([r2])
	r3_list.extend([r3])
	R1.extend([R1[-1]+r1])
	R2.extend([R2[-1]+r2])
	R3.extend([R3[-1]+r3])
	x_norm1_list.extend([x_norm1])
	x_norm2_list.extend([x_norm2])
	x_long_norm_list.extend([x_long_norm])


plt.figure()
plt.plot(beta1[10:], label='beta LinUCB')
plt.plot(beta3[10:], label='beta GOB')
plt.plot(beta2[10:], label='beta G-UCB')
plt.legend(loc=0, fontsize=12)
plt.show()


plt.figure()
plt.plot(x_norm1_list[10:], label='x_norm LinUCB')
plt.plot(x_long_norm_list[10:], label='x_norm GOB')
plt.plot(x_norm2_list[10:], label='x_norm G-UCB')
plt.legend(loc=0, fontsize=12)
plt.show()

plt.figure()
plt.plot(A_det_list[10:], label='A det LinUCB')
plt.plot(np.array(M_det_list[10:]), label='M det GOB')
plt.plot(A_det2_list[10:], label='A det2 G-UCB')
plt.legend(loc=0, fontsize=12)
plt.show()

plt.figure()
plt.plot(r1_list[10:], label='r_t LinUCB')
plt.plot(r3_list[10:], label='r_t GOB')
plt.plot(r2_list[10:], label='r_t G-UCB')
plt.legend(loc=0, fontsize=12)
plt.show()

plt.figure()
plt.plot(R1[10:], label='R_T LinUCB')
plt.plot(R3[10:], label='R_T GOB')
plt.plot(R2[10:], label='R_T G-UCB')
plt.legend(loc=0, fontsize=12)
plt.show()


plt.figure()
plt.plot(old_norm_list[10:], '.-', label='u_norm LinUCB')
plt.plot(u_norm3_list[10:], '.-', label='u_norm GOB')
plt.plot(new_norm_list[10:], '.-', label='u_norm G-UCB')
plt.legend(loc=0, fontsize=12)
plt.show()

plt.figure()
plt.plot(old_noise_list[10:], '.', label='noise norm LinUCB')
plt.plot(noise_norm3_list[10:], '.', label='noise norm GOB')
plt.plot(new_noise_list[10:], '.', label='noise norm G-UCB')
plt.legend(loc=0, fontsize=12)
plt.show()

plt.figure()
plt.plot(theta_avg_list[10:], label='theta_avg_list')
plt.legend(loc=0, fontsize=12)
plt.show()
