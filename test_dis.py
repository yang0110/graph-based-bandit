import numpy as np 
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import Normalizer, MinMaxScaler
from scipy.sparse import csgraph 
import scipy
import os 
os.chdir('C:/Kaige_Research/Code/graph_bandit/code/')
from utils import *
path='../results/regression/'

user_num=20
dimension=5
item_num=100
iteration=1000
alpha=0.1
sigma=0.1
item_f=np.random.normal(size=(item_num, dimension))
item_feature_matrix=Normalizer().fit_transform(item_f)
adj=RBF_graph(user_num, dimension, thres=0.5)
lap=csgraph.laplacian(adj, normed=False)
A=np.kron(lap, np.identity(dimension))
user_feature_matrix=dictionary_matrix_generator(user_num, dimension, lap, 5)
user_feature_vector=user_feature_matrix.flatten()
true_payoffs=np.dot(user_feature_matrix, item_feature_matrix.T)+np.random.normal(scale=sigma, size=(user_num, item_num))

user_seq=np.random.choice(range(user_num), size=iteration)
item_seq=np.random.choice(range(item_num), size=iteration)

cov=alpha*(A+np.identity(user_num*dimension))
bias=np.zeros(user_num*dimension)
user_cov={}
user_bias={}
user_neighbor={}
user_neighbor_index={}
neighbor_cov=0
neighbor_bias=0
bound1={}
bound2={}
bound=np.zeros(iteration)
bound4=np.zeros(iteration)
for u in range(user_num):
	print('initial u', u)
	adj_row=adj[u]
	neighbors=np.where(adj_row>0)[0].tolist()
	user_neighbor[u]=neighbors
	user_neighbor_index[u]=np.where(np.array(neighbors)==u)[0].tolist()[0]
	user_cov[u]=alpha*np.identity(dimension)
	user_bias[u]=np.zeros(dimension)
	bound1[u]=[]
	bound2[u]=[]



f_matrix=np.zeros((user_num, dimension))
f_matrix2=np.zeros((user_num, dimension))
f_matrix3=np.zeros((user_num, dimension))
f_matrix4=np.zeros((user_num, dimension))
f_vector=np.zeros(user_num*dimension)
elist1=[]
elist2=[]
elist3=[]
elist4=[]
elist5=[]
main=np.zeros(iteration)
side=np.zeros(iteration)
for i in range(iteration):
	print('time/iteration', i, iteration)
	user_index=user_seq[i]
	item_index=item_seq[i]
	x=item_feature_matrix[item_index]
	y=true_payoffs[user_index, item_index]
	x_long=np.zeros(user_num*dimension)
	x_long[user_index*dimension:(user_index+1)*dimension]=x
	cov+=np.outer(x_long, x_long)
	bias+=y*x_long
	cov_inv=np.linalg.pinv(cov)
	f_vector=np.dot(cov_inv, bias)
	delta=f_vector-user_feature_vector
	bound[i]=np.dot(np.dot(delta, cov), delta)
	#f_vector[user_index*dimension:(user_index+1)*dimension]=np.dot(cov_inv, bias)[user_index*dimension:(user_index+1)*dimension]
	user_cov[user_index]+=np.outer(x,x)
	user_bias[user_index]+=y*x 
	f_matrix[user_index]=np.dot(np.linalg.pinv(user_cov[user_index]), user_bias[user_index])
	delta1=f_matrix[user_index]-user_feature_matrix[user_index]
	bound1[user_index].extend([np.dot(np.dot(delta1, user_cov[user_index]), delta1)])

	neighbors=user_neighbor[user_index].copy()
	index=user_neighbor_index[user_index]
	neighbor_array=np.zeros((len(neighbors), dimension))
	for j,n in enumerate(neighbors):
		neighbor_array[j]=np.arange(n*dimension,(n+1)*dimension)

	neighbor_array=neighbor_array.flatten().astype(int)
	neighbor_cov=cov[neighbor_array][:, neighbor_array].copy()
	neighbor_bias=bias[neighbor_array].copy()
	f_matrix3[user_index]=np.dot(np.linalg.pinv(neighbor_cov), neighbor_bias).reshape((len(neighbors), dimension))[index]
	#f_matrix3[neighbors]=np.dot(np.linalg.pinv(neighbor_cov), neighbor_bias).reshape((len(neighbors), dimension))
	cov_inv_slice=cov_inv[user_index*dimension:(user_index+1)*dimension].copy()
	cov_slice=cov[user_index*dimension:(user_index+1)*dimension].copy()
	f_matrix2[user_index]=np.zeros(dimension)
	M=np.zeros((dimension, dimension))
	for nei in range(user_num):
		m=cov_slice[:,nei*dimension:(nei+1)*dimension].copy()
		M+=m
		c=cov_inv_slice[:,nei*dimension:(nei+1)*dimension].copy()
		b=bias[nei*dimension:(nei+1)*dimension].copy()
		if nei==user_index:
			main[i]=np.linalg.norm(np.dot(c,b))
		else:
			side[i]+=np.linalg.norm(np.dot(c,b))
		f_matrix2[user_index]+=np.dot(c,b)
	delta2=f_matrix2[user_index]-user_feature_matrix[user_index]
	bound2[user_index].extend([np.dot(np.dot(delta2, M), delta2)])
	for us in range(user_num):
		if us==user_index:
			pass 
		else:
			f_matrix2[us]+=np.dot(cov_inv[us*dimension:(us+1)*dimension, user_index*dimension:(user_index+1)*dimension].copy(), y*x)
	f_matrix4[user_index]=np.dot(cov_inv_slice, bias)
	e1=np.linalg.norm(f_matrix-user_feature_matrix)
	e2=np.linalg.norm(f_vector-user_feature_matrix.flatten())
	e3=np.linalg.norm(f_matrix2-user_feature_matrix)
	e4=np.linalg.norm(f_matrix3-user_feature_matrix)
	e5=np.linalg.norm(f_matrix4-user_feature_matrix)
	elist1.extend([e1])
	elist2.extend([e2])
	elist3.extend([e3])
	elist4.extend([e4])
	elist5.extend([e5])



plt.figure()
plt.plot(elist1, label='ind')
plt.plot(elist2, label='central')
plt.plot(elist3, label='dist')
plt.plot(elist4, label='neighbors')
plt.plot(elist5, label='central-ind')
plt.legend(loc=0)
plt.tight_layout()
plt.savefig(path+'error'+'.png', dpi=200)
plt.show()

total=main+side
plt.figure()
plt.plot(total, label='final')
plt.plot(main, label='posterior')
plt.plot(side, label='prior')
plt.legend(loc=0)
plt.tight_layout()
plt.savefig(path+'prior_and_posterior'+'.png', dpi=200)
plt.show()

plt.figure()
#plt.plot(bound, label='central')
plt.plot(bound1[0][20:], label='ind')
plt.plot(bound2[0][20:], label='half-central')
plt.legend(loc=0)
plt.tight_layout()
plt.savefig(path+'bound'+'.png', dpi=200)
plt.show()


