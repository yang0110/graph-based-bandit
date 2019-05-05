import pandas as pd  
import numpy as np
import json
import matplotlib.pyplot as plt
import os
os.chdir('C:/Kaige_Research/Code/graph_bandit/data_process_code/')
input_path='../original_data/movielens-20m-dataset/'
output_path='../original_data/movielens-20m-dataset/processed_data/'
print(os.listdir(input_path))

user_movie_dict=np.load(output_path+"user_movie_dict.npy").item()
user_rating_dict=np.load(output_path+"user_rating_dict.npy").item()

user_freq_dict=np.load(output_path+"user_freq_dict.npy").item()
freq_list=np.load(output_path+"freq_list.npy")

top_500_user_id=np.load(output_path+'top_500_user_id.npy')
top_500_user_freq=np.load(output_path+'top_500_user_freq.npy')
top_500_user_film=np.load(output_path+'top_500_user_film_id.npy')

df_movie_describe_numeric_small=pd.read_csv(output_path+'df_movie_describe_numeric_small',index_col=False)

X=df_movie_describe_numeric_small.iloc[:,2:].values

film_num=X.shape[0]
user_num=len(freq_list)


movie_list=[]
for u in user_movie_dict.keys():
	movie_list.extend(user_movie_dict[u])

np.save(output_path+'user_history_combine', movie_list)

from collections import Counter 
counted=Counter(movie_list)
film_freq_id=list(counted.keys())
film_freq=list(counted.values())

np.save(output_path+'film_freq', film_freq)
np.save(output_path+'film_freq_id', film_freq_id)


pop_film=list(np.array(film_freq_id)[np.where(np.array(film_freq)>100)])
print(len(pop_film))

np.save(output_path+'popupar_film_list_500', pop_film)


user_freq_dict=np.load(output_path+"user_freq_dict.npy").item()
user_freq_list=np.zeros(len(user_freq_dict.keys()))
for user in list(user_freq_dict.keys()):
	user_freq_list[user]=user_freq_dict[user]

pop_user_list=list(np.where(np.array(user_freq_list)>=800)[0])
print(len(pop_user_list))
np.save(output_path+'popular_user_id_30', pop_user_list)

pop_user=np.load(output_path+'popular_user_id_30.npy')
user_freq_list[pop_user]

rata_matrix=np.zeros((len(pop_user), len(pop_film)))
for user_index, user in enumerate(pop_user):
	for film_index, film in enumerate(pop_film):
		if film in user_movie_dict[user]:
			index=np.where(np.array(user_movie_dict[user])==film)[0][0]
			rata_matrix[user_index, film_index]=user_rating_dict[user][index]
		else: 
			rata_matrix[user_index, film_index]=np.nan

columns=['user_id']+['movie_%s'%s for s in pop_film]
df=pd.DataFrame(columns=columns)
df['user_id']=pop_user
df[['movie_%s'%s for s in pop_film]]=rata_matrix
df=df.fillna(df.mean())

df.to_csv(output_path+'popupar_user_popular_film_rating_matrix_30_500')

np.save(output_path+'popupar_user_popular_film_rating_matrix_30_500', rata_matrix)
film_df=pd.read_csv(output_path+'df_movie_describe_numeric_small',index_col=False)

pop_film_feature_matrix=film_df.loc[film_df['movieId'].isin(pop_film)]
np.save(output_path+'popular_film_feature_matrix_500', pop_film_feature_matrix)

columns=['movieId']+['f_%s'%(s+1) for s in range(pop_film_feature_matrix.shape[1])]
df_popular_film=pd.DataFrame(columns=columns)
df_popular_film['movieId']=pop_film
df_popular_film[['f_%s'%(s+1) for s in range(pop_film_feature_matrix.shape[1])]]=pop_film_feature_matrix
df_popular_film=df_popular_film.fillna(0)

df_popular_film.to_csv(output_path+'df_popular_film_matrix_500')

