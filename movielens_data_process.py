import pandas as pd  
import numpy as np
import json
import matplotlib.pyplot as plt
import os
os.chdir('/Users/KGYaNG/Documents/research/graph_bandit/code/')
input_path='../original_data/movielens/'
output_path='../processed_data/movielens/'
print(os.listdir(input_path))

rating=pd.read_csv(input_path+'user_ratedmovies-timestamps.dat', delimiter="\t")
user_id=rating['userID'].values 
movie_id=rating['movieID'].values 
ratings=rating['rating'].values 

unique_user_id=np.unique(user_id)
user_num=len(np.unique(user_id))

unique_movie_id=np.unique(movie_id)
movie_num=len(unique_movie_id)

most_rated=rating.groupby('movieID').size().sort_values(ascending=False)

most_review=rating.groupby('userID').size().sort_values(ascending=False)

top_100_user=most_review.index[:100].values
top_500_movie=most_rated.index[:500].values
top_100_movie=most_rated.index[:100].values
top_30_user=most_review.index[:30].values 

np.save(output_path+'top_30_user.npy', top_30_user)
np.save(output_path+'top_100_user.npy', top_100_user)
np.save(output_path+'top_500_movie.npy', top_500_movie)
np.save(output_path+'top_100_movie.npy', top_100_movie)

matrix_30=np.zeros((30, 500))
matrix_100=np.zeros((100, 500))
matrix_30_100=np.zeros((30, 100))


for index, user in enumerate(user_id):
	print(index, len(user_id))
	rate=ratings[index]
	if user in top_30_user:
		user_index=list(np.where(top_30_user==user)[0])[0]
		movie=movie_id[index]
		if movie in top_500_movie:
			movie_index=list(np.where(top_500_movie==movie)[0])[0]
			matrix_30[user_index, movie_index]=rate 
		else:
			pass 
	else:
		pass 

np.save(output_path+'rating_matrix_30_user_500_movies.npy', matrix_30)



for index, user in enumerate(user_id):
	print(index, len(user_id))
	rate=ratings[index]
	if user in top_100_user:
		user_index=list(np.where(top_100_user==user)[0])[0]
		movie=movie_id[index]
		if movie in top_500_movie:
			movie_index=list(np.where(top_500_movie==movie)[0])[0]
			matrix_100[user_index, movie_index]=rate 
		else:
			pass 
	else:
		pass 

np.save(output_path+'rating_matrix_100_user_500_movies.npy', matrix_100)


for index, user in enumerate(user_id):
	print(index, len(user_id))
	rate=ratings[index]
	if user in top_30_user:
		user_index=list(np.where(top_30_user==user)[0])[0]
		movie=movie_id[index]
		if movie in top_100_movie:
			movie_index=list(np.where(top_100_movie==movie)[0])[0]
			matrix_30_100[user_index, movie_index]=rate 
		else:
			pass 
	else:
		pass 

np.save(output_path+'rating_matrix_30_user_100_movies.npy', matrix_30_100)






