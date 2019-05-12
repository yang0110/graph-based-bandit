import pandas as pd  
import numpy as np
import json
import matplotlib.pyplot as plt
import os
import gc
os.chdir('C:/Kaige_Research/Code/graph_bandit/code/')
input_path='../original_data/netflix/'
output_path='../processed_data/netflix/'

files = [
    input_path+'combined_data_2.txt',
]

coo_row = []
coo_col = []
coo_val = []

for file_name in files:
    print('processing {0}'.format(file_name))
    with open(file_name, "r") as f:
        movie = -1
        number=0
        for line in f:
        	number+=1
            if line.endswith(':\n'):
                movie = int(line[:-2]) - 1
                continue
            assert movie >= 0
            splitted = line.split(',')
            user = int(splitted[0])
            rating = float(splitted[1])
            coo_row.append(user)
            coo_col.append(movie)
            coo_val.append(rating)
            if number>=2000000:
            	break 
    gc.collect()

print('transformation...')

rating=pd.DataFrame(columns=['userID', 'movieID', 'rate'])
rating['userID']=coo_row
rating['movieID']=coo_col
rating['rate']=coo_val


user_num=len(np.unique(coo_row))
movie_num=len(np.unique(coo_col))

print('user_num', len(np.unique(coo_row)))
print('movie_num', len(np.unique(coo_col)))

multiple_rated=rating.groupby(['userID', 'movieID']).size().sort_values(ascending=False)
m_rated=multiple_rated.reset_index()
columns=['userID', 'movieID', 'rate_times']
m_rated.columns=columns

most_rated=rating.groupby('movieID').size().sort_values(ascending=False)

most_review=rating.groupby('userID').size().sort_values(ascending=False)

top_100_user=most_review.index[:100].values
top_500_movie=most_rated.index
top_100_movie=most_rated.index[:100].values
top_30_user=most_review.index[:30].values 

np.save(output_path+'top_30_user.npy', top_30_user)
np.save(output_path+'top_100_user.npy', top_100_user)
np.save(output_path+'top_500_movie.npy', top_500_movie)
np.save(output_path+'top_100_movie.npy', top_100_movie)

matrix_30=np.zeros((30, movie_num))
matrix_100=np.zeros((100, movie_num))
matrix_30_100=np.zeros((30, 100))

user_id=coo_row
movie_id=coo_col
ratings=coo_val


top_30_user_id=[]
top_30_movie_id=[]
top_30_ratings=[]
counter=0
for user in top_30_user:
    counter+=1
    print(counter)
    list_=list(np.where(user_id==user)[0])
    top_30_user_id.extend(list(np.array(user_id)[list_]))
    top_30_movie_id.extend(list(np.array(movie_id)[list_]))
    top_30_ratings.extend(list(np.array(ratings)[list_]))


top_100_user_id=[]
top_100_movie_id=[]
top_100_ratings=[]
counter=0
for user in top_100_user:
    counter+=1
    print(counter)
    list_=list(np.where(user_id==user)[0])
    top_100_user_id.extend(list(np.array(user_id)[list_]))
    top_100_movie_id.extend(list(np.array(movie_id)[list_]))
    top_100_ratings.extend(list(np.array(ratings)[list_]))



for index, user in enumerate(top_30_user_id):
    print(index, len(top_30_user_id))
    rate=top_30_ratings[index]
    if user in top_30_user:
        user_index=list(np.where(top_30_user==user)[0])[0]
        movie=top_30_movie_id[index]
        if movie in top_100_movie:
            movie_index=list(np.where(top_100_movie==movie)[0])[0]
            matrix_30_100[user_index, movie_index]=rate 
        else:
            pass 
    else:
        pass 

np.save(output_path+'rating_matrix_30_user_100_movies.npy', matrix_30_100)


for index, user in enumerate(top_30_user_id):
    print(index, len(top_30_user_id))
    rate=top_30_ratings[index]
    if user in top_30_user:
        user_index=list(np.where(top_30_user==user)[0])[0]
        movie=top_30_movie_id[index]
        if movie in top_500_movie:
            movie_index=list(np.where(top_500_movie==movie)[0])[0]
            matrix_30[user_index, movie_index]=rate 
        else:
            pass 
    else:
        pass 

np.save(output_path+'rating_matrix_30_user_500_movies.npy', matrix_30)



for index, user in enumerate(top_100_user_id):
    print(index, len(top_100_user_id))
    rate=top_100_ratings[index]
    if user in top_100_user:
        user_index=list(np.where(top_100_user==user)[0])[0]
        movie=top_100_movie_id[index]
        if movie in top_500_movie:
            movie_index=list(np.where(top_500_movie==movie)[0])[0]
            matrix_100[user_index, movie_index]=rate 
        else:
            pass 
    else:
        pass 

np.save(output_path+'rating_matrix_100_user_500_movies.npy', matrix_100)











