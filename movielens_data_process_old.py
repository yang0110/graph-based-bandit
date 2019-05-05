import pandas as pd  
import numpy as np
import json
import matplotlib.pyplot as plt
import os
#os.chdir('~/Documents/research/graph_bandit/code/')
input_path='../original_data/movielens-20m-dataset/'
output_path='../processed_data/movielens/'
print(os.listdir(input_path))


def remove(sentence):
	a=sentence.split(' ')
	b=[]
	for i in a:
		b.extend(i.split('|'))
	c=[]
	for i in b:
		c.extend(i.split('('))

	d=[]
	for i in c:
		d.extend(i.split(')'))
	return d


movie=pd.read_csv(input_path+'movie.csv')
movie.columns
movie=movie.loc[:,['movieId', 'title']]
movie.head(10)

movie_id=movie['movieId'].values
movie_title=movie['title'].values

np.save(output_path+'movie_id', movie_id)
np.save(output_path+'movie_title', movie_title)

user_id=[]
user_movie_id=[]
user_rating=[]

chunksize = 10**5
for chunk in pd.read_csv(input_path+'rating.csv', chunksize=chunksize):
	user_id.extend(list(chunk['userId'].values))
	user_movie_id.extend(list(chunk['movieId'].values))
	user_rating.extend(list(chunk['rating'].values))

np.save(output_path+'user_id', user_id)
np.save(output_path+'user_movie_id', user_movie_id)
np.save(output_path+'user_rating', user_rating)

unique_user_id=[]
size=100000
number=int(len(user_id)/size)+1
for i in range(number):
	print(i, number)
	unique_user_id.extend(list(np.unique(user_id[i*size:(i+1)*size])))

unique_user_id=np.unique(unique_user_id)
np.save(output_path+'unique_user_id', unique_user_id)


user_num=len(unique_user_id)
user_movie_dict={}
user_rating_dict={}

for user in unique_user_id:
	print(user, user_num)
	user_movie_dict[user]=[]
	user_rating_dict[user]=[]


for i in range(int(len(user_id)/100)):
	print(i)
	user=user_id[i]
	m=user_movie_id[i]
	r=user_rating[i]
	user_movie_dict[user].extend([m])
	user_rating_dict[user].extend([r])


np.save(output_path+"user_movie_dict.npy", user_movie_dict)
np.save(output_path+"user_rating_dict.npy", user_rating_dict)

	
user_freq_dict={}
for user in unique_user_id:
	print(user, user_num)
	user_freq_dict[user]=[]

freq_list=np.zeros(len(user_movie_dict.keys())+1)
for user in user_movie_dict.keys():
	print(user)
	user_freq_dict[user]=len(user_movie_dict[user])
	freq_list[user]=len(user_movie_dict[user])

np.save(output_path+"user_freq_dict.npy", user_freq_dict)
np.save(output_path+"freq_list.npy", freq_list)

len(freq_list[freq_list>100])

top_500_user_id=list(np.where(freq_list>=100)[0])
top_500_user_freq=freq_list[freq_list>=100]
np.save(output_path+'top_500_user_id', top_500_user_id)
np.save(output_path+'top_500_user_freq', top_500_user_freq)


top_500_user_film=[]
for user in top_500_user_id:
	films=user_movie_dict[user]
	top_500_user_film.extend(films)

top_500_user_film=np.unique(top_500_user_film)
np.save(output_path+'top_500_user_film_id', top_500_user_film)

movie=pd.read_csv(input_path+'movie.csv')
movie.columns
movie_id=movie['movieId'].values
genres=movie['genres'].values

tag=pd.read_csv(input_path+'tag.csv')
tag.columns
tags=tag['tag'].values
tag_movie_id=tag['movieId'].values

movie_describe_dict={}
for m in movie_id:
	print(m)
	movie_describe_dict[m]=[]

for index in range(len(tag_movie_id)):
	print(index)
	m=tag_movie_id[index]
	try:
		tag=tags[index]
		movie_describe_dict[m]+=[tag]
	except:
		pass

for index in range(len(movie_id)):
	print(index)
	m=movie_id[index]
	try:
		ge=genres[index]
		movie_describe_dict[m]+=[ge]
	except:
		pass

for index in range(len(movie_id)):
	print(index)
	m=movie_id[index]
	try:
		title=movie_title[index]
		movie_describe_dict[m]+=[title]
	except:
		pass


np.save(output_path+'movie_describe_dict.npy', movie_describe_dict)


import nltk 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
stop_words=set(stopwords.words('english'))
ps=PorterStemmer()

new_movie_describe_dict={}
for m in movie_describe_dict.keys():
	try:
		print(m)
		new_movie_describe_dict[m]=remove(movie_describe_dict[m][0])
		filtered_stemm_sent=[]
		for w in new_movie_describe_dict[m]:
			if w not in stop_words:
				filtered_stemm_sent.append(ps.stem(w))
		new_movie_describe_dict[m]=filtered_stemm_sent
	except:
		pass


for i in range(len(movie_id)):
	print(i)
	m=movie_id[i]
	a=''
	for word in new_movie_describe_dict[m]:
		a+=str(word)+' '
	new_movie_describe_dict[m]=a

np.save(output_path+"new_movie_describe_dict.npy", new_movie_describe_dict)	
new_movie_describe_dict=np.load(output_path+'new_movie_describe_dict.npy')
new_movie_describe_dict=new_movie_describe_dict.item()

s1=new_movie_describe_dict.keys()
s2=new_movie_describe_dict.values()
df=pd.DataFrame(columns=['movieId', 'describe'])
df['movieId']=s1
df['describe']=s2

df.to_csv(output_path+'df_moive_describe')


from sklearn.feature_extraction.text import TfidfVectorizer

vec = TfidfVectorizer()
X = vec.fit_transform(df)
X_dense = X.todense()
print (X_dense[100,:])

