from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from sklearn.decomposition import TruncatedSVD
import pandas as pd
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate
import sys
import numpy as np
from scipy.sparse.linalg import svds


data = pd.read_csv('data/Data.csv')
books = pd.read_csv('data/books1.csv')
popular_book_user_rating = pd.read_csv("data/popular_book_user_rating.csv")
#For user based
user_interaction_test = pd.read_csv("data/user_interaction_test.csv")
user_interaction_train = pd.read_csv("data/user_interaction_train.csv")
pivot_mat_users_items = user_interaction_train.pivot(index='User-ID',columns='ISBN', values='Book-Rating').fillna(0)
                                                          
  
#About 1% of the books received 50 or more ratings. Because we have so many books in our data, we will limit it to the top 1%.
# This is why all the books which are comng as part of user based suggeston will not reflect for item or book title based recommendation

def title_rec(title):

	book_title = popular_book_user_rating.columns
	book_list = list(map(str.lower,list(book_title)))
	title = title.lower()

	if title in book_list:	

		coeff = book_list.index(title)
		SVD = TruncatedSVD(n_components=12, random_state=17)
		X = popular_book_user_rating.values.T
		matrix = SVD.fit_transform(X)
		corr = np.corrcoef(matrix)	
		corr_coeff  = corr[coeff]
		rec_books = list(book_title[(corr_coeff<1.0) & (corr_coeff>0.9)])

		#Fetching other details of the book based on its title
		result = []

		if len(rec_books) > 10:
			for i in range(0,10):
				d = data.loc[data['Book-Title'] == rec_books[i]].iloc[0]
				result.append(d)
		else:
			for i in range(0,len(rec_books)):
				d = data.loc[data['Book-Title'] == rec_books[i]].iloc[0]
				result.append(d)	
	else:
		return None
	
	if len(rec_books) > 0:

		return result

	else:
		return None
	

def user_rec(user):

	user_interaction_test_indexed = pd.DataFrame()
	user_interaction_test_indexed = user_interaction_test.set_index('User-ID')

	user = int(user)
	interacted_values = user_interaction_test_indexed.loc[user]

	if type(interacted_values['ISBN']) == pd.Series:
		person_interacted_items_testset = set(interacted_values['ISBN'])
	else:
		person_interacted_items_testset = set([int(interacted_values['ISBN'])])

	#interacted_items_count_testset = len(person_interacted_items_testset) 

	user_rec_df = user_recommend(user, ignore=interaction_items(user, user_interaction_test_indexed),topn=10000000000)
	user_rec_df = user_rec_df.merge(data.drop_duplicates(["Book-Title"]), how="inner")

	if user_rec_df.shape[0] > 0:
		return user_rec_df.head(10)
	else:
		return None
	  
def interaction_items(UserID, interactions):

    items = interactions.loc[UserID]['ISBN']
    return set(items if type(items) == pd.Series else [items])

def user_recommend(user_id, ignore=[], topn=10):

	pivot_mat = pivot_mat_users_items.values

	#Matrix factorization
	U, s, V = svds(pivot_mat, k = 15)

	s = np.diag(s)

	user_pred_rating = np.dot(np.dot(U, s), V) 
	users_ids = list(pivot_mat_users_items.index)


	pred_rat_df = pd.DataFrame(user_pred_rating, columns = pivot_mat_users_items.columns, index=users_ids).transpose()

	# Sorting user predictions
	sorted_pred_rat = pred_rat_df[user_id].sort_values(ascending=False).reset_index().rename(columns={user_id: 'strength'})
	
	# Recommending the highest predicted rating content which the user has not yet read.
	recommendation = sorted_pred_rat[~sorted_pred_rat['ISBN'].isin(ignore)].sort_values('strength', ascending = False).head(topn)
	recommendation=recommendation.merge(books,on='ISBN',how='inner')
	recommendation=recommendation[['ISBN','Book-Title','strength']]

	return recommendation
    

    

   

    
    
    

    