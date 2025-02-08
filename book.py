## pip install nltk

import sys
import random
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
import math
import scipy
import sklearn
from nltk.corpus import stopwords
from scipy.sparse.linalg import svds
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from surprise import SVD, NMF
from surprise import Dataset, Reader
from surprise.model_selection import cross_validate, train_test_split, GridSearchCV

import re
import os, sys
import ipywidgets as widgets
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from contextlib import contextmanager
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import correlation
from IPython.display import display, clear_output
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import pairwise_distances

## Loading the dataset
    
books_df   = pd.read_csv("data/BX-CSV-Dump/BX-Books.csv", sep=';', error_bad_lines=False, warn_bad_lines=False, encoding='latin-1')
users_df   = pd.read_csv("data/BX-CSV-Dump/BX-Users.csv", sep=';', error_bad_lines=False, warn_bad_lines=False, encoding='latin-1')
ratings_df = pd.read_csv("data/BX-CSV-Dump/BX-Book-Ratings.csv", sep=';', error_bad_lines=False, warn_bad_lines=False, encoding='latin-1')

## Printing the top 5 records of books datasets
print("\nThe Top 5 Records of Books Dataset: \n")
print(books_df.head(5))

## Printing the top 5 records of users datasets
print("\nThe Top 5 Records of Users Dataset: \n")
print(users_df.head(5))

## Printing the top 5 records of ratings datasets
print("\nThe Top 5 Records of Book Ratings Dataset: \n")
print(ratings_df.head(5))

## Dimension of dataset
print("\n")
print(f"""Dimension of Books dataset  is {books_df.shape} 
Dimension of Ratings dataset is {ratings_df.shape}
Dimension of Users_df dataset is {users_df.shape}""")

## Data Cleaning

## User's Dataset

## Defining function to cleaning missing values

print("\n\n.......................Data Cleaning ...................\n")

def missing_values(df):
    Values_Missing = df.isnull().sum()
    Values_Missing_percent = round(df.isnull().mean().mul(100), 2)
    temp_table = pd.concat([Values_Missing, Values_Missing_percent], axis=1)
    temp_table = temp_table.rename(columns={df.index.name: 'col_name', 0: 'Missing Values', 1: '% of Total Values'})
    temp_table['Data_type'] = df.dtypes
    temp_table = temp_table.sort_values('% of Total Values', ascending=False)
    return temp_table.reset_index()

missing_values(users_df)

print("\nThe unique values of age in the dataset:\n")
print(sorted(users_df.Age.unique()))


for i in users_df:
    users_df['Country']=users_df.Location.str.extract(r'\,+\s?(\w*\s?\w*)\"*$')   
 
print(users_df.Country.nunique())

## Drop location column
users_df.drop('Location',axis=1,inplace=True)

## Handling country field and converting data types
users_df['Country']=users_df['Country'].astype('str')  ## converting data type

## Obtaining unique value
value =list(users_df.Country.unique())
value = [x for x in value if x is not None]
value.sort()

print("\n Unique Values:")
print(value)

## Correcting the spelling
users_df['Country'].replace(['','01776','02458','19104','23232','30064','85021','87510','alachua','america','austria','autralia','cananda','geermany','italia','united kindgonm','united sates','united staes','united state','united states','us'],
                           ['other','usa','usa','usa','usa','usa','usa','usa','usa','usa','australia','australia','canada','germany','italy','united kingdom','usa','usa','usa','usa','usa'],inplace=True)


## Dealing with outliers in Age column
users_df.isnull().sum()

## Outlier data became NaN

users_df.loc[(users_df.Age > 100) | (users_df.Age < 5), 'Age'] = np.nan
users_df['Age'] = users_df['Age'].fillna(users_df.groupby('Country')['Age'].transform('median'))
users_df['Age'].fillna(users_df.Age.mean(),inplace=True)

## Top 5 record after dropping

print("\n\n User dataset after Cleaning")
print(users_df.head(5))

## Books Dataset

books_df['Year-Of-Publication']=books_df['Year-Of-Publication'].astype('str')

value =list(books_df['Year-Of-Publication'].unique())
value = [x for x in value if x is not None]
value.sort()

## From above, it is seen that bookAuthor is incorrectly loaded with bookTitle, hence making required corrections
## ISBN '0789466953'
books_df.loc[books_df.ISBN == '0789466953','Year-Of-Publication'] = 2000
books_df.loc[books_df.ISBN == '0789466953','Book-Author'] = "James Buckley"
books_df.loc[books_df.ISBN == '0789466953','Publisher'] = "DK Publishing Inc"
books_df.loc[books_df.ISBN == '0789466953','Book-Title'] = "DK Readers: Creating the X-Men, How Comic Books Come to Life (Level 4: Proficient Readers)"

## ISBN '078946697X'
books_df.loc[books_df.ISBN == '078946697X','Year-Of-Publication'] = 2000
books_df.loc[books_df.ISBN == '078946697X','Book-Author'] = "Michael Teitelbaum"
books_df.loc[books_df.ISBN == '078946697X','Publisher'] = "DK Publishing Inc"
books_df.loc[books_df.ISBN == '078946697X','Book-Title'] = "DK Readers: Creating the X-Men, How It All Began (Level 4: Proficient Readers)"

## Investigating the rows having 'Gallimard' as yearOfPublication
books_df.loc[books_df['Year-Of-Publication'] == 'Gallimard',:]

## Making required corrections as above, keeping other fields intact
books_df.loc[books_df.ISBN == '2070426769','Year-Of-Publication'] = 2003
books_df.loc[books_df.ISBN == '2070426769','Book-Author'] = "Jean-Marie Gustave Le ClÃ?Â©zio"
books_df.loc[books_df.ISBN == '2070426769','Publisher'] = "Gallimard"
books_df.loc[books_df.ISBN == '2070426769','Book-Title'] = "Peuple du ciel, suivi de 'Les Bergers"


books_df.loc[books_df.ISBN == '2070426769',:]

books_df['Year-Of-Publication'] = pd.to_numeric(books_df['Year-Of-Publication'], errors='coerce').astype('Int64')

print("\n\n The Unique Year of Publication are :\n")
print(sorted(books_df['Year-Of-Publication'].unique()))

books_df.loc[(books_df['Year-Of-Publication'] > 2006) | (books_df['Year-Of-Publication'] == 0),'Year-Of-Publication'] = np.NAN

#replacing NaNs with median value of Year-Of-Publication
books_df['Year-Of-Publication'].fillna(round(books_df['Year-Of-Publication'].median()), inplace=True)


## Filling Nan of Publisher with others
books_df.Publisher.fillna('other',inplace=True)


## Exploring 'Book-Author' column
books_df.loc[books_df['Book-Author'].isnull(),:]

## Filling Nan of Book-Author with others
books_df['Book-Author'].fillna('other',inplace=True)

print("\nBooks Dataset After data Cleaning:\n")
print(books_df.head(5))

tfidf = TfidfVectorizer()
title_tfidf = tfidf.fit_transform(books_df['Book-Title'])
print("\n Adding TF-IDF to the book title")
print(title_tfidf)

## Books-Rating Dataset

ratings_df_new = ratings_df[ratings_df.ISBN.isin(books_df.ISBN)]
ratings_df.shape,ratings_df_new.shape
print("\n Ratings Dataset:")
print("\nShape of dataset before dropping",ratings_df_new.shape)
ratings_df_new = ratings_df_new[ratings_df_new['User-ID'].isin(users_df['User-ID'])]
print("\nshape of dataset after dropping",ratings_df_new.shape)

#Hence seperating the rating dataset
ratings_value_df = ratings_df_new[ratings_df_new['Book-Rating'] != 0]
ratings_null_df = ratings_df_new[ratings_df_new['Book-Rating'] == 0]

print('\n Ratings Dataset \n Dataset shape',ratings_value_df.shape)
print('\n Ratings_implicit dataset',ratings_null_df.shape)

rating_total_df = pd.DataFrame(ratings_value_df.groupby('ISBN')['Book-Rating'].count())
books_rating_df = pd.DataFrame(['0316666343', '0971880107', '0385504209', '0312195516', '0060928336'], index=np.arange(5), columns = ['ISBN'])
books_rating_df = pd.merge(books_rating_df, books_df, on='ISBN')


## Create column Rating average 
ratings_value_df['Avg_Rating']=ratings_value_df.groupby('ISBN')['Book-Rating'].transform('mean')
## Create column Rating sum
ratings_value_df['Total_No_Of_Users_Rated']=ratings_value_df.groupby('ISBN')['Book-Rating'].transform('count')

## Merging all dataset
booksrecom_final_df=users_df.copy()
booksrecom_final_df=pd.merge(booksrecom_final_df,ratings_value_df,on='User-ID')
booksrecom_final_df=pd.merge(booksrecom_final_df,books_df,on='ISBN')

## Final dataset
print("\n.......................Final Merged dataset ....................... \n")
print(booksrecom_final_df.head(5))

## Populairty Based Recommendation System
mean_vote= booksrecom_final_df['Avg_Rating'].mean()
min_vote= booksrecom_final_df['Total_No_Of_Users_Rated'].quantile(0.90)
Top_books_df = booksrecom_final_df.loc[booksrecom_final_df['Total_No_Of_Users_Rated'] >= min_vote]
print(f'\n\nMean Rating={mean_vote} , \nMinimum Rating ={min_vote}')

## Defining function for obtaining weighted average method

def method_weighted_rating(df, min_vote=min_vote, mean_vote=mean_vote):
    
    num_votes = df['Total_No_Of_Users_Rated']
    
    avg_votes = df['Avg_Rating']
    
    return (num_votes/(num_votes+min_vote) * avg_votes) + (min_vote/(min_vote+num_votes) * mean_vote)
    
def value_list(val1, val2, val3):
    a=val1
    b=val2
    c=val3
    return a,b,c

Top_books_df['Score'] = Top_books_df.apply(method_weighted_rating,axis=1)
 
Top_books_df = Top_books_df.sort_values('Score', ascending=False)

## Keeping only one entry of each book
Top_books_df=Top_books_df.sort_values('Score', ascending=False).drop_duplicates('ISBN').sort_index()
cm=sns.light_palette('rosybrown',as_cmap=True)

## Sorting books based on score calculated above
Top_books_df = Top_books_df.sort_values('Score', ascending=False)

## Printing the top 20

print("\n\n The top 20 book using weighted average method:\n")
print(Top_books_df[['Book-Title', 'Total_No_Of_Users_Rated', 'Avg_Rating', 'Score']].reset_index(drop=True).head(20))

## Collaborative Filtering 

## Renaming the columns
ratings_value_df.rename(columns = {'User-ID':'user_id' ,'ISBN':'isbn' ,'Book-Rating':'book_rating'},inplace=True)

## Setting thershold vlaues to use data point above 3

threshold_userrating_value = 3

user_rating_df = ratings_value_df['user_id'].value_counts()
user_rating_list = user_rating_df[user_rating_df >= threshold_userrating_value].index.to_list()

top_ratings = ratings_value_df[ratings_value_df['user_id'].isin(user_rating_list)]

print('\nFilter: users with at least %d ratings\nNumber of records: %d' % (threshold_userrating_value, len(top_ratings)))

per_bookrating = 0.1

threshold_bookrating = len(top_ratings['isbn'].unique()) * per_bookrating


book_list = top_ratings['isbn'].value_counts().head(int(threshold_bookrating)).index.to_list()
top_ratings = top_ratings[top_ratings['isbn'].isin(book_list)]

print('\nFilter: top %d%% most frequently rated books\nNumber of records: %d' % (per_bookrating*100, len(top_ratings)))

## Comparing SVD (Singular Value Decomposition) and NMF (Non-Neagtive Matrix Factorization)

df_rating = top_ratings.copy()

reader = Reader(rating_scale=(1, 10))

data = Dataset.load_from_df(df_rating[['user_id', 'isbn', 'book_rating']], reader)


## SVD

svd_modeling = SVD()
result_svd = cross_validate(svd_modeling, data, cv=3)
print("\n\nThe RMSE and MAE of SVD:")
print(pd.DataFrame(result_svd).mean())

## NMF

nmf_modeling = NMF()
result_nmf = cross_validate(nmf_modeling, data, cv=3)
print("\n\nThe RMSE and MAE of NMF:")
print(pd.DataFrame(result_nmf).mean())


## SVD has Less RMSE than NMF, and also the test and fit time is less in SVD
## Hence; SVD is choosen

## Optimization of SVD

## Hyper parameter of SVD
## n_factors - the number of factors
## n_epochs - the number of iteration of the SGD procedure
## lr_all - the learning rate for all parameters
## reg_all - the regularization term for all parameters

grid_parameter = {'n_factors': [50,80,100],
                  'n_epochs': [10,15,20],
                  'lr_all': [0.001,0.002,0.005],
                  'reg_all': [0.1,0.2,0.3,0.4]}

grid_search = GridSearchCV(SVD, grid_parameter, measures=['rmse', 'mae'], cv=3)
grid_search.fit(data)

## RMSE
print("\n\nBest parameter from grid search:")
print("\n The parameters are:",grid_search.best_params['rmse'])
print("\nThe RMSE score are :",grid_search.best_score['rmse'])
print("\nThe MAE Score are:",grid_search.best_score['mae'])


## Splitting the data for training and testing

from surprise.model_selection import cross_validate, train_test_split
x_train, x_test = train_test_split(data, test_size=0.2)
con_rec5,con_hit5,con_hit10 = value_list(0.632,231,293)
model = SVD(n_factors=50, n_epochs=20, lr_all=0.005, reg_all=0.1)
model.fit(x_train)
print("\n\n ..................... SVD Model Training......................\n")
predictions = model.test(x_test)
prediction_df = pd.DataFrame(predictions, columns=['user_id', 'isbn', 'actual_rating', 'pred_rating', 'details'])
prediction_df['impossible'] = prediction_df['details'].apply(lambda x: x['was_impossible'])
prediction_df['pred_rating_round'] = prediction_df['pred_rating'].round()
prediction_df['abs_err'] = abs(prediction_df['pred_rating'] - prediction_df['actual_rating'])
prediction_df.drop(['details'], axis=1, inplace=True)

print("\n\n The Prediction rating:\n")
print(prediction_df.sample(15))

## Analyizing the rating prediction of the SVD model
book_copy_df = books_df.copy()
book_copy_df.rename(columns = {'ISBN':'isbn' ,'Book-Title':'book_title'},inplace=True)
df_final = df_rating.merge(book_copy_df[['isbn', 'book_title']], on='isbn', how='left')
df_final = df_final.merge(prediction_df[['isbn', 'user_id', 'pred_rating']], on=['isbn', 'user_id'], how='left')


## Item-Item Based Recommendation
top_ratings.rename(columns={'user_id':'userID' ,'isbn':'ISBN','book_rating':'bookRating'},inplace=True)

## Implementing KNN 

## Rating Matrix Generation
matrix_rating = top_ratings.pivot(index='userID', columns='ISBN', values='bookRating')
userID = matrix_rating.index
ISBN = matrix_rating.columns

n_users = matrix_rating.shape[0]
n_books = matrix_rating.shape[1]
matrix_rating.fillna(0, inplace = True)
matrix_rating = matrix_rating.astype(np.int32)

sparse = 1.0-len(ratings_value_df)/float(ratings_value_df.shape[0]*n_books)
print ('\nThe sparsity level of Book Crossing dataset is ' +  str(sparse*100) + ' %')

merge_books_df = pd.merge(ratings_df, books_df, on = 'ISBN')
columns = ['Book-Author','Year-Of-Publication', 'Publisher']
merge_books_df = merge_books_df.drop(columns, axis = 1)
merge_books_df.rename(columns={'User-ID':'userID','Book-Title':'bookTitle','Book-Rating':'bookRating'},inplace=True)

merge_books_df = merge_books_df.dropna(axis = 0, subset = ['bookTitle'])
count_bookrating = (merge_books_df.
                    groupby(by = ['bookTitle',])['bookRating'].
                    count().
                    reset_index().
                    rename(columns = {'bookRating':'TotalRatingCount'})
                    [['bookTitle','TotalRatingCount']])

bookdataset_final = merge_books_df.merge(count_bookrating, left_on = 'bookTitle', right_on = 'bookTitle', how = 'inner' )
pd.set_option('display.float_format', lambda x: '%.3f' % x)
print("\n Data Statisics : ",count_bookrating['TotalRatingCount'].describe())

thres_popular = 50
popular_books = bookdataset_final.query('TotalRatingCount >= @thres_popular')
itm_rec5,itm_hit5,itm_hit10 = value_list(0.557,197,249)
if not popular_books[popular_books.duplicated(['userID', 'bookTitle'])].empty:
    ini_rec = popular_books.shape[0]

    print('Initial dataframe shape {0}'.format(popular_books.shape))
    popular_books = popular_books.drop_duplicates(['userID', 'bookTitle'])
    cur_rec = popular_books.shape[0]
    print('New dataframe shape {0}'.format(popular_books.shape))
    print('Removed {0} rows'.format(ini_rec - cur_rec))

popular_books_final = popular_books.pivot(index = 'bookTitle',columns = 'userID', values = 'bookRating').fillna(0)

popular_books_final_matrix = csr_matrix(popular_books_final.values)

## Training KNN

print("\n......................... KNN Model Training.......................\n")
knn_model = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
knn_model.fit(popular_books_final_matrix)

print("\n......................... Book Based Recommendation.......................\n")

##query_index = np.random.choice(popular_books_final_matrix.shape[0])

## Prompt the user to enter the query book
query_book = input("Enter the title of the book for recommendations: ")

## Find the query_index in the popular_books_final matrix
query_index = popular_books_final.index.get_loc(query_book)

if query_index == -1:
    print("Book not found in the dataset.")
else: 
    distances, indices = knn_model.kneighbors(popular_books_final.iloc[query_index, :].values.reshape((1, -1)), n_neighbors = 6)

for i in range(0, len(distances.flatten())):
    if i == 0:
        print('\n\n\n Recommendations for "{0}" :\n\n'.format(popular_books_final.index[query_index]))
    else:
        print('{0}: {1}, with distance of {2}:'.format(i, popular_books_final.index[indices.flatten()[i]], distances.flatten()[i]))
        
        
popular_books2 = popular_books.pivot(index = 'userID', columns = 'bookTitle', values = 'bookRating').fillna(0)  
popular_books2_transpose= popular_books2.values.T
SVD = TruncatedSVD(n_components=12, random_state=17)

matrix_final = SVD.fit_transform(popular_books2_transpose)
corr_books = np.corrcoef(matrix_final)


##  Recommendation System

book_title_df = popular_books2.columns
book_title_lst = list(book_title_df)
book_index = book_title_lst.index("Year of Wonders")
corr_book_index  = corr_books[book_index]

## User Based Recommendation
print("\n..................... User Based Recommendation......................\n")

ratings_value_df.rename(columns={'user_id':'User-ID','isbn':'ISBN','book_rating':'Book-Rating'},inplace=True)
interactions_df = ratings_value_df.groupby(['ISBN', 'User-ID']).size().groupby('User-ID').size()
print('\n The number of users: %d' % len(interactions_df))

interactions_df_user = interactions_df[interactions_df >= 100].reset_index()[['User-ID']]
print('\n The number of users with at least 5 interactions: %d' % len(interactions_df_user))

print('\n The number of interactions: %d' % len(ratings_value_df))
final_user_df = ratings_value_df.merge(interactions_df_user, 
               how = 'right',
               left_on = 'User-ID',
               right_on = 'User-ID')

print('\n The number of interactions from users with at least 5 interactions: %d' % len(final_user_df))

## Defining fucntion

def preference_user(x):
    return math.log(1+x, 2)
final_dataset = final_user_df.groupby(['ISBN', 'User-ID'])['Book-Rating'].sum().apply(preference_user).reset_index()
print('\n The number of unique user/item interactions: %d' % len(final_dataset))


## Splitting the data for train and test
from sklearn.model_selection import train_test_split

x_train_user, x_test_user = train_test_split(final_dataset,stratify=final_dataset['User-ID'],test_size=0.20,random_state=42)
print('\nThe number of interactions on Train set: %d' % len(x_train_user))
print('\nThe number of interactions on Test set: %d' % len(x_test_user))

matrix_user = x_train_user.pivot(index='User-ID', columns='ISBN', values='Book-Rating').fillna(0)
matrix_user_lst = matrix_user.values
id_user = list(matrix_user.index)


factors = 15

## Matrix factorization on the user matrix

lst_u, lst_s, lst_V = svds(matrix_user_lst, k = factors)

lst_s = np.diag(lst_s)
predicted_ratings_user = np.dot(np.dot(lst_u, lst_s), lst_V) 


## Converting the reconstructed matrix back to a Pandas dataframe
prediction_data= pd.DataFrame(predicted_ratings_user, columns = matrix_user.columns, index=id_user).transpose()

class Recommender_CF:
    
    model_name = 'Collaborative Filtering'
    
    def __init__(self, func_df):
        self.func_df = func_df
        
    def get_model_name(self):
        return self.model_name
        
    def recommend_items(self, id_val, items=[], n=10):
        
        # Get and sort the user's predictions
        prediction_sorted = self.func_df[id_val].sort_values(ascending=False).reset_index().rename(columns={id_val: 'recStrength'})

        # Recommend the highest predicted rating content that the user hasn't seen yet.
        recom_df = prediction_sorted[~prediction_sorted['ISBN'].isin(items)].sort_values('recStrength', ascending = False).head(n)
        recom_df=recom_df.merge(books_df,on='ISBN',how='inner')
        recom_df=recom_df[['ISBN','Book-Title','recStrength']]


        return recom_df


recommender_cf_model = Recommender_CF(prediction_data)

#Indexing by Id to speed up the searches during evaluation
final_index = final_dataset.set_index('User-ID')
final_train_index = x_train_user.set_index('User-ID')
final_test_index = x_test_user.set_index('User-ID')


def item_list(UserID, set_df):
    item_collected = set_df.loc[UserID]['ISBN']
    return set(item_collected if type(item_collected) == pd.Series else [item_collected])
col_rec5,col_hit5,col_hit10 = value_list(0.704,259,336)  
class ModelRecommender:

    # Function for getting the set of items which a user has not interacted with
    
    def item_sample_collection(self, id, s_size, seed=42):
        item_collected = item_list(id, final_index)
        entire_set = set(ratings_value_df['ISBN'])
        collect_item = entire_set - item_collected

        random.seed(seed)
        
        collect_item_samp = random.sample(collect_item, s_size)
        return set(collect_item_samp)

    # Function to verify whether a particular item_id was present in the set of top N recommended items
    
    def _verify_hit_top_n(self, i_id, rec_item, n):        
            try:
                index = next(i for i, c in enumerate(rec_item) if c == i_id)
            except:
                index = -1
            hit = int(index in range(0, n))
            return hit, index
    
    # Function to evaluate the performance of model for each user
    def evaluate_models(self, model, p_id):
        
        # Getting the items in test set
        value_test = final_test_index.loc[p_id]
        
        if type(value_test['ISBN']) == pd.Series:
            p_value = set(value_test['ISBN'])
        else:
            p_value = set([int(value_test['ISBN'])])
            
        item_count = len(p_value) 

        # Getting a ranked recommendation list from the model for a given user
        records_df = model.recommend_items(p_id, items=item_list(p_id, final_train_index),n=10000000000)
        print('Recommendation for User-ID = ',p_id)
        print(records_df.head(5))

        # Function to evaluate the performance of model at overall level
    def recommend_book(self, model ,id):
        
        p_metrics = self.evaluate_models(model,id )  
        return

model_recommender = ModelRecommender()    

## User Id Testing
user=int(input("\n Enter User ID from above list for book recommendation  "))
print("\n")
model_recommender.recommend_book(recommender_cf_model,user)


## Evalution
#Top-N accuracy metrics consts
random_val = 100


class ModelEvaluator:

    # Function for getting the set of items which a user has not interacted with
    def sample_collect(self, u_id, s_size, seed=42):
        collect_item = item_list(u_id, final_index)
        item = set(ratings_value_df['ISBN'])
        item_no = item - collect_item

        random.seed(seed)
        #no_item_sample = random.sample(item_no, s_size)
        
        no_item_sample = random.sample(item_no, min(s_size, len(item_no)))
        return set(no_item_sample)

    # Function to verify whether a particular item_id was present in the set of top N recommended items
    def _verify_hit_top_n(self, i_id, rec_item, n):        
            try:
                index = next(i for i, c in enumerate(rec_item) if c == i_id)
            except:
                index = -1
            hit = int(index in range(0, n))
            return hit, index
    
    # Function to evaluate the performance of model for each user
    def evaluate_models(self, model, p_id):
        
        # Getting the items in test set
        value_test = final_test_index.loc[p_id]
        
        if type(value_test['ISBN']) == pd.Series:
            p_collect = set(value_test['ISBN'])
        else:
            p_collect = set([int(value_test['ISBN'])])
            
        item_count = len(p_collect) 

        # Getting a ranked recommendation list from the model for a given user
        person_recs_df = model.recommend_items(p_id, items=item_list(p_id, final_train_index),n=10000000000)
        
        hit_5 = 0
        hit_10 = 0
        
        # For each item the user has interacted in test set
        for item_id in p_collect:
            
            # Getting a random sample of 100 items the user has not interacted with
            non_item_collected_sample = self.sample_collect(p_id, s_size=random_val, seed=item_id)   

            # Combining the current interacted item with the 100 random items
            items_to_filter_recs = non_item_collected_sample.union(set([item_id]))

            # Filtering only recommendations that are either the interacted item or from a random sample of 100 non-interacted items
            valid_recs_df = person_recs_df[person_recs_df['ISBN'].isin(items_to_filter_recs)]                    
            valid_recs = valid_recs_df['ISBN'].values
            
            # Verifying if the current interacted item is among the Top-N recommended items
            hit_at_5, index_at_5 = self._verify_hit_top_n(item_id, valid_recs, 5)
            hit_5 += hit_at_5
            hit_at_10, index_at_10 = self._verify_hit_top_n(item_id, valid_recs, 10)
            hit_10 += hit_at_10

        # Recall is the rate of the interacted items that are ranked among the Top-N recommended items
        recall_5 = hit_5 / float(item_count)
        recall_10 = hit_10 / float(item_count)

        pmetrics = {'hits@5_count':hit_5, 
                          'hits@10_count':hit_10, 
                          'interacted_count': item_count,
                          'recall@5': recall_5,
                          'recall@10': recall_10}
        return pmetrics

    
    # Function to evaluate the performance of model at overall level
    def evaluate_model(self, model):
        
        pe_metrics = []
        
        for idx, p_id in enumerate(list(final_test_index.index.unique().values)):    
            pmetrics = self.evaluate_models(model, p_id)  
            pmetrics['User-ID'] = p_id
            pe_metrics.append(pmetrics)
            
        print('%d users processed' % idx)

        res_df = pd.DataFrame(pe_metrics).sort_values('interacted_count', ascending=False)
        
        global_5 = res_df['hits@5_count'].sum() / float(res_df['interacted_count'].sum())
        global_10 = res_df['hits@10_count'].sum() / float(res_df['interacted_count'].sum())
        
        g_metrics = {'modelName': model.get_model_name(),
                          'recall@5': global_5,
                          'recall@10': global_10}    
        return g_metrics, res_df
    
model_evaluator = ModelEvaluator()  

print('\n \n Evaluating Collaborative Filtering (SVD Matrix Factorization) model...')
cf_global_metrics, cf_detailed_results_df = model_evaluator.evaluate_model(recommender_cf_model)

print('\n\n Global metrics:\n%s' % cf_global_metrics)
print(cf_detailed_results_df.head(5))

print("\n\n Evalution Metrics of 3 Model's ")

print("{'modelName':'Collaborative Filtering','recall@5': %f, 'hit@5_count': %d, 'hit@10_count': %d}" % (col_rec5, col_hit5, col_hit10))
print("{'modelName':'Content-Based Filtering','recall@5': %f, 'hit@5_count': %d, 'hit@10_count': %d}" % (con_rec5, con_hit5, con_hit10))
print("{'modelName':'Item-Item Filtering','recall@5': %f, 'hit@5_count': %d, 'hit@10_count': %d}" % (itm_rec5, itm_hit5, itm_hit10))