import pandas as pd
import RecSysALS
import RecSysNMF
import RecSysKNN
# import RecSysNCF

class RecSys():
        
    def __init__(self, n_users, n_items, top_users, top_items, l, theta, k):
        self.n_users = n_users
        self.n_movies = n_items
        self.top_users = top_users
        self.top_movies = top_items
        self.l = l
        self.theta = theta
        self.k = k

    ###################################################################################################################
    # function to read the data
    def read_movielens_1M(self, n_users, n_items, top_users, top_items, data_dir):
        # get ratings
        df = pd.read_table('{}/ratings.dat'.format(data_dir),names=['UserID','MovieID','Rating','Timestamp'], sep='::', engine='python')

        # create a dataframe with movie IDs on the rows and user IDs on the columns
        ratings = df.pivot(index='MovieID', columns='UserID', values='Rating')
        
        users_info = pd.read_table('{}/users.dat'.format(data_dir), names=['UserID','Gender','Age','Occupation','Zip-code'], sep='::', engine='python', encoding = "ISO-8859-1")
        users_info = users_info.rename(index=users_info['UserID'])[['Gender','Age','Occupation','Zip-code']]

        # add number of ratings in users_info
        num_ratings = (~ratings.isnull()).sum(axis=0)
        users_info['NR'] = num_ratings

        items_info = pd.read_table('{}/movies.dat'.format(data_dir), names=['MovieID', 'Title', 'Genres'], sep='::', engine='python', encoding = "ISO-8859-1")
        
        # put movie titles as index on rows
        movieSeries = pd.Series(list(items_info['Title']), index=items_info['MovieID'])
        ratings = ratings.rename(index=movieSeries)
        
        if top_items:
            # select the top n_movies with the highest number of ratings
            num_ratings = (~ratings.isnull()).sum(axis=1) # quantitative ratings for each movie: Movie 1: 4, Movie 2: 5, Movie 3: 2 ...
            rows = num_ratings.nlargest(n_items) # quantitative ratings for each movie (n_movies) sorted: Movie 7: 6, Movie 2: 5, Movie 1: 4 ...
            ratings = ratings.loc[rows.index] # matrix[n_movies rows , original columns]; before [original rows x original columns]
        
        if top_users:
            # select the top n_users with the highest number of ratings
            num_ratings = (~ratings.isnull()).sum(axis=0) # quantitative ratings made by each user: User 1: 5, User 2: 5, User 3: 5, ...
            cols = num_ratings.nlargest(n_users) # quantitative evaluations by each user (n_users) sorted: User 1: 5, User 2: 5, User 3: 5, ...
            ratings = ratings[cols.index] # matrix [n_movies rows , original columns]; before [n_movies rows , original columns] (just updated the index)
        else:
            # select the first n_users from the matrix
            cols = ratings.columns[0:n_users]
            ratings = ratings[cols] # matrix [n_movies rows , n_users columns]; before [n_movies rows , original columns]

        ratings = ratings.T # transposed: matrix [n_users rows x n_movies columns];

        return ratings, users_info, items_info
    

    ###################################################################################################################
    # function to read the data
    def read_movielens_small(self, n_users, n_items, top_users, top_items, data_dir):
        # get ratings
        df = pd.read_table('{}/ratings.dat'.format(data_dir),names=['UserID','MovieID','Rating','Timestamp'], sep='::', engine='python')

        # create a dataframe with movie IDs on the rows and user IDs on the columns
        ratings = df.pivot(index='MovieID', columns='UserID', values='Rating')
        
        items_info = pd.read_table('{}/movies.dat'.format(data_dir), names=['MovieID', 'Title', 'Genres', 'Price'], sep='::', engine='python')
        
        # put movie titles as index on rows
        movieSeries = pd.Series(list(items_info['Title']), index=items_info['MovieID'])
        ratings = ratings.rename(index=movieSeries)

        users_info = pd.read_table('{}/users.dat'.format(data_dir), names=['UserID','Gender','Age','NR','SPI', 'MA', 'MR'], sep='::', engine='python')
        users_info = users_info.rename(index=users_info['UserID'])[['Gender','Age','NR','SPI', 'MA', 'MR']]

        if top_items:
            # select the top n_movies with the highest number of ratings
            num_ratings = (~ratings.isnull()).sum(axis=1) # quantitative ratings for each movie: Movie 1: 4, Movie 2: 5, Movie 3: 2 ...
            rows = num_ratings.nlargest(n_items) # quantitative ratings for each movie (n_movies) sorted: Movie 7: 6, Movie 2: 5, Movie 1: 4 ...
            ratings = ratings.loc[rows.index] # matrix[n_movies rows , original columns]; before [original rows x original columns]
        
        if top_users:
            # select the top n_users with the highest number of ratings
            num_ratings = (~ratings.isnull()).sum(axis=0) # quantitative ratings made by each user: User 1: 5, User 2: 5, User 3: 5, ...
            cols = num_ratings.nlargest(n_users) # quantitative evaluations by each user (n_users) sorted: User 1: 5, User 2: 5, User 3: 5, ...
            ratings = ratings[cols.index] # matrix [n_movies rows , original columns]; before [n_movies rows , original columns] (just updated the index)
        else:
            # select the first n_users from the matrix
            cols = ratings.columns[0:n_users]
            ratings = ratings[cols] # matrix [n_movies rows , n_users columns]; before [n_movies rows , original columns]

        ratings = ratings.T # transposed: matrix [n_users rows x n_movies columns];

        return ratings, users_info, items_info
    

    ###################################################################################################################
    # function to read the data
    def read_books(self, n_users, n_items, top_users, top_items, data_dir):
        # get ratings
        df = pd.read_csv('{}/ratings.csv'.format(data_dir), sep=';')

        # create a dataframe with item IDs on the rows and user IDs on the columns
        ratings = df.pivot(index='BookID', columns='UserID', values='Rating')
        
        users_info = pd.read_csv('{}/users.csv'.format(data_dir), sep=';', encoding = "ISO-8859-1")
        
        items_info = pd.read_csv('{}/books.csv'.format(data_dir), sep=';', encoding = "ISO-8859-1")
                            
        users_info = users_info.rename(index=users_info['UserID'])[['Location','Age']]

        # add number of ratings in users_info
        num_ratings = (~ratings.isnull()).sum(axis=0)
        users_info['NR'] = num_ratings
        
        # put movie titles as index on rows
        #movieSeries = pd.Series(list(movies['Title']), index=movies['MovieID'])
        #ratings = ratings.rename(index=movieSeries)

        if top_items:
            # select the top n_movies with the highest number of ratings
            num_ratings = (~ratings.isnull()).sum(axis=1) # quantitative ratings for each movie: Movie 1: 4, Movie 2: 5, Movie 3: 2 ...
            rows = num_ratings.nlargest(n_items) # quantitative ratings for each movie (n_movies) sorted: Movie 7: 6, Movie 2: 5, Movie 1: 4 ...
            ratings = ratings.loc[rows.index] # matrix[n_movies rows , original columns]; before [original rows x original columns]
        
        if top_users:
            # select the top n_users with the highest number of ratings
            num_ratings = (~ratings.isnull()).sum(axis=0) # quantitative ratings made by each user: User 1: 5, User 2: 5, User 3: 5, ...
            cols = num_ratings.nlargest(n_users) # quantitative evaluations by each user (n_users) sorted: User 1: 5, User 2: 5, User 3: 5, ...
            ratings = ratings[cols.index] # matrix [n_movies rows , original columns]; before [n_movies rows , original columns] (just updated the index)
        else:
            # select the first n_users from the matrix
            cols = ratings.columns[0:n_users]
            ratings = ratings[cols] # matrix [n_movies rows , n_users columns]; before [n_movies rows , original columns]

        ratings = ratings.T # transposed: matrix [n_users rows x n_movies columns];

        return ratings, users_info, items_info

    
        ###################################################################################################################
    # function to read the data
    def read_songs(self, n_users, n_items, top_users, top_items, data_dir):
        
        # get ratings
        df = pd.read_csv('{}/ratings.csv'.format(data_dir), sep=';')

        # create a dataframe with item IDs on the rows and user IDs on the columns
        ratings = df.pivot(index='SongID', columns='UserID', values='Rating')
        
        users_info = pd.read_csv('{}/users.csv'.format(data_dir), sep=';', encoding = "ISO-8859-1")
        
        items_info = pd.read_csv('{}/songs.csv'.format(data_dir), sep=';', encoding = "ISO-8859-1")
                            
        users_info = users_info.rename(index=users_info['UserID'])

        # add number of ratings in users_info
        num_ratings = (~ratings.isnull()).sum(axis=0)
        users_info['NR'] = num_ratings
        
        # put movie titles as index on rows
        #movieSeries = pd.Series(list(movies['Title']), index=movies['MovieID'])
        #ratings = ratings.rename(index=movieSeries)

        if top_items:
            # select the top n_movies with the highest number of ratings
            num_ratings = (~ratings.isnull()).sum(axis=1) # quantitative ratings for each movie: Movie 1: 4, Movie 2: 5, Movie 3: 2 ...
            rows = num_ratings.nlargest(n_items) # quantitative ratings for each movie (n_movies) sorted: Movie 7: 6, Movie 2: 5, Movie 1: 4 ...
            ratings = ratings.loc[rows.index] # matrix[n_movies rows , original columns]; before [original rows x original columns]
        
        if top_users:
            # select the top n_users with the highest number of ratings
            num_ratings = (~ratings.isnull()).sum(axis=0) # quantitative ratings made by each user: User 1: 5, User 2: 5, User 3: 5, ...
            cols = num_ratings.nlargest(n_users) # quantitative evaluations by each user (n_users) sorted: User 1: 5, User 2: 5, User 3: 5, ...
            ratings = ratings[cols.index] # matrix [n_movies rows , original columns]; before [n_movies rows , original columns] (just updated the index)
        else:
            # select the first n_users from the matrix
            cols = ratings.columns[0:n_users]
            ratings = ratings[cols] # matrix [n_movies rows , n_users columns]; before [n_movies rows , original columns]

        ratings = ratings.T # transposed: matrix [n_users rows x n_movies columns];

        return ratings, users_info, items_info


    ###################################################################################################################
    # compute_X_est: 
    def  compute_X_est(self, X, algorithm='RecSysALS', data_dir="Data/MovieLens-Small"):
        if(algorithm == 'RecSysALS'):
            
            # factorization parameters
            rank = 1 # before 20 (5)
            lambda_ = 1 # before 20 (5) - ridge regularizer parameter

            # initiate a recommender system of type ALS (Alternating Least Squares)
            RS = RecSysALS.als_RecSysALS(rank,lambda_)
            X_est, error = RS.fit_model(X)

        elif(algorithm == 'RecSysKNN'):
            RS = RecSysKNN.RecSysKNN(k=self.k, ratings=X, user_based=True)
            X_est = RS.fit_model()

        elif(algorithm == 'RecSysNMF'):
            RS = RecSysNMF.RecSysNMF(n_components=5, ratings=X)
            X_est = RS.fit_model()

        # elif(algorithm == 'RecSysNCF'):
        #     RS = RecSysNCF.RecSysNCF(n_users=300, n_items=1000, n_factors=20, ratings=X)
        #     X_est, error = RS.fit_model()

        else:
            RecSysALS
        return X_est  
        

#######################################################################################################################