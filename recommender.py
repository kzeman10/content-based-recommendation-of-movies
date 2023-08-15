"""module implementing recommender class"""
import warnings
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
warnings.filterwarnings('ignore')

class Recommender():
    """class implementing recommender system"""
    def __init__(self, df_processed, df_orig, metric='euclidean') -> None:
        self.df_processed = df_processed
        self.df_orig = df_orig
        self.watched_movies = []
        # create instance of NearestNeighbors with cosine similarity
        self.nn = NearestNeighbors(n_neighbors=8, algorithm='brute', metric=metric)
        self.nn.fit(self.df_processed)


    def add_movie(self, movie):
        """add movie to watched movies"""
        self.watched_movies.append(movie)


    def recommend(self, use_avg=False, n_recommendations_for_each_movie=8):
        """
        recommend movies based on watched movies
        movies with closest distance to any of the watched movies will be recommended

        Returns:
            dataframe with movie title and its distance
        """
        # list of tuples with movie title and its distance
        all_recommendations = []
        points = []
        for movie in self.watched_movies:

            # get index of the movie in original dataframe
            movie_index = self.df_orig[self.df_orig['Series_Title'] == movie].index[0]

            # get distances and indices of the closest neighbors, unsparse before reshape
            point = self.df_processed.iloc[movie_index, :].values
            # if point is sparse, convert to numpy array
            if isinstance(point, (pd.SparseArray, pd.SparseSeries)):
                point = point.to_numpy()
            point = point.reshape(1, -1)
            distances, indices = self.nn.kneighbors(point, n_neighbors=n_recommendations_for_each_movie)

            # create list of tuples with movie title and its distance
            recommendations = list(zip(self.df_orig.iloc[indices.flatten(), :]['Series_Title'], distances.flatten()))
            all_recommendations += recommendations
            points.append(point)

        if use_avg:
            # avg of all points
            avg_point = np.mean(points, axis=0)
            # get distances and indices of the closest neighbors, unsparse before reshape
            distances, indices = self.nn.kneighbors(avg_point, n_neighbors=n_recommendations_for_each_movie)
            # create list of tuples with movie title and its distance
            recommendations = list(zip(self.df_orig.iloc[indices.flatten(), :]['Series_Title'], distances.flatten()))
            all_recommendations += recommendations

        # create dataframe with movie title and its distance
        df_result = pd.DataFrame(all_recommendations, columns=['movie', 'distance'])

        # drop watched movies
        df_result = df_result[~df_result['movie'].isin(self.watched_movies)]

        # sort by distance
        df_result.sort_values(by='distance', inplace=True)

        # drop duplicates and keep lower distance
        df_result.drop_duplicates(subset='movie', keep='first', inplace=True)

        return df_result
