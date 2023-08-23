import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""**Data Collection**"""

dataset = pd.read_csv('movies.csv')

dataset.head()

dataset.shape

"""**Feature Selection**"""

selected_features = ['genres', 'keywords', 'overview', 'tagline', 'cast', 'director']

# Fulling null value ith empty string
for feature in selected_features:
  dataset[feature] = dataset[feature].fillna('')

# Combining all the selected feature / concatenate
combined_features = dataset['genres']+' '+dataset['keywords']+' '+dataset['overview']+' '+dataset['tagline']+ ' '+dataset['cast']+' '+ dataset['director']


# Converting textual data into feature vector
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()

feature_vector = vectorizer.fit_transform(combined_features)


# Check similarity
from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity(feature_vector)


# Taking  a movie name from user
movie_name = input("Enter movie name : ")

# Create a list that contain movie name that present in our dataset

all_movies = dataset['title'].tolist()


# Finding close match movie that present in movie list

import difflib

finding_close_match = difflib.get_close_matches(movie_name, all_movies)

close_match = finding_close_match[0]

# Finding all the movie and index of movie that present in dataset

index_of_the_movie = dataset[dataset.title == close_match]['index'].values[0]

# Getting a list of similar movie

similarity_score = list(enumerate(similarity[index_of_the_movie]))

# Sorting the movie based on similarity score

sort_similer_movie = sorted(similarity_score, key=lambda x:x[1], reverse=True)

# Print similar movie and suggest to the user

print("Must watch these movie before die : \n")

i = 1
for movie in sort_similer_movie:
  index = movie[0]
  title_of_movie = dataset[dataset.index == index]['title'].values[0]
  if i<=20:
    print(i, '.', title_of_movie)
    i += 1

"""# **Make Recommendation System**"""

movie_name = input("Enter movie name : ")

all_movies = dataset['title'].tolist()

finding_close_match = difflib.get_close_matches(movie_name, all_movies)

close_match = finding_close_match[0]

index_of_the_movie = dataset[dataset.title == close_match]['index'].values[0]

similarity_score = list(enumerate(similarity[index_of_the_movie]))

sort_similer_movie = sorted(similarity_score, key=lambda x:x[1], reverse=True)

print("Must watch these movie before die : \n")

i = 1
for movie in sort_similer_movie:
  index = movie[0]
  title_of_movie = dataset[dataset.index == index]['title'].values[0]
  if i<=20:
    print(i, '.', title_of_movie)
    i += 1

import pickle

# Save the 'sort_similer_movie' list as a pickle file
with open('recommendation.pkl', 'wb') as file:
    pickle.dump(sort_similer_movie, file)

