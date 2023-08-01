from fastapi import FastAPI
from app.schemas import MovieName
import pickle
import difflib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import uvicorn

app = FastAPI()

# Load the 'sort_similer_movie' list from the pickle file
with open('recommendation.pkl', 'rb') as file:
    sort_similer_movie = pickle.load(file)

dataset = pd.read_csv('movies.csv')

selected_features = ['genres', 'keywords', 'overview', 'tagline', 'cast', 'director']

# Fulling null value with an empty string
for feature in selected_features:
    dataset[feature] = dataset[feature].fillna('')

# Combining all the selected features
combined_features = dataset['genres'] + ' ' + dataset['keywords'] + ' ' + dataset['overview'] + ' ' + dataset[
    'tagline'] + ' ' + dataset['cast'] + ' ' + dataset['director']

# Converting textual data into feature vector
vectorizer = TfidfVectorizer()
feature_vector = vectorizer.fit_transform(combined_features)

# Check similarity
similarity = cosine_similarity(feature_vector)


@app.get('/')
def home():
    return "Welcome to Movie Recommendation app"

@app.post('/movie-recommendation')
async def recommend_movie(name: MovieName):
    movie_name = name.movie_name

    # Create a list that contains all movie names present in our dataset
    all_movies = dataset['title'].tolist()

    # Finding close match movie that is present in the movie list
    finding_close_match = difflib.get_close_matches(movie_name, all_movies)
    close_match = finding_close_match[0]

    # Finding the index of the movie that is present in the dataset
    index_of_the_movie = dataset[dataset.title == close_match].index[0]

    # Get the similarity scores for the selected movie
    similarity_score = list(enumerate(similarity[index_of_the_movie]))

    # Sort the movies based on similarity score
    sort_similar_movie = sorted(similarity_score, key=lambda x: x[1], reverse=True)

    # Create a list of recommended movies
    recommended_movies = []
    i = 1
    for movie in sort_similar_movie:
        index = movie[0]
        title_of_movie = dataset.iloc[index]['title']
        if i <= 20:
            recommended_movies.append(f"{i}. {title_of_movie}")
            i += 1

    return {"movie_name": movie_name, "recommended_movies": recommended_movies}

if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
