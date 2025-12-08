import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sp
import numpy as np
import json
import pathlib
import requests

feedback_path = pathlib.Path("movie_rec/data/feedback.json")
extras = ['genres', 'vote_count']

params = {
    "api_key": "fc2e3ee1697f621f0af000955dfb1dc1"
}

def load():
    df = pd.read_csv("movie_rec/data/TMDB_movie_dataset_v11.csv")
    df = df[df['vote_count'] >= 10]
    df = df[['id', 'title', 'genres', 'overview', 'vote_average', 'vote_count', 'runtime', 'poster_path', 'release_date']].fillna('')
    df['genres'] = df['genres'].apply(lambda x: x.split(", "))
    return df

def build_features(movies):
    mlb = MultiLabelBinarizer()
    genre_features = mlb.fit_transform(movies['genres'])

    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    overview_features = tfidf.fit_transform(movies['overview'])
    
    C = movies['vote_average'].mean()
    m = movies['vote_count'].quantile(0.5)
    movies['weighted_rating'] = (
        (movies['vote_count'] / (movies['vote_count'] + m)) * movies['vote_average'] + 
        (m / (movies['vote_count'] + m)) * C
    )

    rating_normalized = (movies['weighted_rating'] - movies['weighted_rating'].min()) / \
                       (movies['weighted_rating'].max() - movies['weighted_rating'].min())
    rating_features = sp.csr_matrix(rating_normalized.values.reshape(-1, 1))

    features = sp.hstack([genre_features, overview_features, rating_features])
    return features, tfidf, mlb

def load_feedback():
    if not feedback_path.exists():
        return {"likes": [], "dislikes": []}
    with open(feedback_path, "r") as f:
        return json.load(f)

def save_feedback(feedback):
    with open(feedback_path, "w") as f:
        json.dump(feedback, f)


def build_user_profile(features, movies, feedback, alpha=0.5):
    liked = [movie for movie in feedback['likes']]
    disliked = [movie for movie in feedback['dislikes']]

    features_csr = features.tocsr()

    liked_ind = movies[movies['id'].isin(liked)].index if liked else np.asarray([])
    disliked_ind = movies[movies['id'].isin(disliked)].index if disliked else np.asarray([])
 

    f_like = np.asarray(features_csr[liked_ind].mean(axis=0)) if liked_ind.any() else 0
    f_dislike = np.asarray(features_csr[disliked_ind].mean(axis=0)) if disliked_ind.any() else 0

    if isinstance(f_like, int) and f_like == 0:
        return None

    user_vector = f_like - (alpha * f_dislike)
    return user_vector


def exploratory_rec(movies, feedback, top_n):
    seen = set(feedback['likes'] + feedback['dislikes'])
    available_movies = movies[~movies['id'].isin(seen)]

    user_liked_movies = movies[movies['id'].isin(feedback['likes'])]
    user_genres = set()
    for genres_list in user_liked_movies['genres']:
        user_genres.update(genres_list)
    
    unexplored_movies = available_movies[available_movies['genres'].apply(lambda x: len(set(x) - user_genres) > 0)]

    if len(unexplored_movies) > 0:
        return unexplored_movies.sort_values('score', ascending=False)[:top_n]
    else:
        return available_movies.sort_values('score', ascending=False)[:top_n]
    
def exploitative_rec(movies, feedback, top_n):
    seen = set(feedback['likes'] + feedback['dislikes'])
    return movies[~movies['id'].isin(seen)].sort_values('score', ascending=False)[:top_n]

def recommend_movies(features, movies, user_vector, feedback, exploration_rate = .4, top_n = 10):
    if user_vector is None:
        movies['score'] = movies['weighted_rating']
    else:
        sims = cosine_similarity(features, user_vector)
        movies['score'] = sims.flatten()

    if np.random.random() < exploration_rate:
        recs = exploratory_rec(movies, feedback, top_n)
        recs['recommendation_type'] = 'exploration'
    else:
        recs = exploitative_rec(movies, feedback, top_n)
        recs['recommendation_type'] = 'exploitation'
    
    recs['poster_path'] = "https://image.tmdb.org/t/p/w500" + recs['poster_path']
    
    cast = []
    director = []
    goThru = recs['id'].values

    for i in goThru:
        response = requests.get(f"https://api.themoviedb.org/3/movie/{i}/credits?language=en-US", params=params)
        data = response.json()
        
        cast_names = [actor["name"] for actor in data.get("cast", [])[:5]]
        cast.append(", ".join(cast_names) if cast_names else "N/A")

        # Get director(s)
        directors = [crew["name"] for crew in data.get("crew", []) 
                    if crew["job"] == "Director" and crew['department'] == "Directing"]
        director.append(", ".join(directors) if directors else "N/A")
        
    print(recs.head())
    recs["cast"] = cast
    recs["director"] = director
    
    result = recs[['id', 'title', 'overview', 'vote_average', 'runtime', 'poster_path', 'cast', 'director', 'release_date']].to_dict(orient='records')

    return result


def router_function():
    features_path = pathlib.Path("movie_rec/user_data/features.pkl")
    movies = load()

    C = movies['vote_average'].mean()
    m = movies['vote_count'].quantile(0.5)
    movies['weighted_rating'] = (
        (movies['vote_count'] / (movies['vote_count'] + m)) * movies['vote_average'] + 
        (m / (movies['vote_count'] + m)) * C
    )

    with open(features_path, 'rb') as f:
        features = pickle.load(f)


    feedback = load_feedback()
    user_vec = build_user_profile(features, movies, feedback)
    return recommend_movies(features, movies, user_vec, feedback, top_n = 10)
