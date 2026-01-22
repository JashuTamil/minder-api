from data_processing import *
import pathlib
import pickle

features_path = pathlib.Path("movie_rec/user_data/features.pkl")
movies = load()

features, _, _ = build_features(movies)

with open(features_path, 'wb') as f:
    pickle.dump(features, f)


feedback = load_feedback()

user_vec, feedback = build_user_profile(features, movies, feedback)
recs = recommend_movies(features, movies, user_vec, feedback, top_n = 10)

print(recs)
