from data_processing import *
import pathlib
import pickle

features_path = pathlib.Path("movie_rec/user_data/features.pkl")
movies = load()

if not features_path.exists():
    features, tfidf, mlb = build_features(movies)
    with open(features_path, 'wb') as f:
        pickle.dump(features, f)
else:
    with open(features_path, 'rb') as f:
        features = pickle.load(f)

feedback = load_feedback()

if len(feedback['likes']) == 0:
    feedback['likes'] = [441717, 334541, 324857, 335984, 1091]
    feedback['dislikes'] = [1428301, 1153399, 758323, 868759, 719221]

user_vec = build_user_profile(features, movies, feedback)
recs = recommend_movies(features, movies, user_vec, feedback, top_n = 10)

print(recs)
