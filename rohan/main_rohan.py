import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MultiLabelBinarizer
from flask import Flask,request,jsonify


data = pd.read_csv('../data/TMDB_cleaned.csv')
app = Flask(__name__)

def split_genres(genre_entry):
    if isinstance(genre_entry, str):  # Only apply split if it's a string
        return genre_entry.split(', ')  # Split by comma and space
    return genre_entry  # Return the list as-is if it's already a list
data['genres'] = data['genres'].apply(split_genres)

mlb = MultiLabelBinarizer()
genres_encoded = mlb.fit_transform(data['genres'])
knn_model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=5)
# Fit the model based on the one-hot encoded genres
knn_model.fit(genres_encoded)

def recommend_movies_based_on_title(title, data, knn_model, n_recommendations=5):
    # Get the index of the movie that matches the title
    try:
        movie_index = data[data['title'] == title].index[0]
    except IndexError:
        return []
    
    # Get the genre vector for that movie
    genre_vector = genres_encoded[movie_index].reshape(1, -1)
    
    # Find the nearest neighbors (most similar movies)
    distances, indices = knn_model.kneighbors(genre_vector, n_neighbors=n_recommendations + 1)
    
    # Get the recommended movie indices (excluding the first, which is the movie itself)
    recommended_movie_indices = indices.flatten()[1:]

    # Prepare detailed information for each recommendation
    recommended_movies = []
    for idx in recommended_movie_indices:
        movie_details = {
            "id": data.at[idx, 'id'],
            "title": data.at[idx, 'title'],
            "adult": data.at[idx, 'adult'],
            "backdrop_path": data.at[idx, 'backdrop_path'],
            "imdb_id": data.at[idx, 'imdb_id'],
            "overview": data.at[idx, 'overview'],
            "genres": data.at[idx, 'genres'],
            "production_companies": data.at[idx, 'production_companies']
        }
        recommended_movies.append(movie_details)
    
    return recommended_movies

def recommend_movies_based_on_genre(genre, data, knn_model, n_recommendations=5):
    # Check if the genre exists in the dataset
    if genre not in mlb.classes_:
        return []
    
    # Create a genre vector for the given genre
    genre_vector = np.zeros(len(mlb.classes_))
    genre_vector[mlb.classes_.tolist().index(genre)] = 1
    genre_vector = genre_vector.reshape(1, -1)
    
    # Find the nearest neighbors (most similar movies)
    distances, indices = knn_model.kneighbors(genre_vector, n_neighbors=n_recommendations)
    
    # Get the recommended movie indices
    recommended_movie_indices = indices.flatten()

    # Prepare detailed information for each recommendation
    recommended_movies = []
    for idx in recommended_movie_indices:
        movie_details = {
            "id": data.at[idx, 'id'],
            "title": data.at[idx, 'title'],
            "adult": data.at[idx, 'adult'],
            "backdrop_path": data.at[idx, 'backdrop_path'],
            "imdb_id": data.at[idx, 'imdb_id'],
            "overview": data.at[idx, 'overview'],
            "genres": data.at[idx, 'genres'],
            "production_companies": data.at[idx, 'production_companies']
        }
        recommended_movies.append(movie_details)
    
    return recommended_movies

def recommend_movies_based_on_title_modded(title, data, knn_model, n_recommendations=5):
    # Get the index of the movie that matches the title
    try:
        movie_index = data[data['title'] == title].index[0]
    except IndexError:
        return []
    
    # Get the genre vector for that movie
    genre_vector = genres_encoded[movie_index].reshape(1, -1)
    
    # Find the nearest neighbors (most similar movies)
    distances, indices = knn_model.kneighbors(genre_vector, n_neighbors=n_recommendations + 1)
    
    # Get the recommended movie indices (excluding the first, which is the movie itself)
    recommended_movie_indices = indices.flatten()[1:]

    # Convert to list to avoid JSON serialization issues
    return data['title'].iloc[recommended_movie_indices].tolist()

def recommend_movies_based_on_genre_modded(genre, data, knn_model, n_recommendations=5):
    # Check if the genre exists in the dataset
    if genre not in mlb.classes_:
        return []
    
    # Create a genre vector for the given genre
    genre_vector = np.zeros(len(mlb.classes_))
    genre_vector[mlb.classes_.tolist().index(genre)] = 1
    genre_vector = genre_vector.reshape(1, -1)
    
    # Find the nearest neighbors (most similar movies)
    distances, indices = knn_model.kneighbors(genre_vector, n_neighbors=n_recommendations)
    
    # Get the recommended movie indices
    recommended_movie_indices = indices.flatten()

    # Convert to list to avoid JSON serialization issues
    return data['title'].iloc[recommended_movie_indices].tolist()


@app.route('/recommend_by_title', methods=['GET'])
def recommend_by_title():
    title = request.args.get('title')
    if not title:
        return jsonify({'error': 'Title is required'}), 400
    recommendations = recommend_movies_based_on_title_modded(title, data, knn_model)
    return jsonify({'recommendations': recommendations})

@app.route('/recommend_by_genre', methods=['GET'])
def recommend_by_genre():
    genre = request.args.get('genre')
    if not genre:
        return jsonify({'error': 'Genre is required'}), 400
    recommendations = recommend_movies_based_on_genre_modded(genre, data, knn_model)
    return jsonify({'recommendations': recommendations})

if __name__ == '__main__':

    # message = "Enter 1 to get recommendations based on movie title, 2 to get recommendations based on genre, or 0 to exit: "
    # while(True):
    #     print(message)
    #     choice = input()
    #     if choice == '0':
    #         break
    #     elif choice == '1':
    #         title = input("Enter the movie title: ")
    #         recommendations = recommend_movies_based_on_title(title, data, knn_model)
    #         print("Recommended movies based on title:")
    #         print(recommendations)
    #         print(type(recommendations))
    #     elif choice == '2':
    #         genre = input("Enter the genre: ")
    #         recommendations = recommend_movies_based_on_genre(genre, data, knn_model)
    #         print("Recommended movies based on genre:")
    #         print(recommendations)
    #     else:
    #         print("Invalid choice. Please try again.")
    app.run(debug=True)