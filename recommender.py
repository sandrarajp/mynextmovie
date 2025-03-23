import pandas as pd

# Load Datasets
try:
    movies = pd.read_csv('movies.csv')
except Exception as e:
    print("Error reading movies.csv:", e)


movies = pd.read_csv('movies.csv')  # Ensure this file is in the same directory
ratings = pd.read_csv('ratings.csv')  # Ensure this file is in the same directory

# -------------------------- Popularity-Based Recommender -------------------------- #
def popularity_based_recommender(genre, min_ratings, top_n):
    """
    Recommends top N popular movies in a specific genre.
    
    Parameters:
        genre (str): The genre to filter movies (e.g., 'Comedy').
        min_ratings (int): Minimum number of ratings required for a movie.
        top_n (int): Number of movies to recommend.

    Returns:
        DataFrame: Top N popular movies in the specified genre.
    """
    # Filter movies by genre
    genre_movies = movies[movies['genres'].str.contains(genre, case=False, na=False)]
    
    # Merge movies with ratings
    genre_ratings = pd.merge(genre_movies, ratings, on='movieId')
    
    # Calculate average rating and rating count for each movie
    movie_stats = genre_ratings.groupby('movieId').agg({
        'rating': ['mean', 'count']
    }).reset_index()
    movie_stats.columns = ['movieId', 'avg_rating', 'rating_count']
    
    # Filter movies by minimum ratings threshold
    popular_movies = movie_stats[movie_stats['rating_count'] >= min_ratings]
    
    # Sort movies by average rating in descending order
    popular_movies = popular_movies.sort_values(by='avg_rating', ascending=False)
    
    # Merge with movie titles
    result = pd.merge(popular_movies, movies[['movieId', 'title']], on='movieId')
    
    # Return top N movies
    return result[['title', 'avg_rating', 'rating_count']].head(top_n)

# -------------------------- Content-Based Recommender -------------------------- #
def content_based_recommender(movie_title, top_n):
    """
    Recommends top N movies based on similar genres to a given movie.
    
    Parameters:
        movie_title (str): The title of the reference movie.
        top_n (int): Number of movies to recommend.

    Returns:
        DataFrame: Top N similar movies based on genres.
    """
    # Get the genres of the input movie
    input_movie = movies[movies['title'].str.contains(movie_title, case=False, na=False)]
    if input_movie.empty:
        return f"Movie '{movie_title}' not found in the dataset."
    
    input_genres = input_movie.iloc[0]['genres']
    
    # Filter movies with matching genres
    similar_movies = movies[movies['genres'] == input_genres]
    
    # Exclude the input movie itself
    similar_movies = similar_movies[similar_movies['title'] != input_movie.iloc[0]['title']]
    
    # Return top N similar movies
    return similar_movies[['title', 'genres']].head(top_n)

# -------------------------- Main Program -------------------------- #
if __name__ == "__main__":
    print("Welcome to the MyNextMovie Recommender System!")
    
    # Example Usage: Popularity-Based Recommender
    genre = input("Enter a genre (e.g., Comedy, Action): ")
    min_ratings = int(input("Enter the minimum number of ratings: "))
    top_n = int(input("Enter the number of recommendations: "))
    
    print("\nPopularity-Based Recommendations:")
    popular_recommendations = popularity_based_recommender(genre, min_ratings, top_n)
    print(popular_recommendations)
    
    # Save Popular Recommendations to File
    popular_recommendations.to_csv('outputs/popular_recommendations.csv', index=False)
    
    # Example Usage: Content-Based Recommender
    movie_title = input("\nEnter a movie title to find similar movies: ")
    top_n = int(input("Enter the number of similar movie recommendations: "))
    
    print("\nContent-Based Recommendations:")
    similar_recommendations = content_based_recommender(movie_title, top_n)
    print(similar_recommendations)
    
    # Save Content-Based Recommendations to File
    if not isinstance(similar_recommendations, str):  # Only save if it's a DataFrame
        similar_recommendations.to_csv('outputs/similar_recommendations.csv', index=False)

