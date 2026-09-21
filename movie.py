from pathlib import Path

code = r'''# movie.py
# Movie Recommendation System using Collaborative Filtering
# Requirements:
#   pip install pandas numpy scikit-learn matplotlib

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error


# ============================================================
# 1. FILE PATHS
# ============================================================

MOVIES_FILE = "movie.csv"
RATINGS_FILE = "ratings.csv"

RECOMMENDATION_FILE = "Top_5_Recommended_Movies.csv"


# ============================================================
# 2. LOAD DATA
# ============================================================

def load_data():
    if not os.path.exists(MOVIES_FILE):
        raise FileNotFoundError(
            f"{MOVIES_FILE} not found. Put movie.csv in the same folder as movie.py."
        )

    if not os.path.exists(RATINGS_FILE):
        raise FileNotFoundError(
            f"{RATINGS_FILE} not found. Put ratings.csv in the same folder as movie.py."
        )

    movies = pd.read_csv(MOVIES_FILE)
    ratings = pd.read_csv(RATINGS_FILE)

    print("\n========== DATA LOADED ==========")
    print("Movies shape:", movies.shape)
    print("Ratings shape:", ratings.shape)

    return movies, ratings


# ============================================================
# 3. DATA CLEANING
# ============================================================

def clean_data(movies, ratings):

    print("\n========== DATA CLEANING ==========")

    # Remove duplicate records
    movies = movies.drop_duplicates()
    ratings = ratings.drop_duplicates()

    # Remove rows with important missing values
    movies = movies.dropna(subset=["movieId", "title"])
    ratings = ratings.dropna(subset=["userId", "movieId", "rating"])

    # Convert IDs to integers
    movies["movieId"] = pd.to_numeric(
        movies["movieId"], errors="coerce"
    )

    ratings["userId"] = pd.to_numeric(
        ratings["userId"], errors="coerce"
    )

    ratings["movieId"] = pd.to_numeric(
        ratings["movieId"], errors="coerce"
    )

    ratings["rating"] = pd.to_numeric(
        ratings["rating"], errors="coerce"
    )

    movies = movies.dropna(subset=["movieId"])
    ratings = ratings.dropna(
        subset=["userId", "movieId", "rating"]
    )

    movies["movieId"] = movies["movieId"].astype(int)
    ratings["userId"] = ratings["userId"].astype(int)
    ratings["movieId"] = ratings["movieId"].astype(int)

    # Keep valid ratings
    ratings = ratings[
        (ratings["rating"] >= 0.5) &
        (ratings["rating"] <= 5)
    ]

    # Remove duplicate movie IDs
    movies = movies.drop_duplicates(
        subset=["movieId"],
        keep="first"
    )

    print("Clean Movies:", movies.shape)
    print("Clean Ratings:", ratings.shape)

    return movies, ratings


# ============================================================
# 4. EXTRACT YEAR FROM MOVIE TITLE
# ============================================================

def extract_year(movies):

    movies["Year"] = movies["title"].str.extract(
        r"\((\d{4})\)"
    )

    movies["Year"] = pd.to_numeric(
        movies["Year"],
        errors="coerce"
    )

    return movies


# ============================================================
# 5. BASIC EDA
# ============================================================

def perform_eda(movies, ratings):

    print("\n========== EDA ==========")

    total_movies = movies["movieId"].nunique()
    total_users = ratings["userId"].nunique()
    total_ratings = len(ratings)
    average_rating = ratings["rating"].mean()

    print("Total Movies :", total_movies)
    print("Total Users  :", total_users)
    print("Total Ratings:", total_ratings)
    print("Average Rating:", round(average_rating, 2))

    # Most rated movies
    rating_counts = (
        ratings.groupby("movieId")
        .size()
        .reset_index(name="Rating_Count")
    )

    most_rated = (
        rating_counts
        .merge(
            movies[["movieId", "title"]],
            on="movieId",
            how="left"
        )
        .sort_values(
            "Rating_Count",
            ascending=False
        )
        .head(10)
    )

    print("\n========== TOP 10 MOST RATED MOVIES ==========")
    print(
        most_rated[
            ["title", "Rating_Count"]
        ].to_string(index=False)
    )

    # Highest rated movies with minimum number of ratings
    movie_stats = (
        ratings.groupby("movieId")
        .agg(
            Average_Rating=("rating", "mean"),
            Rating_Count=("rating", "count")
        )
        .reset_index()
    )

    # Minimum 10 ratings to avoid movies with only one rating
    highest_rated = movie_stats[
        movie_stats["Rating_Count"] >= 10
    ].sort_values(
        ["Average_Rating", "Rating_Count"],
        ascending=[False, False]
    ).head(10)

    highest_rated = highest_rated.merge(
        movies[["movieId", "title"]],
        on="movieId",
        how="left"
    )

    print("\n========== TOP 10 HIGHEST RATED MOVIES ==========")
    print(
        highest_rated[
            ["title", "Average_Rating", "Rating_Count"]
        ].to_string(index=False)
    )

    # Rating distribution
    plt.figure(figsize=(8, 5))
    ratings["rating"].value_counts().sort_index().plot(
        kind="bar"
    )
    plt.title("Rating Distribution")
    plt.xlabel("Rating")
    plt.ylabel("Number of Ratings")
    plt.tight_layout()
    plt.show()

    return movie_stats


# ============================================================
# 6. CREATE USER-ITEM MATRIX
# ============================================================

def create_user_item_matrix(ratings):

    print("\n========== USER-ITEM MATRIX ==========")

    user_item_matrix = ratings.pivot_table(
        index="userId",
        columns="movieId",
        values="rating"
    )

    print(
        "Original matrix shape:",
        user_item_matrix.shape
    )

    # Missing ratings are represented by 0
    user_item_filled = user_item_matrix.fillna(0)

    return user_item_matrix, user_item_filled


# ============================================================
# 7. USER-BASED COLLABORATIVE FILTERING
# ============================================================

def create_user_similarity(user_item_filled):

    print("\n========== USER SIMILARITY ==========")

    user_similarity = cosine_similarity(
        user_item_filled
    )

    user_similarity_df = pd.DataFrame(
        user_similarity,
        index=user_item_filled.index,
        columns=user_item_filled.index
    )

    return user_similarity_df


def user_based_recommendations(
    user_id,
    ratings,
    movies,
    user_item_filled,
    user_similarity_df,
    n=5
):

    if user_id not in user_item_filled.index:
        return pd.DataFrame()

    # Similar users
    similar_users = (
        user_similarity_df.loc[user_id]
        .drop(user_id)
        .sort_values(ascending=False)
    )

    # Select top 10 similar users
    similar_users = similar_users.head(10)

    watched_movies = set(
        ratings.loc[
            ratings["userId"] == user_id,
            "movieId"
        ]
    )

    recommendation_scores = {}

    for similar_user, similarity_score in similar_users.items():

        if similarity_score <= 0:
            continue

        similar_user_ratings = ratings[
            ratings["userId"] == similar_user
        ]

        for _, row in similar_user_ratings.iterrows():

            movie_id = int(row["movieId"])
            rating = float(row["rating"])

            # Do not recommend already watched movies
            if movie_id in watched_movies:
                continue

            if movie_id not in recommendation_scores:
                recommendation_scores[movie_id] = []

            recommendation_scores[movie_id].append(
                rating * similarity_score
            )

    if not recommendation_scores:
        return pd.DataFrame()

    recommendations = []

    for movie_id, scores in recommendation_scores.items():

        predicted_rating = sum(scores) / len(scores)

        recommendations.append(
            {
                "movieId": movie_id,
                "Predicted_Rating": predicted_rating
            }
        )

    recommendations = pd.DataFrame(
        recommendations
    )

    recommendations = recommendations.sort_values(
        "Predicted_Rating",
        ascending=False
    ).head(n)

    recommendations = recommendations.merge(
        movies[["movieId", "title"]],
        on="movieId",
        how="left"
    )

    recommendations = recommendations[
        [
            "movieId",
            "title",
            "Predicted_Rating"
        ]
    ]

    return recommendations


# ============================================================
# 8. ITEM-BASED COLLABORATIVE FILTERING
# ============================================================

def create_movie_similarity(user_item_filled):

    print("\n========== ITEM-BASED SIMILARITY ==========")

    # Movies are rows after transpose
    movie_item_matrix = user_item_filled.T

    # Cosine similarity between movies
    movie_similarity = cosine_similarity(
        movie_item_matrix
    )

    movie_similarity_df = pd.DataFrame(
        movie_similarity,
        index=movie_item_matrix.index,
        columns=movie_item_matrix.index
    )

    return movie_similarity_df


def item_based_recommendations(
    user_id,
    ratings,
    movies,
    movie_similarity_df,
    n=5
):

    if user_id not in ratings["userId"].unique():
        return pd.DataFrame()

    user_ratings = ratings[
        ratings["userId"] == user_id
    ].sort_values(
        "rating",
        ascending=False
    )

    watched_movies = set(
        user_ratings["movieId"]
    )

    scores = {}

    for _, row in user_ratings.head(10).iterrows():

        movie_id = int(row["movieId"])
        user_rating = float(row["rating"])

        if movie_id not in movie_similarity_df.index:
            continue

        similar_movies = (
            movie_similarity_df[movie_id]
            .drop(movie_id)
            .sort_values(ascending=False)
            .head(20)
        )

        for similar_movie_id, similarity in similar_movies.items():

            similar_movie_id = int(similar_movie_id)

            # Exclude movies already watched
            if similar_movie_id in watched_movies:
                continue

            if similar_movie_id not in scores:
                scores[similar_movie_id] = []

            scores[similar_movie_id].append(
                similarity * user_rating
            )

    if not scores:
        return pd.DataFrame()

    recommendations = []

    for movie_id, values in scores.items():

        predicted_rating = sum(values) / len(values)

        recommendations.append(
            {
                "movieId": movie_id,
                "Predicted_Rating": predicted_rating
            }
        )

    recommendations = pd.DataFrame(
        recommendations
    )

    recommendations = recommendations.sort_values(
        "Predicted_Rating",
        ascending=False
    ).head(n)

    recommendations = recommendations.merge(
        movies[["movieId", "title"]],
        on="movieId",
        how="left"
    )

    recommendations = recommendations[
        [
            "movieId",
            "title",
            "Predicted_Rating"
        ]
    ]

    return recommendations


# ============================================================
# 9. GENERATE RECOMMENDATIONS FOR USERS
# ============================================================

def generate_recommendations(
    movies,
    ratings,
    user_item_filled,
    user_similarity_df,
    movie_similarity_df
):

    print("\n========== RECOMMENDATIONS ==========")

    # Select users who have ratings
    user_ids = ratings["userId"].unique()

    all_recommendations = []

    # Generate recommendations for maximum 20 users
    for user_id in user_ids[:20]:

        user_recs = item_based_recommendations(
            user_id,
            ratings,
            movies,
            movie_similarity_df,
            n=5
        )

        if user_recs.empty:
            # Try user-based recommendations
            user_recs = user_based_recommendations(
                user_id,
                ratings,
                movies,
                user_item_filled,
                user_similarity_df,
                n=5
            )

        if user_recs.empty:
            continue

        user_recs.insert(
            0,
            "userId",
            user_id
        )

        all_recommendations.append(
            user_recs
        )

    if not all_recommendations:
        print("No recommendations generated.")
        return pd.DataFrame()

    recommendation_df = pd.concat(
        all_recommendations,
        ignore_index=True
    )

    recommendation_df.to_csv(
        RECOMMENDATION_FILE,
        index=False
    )

    print(
        f"\nRecommendations saved to: "
        f"{RECOMMENDATION_FILE}"
    )

    print("\n========== SAMPLE RECOMMENDATIONS ==========")
    print(
        recommendation_df.head(20).to_string(
            index=False
        )
    )

    return recommendation_df


# ============================================================
# 10. RMSE EVALUATION
# ============================================================

def calculate_rmse(ratings):

    print("\n========== RMSE EVALUATION ==========")

    # Simple baseline prediction:
    # predict every rating using the global average rating.
    actual = ratings["rating"].values

    predicted = np.full(
        len(actual),
        ratings["rating"].mean()
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    print(
        "Baseline RMSE:",
        round(rmse, 4)
    )

    print(
        "\nNote: This is a baseline RMSE. "
        "For model-specific RMSE, create train/test "
        "predictions from the recommendation algorithm."
    )

    return rmse


# ============================================================
# 11. RECOMMEND MOVIES FOR ONE USER
# ============================================================

def recommend_for_user(
    user_id,
    ratings,
    movies,
    movie_similarity_df,
    n=5
):

    recommendations = item_based_recommendations(
        user_id,
        ratings,
        movies,
        movie_similarity_df,
        n=n
    )

    if recommendations.empty:
        print(
            f"No recommendation available for User {user_id}"
        )
    else:
        print(
            f"\nTop {n} recommendations for User {user_id}:"
        )
        print(
            recommendations.to_string(index=False)
        )

    return recommendations


# ============================================================
# 12. MAIN PROGRAM
# ============================================================

def main():

    print("=" * 60)
    print("       MOVIE RECOMMENDATION SYSTEM")
    print("       Collaborative Filtering")
    print("=" * 60)

    # Load
    movies, ratings = load_data()

    # Clean
    movies, ratings = clean_data(
        movies,
        ratings
    )

    # Extract year
    movies = extract_year(movies)

    # EDA
    perform_eda(
        movies,
        ratings
    )

    # User-item matrix
    user_item_matrix, user_item_filled = (
        create_user_item_matrix(ratings)
    )

    # User-based similarity
    user_similarity_df = create_user_similarity(
        user_item_filled
    )

    # Item-based similarity
    movie_similarity_df = create_movie_similarity(
        user_item_filled
    )

    # Generate recommendations
    recommendation_df = generate_recommendations(
        movies,
        ratings,
        user_item_filled,
        user_similarity_df,
        movie_similarity_df
    )

    # RMSE baseline
    calculate_rmse(ratings)

    # Example recommendation for first available user
    first_user = ratings["userId"].iloc[0]

    print("\n========== SINGLE USER TEST ==========")

    recommend_for_user(
        first_user,
        ratings,
        movies,
        movie_similarity_df,
        n=5
    )

    print("\n" + "=" * 60)
    print("PROJECT COMPLETED")
    print("=" * 60)
    print(
        "Power BI can import:",
        RECOMMENDATION_FILE
    )


if __name__ == "__main__":
    main()
'''

path = "/mnt/data/movie.py"
Path(path).write_text(code, encoding="utf-8")
print(path)
