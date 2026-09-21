# ============================================================
# MOVIE RECOMMENDATION SYSTEM
# Collaborative Filtering + Cosine Similarity
# ============================================================

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error


# ============================================================
# 1. LOAD DATA
# ============================================================

print("Loading data...")

movies = pd.read_csv("movie.csv")
ratings = pd.read_csv("ratings.csv")

print("\nMovies:")
print(movies.head())

print("\nRatings:")
print(ratings.head())


# ============================================================
# 2. CHECK DATA
# ============================================================

print("\n========== DATA INFORMATION ==========")

print("Movies shape:", movies.shape)
print("Ratings shape:", ratings.shape)

print("\nMovie columns:")
print(movies.columns.tolist())

print("\nRating columns:")
print(ratings.columns.tolist())


# ============================================================
# 3. DATA CLEANING
# ============================================================

print("\n========== DATA CLEANING ==========")

# Remove duplicate rows
movies = movies.drop_duplicates()
ratings = ratings.drop_duplicates()

# Remove missing important values
movies = movies.dropna(subset=["movieId", "title"])

ratings = ratings.dropna(
    subset=["userId", "movieId", "rating"]
)

# Convert data types
movies["movieId"] = pd.to_numeric(
    movies["movieId"],
    errors="coerce"
)

ratings["userId"] = pd.to_numeric(
    ratings["userId"],
    errors="coerce"
)

ratings["movieId"] = pd.to_numeric(
    ratings["movieId"],
    errors="coerce"
)

ratings["rating"] = pd.to_numeric(
    ratings["rating"],
    errors="coerce"
)

# Remove invalid values
movies = movies.dropna(subset=["movieId"])
ratings = ratings.dropna(
    subset=["userId", "movieId", "rating"]
)

movies["movieId"] = movies["movieId"].astype(int)
ratings["userId"] = ratings["userId"].astype(int)
ratings["movieId"] = ratings["movieId"].astype(int)

# Keep ratings between 0.5 and 5
ratings = ratings[
    (ratings["rating"] >= 0.5) &
    (ratings["rating"] <= 5)
]

print("Clean movie data:", movies.shape)
print("Clean rating data:", ratings.shape)


# ============================================================
# 4. EXTRACT MOVIE YEAR
# ============================================================

print("\n========== MOVIE YEAR ==========")

movies["Year"] = movies["title"].str.extract(
    r"\((\d{4})\)"
)

movies["Year"] = pd.to_numeric(
    movies["Year"],
    errors="coerce"
)

print(
    movies[["title", "Year"]].head(10)
)


# ============================================================
# 5. BASIC EDA
# ============================================================

print("\n========== EDA ==========")

total_movies = movies["movieId"].nunique()
total_users = ratings["userId"].nunique()
total_ratings = len(ratings)
average_rating = ratings["rating"].mean()

print("Total Movies:", total_movies)
print("Total Users:", total_users)
print("Total Ratings:", total_ratings)
print(
    "Average Rating:",
    round(average_rating, 2)
)


# ============================================================
# 6. MOST RATED MOVIES
# ============================================================

movie_rating_count = (
    ratings.groupby("movieId")
    .size()
    .reset_index(name="Rating_Count")
)

most_rated = movie_rating_count.merge(
    movies[["movieId", "title"]],
    on="movieId",
    how="left"
)

most_rated = most_rated.sort_values(
    "Rating_Count",
    ascending=False
)

print("\n========== TOP 10 MOST RATED MOVIES ==========")

print(
    most_rated[
        ["title", "Rating_Count"]
    ].head(10).to_string(index=False)
)


# ============================================================
# 7. HIGHEST RATED MOVIES
# ============================================================

movie_rating_stats = (
    ratings.groupby("movieId")
    .agg(
        Average_Rating=("rating", "mean"),
        Rating_Count=("rating", "count")
    )
    .reset_index()
)

# Minimum 10 ratings
highest_rated = movie_rating_stats[
    movie_rating_stats["Rating_Count"] >= 10
]

highest_rated = highest_rated.sort_values(
    "Average_Rating",
    ascending=False
)

highest_rated = highest_rated.merge(
    movies[["movieId", "title"]],
    on="movieId",
    how="left"
)

print("\n========== TOP 10 HIGHEST RATED MOVIES ==========")

print(
    highest_rated[
        [
            "title",
            "Average_Rating",
            "Rating_Count"
        ]
    ].head(10).to_string(index=False)
)


# ============================================================
# 8. CREATE USER-ITEM MATRIX
# ============================================================

print("\n========== USER-ITEM MATRIX ==========")

user_item_matrix = ratings.pivot_table(
    index="userId",
    columns="movieId",
    values="rating"
)

print(
    "User-item matrix:",
    user_item_matrix.shape
)

# Replace missing values with 0
user_item_matrix_filled = user_item_matrix.fillna(0)


# ============================================================
# 9. USER-BASED COLLABORATIVE FILTERING
# ============================================================

print("\n========== USER SIMILARITY ==========")

user_similarity = cosine_similarity(
    user_item_matrix_filled
)

user_similarity_df = pd.DataFrame(
    user_similarity,
    index=user_item_matrix_filled.index,
    columns=user_item_matrix_filled.index
)

print(
    "User similarity matrix:",
    user_similarity_df.shape
)


# ============================================================
# 10. USER-BASED RECOMMENDATION FUNCTION
# ============================================================

def user_based_recommendation(
    user_id,
    number_of_recommendations=5
):

    if user_id not in user_item_matrix_filled.index:
        print("User ID not found.")
        return pd.DataFrame()

    # Find similar users
    similar_users = (
        user_similarity_df.loc[user_id]
        .drop(user_id)
        .sort_values(ascending=False)
        .head(10)
    )

    # Movies already watched
    watched_movies = set(
        ratings[
            ratings["userId"] == user_id
        ]["movieId"]
    )

    recommendation_scores = {}

    for similar_user, similarity_score in similar_users.items():

        if similarity_score <= 0:
            continue

        similar_user_movies = ratings[
            ratings["userId"] == similar_user
        ]

        for _, row in similar_user_movies.iterrows():

            movie_id = int(row["movieId"])

            # Don't recommend watched movies
            if movie_id in watched_movies:
                continue

            rating = float(row["rating"])

            score = rating * similarity_score

            if movie_id not in recommendation_scores:
                recommendation_scores[movie_id] = []

            recommendation_scores[movie_id].append(score)

    if len(recommendation_scores) == 0:
        return pd.DataFrame()

    # Calculate predicted rating
    recommendations = []

    for movie_id, scores in recommendation_scores.items():

        predicted_rating = np.mean(scores)

        recommendations.append(
            [
                movie_id,
                predicted_rating
            ]
        )

    recommendations = pd.DataFrame(
        recommendations,
        columns=[
            "movieId",
            "Predicted_Rating"
        ]
    )

    recommendations = recommendations.sort_values(
        "Predicted_Rating",
        ascending=False
    )

    recommendations = recommendations.head(
        number_of_recommendations
    )

    # Add movie title
    recommendations = recommendations.merge(
        movies[["movieId", "title"]],
        on="movieId",
        how="left"
    )

    return recommendations[
        [
            "movieId",
            "title",
            "Predicted_Rating"
        ]
    ]


# ============================================================
# 11. ITEM-BASED COLLABORATIVE FILTERING
# ============================================================

print("\n========== ITEM SIMILARITY ==========")

# Transpose matrix
movie_item_matrix = user_item_matrix_filled.T

# Calculate movie-to-movie similarity
movie_similarity = cosine_similarity(
    movie_item_matrix
)

movie_similarity_df = pd.DataFrame(
    movie_similarity,
    index=movie_item_matrix.index,
    columns=movie_item_matrix.index
)

print(
    "Movie similarity matrix:",
    movie_similarity_df.shape
)


# ============================================================
# 12. ITEM-BASED RECOMMENDATION FUNCTION
# ============================================================

def item_based_recommendation(
    user_id,
    number_of_recommendations=5
):

    if user_id not in ratings["userId"].values:
        print("User ID not found.")
        return pd.DataFrame()

    # Get user's ratings
    user_ratings = ratings[
        ratings["userId"] == user_id
    ].sort_values(
        "rating",
        ascending=False
    )

    # Movies already watched
    watched_movies = set(
        user_ratings["movieId"]
    )

    recommendation_scores = {}

    # Use user's top rated movies
    top_movies = user_ratings.head(10)

    for _, row in top_movies.iterrows():

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

            # Don't recommend already watched movies
            if similar_movie_id in watched_movies:
                continue

            score = similarity * user_rating

            if similar_movie_id not in recommendation_scores:
                recommendation_scores[
                    similar_movie_id
                ] = []

            recommendation_scores[
                similar_movie_id
            ].append(score)

    if len(recommendation_scores) == 0:
        return pd.DataFrame()

    recommendations = []

    for movie_id, scores in recommendation_scores.items():

        predicted_rating = np.mean(scores)

        recommendations.append(
            [
                movie_id,
                predicted_rating
            ]
        )

    recommendations = pd.DataFrame(
        recommendations,
        columns=[
            "movieId",
            "Predicted_Rating"
        ]
    )

    recommendations = recommendations.sort_values(
        "Predicted_Rating",
        ascending=False
    )

    recommendations = recommendations.head(
        number_of_recommendations
    )

    recommendations = recommendations.merge(
        movies[["movieId", "title"]],
        on="movieId",
        how="left"
    )

    return recommendations[
        [
            "movieId",
            "title",
            "Predicted_Rating"
        ]
    ]


# ============================================================
# 13. TEST USER RECOMMENDATION
# ============================================================

print("\n========== TEST RECOMMENDATION ==========")

# Take first available user
test_user = ratings["userId"].iloc[0]

print("Testing User ID:", test_user)

recommendations = item_based_recommendation(
    test_user,
    5
)

print("\nTop 5 Recommended Movies:")

print(
    recommendations.to_string(index=False)
)


# ============================================================
# 14. USER-BASED TEST
# ============================================================

print("\n========== USER-BASED RECOMMENDATION ==========")

user_recommendations = user_based_recommendation(
    test_user,
    5
)

print(
    user_recommendations.to_string(index=False)
)


# ============================================================
# 15. CREATE RECOMMENDATIONS FOR MULTIPLE USERS
# ============================================================

print("\n========== CREATING RECOMMENDATION FILE ==========")

all_recommendations = []

# First 20 users
users_to_process = (
    ratings["userId"]
    .drop_duplicates()
    .head(20)
)

for user_id in users_to_process:

    recs = item_based_recommendation(
        user_id,
        5
    )

    if recs.empty:
        continue

    recs.insert(
        0,
        "userId",
        user_id
    )

    all_recommendations.append(
        recs
    )


if len(all_recommendations) > 0:

    final_recommendations = pd.concat(
        all_recommendations,
        ignore_index=True
    )

    final_recommendations.to_csv(
        "Top_5_Recommended_Movies.csv",
        index=False
    )

    print(
        "\nFile created:"
        " Top_5_Recommended_Movies.csv"
    )

    print("\nSample:")
    print(
        final_recommendations.head(20)
        .to_string(index=False)
    )

else:

    print("No recommendations generated.")


# ============================================================
# 16. BASELINE RMSE
# ============================================================

print("\n========== RMSE ==========")

# Simple baseline:
# Predict every rating using average rating

actual_ratings = ratings["rating"]

average_rating = ratings["rating"].mean()

predicted_ratings = np.full(
    len(actual_ratings),
    average_rating
)

rmse = np.sqrt(
    mean_squared_error(
        actual_ratings,
        predicted_ratings
    )
)

print(
    "Baseline RMSE:",
    round(rmse, 4)
)


# ============================================================
# 17. FINAL OUTPUT
# ============================================================

print("\n==========================================")
print("MOVIE RECOMMENDATION SYSTEM COMPLETED")
print("==========================================")

print("\nFiles required:")
print("1. movie.csv")
print("2. ratings.csv")

print("\nOutput file:")
print("Top_5_Recommended_Movies.csv")

print("\nMethod used:")
print("User-Based Collaborative Filtering")
print("Item-Based Collaborative Filtering")
print("Cosine Similarity")

print("\nYou can import")
print("Top_5_Recommended_Movies.csv")
print("into Power BI.")
