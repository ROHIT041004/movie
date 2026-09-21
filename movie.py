import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

# ==========================================================
# TITLE
# ==========================================================

st.title("🎬 Movie Recommendation System")
st.markdown(
    "### Collaborative Filtering using Cosine Similarity"
)

st.write(
    "Upload your MovieLens movie and ratings datasets "
    "to get personalized movie recommendations."
)

st.divider()


# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.header("📂 Upload Dataset")

movie_file = st.sidebar.file_uploader(
    "Upload movie.csv",
    type=["csv"]
)

ratings_file = st.sidebar.file_uploader(
    "Upload ratings.csv",
    type=["csv"]
)


# ==========================================================
# MAIN APPLICATION
# ==========================================================

if movie_file is None or ratings_file is None:

    st.info(
        "👈 Please upload both **movie.csv** and **ratings.csv** "
        "from the sidebar."
    )

    st.markdown("### Required columns")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**movie.csv**")
        st.code(
            "movieId\n"
            "title\n"
            "genres"
        )

    with col2:
        st.write("**ratings.csv**")
        st.code(
            "userId\n"
            "movieId\n"
            "rating\n"
            "timestamp"
        )

    st.stop()


# ==========================================================
# LOAD DATA
# ==========================================================

try:

    movies = pd.read_csv(movie_file)
    ratings = pd.read_csv(ratings_file)

except Exception as e:

    st.error(f"Error loading files: {e}")
    st.stop()


# ==========================================================
# CHECK REQUIRED COLUMNS
# ==========================================================

movie_columns = ["movieId", "title"]

rating_columns = [
    "userId",
    "movieId",
    "rating"
]

missing_movie_columns = [
    col for col in movie_columns
    if col not in movies.columns
]

missing_rating_columns = [
    col for col in rating_columns
    if col not in ratings.columns
]

if missing_movie_columns:

    st.error(
        "movie.csv is missing columns: "
        + ", ".join(missing_movie_columns)
    )

    st.stop()


if missing_rating_columns:

    st.error(
        "ratings.csv is missing columns: "
        + ", ".join(missing_rating_columns)
    )

    st.stop()


# ==========================================================
# DATA CLEANING
# ==========================================================

movies = movies.drop_duplicates()
ratings = ratings.drop_duplicates()

movies = movies.dropna(
    subset=["movieId", "title"]
)

ratings = ratings.dropna(
    subset=["userId", "movieId", "rating"]
)

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

movies = movies.dropna(
    subset=["movieId"]
)

ratings = ratings.dropna(
    subset=[
        "userId",
        "movieId",
        "rating"
    ]
)

movies["movieId"] = movies[
    "movieId"
].astype(int)

ratings["userId"] = ratings[
    "userId"
].astype(int)

ratings["movieId"] = ratings[
    "movieId"
].astype(int)

ratings = ratings[
    (ratings["rating"] >= 0.5) &
    (ratings["rating"] <= 5)
]


# ==========================================================
# MOVIE YEAR
# ==========================================================

movies["Year"] = movies["title"].str.extract(
    r"\((\d{4})\)"
)

movies["Year"] = pd.to_numeric(
    movies["Year"],
    errors="coerce"
)


# ==========================================================
# KPI SECTION
# ==========================================================

total_movies = movies["movieId"].nunique()
total_users = ratings["userId"].nunique()
total_ratings = len(ratings)
average_rating = ratings["rating"].mean()

st.subheader("📊 Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "🎬 Total Movies",
        f"{total_movies:,}"
    )

with col2:
    st.metric(
        "👥 Total Users",
        f"{total_users:,}"
    )

with col3:
    st.metric(
        "⭐ Total Ratings",
        f"{total_ratings:,}"
    )

with col4:
    st.metric(
        "⭐ Average Rating",
        f"{average_rating:.2f}"
    )


st.divider()


# ==========================================================
# DATA PREVIEW
# ==========================================================

with st.expander("🔍 View Dataset"):

    tab1, tab2 = st.tabs(
        ["Movies", "Ratings"]
    )

    with tab1:
        st.dataframe(
            movies.head(100),
            use_container_width=True
        )

    with tab2:
        st.dataframe(
            ratings.head(100),
            use_container_width=True
        )


# ==========================================================
# TOP MOVIES
# ==========================================================

st.subheader("🏆 Movie Analysis")

movie_stats = (
    ratings
    .groupby("movieId")
    .agg(
        Average_Rating=("rating", "mean"),
        Rating_Count=("rating", "count")
    )
    .reset_index()
)

movie_stats = movie_stats.merge(
    movies[
        ["movieId", "title"]
    ],
    on="movieId",
    how="left"
)

movie_stats = movie_stats.dropna(
    subset=["title"]
)


col1, col2 = st.columns(2)


with col1:

    st.write("### ⭐ Highest Rated Movies")

    highest_rated = movie_stats[
        movie_stats["Rating_Count"] >= 10
    ].sort_values(
        "Average_Rating",
        ascending=False
    ).head(10)

    st.dataframe(
        highest_rated[
            [
                "title",
                "Average_Rating",
                "Rating_Count"
            ]
        ].reset_index(drop=True),
        use_container_width=True
    )


with col2:

    st.write("### 🔥 Most Rated Movies")

    most_rated = movie_stats.sort_values(
        "Rating_Count",
        ascending=False
    ).head(10)

    st.dataframe(
        most_rated[
            [
                "title",
                "Rating_Count",
                "Average_Rating"
            ]
        ].reset_index(drop=True),
        use_container_width=True
    )


st.divider()


# ==========================================================
# USER-ITEM MATRIX
# ==========================================================

st.subheader(
    "🤖 Building Recommendation System"
)

with st.spinner(
    "Creating user-item matrix..."
):

    user_item_matrix = ratings.pivot_table(
        index="userId",
        columns="movieId",
        values="rating"
    )

    user_item_filled = user_item_matrix.fillna(0)


st.success(
    f"User-item matrix created: "
    f"{user_item_matrix.shape[0]} users × "
    f"{user_item_matrix.shape[1]} movies"
)


# ==========================================================
# ITEM SIMILARITY
# ==========================================================

with st.spinner(
    "Calculating movie similarity..."
):

    movie_item_matrix = user_item_filled.T

    movie_similarity = cosine_similarity(
        movie_item_matrix
    )

    movie_similarity_df = pd.DataFrame(
        movie_similarity,
        index=movie_item_matrix.index,
        columns=movie_item_matrix.index
    )

st.success(
    "✅ Movie similarity calculated using Cosine Similarity"
)


# ==========================================================
# RECOMMENDATION FUNCTION
# ==========================================================

def get_recommendations(
    user_id,
    number_of_recommendations
):

    user_ratings = ratings[
        ratings["userId"] == user_id
    ].sort_values(
        "rating",
        ascending=False
    )

    if user_ratings.empty:
        return pd.DataFrame()

    watched_movies = set(
        user_ratings["movieId"]
    )

    recommendation_scores = {}

    # Take user's highest-rated movies
    top_movies = user_ratings.head(10)

    for _, row in top_movies.iterrows():

        movie_id = int(row["movieId"])

        user_rating = float(
            row["rating"]
        )

        if movie_id not in movie_similarity_df.index:
            continue

        similar_movies = (
            movie_similarity_df[
                movie_id
            ]
            .drop(movie_id)
            .sort_values(
                ascending=False
            )
            .head(20)
        )

        for similar_movie_id, similarity in (
            similar_movies.items()
        ):

            similar_movie_id = int(
                similar_movie_id
            )

            # Don't recommend watched movies
            if similar_movie_id in watched_movies:
                continue

            score = (
                similarity *
                user_rating
            )

            if (
                similar_movie_id
                not in recommendation_scores
            ):

                recommendation_scores[
                    similar_movie_id
                ] = []

            recommendation_scores[
                similar_movie_id
            ].append(score)

    if not recommendation_scores:
        return pd.DataFrame()

    results = []

    for movie_id, scores in (
        recommendation_scores.items()
    ):

        predicted_rating = np.mean(scores)

        results.append(
            {
                "movieId": movie_id,
                "Predicted_Rating": predicted_rating
            }
        )

    recommendations = pd.DataFrame(
        results
    )

    recommendations = (
        recommendations
        .sort_values(
            "Predicted_Rating",
            ascending=False
        )
        .head(number_of_recommendations)
    )

    recommendations = recommendations.merge(
        movies[
            ["movieId", "title", "genres"]
        ],
        on="movieId",
        how="left"
    )

    recommendations["Predicted_Rating"] = (
        recommendations[
            "Predicted_Rating"
        ].clip(0, 5)
    )

    return recommendations[
        [
            "movieId",
            "title",
            "genres",
            "Predicted_Rating"
        ]
    ]


# ==========================================================
# RECOMMENDATION SECTION
# ==========================================================

st.divider()

st.subheader("🎯 Personalized Movie Recommendation")


user_list = sorted(
    ratings["userId"].unique()
)

selected_user = st.selectbox(
    "👤 Select User ID",
    user_list
)

number_of_movies = st.slider(
    "Number of Recommendations",
    min_value=1,
    max_value=10,
    value=5
)


if st.button(
    "🎬 Get Recommendations",
    type="primary"
):

    with st.spinner(
        "Finding movies for you..."
    ):

        recommendations = get_recommendations(
            selected_user,
            number_of_movies
        )

    if recommendations.empty:

        st.warning(
            "No recommendations found for this user."
        )

    else:

        st.success(
            f"Found {len(recommendations)} recommendations!"
        )

        st.dataframe(
            recommendations,
            use_container_width=True,
            hide_index=True
        )


# ==========================================================
# USER RATING HISTORY
# ==========================================================

st.divider()

st.subheader("📜 User Rating History")

history = ratings[
    ratings["userId"] == selected_user
].merge(
    movies[
        ["movieId", "title", "genres"]
    ],
    on="movieId",
    how="left"
)

history = history.sort_values(
    "rating",
    ascending=False
)

st.dataframe(
    history[
        [
            "movieId",
            "title",
            "genres",
            "rating"
        ]
    ].head(20),
    use_container_width=True,
    hide_index=True
)


# ==========================================================
# DOWNLOAD RECOMMENDATIONS
# ==========================================================

if "recommendations" in locals():

    if not recommendations.empty:

        csv_data = recommendations.to_csv(
            index=False
        )

        st.download_button(
            label="⬇️ Download Recommendations CSV",
            data=csv_data,
            file_name="Top_5_Recommended_Movies.csv",
            mime="text/csv"
        )


# ==========================================================
# FOOTER
# ==========================================================

st.divider()

st.caption(
    "🎬 Movie Recommendation System | "
    "Collaborative Filtering + Cosine Similarity"
)
