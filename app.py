import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time

# Load the Bollywood movie dataset
@st.cache_data  # Updated caching method
def load_data():
    try:
        file_path = 'datasets/IMDB-Movie-Dataset(2023-1951).csv'  # Update with the correct file path
        movies_df = pd.read_csv(file_path)
        print(movies_df.columns)  # Print the column names
        print(movies_df.head())  # Print the first few rows
        movies_df['combined_features'] = movies_df['genre'] + ' ' + movies_df['director'] + ' ' + movies_df['cast']
        return movies_df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return pd.DataFrame()

movies_df = load_data()

# Check if 'movie_name' column exists
if 'movie_name' not in movies_df.columns:
    st.error("The 'movie_name' column does not exist in the DataFrame.")
else:
    # Content-Based Filtering: Cosine Similarity on combined features
    @st.cache_data
    def calculate_cosine_similarity(df):
        try:
            tfidf = TfidfVectorizer(stop_words='english')
            tfidf_matrix = tfidf.fit_transform(df['combined_features'])
            cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
            return cosine_sim
        except Exception as e:
            st.error(f"Error calculating cosine similarity: {e}")
            return None

    cosine_sim = calculate_cosine_similarity(movies_df)

    # Cosine Similarity based on the cast (actors only)
    @st.cache_data
    def calculate_actor_similarity(df):
        try:
            actor_tfidf = TfidfVectorizer(stop_words='english')
            actor_tfidf_matrix = actor_tfidf.fit_transform(df['cast'])
            actor_cosine_sim = cosine_similarity(actor_tfidf_matrix, actor_tfidf_matrix)
            return actor_cosine_sim
        except Exception as e:
            st.error(f"Error calculating actor similarity: {e}")
            return None

    actor_cosine_sim = calculate_actor_similarity(movies_df)

    # Get recommendations based on content similarity
    def get_content_based_recommendations(title, top_n=5):
        try:
            idx = movies_df[movies_df['movie_name'] == title].index[0]
            sim_scores = list(enumerate(cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
            movie_indices = [i[0] for i in sim_scores]
            return movies_df['movie_name'].iloc[movie_indices].values.tolist()
        except Exception as e:
            st.error(f"Error fetching content-based recommendations: {e}")
            return []

    # Get recommendations based on actor similarity
    def get_actor_based_recommendations(title, top_n=5):
        try:
            idx = movies_df[movies_df['movie_name'] == title].index[0]
            actor_sim_scores = list(enumerate(actor_cosine_sim[idx]))
            actor_sim_scores = sorted(actor_sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
            movie_indices = [i[0] for i in actor_sim_scores]
            return movies_df['movie_name'].iloc[movie_indices].values.tolist()
        except Exception as e:
            st.error(f"Error fetching actor-based recommendations: {e}")
            return []

    # Streamlit frontend layout with enhanced UI
    def app_layout():
        st.markdown(
            """
            <style>
            .main {
                background-color: #0e1117;
                color: white;
            }
            .stButton button {
                background-color: #0d6efd;
                color: white;
                font-weight: bold;
            }
            .stButton button :hover {
                background-color: #0b5ed7;
                color: white;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        st.title("Movie Recommendation System")
        st.write("This system recommends movies based on content similarity and actor similarity.")

        # User input: movie title
        movie_list = movies_df['movie_name'].tolist()
        selected_movie = st.selectbox("Choose a movie", movie_list)

        # Show recommendations on button click
        if st.button("Get Recommendations"):
            with st.spinner("Loading recommendations..."):  # Add a loading spinner
                time.sleep(5)  # Simulate a 5-second delay
                content_based_recommendations = get_content_based_recommendations(selected_movie)
                actor_based_recommendations = get_actor_based_recommendations(selected_movie)

            col1, col2 = st.columns (2)  # Create two columns
            with col1:
                st.write("Content-based Recommendations:")
                for movie in content_based_recommendations:
                    st.write(movie)
            with col2:
                st.write("Actor-based Recommendations:")
                for movie in actor_based_recommendations:
                    st.write(movie)

    if __name__ == '__main__':
        app_layout()
