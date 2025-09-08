import streamlit as st
import pandas as pd
import pickle

# Load dataset
data = pd.read_csv("movie_recommendation/movies.csv")

# Load cosine similarity matrix
with open("movie_recommendation/cosine_sim.pkl", "rb") as f:
    cosine_sim = pickle.load(f)


#  Recommendation function

def recommend_movies(movie_name, cosine_sim=cosine_sim, df=data, top_n=5):
    # Find index of the movie
    idx = df[df["original_title"].str.lower() == movie_name.lower()].index
    if len(idx) == 0:
        return None
    idx = idx[0]

    # Get similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda i: i[1], reverse=True)
    sim_scores = sim_scores[1: top_n + 1]

    # Get movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return top N recommended movies
    return df["original_title"].iloc[movie_indices].tolist()



#app UI

st.title("Movie Recommendation System")

movie_input = st.selectbox("Choose a movie you like:", data['original_title'].tolist())

if st.button("Recommend"):
    recommendations = recommend_movies(movie_input)
    if recommendations:
        st.write("✅ We recommend:")
        for movie in recommendations:
            st.write(f"- {movie}")
    else:
        st.write("Movie not found in the database.")
