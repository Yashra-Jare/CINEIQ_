
import streamlit as st
import requests
from visuals import plot_genre_radar, plot_decade_preferences, plot_actor_director_affinities

# Configure the page
st.set_page_config(page_title="CINEIQ Dashboard", layout="wide")

st.title("🎬 CINEIQ: Explainable Recommendation Engine")
st.markdown("Discover movies tailored to your taste, with full transparency on *why* they were chosen.")

# --- SIDEBAR: User Context ---
st.sidebar.header("👤 User Context")
st.sidebar.markdown(
    "**Why do we need an ID?**\n\n"
    "To provide *Collaborative* recommendations, the ML engine needs to analyze past rating history. "
    "Enter a User ID below to simulate logging in as different users from the MovieLens dataset."
)
user_id = st.sidebar.number_input("Simulate User ID:", min_value=1, value=11, step=1)
st.sidebar.info("Try switching between User 11, User 42, etc., to see how the recommendations change based on different taste profiles!")

# Create layout tabs
tab1, tab2 = st.tabs(["🍿 Discover Movies", "📊 My Taste Profile"])

# --- TAB 1: Recommendations ---
with tab1:
    st.subheader("Get Personalized Recommendations")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        target_movie = st.text_input("Enter a movie you recently loved:", "The Dark Knight", 
                                     help="The engine will find movies with similar content and then re-rank them based on your specific taste.")
    with col2:
        top_n = st.number_input("Number of Results:", min_value=1, max_value=10, value=5,
                                help="How many recommendations should the API return?")
        
    if st.button("Generate Explainable Recommendations", type="primary"):
        with st.spinner(f"Analyzing ML Ensemble and NLP Sentiment for '{target_movie}'..."):
            
            api_url = "http://127.0.0.1:8000/recommend"
            payload = {"user_id": user_id, "movie_title": target_movie, "top_n": top_n}
            
            try:
                response = requests.post(api_url, json=payload)
                response.raise_for_status()
                data = response.json()
                
                st.success("Recommendations Generated!")
                for i, item in enumerate(data, 1):
                    with st.expander(f"**{i}. {item['movie']}**", expanded=True):
                        st.write(f"💡 **Why we recommend this:** {item['explanation']}")
                        
            except requests.exceptions.ConnectionError:
                st.error("Error: Could not connect to the API. Make sure your FastAPI server is running in another terminal!")

# --- TAB 2: Taste Profile (Visuals) ---
# --- TAB 2: Taste Profile (Visuals) ---
with tab2:
    st.subheader("Your Evolving Taste Profile")
    st.markdown(f"Visualizing the rating history and ML-derived preferences for **User {user_id}**.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.plotly_chart(plot_genre_radar(user_id), use_container_width=True)
    with col2:
        st.plotly_chart(plot_decade_preferences(user_id), use_container_width=True)
    with col3:
        st.plotly_chart(plot_actor_director_affinities(user_id), use_container_width=True)