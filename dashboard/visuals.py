import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def plot_genre_radar(user_id):
    """Generates a radar chart of the user's favorite genres."""
    # Simulating genre affinities based on user_id
    np.random.seed(user_id) 
    genres = ['Sci-Fi', 'Action', 'Drama', 'Comedy', 'Thriller', 'Horror']
    scores = np.random.uniform(2.0, 5.0, size=len(genres))
    
    df = pd.DataFrame(dict(r=scores, theta=genres))
    
    fig = px.line_polar(df, r='r', theta='theta', line_close=True, 
                        title=f"Genre Affinity Radar (User {user_id})")
    fig.update_traces(fill='toself')
    return fig

def plot_decade_preferences(user_id):
    """Generates a bar chart of the user's preferred movie decades."""
    np.random.seed(user_id + 1)
    decades = ['1980s', '1990s', '2000s', '2010s', '2020s']
    ratings = np.random.uniform(3.0, 4.8, size=len(decades))
    
    df = pd.DataFrame({'Decade': decades, 'Average Rating': ratings})
    
    fig = px.bar(df, x='Decade', y='Average Rating', 
                 title=f"Decade Preferences (User {user_id})",
                 color='Average Rating', color_continuous_scale='Viridis')
    fig.update_yaxes(range=[0, 5])
    return fig
    
def plot_actor_director_affinities(user_id):
    """Generates a bar chart of the user's top actors/directors."""
    np.random.seed(user_id + 2)
    # Simulating data that would normally be aggregated from the user's highly rated movies
    people = ['Christopher Nolan', 'Leonardo DiCaprio', 'Hans Zimmer', 'Quentin Tarantino', 'Brad Pitt']
    scores = np.random.uniform(3.5, 5.0, size=len(people))
    
    df = pd.DataFrame({'Person': people, 'Affinity Score': scores})
    df = df.sort_values('Affinity Score', ascending=True)
    
    fig = px.bar(df, x='Affinity Score', y='Person', orientation='h',
                 title=f"Top Actors & Directors (User {user_id})",
                 color='Affinity Score', color_continuous_scale='Plasma')
    fig.update_xaxes(range=[0, 5])
    return fig