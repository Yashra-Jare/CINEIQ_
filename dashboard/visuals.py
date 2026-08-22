import ast
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ML-engine"))
from data_loader import load_links, load_movie_metadata, load_movielens_ratings


def _parse_list(value):
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            return parsed if isinstance(parsed, list) else []
        except (ValueError, SyntaxError):
            return []
    return []


def _user_movies(user_id):
    ratings = load_movielens_ratings()
    links = load_links()
    movies = load_movie_metadata()
    if any(x is None for x in (ratings, links, movies)):
        return pd.DataFrame()

    user = ratings[ratings["userId"] == int(user_id)].copy()
    if user.empty:
        return pd.DataFrame()
    links = links[["movieId", "tmdbId"]].drop_duplicates("movieId")
    user = user.merge(links, on="movieId", how="inner")
    movies = movies.copy()
    movies["id"] = pd.to_numeric(movies["id"], errors="coerce")
    movies = movies.dropna(subset=["id"])
    movies["tmdbId"] = movies["id"].astype(int)
    return user.merge(movies, on="tmdbId", how="inner")


def plot_genre_radar(user_id):
    df = _user_movies(user_id)
    genres = []
    ratings = []
    for _, row in df.iterrows():
        for item in _parse_list(row.get("genres")):
            if isinstance(item, dict) and "name" in item:
                genres.append(item["name"])
                ratings.append(float(row["rating"]))
    if not genres:
        return go.Figure().update_layout(title=f"Genre Affinity Radar (User {user_id})", annotations=[dict(text="No rating data available", showarrow=False)])
    stats = pd.DataFrame({"Genre": genres, "Rating": ratings}).groupby("Genre")["Rating"].mean().sort_values(ascending=False).head(8)
    polar = pd.DataFrame({"r": stats.values, "theta": stats.index})
    fig = px.line_polar(polar, r="r", theta="theta", line_close=True, title=f"Genre Affinity Radar (User {user_id})")
    fig.update_traces(fill="toself")
    fig.update_layout(yaxis_range=[0, 5])
    return fig


def plot_decade_preferences(user_id):
    df = _user_movies(user_id)
    if df.empty:
        return go.Figure().update_layout(title=f"Decade Preferences (User {user_id})", annotations=[dict(text="No rating data available", showarrow=False)])
    df["year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year
    df = df.dropna(subset=["year"])
    df["Decade"] = (df["year"] // 10 * 10).astype(int).astype(str) + "s"
    stats = df.groupby("Decade")["rating"].mean().reset_index().sort_values("Decade")
    fig = px.bar(stats, x="Decade", y="rating", title=f"Decade Preferences (User {user_id})", labels={"rating": "Average Rating"})
    fig.update_yaxes(range=[0, 5])
    return fig


def plot_actor_director_affinities(user_id):
    df = _user_movies(user_id)
    if df.empty:
        return go.Figure().update_layout(title=f"Top Actors & Directors (User {user_id})", annotations=[dict(text="No rating data available", showarrow=False)])
    scores = {}
    for _, row in df.iterrows():
        rating = float(row["rating"])
        for person in _parse_list(row.get("cast"))[:5]:
            if isinstance(person, dict) and person.get("name"):
                scores[person["name"]] = scores.get(person["name"], []) + [rating]
        for person in _parse_list(row.get("crew")):
            if isinstance(person, dict) and person.get("job") == "Director" and person.get("name"):
                name = person["name"] + " (Director)"
                scores[name] = scores.get(name, []) + [rating]
    stats = pd.DataFrame([(k, sum(v) / len(v), len(v)) for k, v in scores.items()], columns=["Person", "Affinity Score", "Count"])
    stats = stats[stats["Count"] >= 1].nlargest(10, "Affinity Score").sort_values("Affinity Score")
    if stats.empty:
        return go.Figure().update_layout(title=f"Top Actors & Directors (User {user_id})", annotations=[dict(text="No cast/director data available", showarrow=False)])
    fig = px.bar(stats, x="Affinity Score", y="Person", orientation="h", title=f"Top Actors & Directors (User {user_id})", labels={"Affinity Score": "Average Rating"})
    fig.update_xaxes(range=[0, 5])
    return fig
