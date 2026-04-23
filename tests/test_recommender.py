import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import pandas as pd
import numpy as np
from app import cosine_similarity_np, load_data

def test_cosine_similarity_np():
    a = np.array([1.0, 0.0])
    b = np.array([[1.0, 0.0], [0.0, 1.0]])
    sim = cosine_similarity_np(a, b)
    assert np.isclose(sim[0], 1.0)
    assert np.isclose(sim[1], 0.0)

def test_load_data():
    df, embeddings = load_data()
    assert isinstance(df, pd.DataFrame)
    assert isinstance(embeddings, np.ndarray)
    assert len(df) == len(embeddings)
    assert 'embedding' in df.columns

def test_recommendation_flow():
    from app import movies_df, embeddings, movie_titles
    first_movie_id = list(movie_titles.keys())[0]
    idx = movies_df[movies_df['movieId'] == first_movie_id].index[0]
    query_embedding = embeddings[idx]
    similarities = cosine_similarity_np(query_embedding, embeddings)
    sim_scores = list(enumerate(similarities))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = [s for s in sim_scores if s[0] != idx][:2]
    assert len(sim_scores) == 2
    for i, _ in sim_scores:
        title = movies_df.iloc[i]['title']
        assert isinstance(title, str)
