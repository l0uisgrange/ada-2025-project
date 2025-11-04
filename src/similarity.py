"""
Similarity analysis utilities for finding related subreddits.

This module provides functions for computing similarity between subreddits
using their embedding vectors (centroid-based and max-similarity methods).
"""

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def similar_by_centroid(picked_subreddits, embeddings, top_n=20):
    """
    Find similar subreddits based on centroid of picked subreddits.
    
    All subreddits are equally weighted regardless of their proximity to each other.
    
    Args:
        picked_subreddits (list of str): List of subreddit names to compute centroid from
        embeddings (pd.DataFrame): DataFrame with 'SUBREDDIT' column and embedding columns
        top_n (int): Number of top similar subreddits to return (default: 10)
        
    Returns:
        pd.DataFrame
            Top N similar subreddits with columns: SUBREDDIT, SIMILARITY, ORIGINAL
    """
    # Isolate embeddings of picked subreddits
    isolate_subreddits = embeddings[embeddings['SUBREDDIT'].isin(picked_subreddits)]
    
    if len(isolate_subreddits) == 0:
        raise ValueError("None of the picked subreddits found in embeddings")
    
    # Compute centroid (mean of all picked subreddit embeddings)
    centroids = isolate_subreddits.drop(columns=['SUBREDDIT']).mean(axis=0).values
    
    # Compute cosine similarity between centroid and all subreddits
    similarities = cosine_similarity(
        embeddings.drop(columns=['SUBREDDIT']), 
        centroids.reshape(1, -1)
    )
    
    # Create results DataFrame
    results = pd.DataFrame({
        'SUBREDDIT': embeddings['SUBREDDIT'],
        'SIMILARITY': similarities.flatten(),
        'ORIGINAL': embeddings['SUBREDDIT'].isin(picked_subreddits)
    })
    
    # Remove picked subreddits from results
    results = results[~results['SUBREDDIT'].isin(picked_subreddits)]
    
    # Sort by similarity and return top N
    results = results.sort_values(by='SIMILARITY', ascending=False).head(top_n)
    
    return results


def similar_by_max_similarity(picked_subreddits, embeddings, top_n=20):
    """
    Find similar subreddits based on maximum similarity to any picked subreddit.
    
    For each subreddit, finds its closest match among the picked subreddits.
    Useful when picked subreddits cover different sub-topics.
    
    Args:
        picked_subreddits (list of str): List of seed subreddit names
        embeddings (pd.DataFrame): DataFrame with 'SUBREDDIT' column and embedding columns
        top_n (int): Number of top similar subreddits to return (default: 10)
        
    Returns:
        pd.DataFrame
            Top N similar subreddits with columns: subreddit, max_similarity, closest_seed, is_seed
    """
    # Isolate embeddings of picked subreddits
    isolate_subreddits = embeddings[embeddings['SUBREDDIT'].isin(picked_subreddits)]
    
    if len(isolate_subreddits) == 0:
        raise ValueError("None of the picked subreddits found in embeddings")
    
    # Compute cosine similarity matrix
    similarities = cosine_similarity(
        isolate_subreddits.drop(columns=['SUBREDDIT']), 
        embeddings.drop(columns=['SUBREDDIT'])
    )
    
    # Create DataFrame with similarities to each seed
    similarity_df = pd.DataFrame(
        similarities.T,
        columns=picked_subreddits
    )
    
    # Get max similarity for each subreddit
    results = pd.DataFrame({
        'SUBREDDIT': embeddings['SUBREDDIT'],
        'MAX_SIMILARITY': similarity_df.max(axis=1),
        'CLOSEST_SEED': similarity_df.idxmax(axis=1),
        'IS_SEED': embeddings['SUBREDDIT'].isin(picked_subreddits)
    })
    
    # Remove seed subreddits and sort
    results = results[~results['IS_SEED']]
    results = results.sort_values(by='MAX_SIMILARITY', ascending=False).head(top_n)
    
    return results




