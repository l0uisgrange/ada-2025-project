"""
Statistical metrics and evaluation utilities.

This module provides functions for calculating clustering metrics,
activity statistics, and performance evaluations.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score


def calculate_clustering_metrics(embeddings, labels):
    """
    Calculate clustering quality metrics.
    
    Parameters:
    -----------
    embeddings : np.ndarray
        Embedding vectors
    labels : np.ndarray
        Cluster assignments
        
    Returns:
    --------
    dict
        Dictionary with silhouette_score and davies_bouldin_score
    """
    metrics = {}
    
    if len(set(labels)) > 1:  # Need at least 2 clusters
        metrics['silhouette_score'] = silhouette_score(embeddings, labels)
        metrics['davies_bouldin_score'] = davies_bouldin_score(embeddings, labels)
    else:
        metrics['silhouette_score'] = None
        metrics['davies_bouldin_score'] = None
    
    return metrics


def get_top_subreddits_per_cluster(viz_df, total_counts, top_n=5):
    """
    Get top active subreddits for each cluster.
    
    Parameters:
    -----------
    viz_df : pd.DataFrame
        Visualization DataFrame with 'cluster' and 'subreddit' columns
    total_counts : pd.DataFrame
        Activity counts with 'SUBREDDIT' and 'POST_COUNT_TOTAL' columns
    top_n : int, optional
        Number of top subreddits per cluster (default: 5)
        
    Returns:
    --------
    dict
        Dictionary {cluster_id: top_subreddits_df}
    """
    n_clusters = viz_df['cluster'].nunique()
    cluster_tops = {}
    
    for cluster_id in range(n_clusters):
        # Get subreddits in this cluster
        cluster_subreddits = viz_df[viz_df['cluster'] == cluster_id]['subreddit'].tolist()
        
        # Filter and sort by activity
        cluster_activity = total_counts[
            total_counts['SUBREDDIT'].isin(cluster_subreddits)
        ].sort_values(by='POST_COUNT_TOTAL', ascending=False).head(top_n)
        
        cluster_tops[cluster_id] = cluster_activity
    
    return cluster_tops


def calculate_activity_stats(total_counts):
    """
    Calculate summary statistics for subreddit activity.
    
    Parameters:
    -----------
    total_counts : pd.DataFrame
        Activity counts DataFrame
        
    Returns:
    --------
    dict
        Dictionary with mean, median, std, min, max statistics
    """
    stats = {
        'mean_total': total_counts['POST_COUNT_TOTAL'].mean(),
        'median_total': total_counts['POST_COUNT_TOTAL'].median(),
        'std_total': total_counts['POST_COUNT_TOTAL'].std(),
        'min_total': total_counts['POST_COUNT_TOTAL'].min(),
        'max_total': total_counts['POST_COUNT_TOTAL'].max(),
        'total_subreddits': len(total_counts)
    }
    
    if 'POST_COUNT_SOURCE' in total_counts.columns:
        stats['mean_source'] = total_counts['POST_COUNT_SOURCE'].mean()
        stats['mean_target'] = total_counts['POST_COUNT_TARGET'].mean()
    
    return stats


def get_cluster_size_distribution(viz_df):
    """
    Calculate size distribution of clusters.
    
    Parameters:
    -----------
    viz_df : pd.DataFrame
        Visualization DataFrame with 'cluster' column
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with cluster_id, size, percentage columns
    """
    cluster_sizes = viz_df['cluster'].value_counts().sort_index()
    total_items = len(viz_df)
    
    distribution = pd.DataFrame({
        'cluster_id': cluster_sizes.index,
        'size': cluster_sizes.values,
        'percentage': (cluster_sizes.values / total_items * 100).round(2)
    })
    
    return distribution


def compare_cluster_metrics(viz_df, total_counts):
    """
    Compare average activity metrics across clusters.
    
    Parameters:
    -----------
    viz_df : pd.DataFrame
        Visualization DataFrame with 'cluster' and 'subreddit' columns
    total_counts : pd.DataFrame
        Activity counts DataFrame
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with cluster-level aggregated metrics
    """
    # Merge cluster info with activity counts
    merged = viz_df.merge(total_counts, left_on='subreddit', right_on='SUBREDDIT', how='left')
    
    # Aggregate by cluster
    cluster_metrics = merged.groupby('cluster').agg({
        'POST_COUNT_TOTAL': ['mean', 'median', 'std', 'sum'],
        'POST_COUNT_SOURCE': ['mean', 'sum'],
        'POST_COUNT_TARGET': ['mean', 'sum'],
        'subreddit': 'count'  # Number of subreddits in cluster
    }).round(2)
    
    cluster_metrics.columns = ['_'.join(col).strip() for col in cluster_metrics.columns.values]
    cluster_metrics = cluster_metrics.rename(columns={'subreddit_count': 'num_subreddits'})
    
    return cluster_metrics


def find_outlier_subreddits(total_counts, threshold_std=3.0):
    """
    Identify outlier subreddits with unusually high activity.
    
    Parameters:
    -----------
    total_counts : pd.DataFrame
        Activity counts DataFrame
    threshold_std : float, optional
        Number of standard deviations to consider outlier (default: 3.0)
        
    Returns:
    --------
    pd.DataFrame
        Outlier subreddits with their z-scores
    """
    mean_activity = total_counts['POST_COUNT_TOTAL'].mean()
    std_activity = total_counts['POST_COUNT_TOTAL'].std()
    
    if std_activity == 0:
        return pd.DataFrame()  # No variation
    
    total_counts = total_counts.copy()
    total_counts['zscore'] = (total_counts['POST_COUNT_TOTAL'] - mean_activity) / std_activity
    
    outliers = total_counts[
        total_counts['zscore'] > threshold_std
    ].sort_values(by='zscore', ascending=False)
    
    return outliers[['SUBREDDIT', 'POST_COUNT_TOTAL', 'zscore']]