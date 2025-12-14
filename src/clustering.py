import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from src import consts

def reduce_dimensions_pca(embeddings, n_components=50, random_state=42):
    """
    Reduce embedding dimensions using PCA.
    
    Parameters:
    -----------
    embeddings : pd.DataFrame or np.ndarray
        Embedding vectors (without SUBREDDIT column if DataFrame)
    n_components : int, optional
        Number of principal components (default: 50)
    random_state : int, optional
        Random seed for reproducibility (default: 42)
        
    Returns:
    --------
    tuple of (np.ndarray, PCA)
        - Reduced embeddings
        - Fitted PCA object
    """
    # Normalize embeddings
    embeddings_norm = normalize(embeddings, norm='l2')
    
    # Apply PCA
    pca = PCA(n_components=n_components, random_state=random_state)
    embeddings_reduced = pca.fit_transform(embeddings_norm)
    
    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_.sum():.2%}")
    
    return embeddings_reduced, pca


def perform_kmeans_clustering(embeddings, n_clusters=consts.OPTIMAL_K, n_init=50, max_iter=1000, random_state=42):
    """
    Perform K-Means clustering on embeddings.
    
    Args:
        embeddings (np.ndarray): Embedding vectors
        n_clusters (int): Number of clusters
        n_init (int): Number of times K-Means runs with different centroids (default: 50)
        max_iter (int): Maximum iterations per run (default: 1000)
        random_state (int): Random seed (default: 42)
        
    Returns:
        tuple of (np.ndarray, KMeans)
            - Cluster labels
            - Fitted KMeans object
    """
    kmeans = KMeans(n_clusters=n_clusters, n_init=n_init, 
                   max_iter=max_iter, random_state=random_state)
    labels = kmeans.fit_predict(embeddings)
    
    return labels, kmeans


def evaluate_clustering_range(embeddings, k_range=None,
                              n_init=50, max_iter=1000, random_state=42):
    """
    Test K-Means with different K values and evaluate metrics.
    
    Args: 
        embeddings (np.ndarray): Embedding vectors
        k_range (list of int): List of K values to test
        n_init (int): Number of K-Means initializations (default: 50)
        max_iter (int): Max iterations (default: 1000)
        random_state (int): Random seed (default: 42)
        
    Returns:
        pd.DataFrame
            Results with columns: n_clusters, silhouette_score, davies_bouldin_score, labels
    """
    if k_range is None:
        k_range = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

    results = []
    
    for k in k_range:
        labels, _ = perform_kmeans_clustering(
            embeddings, n_clusters=k, n_init=n_init, 
            max_iter=max_iter, random_state=random_state
        )
        
        if len(set(labels)) > 1:  # Only calculate if we have multiple clusters
            sil_score = silhouette_score(embeddings, labels)
            db_score = davies_bouldin_score(embeddings, labels)
            
            results.append({
                'n_clusters': k,
                'silhouette_score': sil_score,
                'davies_bouldin_score': db_score,
                'labels': labels
            })
            
            #print(f"K={k:2d} | Silhouette: {sil_score:.3f} | Davies-Bouldin: {db_score:.3f}")
    
    return pd.DataFrame(results)


def reduce_to_2d_tsne(embeddings, random_state=42, perplexity=30, verbose=True):
    """
    Reduce embeddings to 2D using t-SNE for visualization.
    
    Args:
        embeddings (np.ndarray): High-dimensional embeddings
        random_state (int): Random seed (default: 42)
        perplexity (int): t-SNE perplexity parameter (default: 30)
        verbose (bool): Print progress messages (default: True)
        
    Returns:
        np.ndarray
            2D embeddings (n_samples, 2)
    """
    if verbose:
        print(f"Reducing {embeddings.shape[1]}D embeddings to 2D using t-SNE...")
        print("This may take a few minutes...")
    
    tsne = TSNE(n_components=2, random_state=random_state, perplexity=perplexity)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    return embeddings_2d


def create_cluster_visualization_df(embeddings_2d, subreddit_names, cluster_labels):
    """
    Create a DataFrame for cluster visualization.
    
    Args:
        embeddings_2d (np.ndarray): 2D embeddings (from t-SNE)
        subreddit_names (pd.Series or list): Subreddit names
        cluster_labels (np.ndarray): Cluster assignments
        
    Returns:
        pd.DataFrame
            DataFrame with columns: subreddit, x, y, cluster
    """
    viz_df = pd.DataFrame({
        'subreddit': subreddit_names,
        'x': embeddings_2d[:, 0],
        'y': embeddings_2d[:, 1],
        'cluster': cluster_labels
    })
    
    return viz_df


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


def get_cluster_colors(n_clusters):
    """
    Generate distinct colors for clusters.
    
    Args:
        n_clusters (int): Number of clusters
        
    Returns:
        np.ndarray Array of RGBA colors (n_clusters, 4)
    """
    
    colors = np.vstack((
        plt.cm.get_cmap('tab20')(np.linspace(0, 1, 20)),
        plt.cm.get_cmap('tab20b')(np.linspace(0, 1, 20)),
        plt.cm.get_cmap('tab20c')(np.linspace(0, 1, 20))
    ))
    
    if n_clusters > len(colors):
        # Fallback to continuous colormap for >60 clusters
        colors = plt.cm.get_cmap('nipy_spectral')(np.linspace(0, 1, n_clusters))
    else:
        colors = colors[:n_clusters]
    
    return colors

def similarity_subreddits(df_subreddits, top_n = consts.SIMILARITY_N_CLUSTERS):
    """
    Filters subreddits with

    Args:
        df_subreddits (DataFrame): the embeddings dataset
        top_n (int): Similarity clustering parameter

    Returns:
        df_gaming, df_politics, df_sports
    """

    # Dictionary to hold merged dataframes
    merged_dfs = {}

    # Merge with the neighbors found (centroid)
    for theme, seeds in consts.THEMES_SEEDS.items():
        max_results = similar_by_max_similarity(
            seeds,
            df_subreddits,
            top_n=top_n
        )

        # Put all names in a list to filter duplicates
        merged_list = list(set(seeds + max_results['SUBREDDIT'].tolist()))
        merged_df = df_subreddits[df_subreddits['SUBREDDIT'].isin(merged_list)].copy()
        merged_dfs[theme] = merged_df

    return merged_dfs.get('politics'), merged_dfs.get('sports')