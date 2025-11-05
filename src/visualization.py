"""
Visualization utilities for Reddit data analysis.

This module provides reusable plotting functions for clusters, temporal trends,
and network visualizations.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
from scipy.spatial import ConvexHull

from .clustering import get_cluster_colors
from .consts import OPTIMAL_K


def plot_mentions_over_time(counts_df, pattern, period='M', figsize=None):
    """
    Plot stacked bar chart of subreddit mentions over time.
    
    Args:
        counts_df (pd.DataFrame):
        DataFrame with 'Title Mentions' and 'Body Mentions' columns
        pattern (str):
        Subreddit name being analyzed
        period (str)
        Time period ('M' for monthly, 'W' for weekly) (default: 'M')
        save_path (str):
        Path to save figure (default: None, no save)
        figsize (tuple):
        Figure size (width, height) (default: auto-detect based on period)
        
    Returns:
        matplotlib.figure.Figure
            The generated figure
    """
    # Auto-detect figure size
    if figsize is None:
        figsize = (14, 6) if period == 'M' else (24, 8)

    counts_df.plot(
        kind="bar",
        stacked=True,
        figsize=figsize,
        color=["#66c2a5", "#fc8d62"],
        alpha=0.85
    )    
    plt.title(f"Stacked Mentions of '{pattern}' Over Time", 
             fontsize=14, fontweight='bold')
    plt.xlabel('Time Period')
    plt.ylabel('Number of Mentions')
    plt.xticks(rotation=90)
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()
    return 

def plot_cluster_overview(viz_df, n_clusters=OPTIMAL_K, save_path=None):
    """
    Plot all clusters in a single 2D scatter plot.
    
    Args:
        viz_df (pd.DataFrame): Visualization DataFrame with 'x', 'y', 'cluster' columns
        n_clusters (int): Number of clusters
        save_path (str): Path to save figure (default: None)
        figsize (tuple): Figure size (default: (16, 12))
        
    Returns:
        The generated figure
    """
    plt.figure(figsize=(16, 12))
    colors = get_cluster_colors(n_clusters)
    
    # Plot each cluster
    for cluster_id in range(n_clusters):
        cluster_data = viz_df[viz_df['cluster'] == cluster_id]
        plt.scatter(cluster_data['x'], cluster_data['y'],
                   c=[colors[cluster_id]],
                   label=f'Cluster {cluster_id}',
                   alpha=0.8,
                   s=20)
    
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.title(f'Subreddit Clusters (K={n_clusters}) - t-SNE Visualization',
              fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    


def plot_cluster_grid(viz_df, n_clusters=OPTIMAL_K, boundary_type='ellipse', 
                     save_path=None, figsize=(20, 24)):
    """
    Plot each cluster in a separate subplot grid with boundaries.
    
    Args:
        viz_df (pd.DataFrame): Visualization DataFrame with 'x', 'y', 'cluster' columns
        n_clusters (int): Number of clusters
        boundary_type (str): Type of boundary: 'ellipse', 'kde', or 'hull' (default: 'ellipse')
        save_path (str): Path to save figure (default: None)
        figsize (tuple): Figure size (default: (20, 24))
        
    Returns:
        matplotlib.figure.Figure
            The generated figure
    """
    colors = get_cluster_colors(n_clusters)
    
    # Compute axis limits (consistent across all subplots)
    x_min, x_max = viz_df['x'].min(), viz_df['x'].max()
    y_min, y_max = viz_df['y'].min(), viz_df['y'].max()
    
    x_padding = (x_max - x_min) * 0.05
    y_padding = (y_max - y_min) * 0.05
    x_min, x_max = x_min - x_padding, x_max + x_padding
    y_min, y_max = y_min - y_padding, y_max + y_padding
    
    # Create grid (6×5 for 30 clusters, adjust as needed)
    n_rows = int(np.ceil(n_clusters / 5))
    fig, axes = plt.subplots(n_rows, 5, figsize=figsize)
    axes = axes.flatten()
    
    for cluster_id in range(n_clusters):
        ax = axes[cluster_id]
        cluster_data = viz_df[viz_df['cluster'] == cluster_id]
        
        # Plot cluster points
        ax.scatter(cluster_data['x'], cluster_data['y'],
                  c=[colors[cluster_id]], alpha=0.6, s=10)
        
        # Add boundaries
        if boundary_type == 'ellipse' and len(cluster_data) > 1:
            _add_ellipse_boundary(ax, cluster_data, colors[cluster_id])
        elif boundary_type == 'kde' and len(cluster_data) > 5:
            _add_kde_boundary(ax, cluster_data, colors[cluster_id])
        elif boundary_type == 'hull' and len(cluster_data) >= 3:
            _add_hull_boundary(ax, cluster_data, colors[cluster_id])
        
        # Style subplot
        ax.set_title(f'Cluster {cluster_id} (n={len(cluster_data)})', 
                    fontsize=10, fontweight='bold')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.grid(False)
    
    # Hide extra subplots
    for idx in range(n_clusters, len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle(f't-SNE Clusters ({boundary_type.capitalize()} Boundaries)',
                fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def _add_ellipse_boundary(ax, cluster_data, color):
    """Add elliptical boundary to cluster subplot."""
    x_mean, y_mean = cluster_data['x'].mean(), cluster_data['y'].mean()
    x_std, y_std = cluster_data['x'].std(), cluster_data['y'].std()
    
    ellipse = mpatches.Ellipse(
        (x_mean, y_mean),
        width=3 * x_std,
        height=3 * y_std,
        fill=False,
        edgecolor=color,
        linewidth=1.5,
        alpha=0.9
    )
    ax.add_patch(ellipse)


def _add_kde_boundary(ax, cluster_data, color):
    """Add KDE contour boundary to cluster subplot."""
    sns.kdeplot(
        x=cluster_data['x'],
        y=cluster_data['y'],
        ax=ax,
        levels=1,
        color=color,
        linewidths=1.2,
        alpha=0.8
    )


def _add_hull_boundary(ax, cluster_data, color):
    """Add convex hull boundary to cluster subplot."""
    points = cluster_data[['x', 'y']].values
    hull = ConvexHull(points)
    for simplex in hull.simplices:
        ax.plot(points[simplex, 0], points[simplex, 1],
               color=color, alpha=0.8, linewidth=1.2)



def active_subreddit_counts(df_body, df_title, df_subreddits):
    """Plot active subreddit counts vs. activity thresholds."""
    from src.preprocessing import filter_active_subreddits
    active_subreddits, total_counts = filter_active_subreddits(df_body, df_title, df_subreddits)
    total = (total_counts['POST_COUNT_TOTAL'] < 20).sum()
    print('These subreddits have less than 20 interactions:', total, 'out of', len(total_counts), 'total, but these are the top')

    thresholds = list(range(10, 210, 10))

    counts_source = []
    counts_target = []
    counts_both = []
    counts_total = []

    for threshold in thresholds:
        # Subreddits with POST_COUNT_SOURCE >= threshold
        count_source = len(total_counts[total_counts['POST_COUNT_SOURCE'] > threshold])
        counts_source.append(count_source)
        
        # Subreddits with POST_COUNT_TARGET >= threshold
        count_target = len(total_counts[total_counts['POST_COUNT_TARGET'] > threshold])
        counts_target.append(count_target)
        
        # Subreddits meeting BOTH thresholds
        count_both = len(total_counts[
            (total_counts['POST_COUNT_SOURCE'] > threshold) & 
            (total_counts['POST_COUNT_TARGET'] > threshold)
        ])
        counts_both.append(count_both)

        # Subreddits with POST_COUNT_TOTAL >= threshold
        count_total = len(total_counts[total_counts['POST_COUNT_TOTAL'] > threshold])
        counts_total.append(count_total)


    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, counts_source, marker='o', label='Source > threshold', linewidth=2)
    plt.plot(thresholds, counts_target, marker='s', label='Target > threshold', linewidth=2)
    plt.plot(thresholds, counts_both, marker='^', label='(Source > threshold) & (Target > treshold)', linewidth=2)
    plt.plot(thresholds, counts_total, marker='D', label='(Source + Target) > threshold', linewidth=2)

    plt.xlabel('Minimum Post Count Threshold', fontsize=12)
    plt.ylabel('Number of Subreddits', fontsize=12)
    plt.title('Active Subreddit Count vs. Activity Threshold', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    return