import matplotlib.patches as mpatches
import seaborn as sns
from scipy.spatial import ConvexHull

from .clustering import *
from .consts import *

def set_plt_style():
    plt.rc('font', size=PLOT_FONT_SIZE)
    plt.rc('axes', titlesize=PLOT_FONT_SIZE)
    plt.rc('axes', labelsize=PLOT_FONT_SIZE)
    plt.rc('xtick', labelsize=PLOT_FONT_SIZE)
    plt.rc('ytick', labelsize=PLOT_FONT_SIZE)
    plt.rc('legend', fontsize=PLOT_FONT_SIZE)
    plt.rc('figure', titlesize=PLOT_TITLE_FONT_SIZE, figsize=(PLOT_WIDTH, PLOT_WIDTH/3), dpi=PLOT_DPI)

    plt.style.use('seaborn-v0_8-whitegrid')

def plot_activity_over_time(df: pd.DataFrame, pattern):
    """
    Plot stacked bar chart of subreddit mentions over time.
    """

    df.plot(
        kind="bar",
        stacked=True,
        alpha=0.85
    )
    plt.title(f"Stacked Mentions of '{pattern}' Over Time")
    plt.xlabel('Time Period')
    plt.ylabel('Number of Hyperlinks')
    plt.xticks(rotation=90)
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()
    return


def plot_cluster_overview(viz_df, n_clusters=OPTIMAL_K):
    """
    Plot all clusters in a single 2D scatter plot.
    """
    plt.figure(figsize=(12, 12))
    colors = get_cluster_colors(n_clusters)

    # Plot each cluster
    for cluster_id in range(n_clusters):
        cluster_data = viz_df[viz_df['cluster'] == cluster_id]
        plt.scatter(cluster_data['x'], cluster_data['y'],
                    c=[colors[cluster_id]],
                    label=f'Cluster {cluster_id}',
                    alpha=0.8,
                    s=20)

    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.title(f'Subreddit Clusters (K={n_clusters}) - t-SNE Visualization')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


def plot_cluster_grid(viz_df, n_clusters=OPTIMAL_K, boundary_type='ellipse', figsize=(20, 24)):
    """
    Plot each cluster in a separate subplot grid with boundaries.
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

    return fig


def _add_ellipse_boundary(ax, cluster_data, color):
    """
    Add elliptical boundary to cluster subplot.
    """
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
    """
    Add convex hull boundary to cluster subplot.
    """
    points = cluster_data[['x', 'y']].values
    hull = ConvexHull(points)
    for simplex in hull.simplices:
        ax.plot(points[simplex, 0], points[simplex, 1],
                color=color, alpha=0.8, linewidth=1.2)


def active_subreddit_counts(df_body, df_title, df_subreddits):
    """
    Plot active subreddit counts vs. activity thresholds.
    """
    from src.preprocessing import filter_active_subreddits
    active_subreddits, total_counts = filter_active_subreddits(df_body, df_title, df_subreddits)
    total = (total_counts['POST_COUNT_TOTAL'] < 20).sum()
    print('These subreddits have less than 20 interactions:', total, 'out of', len(total_counts),
          'total, but these are the top')

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


def plot_neighbors_2d(ax, seed_subreddits, neighbors_df, top_n = 20):
    """
    Displays picked subreddits and their nearest neighbors in 2D space.
    """

    # Prepare data
    all_subs = seed_subreddits + list(neighbors_df.head(top_n)['SUBREDDIT'])
    viz_data = neighbors_df
    embeddings = viz_data.drop(columns=['SUBREDDIT']).values

    # TSNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(15, len(all_subs) - 1))
    coords_2d = tsne.fit_transform(embeddings)

    # Separate seed and neighbor points
    mask = viz_data['SUBREDDIT'].isin(seed_subreddits)

    ax.scatter(coords_2d[~mask, 0], coords_2d[~mask, 1], label='Neighbors')
    ax.scatter(coords_2d[mask, 0], coords_2d[mask, 1], label='Seeds')

    # Annotations
    for i, subreddit in enumerate(viz_data['SUBREDDIT']):
        plt.annotate(subreddit, (coords_2d[i, 0], coords_2d[i, 1]), alpha=0.8, ha='center')

    # ax.title(f'Embedding Space: {theme.upper()}')
    ax.set(xlabel = 't-SNE Dimension 1', ylabel = 't-SNE Dimension 2')
    ax.legend()
    ax.legend()


def plot_liwc_heatmaps(
        ax,
        df,
        subreddits,
        start_date=DATA_START_DATE,
        end_date=DATA_END_DATE,
        subreddit_col='SOURCE_SUBREDDIT',
        start_properties=0,
        size_properties=10,
        title=None,
        vmin=0,
        vmax=None
):
    """
    Shows a heatmap of LIWC properties based on given subreddits for a given period of time.
    """

    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])
    df_time_filtered = df[
        (df["TIMESTAMP"] >= pd.to_datetime(start_date)) &
        (df["TIMESTAMP"] <= pd.to_datetime(end_date))
        ]
    df_redd_filtered = df_time_filtered[df_time_filtered[subreddit_col].isin(subreddits)].copy()

    properties_list = list(PROPERTIES.keys())[start_properties:start_properties + size_properties]
    labels_list = list(PROPERTIES.values())[start_properties:start_properties + size_properties]

    sum_by_source = df_redd_filtered.groupby(subreddit_col)[properties_list].sum()
    mat_values = sum_by_source.values
    ax.imshow(mat_values, aspect='auto', origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)

    rows, cols = mat_values.shape
    for i in range(rows):
        for j in range(cols):
            value_to_display = f"{mat_values[i, j]:.2f}"
            ax.text(j, i, value_to_display, ha="center", va="center", color='white')
    ax.set_xticks(np.arange(sum_by_source.shape[1]))

    ax.set_xticklabels(labels_list, rotation=45, ha='right')
    ax.set_yticks(np.arange(sum_by_source.shape[0]))
    ax.set_yticklabels(sum_by_source.index)
    ax.set_title(title)
    ax.grid(False)

    return mat_values


def plot_subreddits_properties_trend(ax, tf, subreddits_list, start_time, end_time, features_list, period = "D"):
    """
    Shows a distribution of PROPERTIES on a given period of time.
    """
    period_map = {"D": "Day", "W": "Week", "M": "Month", "Q": "Quarter", "Y": "Year"}

    start_time = pd.Timestamp(start_time)
    end_time = pd.Timestamp(end_time)

    # apply filters
    df = tf[
        (
                tf["SOURCE_SUBREDDIT"].isin(subreddits_list) |
                tf["TARGET_SUBREDDIT"].isin(subreddits_list)
        )
        & (tf["TIMESTAMP"] >= start_time)
        & (tf["TIMESTAMP"] <= end_time)
        ]
    df = df.assign(**{period_map[period]: df["TIMESTAMP"].dt.to_period(period)})

    period_stats = (
        df.groupby(period_map[period])[features_list]
        .sum()
        .assign(Count=df.groupby(period_map[period]).size())
    )
    period_stats.index = period_stats.index.to_timestamp()

    ax.bar(period_stats.index, period_stats["Count"], color="gray", alpha=0.3)
    ax.set_ylabel("Post Count")
    ax.set_xlabel(period_map[period])
    ax.tick_params(axis="y")

    ax2 = ax.twinx()
    for f in features_list:
        ax2.plot(period_stats.index, period_stats[f], label=f, linewidth=2)
    ax2.set_ylabel("Sum of Feature Values")
    ax2.tick_params(axis="y")
    plt.legend()
    # Title and legend
    plt.title(
        f"{period_map[period]} properties within {','.join(subreddits_list)}")

def plot_liwc_heatmaps_for_features_list(
    ax,
    df,
    subreddits,
    properties_list,
    start_date=DATA_START_DATE,
    end_date=DATA_END_DATE,
    subreddit_col='TARGET_SUBREDDIT',
    title=None,
    vmin=0,
    vmax=None,
    normalize=False,
):
    import numpy as np
    """
    Shows a heatmap of LIWC properties based on given subreddits for a given period of time.
    """

    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])
    df_time_filtered = df[
        (df["TIMESTAMP"] >= pd.to_datetime(start_date)) &
        (df["TIMESTAMP"] <= pd.to_datetime(end_date))
        ]
    df_redd_filtered = df_time_filtered[df_time_filtered[subreddit_col].isin(subreddits)].copy()
 
    # properties_list = list(PROPERTIES.keys())[start_properties:start_properties + size_properties]
    # labels_list = list(PROPERTIES.values())[start_properties:start_properties + size_properties]

    sum_by_source = df_redd_filtered.groupby(subreddit_col)[properties_list].sum()
    mat_values = sum_by_source.values
    # print(mat_values)
    if normalize:
        row_sums = mat_values.sum(axis=1, keepdims=True)
        mat_values = mat_values / row_sums
    ax.imshow(mat_values, aspect='auto', origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)

    rows, cols = mat_values.shape
    for i in range(rows):
        for j in range(cols):
            value_to_display = f"{mat_values[i, j]:.2f}"
            ax.text(j, i, value_to_display, ha="center", va="center", color='white')
    ax.set_xticks(np.arange(sum_by_source.shape[1]))

    ax.set_xticklabels(properties_list, rotation=45, ha='right')
    ax.set_yticks(np.arange(sum_by_source.shape[0]))
    ax.set_yticklabels(sum_by_source.index)
    if normalize:
        ax.set_title(f'{title} (Normalized)')
    else:
        ax.set_title(f'{title} (Sum)')
    ax.grid(False)
    return  ax