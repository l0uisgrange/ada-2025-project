"""
Phase 2: Baseline Interaction Analysis

This module provides:
1. Interaction matrices (sentiment heatmaps)
2. LIWC profiling and comparison
3. Reciprocity analysis (asymmetric rivalries)
4. Cross-domain comparison visualizations

Usage:
    from interaction_analysis import (
        build_interaction_matrix,
        analyze_reciprocity,
        plot_sentiment_heatmap,
        plot_liwc_comparison
    )
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


# =============================================================================
# INTERACTION MATRICES
# =============================================================================

def build_interaction_matrix(df, metric='negative_rate', min_count=30):
    """
    Build camp-to-camp interaction matrix.

    Parameters:
        df: DataFrame with source_camp, target_camp, LINK_SENTIMENT
        metric: 'negative_rate', 'mean_sentiment', or 'count'
        min_count: minimum interactions to include a cell

    Returns:
        DataFrame pivot table (source as rows, target as columns)
    """
    # Filter to labeled camps only
    df_filtered = df[(df['source_camp'] != 'other') & (df['target_camp'] != 'other')]

    # Aggregate
    agg = df_filtered.groupby(['source_camp', 'target_camp']).agg(
        count=('LINK_SENTIMENT', 'size'),
        mean_sentiment=('LINK_SENTIMENT', 'mean'),
        negative_count=('LINK_SENTIMENT', lambda x: (x == -1).sum())
    ).reset_index()

    agg['negative_rate'] = agg['negative_count'] / agg['count']

    # Apply minimum count filter
    agg_filtered = agg[agg['count'] >= min_count]

    # Pivot
    matrix = agg_filtered.pivot(
        index='source_camp',
        columns='target_camp',
        values=metric
    )

    return matrix


def get_interaction_summary(df, min_count=30):
    """
    Get detailed interaction statistics for all camp pairs.

    Returns:
        DataFrame with source_camp, target_camp, count, negative_rate,
        mean_sentiment, and VADER means
    """
    df_filtered = df[(df['source_camp'] != 'other') & (df['target_camp'] != 'other')]

    agg = df_filtered.groupby(['source_camp', 'target_camp']).agg(
        count=('LINK_SENTIMENT', 'size'),
        mean_sentiment=('LINK_SENTIMENT', 'mean'),
        negative_count=('LINK_SENTIMENT', lambda x: (x == -1).sum()),
        vader_neg_mean=('VADER_NEG', 'mean'),
        vader_comp_mean=('VADER_COMP', 'mean')
    ).reset_index()

    agg['negative_rate'] = agg['negative_count'] / agg['count']

    return agg[agg['count'] >= min_count].sort_values('count', ascending=False)


# =============================================================================
# RECIPROCITY ANALYSIS
# =============================================================================

def analyze_reciprocity(df, min_count=30):
    """
    Analyze reciprocity in camp interactions.
    Compares sentiment A→B vs B→A for all camp pairs.

    Parameters:
        df: DataFrame with source_camp, target_camp, LINK_SENTIMENT
        min_count: minimum interactions in BOTH directions

    Returns:
        DataFrame with columns:
            - camp_a, camp_b: the two camps
            - a_to_b_count, b_to_a_count: interaction counts
            - a_to_b_neg_rate, b_to_a_neg_rate: hostility rates
            - asymmetry: difference (a_to_b - b_to_a), positive = A more hostile
            - relationship_type: 'mutual_hostile', 'mutual_friendly',
                                'a_attacks_b', 'b_attacks_a'
    """
    interactions = get_interaction_summary(df, min_count=1)

    results = []
    seen_pairs = set()

    for _, row in interactions.iterrows():
        src, tgt = row['source_camp'], row['target_camp']

        # Skip self-interactions and already processed pairs
        if src == tgt:
            continue
        pair_key = tuple(sorted([src, tgt]))
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)

        # Get A→B
        a_to_b = interactions[(interactions['source_camp'] == src) &
                              (interactions['target_camp'] == tgt)]
        # Get B→A
        b_to_a = interactions[(interactions['source_camp'] == tgt) &
                              (interactions['target_camp'] == src)]

        if len(a_to_b) == 0 or len(b_to_a) == 0:
            continue

        a_to_b = a_to_b.iloc[0]
        b_to_a = b_to_a.iloc[0]

        # Apply min_count filter
        if a_to_b['count'] < min_count or b_to_a['count'] < min_count:
            continue

        a_neg = a_to_b['negative_rate']
        b_neg = b_to_a['negative_rate']
        asymmetry = a_neg - b_neg

        # Classify relationship
        hostile_threshold = 0.15
        if a_neg > hostile_threshold and b_neg > hostile_threshold:
            rel_type = 'mutual_hostile'
        elif a_neg <= hostile_threshold and b_neg <= hostile_threshold:
            rel_type = 'mutual_friendly'
        elif a_neg > b_neg:
            rel_type = f'{src}_attacks'
        else:
            rel_type = f'{tgt}_attacks'

        results.append({
            'camp_a': src,
            'camp_b': tgt,
            'a_to_b_count': a_to_b['count'],
            'b_to_a_count': b_to_a['count'],
            'a_to_b_neg_rate': a_neg,
            'b_to_a_neg_rate': b_neg,
            'asymmetry': asymmetry,
            'total_interactions': a_to_b['count'] + b_to_a['count'],
            'relationship_type': rel_type
        })

    return pd.DataFrame(results).sort_values('total_interactions', ascending=False)


def get_camp_hostility_profile(df):
    """
    Calculate overall hostility profile for each camp.

    Returns:
        DataFrame with:
            - camp: camp name
            - outgoing_neg_rate: how hostile when posting about others
            - incoming_neg_rate: how much hostility received
            - net_hostility: outgoing - incoming (positive = aggressor)
    """
    df_filtered = df[(df['source_camp'] != 'other') & (df['target_camp'] != 'other')]

    # Outgoing hostility (as source)
    outgoing = df_filtered.groupby('source_camp').agg(
        outgoing_count=('LINK_SENTIMENT', 'size'),
        outgoing_neg=('LINK_SENTIMENT', lambda x: (x == -1).sum())
    )
    outgoing['outgoing_neg_rate'] = outgoing['outgoing_neg'] / outgoing['outgoing_count']

    # Incoming hostility (as target)
    incoming = df_filtered.groupby('target_camp').agg(
        incoming_count=('LINK_SENTIMENT', 'size'),
        incoming_neg=('LINK_SENTIMENT', lambda x: (x == -1).sum())
    )
    incoming['incoming_neg_rate'] = incoming['incoming_neg'] / incoming['incoming_count']

    # Combine
    profile = outgoing.join(incoming, how='outer').fillna(0)
    profile['net_hostility'] = profile['outgoing_neg_rate'] - profile['incoming_neg_rate']
    profile = profile.reset_index().rename(columns={'index': 'camp', 'source_camp': 'camp'})

    return profile.sort_values('outgoing_neg_rate', ascending=False)


# =============================================================================
# LIWC ANALYSIS
# =============================================================================

def get_liwc_by_sentiment(df):
    """
    Get mean LIWC values split by sentiment.

    Returns:
        dict with 'positive' and 'negative' DataFrames
    """
    liwc_cols = [c for c in df.columns if c.startswith('LIWC_')]

    pos = df[df['LINK_SENTIMENT'] == 1][liwc_cols].mean()
    neg = df[df['LINK_SENTIMENT'] == -1][liwc_cols].mean()

    return {'positive': pos, 'negative': neg, 'difference': neg - pos}


def get_top_hostile_features(df, n=15):
    """
    Get the LIWC features most elevated in hostile posts.

    Returns:
        Series of top n features sorted by (negative - positive) difference
    """
    liwc_data = get_liwc_by_sentiment(df)
    return liwc_data['difference'].sort_values(ascending=False).head(n)


def compare_liwc_signatures(df1, df2, name1='Domain 1', name2='Domain 2'):
    """
    Compare LIWC hostility signatures between two domains.

    Returns:
        DataFrame with feature, domain1_diff, domain2_diff, and correlation stats
    """
    sig1 = get_liwc_by_sentiment(df1)['difference']
    sig2 = get_liwc_by_sentiment(df2)['difference']

    comparison = pd.DataFrame({
        name1: sig1,
        name2: sig2
    }).dropna()

    comparison['both_increase'] = (comparison[name1] > 0) & (comparison[name2] > 0)
    comparison['both_decrease'] = (comparison[name1] < 0) & (comparison[name2] < 0)
    comparison['same_direction'] = comparison['both_increase'] | comparison['both_decrease']

    return comparison.sort_values(name1, ascending=False)


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_sentiment_heatmap(matrix, title, figsize=(12, 10), cmap='RdYlGn_r',
                           vmin=0, vmax=0.4, save_path=None):
    """
    Plot camp-to-camp sentiment heatmap.

    Parameters:
        matrix: DataFrame pivot table from build_interaction_matrix
        title: plot title
        cmap: colormap (RdYlGn_r = red=hostile, green=friendly)
        vmin, vmax: color scale limits
        save_path: if provided, saves figure

    Returns:
        matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        matrix,
        annot=True,
        fmt='.2f',
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        ax=ax,
        cbar_kws={'label': 'Negative Rate'}
    )

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Target Camp', fontsize=12)
    ax.set_ylabel('Source Camp', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_liwc_comparison(df1, df2, name1='Politics', name2='Sports',
                         n_features=12, figsize=(14, 6), save_path=None):
    """
    Plot side-by-side LIWC hostility signature comparison.

    Parameters:
        df1, df2: DataFrames for each domain
        name1, name2: domain names for labels
        n_features: number of top features to show
        save_path: if provided, saves figure

    Returns:
        matplotlib figure
    """
    sig1 = get_liwc_by_sentiment(df1)['difference']
    sig2 = get_liwc_by_sentiment(df2)['difference']

    # Get top features from domain 1
    top_features = sig1.sort_values(ascending=False).head(n_features).index.tolist()

    # Prepare data
    x = np.arange(len(top_features))
    width = 0.35

    fig, ax = plt.subplots(figsize=figsize)

    bars1 = ax.bar(x - width / 2, [sig1[f] for f in top_features], width,
                   label=name1, color='#e74c3c', alpha=0.8)
    bars2 = ax.bar(x + width / 2, [sig2[f] for f in top_features], width,
                   label=name2, color='#3498db', alpha=0.8)

    ax.set_ylabel('Change in Hostile Posts\n(negative - positive)', fontsize=11)
    ax.set_title('LIWC Hostility Signatures: Politics vs Sports', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f.replace('LIWC_', '') for f in top_features], rotation=45, ha='right')
    ax.legend()
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(axis='y', alpha=0.3)

    # Add correlation annotation
    corr = sig1.corr(sig2)
    ax.annotate(f'Correlation: {corr:.3f}', xy=(0.98, 0.98), xycoords='axes fraction',
                ha='right', va='top', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_reciprocity(reciprocity_df, title, top_n=15, figsize=(12, 8), save_path=None):
    """
    Plot asymmetry in camp rivalries.

    Parameters:
        reciprocity_df: DataFrame from analyze_reciprocity
        title: plot title
        top_n: number of pairs to show
        save_path: if provided, saves figure

    Returns:
        matplotlib figure
    """
    # Take top pairs by total interactions
    df = reciprocity_df.head(top_n).copy()
    df['pair_label'] = df['camp_a'] + ' ↔ ' + df['camp_b']

    fig, ax = plt.subplots(figsize=figsize)

    y = np.arange(len(df))
    height = 0.35

    # A→B bars (right side)
    bars1 = ax.barh(y - height / 2, df['a_to_b_neg_rate'], height,
                    label='A → B hostility', color='#e74c3c', alpha=0.8)
    # B→A bars (right side, different color)
    bars2 = ax.barh(y + height / 2, df['b_to_a_neg_rate'], height,
                    label='B → A hostility', color='#3498db', alpha=0.8)

    ax.set_xlabel('Negative Rate', fontsize=11)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_yticks(y)
    ax.set_yticklabels(df['pair_label'])
    ax.legend(loc='lower right')
    ax.axvline(x=0.15, color='red', linestyle='--', alpha=0.5, label='Hostile threshold')
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_camp_profiles(profile_df, title, figsize=(12, 6), save_path=None):
    """
    Plot camp hostility profiles (outgoing vs incoming).

    Parameters:
        profile_df: DataFrame from get_camp_hostility_profile
        title: plot title
        save_path: if provided, saves figure

    Returns:
        matplotlib figure
    """
    df = profile_df.sort_values('outgoing_neg_rate', ascending=True)

    fig, ax = plt.subplots(figsize=figsize)

    y = np.arange(len(df))
    height = 0.35

    bars1 = ax.barh(y - height / 2, df['outgoing_neg_rate'], height,
                    label='Outgoing (as attacker)', color='#e74c3c', alpha=0.8)
    bars2 = ax.barh(y + height / 2, df['incoming_neg_rate'], height,
                    label='Incoming (as target)', color='#3498db', alpha=0.8)

    ax.set_xlabel('Negative Rate', fontsize=11)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_yticks(y)
    ax.set_yticklabels(df['camp'])
    ax.legend(loc='lower right')
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


# =============================================================================
# PLOTLY INTERACTIVE VISUALIZATIONS
# =============================================================================

def create_plotly_heatmap(matrix, title):
    """
    Create interactive Plotly heatmap.

    Returns:
        dict with Plotly figure specification (for JSON export)
    """
    import json

    # Prepare data for plotly
    z_data = matrix.fillna(0).values.tolist()
    x_labels = matrix.columns.tolist()
    y_labels = matrix.index.tolist()

    plotly_spec = {
        'data': [{
            'type': 'heatmap',
            'z': z_data,
            'x': x_labels,
            'y': y_labels,
            'colorscale': 'RdYlGn_r',
            'zmin': 0,
            'zmax': 0.4,
            'hovertemplate': 'Source: %{y}<br>Target: %{x}<br>Neg Rate: %{z:.2%}<extra></extra>'
        }],
        'layout': {
            'title': title,
            'xaxis': {'title': 'Target Camp'},
            'yaxis': {'title': 'Source Camp'}
        }
    }

    return plotly_spec


def create_plotly_bar_comparison(sig1, sig2, name1, name2, n_features=12):
    """
    Create interactive Plotly grouped bar chart for LIWC comparison.

    Returns:
        dict with Plotly figure specification
    """
    top_features = sig1.sort_values(ascending=False).head(n_features).index.tolist()
    feature_labels = [f.replace('LIWC_', '') for f in top_features]

    plotly_spec = {
        'data': [
            {
                'type': 'bar',
                'name': name1,
                'x': feature_labels,
                'y': [sig1[f] for f in top_features],
                'marker': {'color': '#e74c3c'}
            },
            {
                'type': 'bar',
                'name': name2,
                'x': feature_labels,
                'y': [sig2[f] for f in top_features],
                'marker': {'color': '#3498db'}
            }
        ],
        'layout': {
            'title': 'LIWC Hostility Signatures Comparison',
            'barmode': 'group',
            'xaxis': {'title': 'LIWC Feature'},
            'yaxis': {'title': 'Change in Hostile Posts'},
            'legend': {'x': 0.8, 'y': 0.95}
        }
    }

    return plotly_spec


# =============================================================================
# MAIN EXECUTION / DEMO
# =============================================================================

if __name__ == "__main__":
    from data_prep import load_prepared_data

    print("Loading data...")
    politics, sports = load_prepared_data()

    print("\n" + "=" * 70)
    print("PHASE 2: BASELINE INTERACTION ANALYSIS")
    print("=" * 70)

    # 1. Interaction Matrices
    print("\n1. INTERACTION MATRICES")
    print("-" * 70)

    pol_matrix = build_interaction_matrix(politics, metric='negative_rate', min_count=50)
    sport_matrix = build_interaction_matrix(sports, metric='negative_rate', min_count=50)

    print(f"\nPolitics matrix shape: {pol_matrix.shape}")
    print(f"Sports matrix shape: {sport_matrix.shape}")

    # 2. Camp Hostility Profiles
    print("\n2. CAMP HOSTILITY PROFILES")
    print("-" * 70)

    pol_profile = get_camp_hostility_profile(politics)
    sport_profile = get_camp_hostility_profile(sports)

    print("\nPolitics - Most hostile camps (when posting):")
    print(
        pol_profile[['camp', 'outgoing_neg_rate', 'incoming_neg_rate', 'net_hostility']].head(8).to_string(index=False))

    print("\nSports - Most hostile camps (when posting):")
    print(sport_profile[['camp', 'outgoing_neg_rate', 'incoming_neg_rate', 'net_hostility']].head(8).to_string(
        index=False))

    # 3. Reciprocity Analysis
    print("\n3. RECIPROCITY ANALYSIS")
    print("-" * 70)

    pol_recip = analyze_reciprocity(politics, min_count=50)
    sport_recip = analyze_reciprocity(sports, min_count=50)

    print("\nPolitics - Most asymmetric rivalries:")
    pol_asym = pol_recip.sort_values('asymmetry', key=abs, ascending=False)
    print(pol_asym[['camp_a', 'camp_b', 'a_to_b_neg_rate', 'b_to_a_neg_rate', 'asymmetry']].head(8).to_string(
        index=False))

    print("\nSports - Most asymmetric rivalries:")
    sport_asym = sport_recip.sort_values('asymmetry', key=abs, ascending=False)
    print(sport_asym[['camp_a', 'camp_b', 'a_to_b_neg_rate', 'b_to_a_neg_rate', 'asymmetry']].head(8).to_string(
        index=False))

    # 4. LIWC Comparison
    print("\n4. LIWC HOSTILITY SIGNATURES")
    print("-" * 70)

    pol_liwc = get_liwc_by_sentiment(politics)
    sport_liwc = get_liwc_by_sentiment(sports)

    print("\nTop LIWC features elevated in hostile posts:")
    comparison = compare_liwc_signatures(politics, sports, 'Politics', 'Sports')
    print(comparison[['Politics', 'Sports', 'same_direction']].head(12).to_string())

    # Correlation
    corr = pol_liwc['difference'].corr(sport_liwc['difference'])
    print(f"\nCross-domain LIWC correlation: {corr:.3f}")

    print("\n" + "=" * 70)
    print("PHASE 2 COMPLETE")
    print("=" * 70)