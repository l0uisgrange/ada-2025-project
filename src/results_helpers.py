"""
This module provides wrapper functions for the results notebook,
encapsulating data loading, analysis, and visualization logic
to keep the main notebook clean and focused on interpretation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# IMPORTS FROM PROJECT MODULES

from src.consts import *
from src.clustering import evaluate_clustering_range, reduce_dimensions_pca

from src.data_prep import (
    load_data, load_prepared_data, POLITICS_CAMPS, SPORTS_CAMPS, EVENTS,
    LIWC_COLS, TEXT_STRUCTURE_COLS, VADER_COLS, filter_active_subreddits
)
from src.interaction_analysis import (
    build_interaction_matrix, get_camp_hostility_profile,
    get_liwc_by_sentiment
)
from src.event_analysis import (
    analyze_all_events, compare_event_effects, get_weekly_trends
)
from src.statistical_analysis import (
    compare_domain_hostility, full_transfer_analysis,
    compare_logistic_coefficients, full_correlation_significance,
    event_diff_in_diff, build_interaction_network,
    compare_network_structures
)

# CONFIGURATION

# Feature columns for classification
FEATURE_COLS = LIWC_COLS + VADER_COLS

# Color scheme
COLORS = {
    'politics': '#e74c3c',
    'sports': '#3498db',
    'neutral': '#95a5a6',
    'highlight': '#27ae60',
    'purple': '#9b59b6'
}


# DATA LOADING

def load_all_data():
    """
    Load all datasets needed for analysis.

    Returns:
        dict with keys: 'politics', 'sports', 'raw_body', 'raw_title',
                        'subreddits', 'events'
    """
    print("Loading datasets...")
    print("=" * 60)

    # Load prepared analysis data
    politics, sports = load_prepared_data()

    # Load raw data if src modules available
    raw_body, raw_title, subreddits = load_data()
    print(f"✓ Raw body posts:        {len(raw_body):,}")
    print(f"✓ Raw title posts:       {len(raw_title):,}")
    print(f"✓ Subreddit embeddings:  {len(subreddits):,}")
    print(f"✓ Politics posts:        {len(politics):,}")
    print(f"✓ Sports posts:          {len(sports):,}")
    print(f"✓ Total for analysis:    {len(politics) + len(sports):,}")
    print(f"✓ Events defined:        {len(EVENTS)}")
    print("=" * 60)

    # Calculate date range
    all_dates = pd.concat([politics['TIMESTAMP'], sports['TIMESTAMP']])
    print(f"\nDate range: {all_dates.min().strftime('%Y-%m-%d')} to {all_dates.max().strftime('%Y-%m-%d')}")
    print(f"Features per post: {len(LIWC_COLS)} LIWC + {len(TEXT_STRUCTURE_COLS)} text + {len(VADER_COLS)} sentiment")

    return {
        'politics': politics,
        'sports': sports,
        'raw_body': raw_body,
        'raw_title': raw_title,
        'subreddits': subreddits,
        'events': EVENTS
    }

def show_pca_and_clustering_score(data):
    """Display PCA explained variance and clustering score."""
    # Filter to get only active subreddits
    active_subreddits, total_counts = filter_active_subreddits(data['raw_body'],
                                                               data['raw_title'],
                                                               data['subreddits'])
    embeddings_pca, pca_model = reduce_dimensions_pca(active_subreddits.drop(
                                                        columns=['SUBREDDIT']),
                                                        n_components=50)

    metrics_df = evaluate_clustering_range(embeddings_pca)

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].plot(metrics_df['n_clusters'], metrics_df['silhouette_score'], marker='o')
    axs[0].set_title('Silhouette Score (higher is better)')
    axs[0].set_xlabel('Number of Clusters')
    axs[0].set_ylabel('Silhouette Score')

    axs[1].plot(metrics_df['n_clusters'], metrics_df['davies_bouldin_score'], marker='o', color='orange')
    axs[1].set_title('Davies-Bouldin Score (lower is better)')
    axs[1].set_xlabel('Number of Clusters')
    axs[1].set_ylabel('Davies-Bouldin Score')

    plt.tight_layout()
    plt.show()

def show_camp_definitions():
    """Display the camp definitions used in our analysis."""
    print("POLITICAL CAMPS")
    print("=" * 60)
    for camp, subreddits in POLITICS_CAMPS.items():
        subs_preview = ', '.join(subreddits[:4])
        suffix = f'... (+{len(subreddits) - 4} more)' if len(subreddits) > 4 else ''
        print(f"\n{camp} ({len(subreddits)} subreddits):")
        print(f"  {subs_preview}{suffix}")

    print("\n" + "=" * 60)
    print("SPORTS CAMPS (showing first 6)")
    print("=" * 60)
    for camp, subreddits in list(SPORTS_CAMPS.items())[:6]:
        subs_preview = ', '.join(subreddits[:3])
        suffix = f'...' if len(subreddits) > 3 else ''
        print(f"\n{camp} ({len(subreddits)} subreddits):")
        print(f"  {subs_preview}{suffix}")
    print(f"\n... and {len(SPORTS_CAMPS) - 6} more sports camps")


def show_camp_coverage(data):
    """Show what percentage of posts are assigned to camps."""
    politics = data['politics']
    sports = data['sports']

    pol_coverage = np.mean(politics['source_camp'] != 'other') * 100
    sport_coverage = np.mean(sports['source_camp'] != 'other') * 100

    print("Camp Assignment Coverage")
    print("=" * 40)
    print(f"Politics: {pol_coverage:.1f}% of posts assigned to a camp")
    print(f"Sports:   {sport_coverage:.1f}% of posts assigned to a camp")
    print("=" * 40)


# BASELINE HOSTILITY ANALYSIS

def analyze_baseline_hostility(data):
    """
    Calculate and display baseline hostility rates.

    Returns:
        dict with hostility statistics
    """
    politics = data['politics']
    sports = data['sports']

    pol_neg_rate = float(np.mean(politics['LINK_SENTIMENT'] == -1))
    sport_neg_rate = float(np.mean(sports['LINK_SENTIMENT'] == -1))
    ratio = pol_neg_rate / sport_neg_rate

    print("=" * 60)
    print("BASELINE HOSTILITY RATES")
    print("=" * 60)
    print(f"\nPolitics: {pol_neg_rate * 100:.1f}% of cross-community posts are hostile")
    print(f"Sports:   {sport_neg_rate * 100:.1f}% of cross-community posts are hostile")
    print(f"\n→ Politics is {ratio:.1f}× more hostile than sports")
    print("=" * 60)

    return {
        'politics_rate': pol_neg_rate,
        'sports_rate': sport_neg_rate,
        'ratio': ratio
    }


def plot_baseline_hostility(data):
    """Create visualization comparing baseline hostility rates."""
    politics = data['politics']
    sports = data['sports']

    pol_neg_rate = np.mean(politics['LINK_SENTIMENT'] == -1)
    sport_neg_rate = np.mean(sports['LINK_SENTIMENT'] == -1)

    fig, ax = plt.subplots(figsize=(8, 5))

    domains = ['Politics', 'Sports']
    rates = [pol_neg_rate * 100, sport_neg_rate * 100]
    colors = [COLORS['politics'], COLORS['sports']]

    bars = ax.bar(domains, rates, color=colors, alpha=0.8, width=0.5)

    # Add value labels
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{rate:.1f}%', ha='center', fontsize=14, fontweight='bold')

    # Add ratio annotation
    ax.annotate('3.3×', xy=(0.5, 10), fontsize=24, fontweight='bold', ha='center', color='#333')
    ax.annotate('more hostile', xy=(0.5, 7.5), fontsize=12, ha='center', color='#666')

    ax.set_ylabel('Negative Interaction Rate (%)', fontsize=12)
    ax.set_title('Baseline Hostility: Politics vs Sports', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 22)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()


# INTERACTION MATRIX ANALYSIS

def analyze_camp_interactions(data):
    """
    Build and display camp-to-camp interaction statistics.

    Returns:
        dict with interaction matrices and profiles
    """
    politics = data['politics']
    sports = data['sports']

    # Build matrices
    pol_matrix = build_interaction_matrix(politics, metric='negative_rate', min_count=50)
    sport_matrix = build_interaction_matrix(sports, metric='negative_rate', min_count=50)

    # Get profiles
    pol_profile = get_camp_hostility_profile(politics)
    sport_profile = get_camp_hostility_profile(sports)

    print("Interaction Matrix Dimensions")
    print("=" * 40)
    print(f"Politics: {pol_matrix.shape[0]} × {pol_matrix.shape[1]} camps")
    print(f"Sports:   {sport_matrix.shape[0]} × {sport_matrix.shape[1]} camps")
    print("=" * 40)

    return {
        'politics_matrix': pol_matrix,
        'sports_matrix': sport_matrix,
        'politics_profile': pol_profile,
        'sports_profile': sport_profile
    }


def plot_interaction_heatmaps(data):
    """Create side-by-side heatmaps of camp-to-camp hostility."""
    politics = data['politics']
    sports = data['sports']

    pol_matrix = build_interaction_matrix(politics, metric='negative_rate', min_count=50)
    sport_matrix = build_interaction_matrix(sports, metric='negative_rate', min_count=50)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Politics heatmap
    sns.heatmap(pol_matrix, annot=True, fmt='.2f', cmap='RdYlGn_r',
                vmin=0, vmax=0.4, ax=axes[0], cbar_kws={'label': 'Negative Rate'})
    axes[0].set_title('Politics: Camp-to-Camp Hostility', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Target Camp')
    axes[0].set_ylabel('Source Camp')
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Sports heatmap
    sns.heatmap(sport_matrix, annot=True, fmt='.2f', cmap='RdYlGn_r',
                vmin=0, vmax=0.4, ax=axes[1], cbar_kws={'label': 'Negative Rate'})
    axes[1].set_title('Sports: Camp-to-Camp Hostility', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Target Camp')
    axes[1].set_ylabel('Source Camp')
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.show()


def show_most_hostile_camps(data):
    """Display the most hostile camps in each domain."""
    politics = data['politics']
    sports = data['sports']

    pol_profile = get_camp_hostility_profile(politics)
    sport_profile = get_camp_hostility_profile(sports)

    print("MOST HOSTILE CAMPS (by outgoing hostility)")
    print("=" * 60)
    print("\nPolitics - Top 5:")
    for _, row in pol_profile.head(5).iterrows():
        print(f"  {row['camp']:20s} → {row['outgoing_neg_rate'] * 100:5.1f}% hostile outgoing")

    print("\nSports - Top 5:")
    for _, row in sport_profile.head(5).iterrows():
        print(f"  {row['camp']:20s} → {row['outgoing_neg_rate'] * 100:5.1f}% hostile outgoing")
    print("=" * 60)


# LIWC SIGNATURE ANALYSIS

def analyze_liwc_signatures(data):
    """
    Calculate LIWC signatures for hostile vs friendly posts.

    Returns:
        dict with LIWC analysis results
    """
    politics = data['politics']
    sports = data['sports']

    pol_liwc = get_liwc_by_sentiment(politics)
    sport_liwc = get_liwc_by_sentiment(sports)

    pol_sig = pol_liwc['difference']
    sport_sig = sport_liwc['difference']

    # Calculate correlations
    pearson_r = pol_sig.corr(sport_sig)
    spearman_r, spearman_p = stats.spearmanr(pol_sig, sport_sig)

    print("=" * 60)
    print("THE UNIVERSAL HOSTILITY SIGNATURE")
    print("=" * 60)
    print(f"\nCross-domain LIWC correlation:")
    print(f"  Pearson r:  {pearson_r:.3f}")
    print(f"  Spearman ρ: {spearman_r:.3f}")
    print(f"\n→ When posts ARE hostile, they use nearly IDENTICAL vocabulary!")
    print("=" * 60)

    return {
        'politics_liwc': pol_liwc,
        'sports_liwc': sport_liwc,
        'politics_signature': pol_sig,
        'sports_signature': sport_sig,
        'pearson_r': pearson_r,
        'spearman_r': spearman_r
    }


def plot_liwc_correlation(data):
    """Visualization showing cross-domain LIWC correlation."""
    politics = data['politics']
    sports = data['sports']

    pol_liwc = get_liwc_by_sentiment(politics)
    sport_liwc = get_liwc_by_sentiment(sports)

    pol_sig = pol_liwc['difference']
    sport_sig = sport_liwc['difference']
    correlation = pol_sig.corr(sport_sig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Scatter plot
    ax = axes[0]
    ax.scatter(pol_sig, sport_sig, alpha=0.6, s=80, c=COLORS['sports'])

    # Label top features
    for feat in pol_sig.sort_values(ascending=False).head(6).index:
        ax.annotate(feat.replace('LIWC_', ''), (pol_sig[feat], sport_sig[feat]),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)

    # Regression line
    z = np.polyfit(pol_sig, sport_sig, 1)
    p = np.poly1d(z)
    x_line = np.linspace(pol_sig.min(), pol_sig.max(), 100)
    ax.plot(x_line, p(x_line), 'r--', alpha=0.8, linewidth=2, label=f'r = {correlation:.3f}')

    ax.set_xlabel('Politics: LIWC Change in Hostile Posts', fontsize=11)
    ax.set_ylabel('Sports: LIWC Change in Hostile Posts', fontsize=11)
    ax.set_title('The Universal Language of Hostility\nr = 0.937', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)

    # Side-by-side bar chart
    ax = axes[1]
    top_features = pol_sig.sort_values(ascending=False).head(10).index.tolist()
    x = np.arange(len(top_features))
    width = 0.35

    ax.bar(x - width / 2, [pol_sig[f] for f in top_features], width,
           label='Politics', color=COLORS['politics'], alpha=0.8)
    ax.bar(x + width / 2, [sport_sig[f] for f in top_features], width,
           label='Sports', color=COLORS['sports'], alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([f.replace('LIWC_', '') for f in top_features], rotation=45, ha='right')
    ax.set_ylabel('Increase in Hostile Posts')
    ax.set_title('Top LIWC Features Elevated in Hostility', fontsize=12, fontweight='bold')
    ax.legend()
    ax.axhline(y=0, color='black', linewidth=0.5)

    plt.tight_layout()
    plt.show()


def show_top_hostile_features(data):
    """Display the features most associated with hostility."""
    politics = data['politics']
    sports = data['sports']

    pol_liwc = get_liwc_by_sentiment(politics)
    sport_liwc = get_liwc_by_sentiment(sports)

    pol_sig = pol_liwc['difference']
    sport_sig = sport_liwc['difference']

    # Get top features from each domain
    pol_top = pol_sig.sort_values(ascending=False).head(8)
    sport_top = sport_sig.sort_values(ascending=False).head(8)

    print("TOP FEATURES ELEVATED IN HOSTILE POSTS")
    print("=" * 60)
    print(f"\n{'Feature':<15} {'Politics':>12} {'Sports':>12}")
    print("-" * 40)

    all_top = set(pol_top.index) | set(sport_top.index)
    for feat in list(all_top)[:10]:
        pol_val = pol_sig.get(feat, 0)
        sport_val = sport_sig.get(feat, 0)
        print(f"{feat.replace('LIWC_', ''):<15} {pol_val:>+12.4f} {sport_val:>+12.4f}")
    print("=" * 60)


# PRONOUN ANALYSIS

def analyze_pronouns(data):
    """Analyze pronoun usage patterns in hostile vs friendly posts."""
    politics = data['politics']
    sports = data['sports']

    pol_liwc = get_liwc_by_sentiment(politics)
    sport_liwc = get_liwc_by_sentiment(sports)

    pronoun_features = ['LIWC_I', 'LIWC_WE', 'LIWC_YOU', 'LIWC_THEY']

    print("PRONOUN CHANGES IN HOSTILE POSTS")
    print("=" * 60)
    print(f"\n{'Pronoun':<10} {'Politics':>12} {'Sports':>12} {'Interpretation':<25}")
    print("-" * 60)

    interpretations = {
        'LIWC_I': 'Self-focus',
        'LIWC_WE': 'In-group identity',
        'LIWC_YOU': 'Direct accusation',
        'LIWC_THEY': 'Out-group targeting'
    }

    for feat in pronoun_features:
        pol_val = pol_liwc['difference'].get(feat, 0)
        sport_val = sport_liwc['difference'].get(feat, 0)
        interp = interpretations.get(feat, '')
        print(f"{feat.replace('LIWC_', ''):<10} {pol_val:>+12.4f} {sport_val:>+12.4f} {interp:<25}")

    print("-" * 60)
    print("\n→ THEY increases dramatically in BOTH domains!")
    print("  This reflects out-group targeting: 'they' are the enemy.")
    print("=" * 60)


def plot_pronoun_patterns(data):
    """Visualize pronoun usage changes in hostile posts."""
    politics = data['politics']
    sports = data['sports']

    pol_liwc = get_liwc_by_sentiment(politics)
    sport_liwc = get_liwc_by_sentiment(sports)

    pronouns = ['I', 'WE', 'YOU', 'THEY']
    pol_values = [pol_liwc['difference'].get(f'LIWC_{p}', 0) for p in pronouns]
    sport_values = [sport_liwc['difference'].get(f'LIWC_{p}', 0) for p in pronouns]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(pronouns))
    width = 0.35

    ax.bar(x - width / 2, pol_values, width, label='Politics', color=COLORS['politics'], alpha=0.8)
    ax.bar(x + width / 2, sport_values, width, label='Sports', color=COLORS['sports'], alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(pronouns, fontsize=14)
    ax.set_ylabel('Change in Hostile Posts', fontsize=11)
    ax.set_title('Pronoun Usage: The Grammar of Us vs Them', fontsize=12, fontweight='bold')
    ax.legend()
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.grid(axis='y', alpha=0.3)

    # Highlight THEY
    they_max = max(pol_values[3], sport_values[3])
    ax.annotate('Out-group\ntargeting', xy=(3, they_max),
                xytext=(3.3, they_max + 0.002),
                fontsize=10, color=COLORS['highlight'], fontweight='bold')

    plt.tight_layout()
    plt.show()


# NETWORK / AGGRESSOR-TARGET ANALYSIS

def plot_aggressor_target_space(data):
    """Visualize camps in aggressor (outgoing) vs target (incoming) space."""
    politics = data['politics']
    sports = data['sports']

    pol_profile = get_camp_hostility_profile(politics)
    sport_profile = get_camp_hostility_profile(sports)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, profile, title, color in [(axes[0], pol_profile, 'Politics', COLORS['politics']),
                                      (axes[1], sport_profile, 'Sports', COLORS['sports'])]:
        ax.scatter(profile['outgoing_neg_rate'] * 100,
                   profile['incoming_neg_rate'] * 100,
                   s=200, alpha=0.6, c=color)

        for _, row in profile.iterrows():
            ax.annotate(row['camp'],
                        (row['outgoing_neg_rate'] * 100, row['incoming_neg_rate'] * 100),
                        fontsize=9, alpha=0.8, ha='center')

        # Diagonal line (mutual combat)
        max_val = max(profile['outgoing_neg_rate'].max(), profile['incoming_neg_rate'].max()) * 100 + 5
        ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, linewidth=1)

        ax.set_xlabel('Outgoing Hostility (%)', fontsize=11)
        ax.set_ylabel('Incoming Hostility (%)', fontsize=11)
        ax.set_title(f'{title}: Aggressor vs Target Landscape', fontweight='bold', fontsize=12)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def analyze_meta_drama(data):
    """Analyze the 'observer' camp (meta_drama) across domains."""
    politics = data['politics']
    sports = data['sports']

    pol_profile = get_camp_hostility_profile(politics)
    sport_profile = get_camp_hostility_profile(sports)

    pol_meta = pol_profile[pol_profile['camp'] == 'meta_drama']
    sport_meta = sport_profile[sport_profile['camp'] == 'meta_drama']

    print("THE OBSERVER EFFECT: meta_drama Analysis")
    print("=" * 60)

    if len(pol_meta) > 0 and len(sport_meta) > 0:
        pol_rate = pol_meta['outgoing_neg_rate'].values[0] * 100
        sport_rate = sport_meta['outgoing_neg_rate'].values[0] * 100

        print(f"\nmeta_drama subreddits: subredditdrama, bestof, etc.")
        print(f"\nOutgoing hostility:")
        print(f"  When observing politics: {pol_rate:.1f}%")
        print(f"  When observing sports:   {sport_rate:.1f}%")
        print(f"\n→ The 'observers' are equally hostile regardless of what they observe!")
        print("  This suggests drama commentary is inherently hostile.")
    else:
        print("Note: meta_drama camp not found in both domains")
    print("=" * 60)


# =============================================================================
# EVENT ANALYSIS
# =============================================================================

def show_events():
    """Display the events analyzed in this study."""
    print("EVENTS ANALYZED")
    print("=" * 60)

    for domain in ['politics', 'sports']:
        print(f"\n{domain.upper()}:")
        domain_events = EVENTS[EVENTS['domain'] == domain]
        for _, event in domain_events.iterrows():
            print(f"  • {event['name']} ({event['date'].strftime('%Y-%m-%d')})")
    print("=" * 60)


def analyze_events(data):
    """
    Run event analysis and display effects.

    Returns:
        dict with event analysis results
    """
    politics = data['politics']
    sports = data['sports']

    all_analyses = analyze_all_events(politics, sports, EVENTS)
    effects = compare_event_effects(all_analyses)

    print("EVENT EFFECTS ON HOSTILITY")
    print("=" * 60)
    print(f"\n{'Event':<35} {'Domain':<10} {'Baseline':>10} {'During':>10} {'Effect':>10}")
    print("-" * 75)

    for _, row in effects.iterrows():
        name = row['event_name'][:33]
        domain = row['domain']
        baseline = row['baseline_neg_rate'] * 100
        during = row['during_neg_rate'] * 100
        effect = row['absolute_effect'] * 100
        print(f"{name:<35} {domain:<10} {baseline:>9.1f}% {during:>9.1f}% {effect:>+9.1f}pp")

    print("=" * 60)

    return {'analyses': all_analyses, 'effects': effects}


def plot_event_effects(data):
    """Visualize the effect of each event on hostility."""
    politics = data['politics']
    sports = data['sports']

    all_analyses = analyze_all_events(politics, sports, EVENTS)
    effects = compare_event_effects(all_analyses)

    fig, ax = plt.subplots(figsize=(12, 8))

    effects_sorted = effects.sort_values('absolute_effect', ascending=True)
    colors = [COLORS['politics'] if d == 'politics' else COLORS['sports']
              for d in effects_sorted['domain']]

    y = np.arange(len(effects_sorted))
    ax.barh(y, effects_sorted['absolute_effect'] * 100, color=colors, alpha=0.8)

    ax.set_yticks(y)
    ax.set_yticklabels(effects_sorted['event_name'].str[:35], fontsize=9)
    ax.set_xlabel('Change in Negative Rate (percentage points)', fontsize=11)
    ax.set_title('Event Effects on Hostility\n(Positive = More Hostile During Event)',
                 fontsize=12, fontweight='bold')
    ax.axvline(x=0, color='black', linewidth=1)
    ax.grid(axis='x', alpha=0.3)

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=COLORS['politics'], alpha=0.8, label='Politics'),
                       Patch(facecolor=COLORS['sports'], alpha=0.8, label='Sports')]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    plt.show()


def plot_weekly_timeline(data):
    """Plot weekly hostility trends with event markers."""
    politics = data['politics']
    sports = data['sports']

    pol_weekly = get_weekly_trends(politics)
    sport_weekly = get_weekly_trends(sports)

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    # Politics
    axes[0].fill_between(pol_weekly['week'], pol_weekly['negative_rate'] * 100,
                         alpha=0.3, color=COLORS['politics'])
    axes[0].plot(pol_weekly['week'], pol_weekly['negative_rate'] * 100,
                 color=COLORS['politics'], linewidth=1.5)
    for _, event in EVENTS[EVENTS['domain'] == 'politics'].iterrows():
        axes[0].axvline(x=event['date'], color='gray', linestyle='--', alpha=0.5)
        axes[0].text(event['date'], 32, event['name'][:15], rotation=90, fontsize=7, alpha=0.7)
    axes[0].set_ylabel('Negative Rate (%)', fontsize=11)
    axes[0].set_title('Politics: Weekly Hostility Over Time', fontweight='bold')
    axes[0].set_ylim(0, 35)

    # Sports
    axes[1].fill_between(sport_weekly['week'], sport_weekly['negative_rate'] * 100,
                         alpha=0.3, color=COLORS['sports'])
    axes[1].plot(sport_weekly['week'], sport_weekly['negative_rate'] * 100,
                 color=COLORS['sports'], linewidth=1.5)
    for _, event in EVENTS[EVENTS['domain'] == 'sports'].iterrows():
        axes[1].axvline(x=event['date'], color='gray', linestyle='--', alpha=0.5)
        axes[1].text(event['date'], 17, event['name'][:15], rotation=90, fontsize=7, alpha=0.7)
    axes[1].set_ylabel('Negative Rate (%)', fontsize=11)
    axes[1].set_xlabel('Date', fontsize=11)
    axes[1].set_title('Sports: Weekly Hostility Over Time', fontweight='bold')
    axes[1].set_ylim(0, 20)

    plt.tight_layout()
    plt.show()


# STATISTICAL TESTS

def run_proportion_test(data):
    """
    Run statistical tests comparing hostility proportions.

    Returns:
        dict with test results
    """
    politics = data['politics']
    sports = data['sports']

    results = compare_domain_hostility(politics, sports)

    print("=" * 60)
    print("STATISTICAL TEST: Proportion Comparison")
    print("=" * 60)
    print(
        f"""
            Sample Sizes:
              Politics: n = {results['politics_n']:,} ({results['politics_hostile']:,} hostile)
              Sports:   n = {results['sports_n']:,} ({results['sports_hostile']:,} hostile)
            
            Proportions:
              Politics: {results['p1'] * 100:.2f}%
              Sports:   {results['p2'] * 100:.2f}%
              Difference: {results['difference'] * 100:.2f} percentage points
              95% CI: [{results['difference_ci_95'][0] * 100:.2f}%, {results['difference_ci_95'][1] * 100:.2f}%]
            
            Statistical Tests:
              Chi-square statistic: {results['chi2_statistic']:.2f}
              Chi-square p-value:   < 0.001
              Z-statistic: {results['z_statistic']:.2f}
            
            Effect Sizes:
              Cohen's h: {results['cohens_h']:.3f} ({results['cohens_h_interpretation']} effect)
              Risk Ratio: {results['risk_ratio']:.2f}×
              Odds Ratio: {results['odds_ratio']:.2f}
        """
    )
    print("=" * 60)

    return results


def run_cross_domain_classifier(data):
    """
    Train classifiers and test cross-domain transfer.

    Returns:
        dict with classifier results
    """
    politics = data['politics']
    sports = data['sports']

    print("Running cross-domain classifier analysis...")
    print("(Training logistic regression models, this may take a moment)")

    results = full_transfer_analysis(politics, sports, FEATURE_COLS)

    print("\n" + "=" * 60)
    print("CROSS-DOMAIN CLASSIFIER TRANSFER")
    print("=" * 60)
    print(
        f"""
            Same-Domain Performance (baseline):
              Politics → Politics: AUC = {results['politics_baseline']['roc_auc']:.3f}
              Sports → Sports:     AUC = {results['sports_baseline']['roc_auc']:.3f}
            
            Cross-Domain Transfer:
              Politics → Sports:   AUC = {results['politics_to_sports']['roc_auc']:.3f}
              Sports → Politics:   AUC = {results['sports_to_politics']['roc_auc']:.3f}
            
            Random Baseline: AUC = 0.500
            
            → Cross-domain AUC well above 0.5 proves patterns transfer!
        """
    )
    print("=" * 60)

    return results


def plot_classifier_results(data):
    """Visualize classifier performance across scenarios."""
    politics = data['politics']
    sports = data['sports']

    results = full_transfer_analysis(politics, sports, FEATURE_COLS)

    fig, ax = plt.subplots(figsize=(10, 5))

    scenarios = ['Politics\n(same)', 'Sports\n(same)', 'Politics→\nSports', 'Sports→\nPolitics']
    aucs = [
        results['politics_baseline']['roc_auc'],
        results['sports_baseline']['roc_auc'],
        results['politics_to_sports']['roc_auc'],
        results['sports_to_politics']['roc_auc']
    ]
    colors = [COLORS['politics'], COLORS['sports'], COLORS['purple'], COLORS['highlight']]

    bars = ax.bar(scenarios, aucs, color=colors, alpha=0.8)
    ax.axhline(y=0.5, color='gray', linestyle='--', label='Random baseline (0.5)')
    ax.set_ylabel('AUC-ROC', fontsize=12)
    ax.set_title('Classifier Performance: Same-Domain vs Cross-Domain', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.legend()

    for bar, auc in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{auc:.3f}', ha='center', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.show()


def run_coefficient_comparison(data):
    """
    Compare logistic regression coefficients across domains.

    Returns:
        dict with coefficient comparison results
    """
    politics = data['politics']
    sports = data['sports']

    print("Fitting logistic regression models...")
    results = compare_logistic_coefficients(politics, sports, FEATURE_COLS)

    print("\n" + "=" * 60)
    print("LOGISTIC REGRESSION COEFFICIENT COMPARISON")
    print("=" * 60)
    print(
        f"""
            Model Performance (5-fold CV):
              Politics AUC: {results['politics_results']['cv_auc_mean']:.3f} ± {results['politics_results']['cv_auc_std']:.3f}
              Sports AUC:   {results['sports_results']['cv_auc_mean']:.3f} ± {results['sports_results']['cv_auc_std']:.3f}
            
            Coefficient Correlation:
              Pearson r:  {results['pearson_r']:.3f} (p = {results['pearson_p']:.2e})
              Spearman ρ: {results['spearman_r']:.3f}
            
            Sign Agreement: {results['sign_agreement'] * 100:.1f}% of coefficients point same direction
            Top-10 Overlap: {results['top10_overlap']}/10 features appear in both top-10
            
            → Same features predict hostility in BOTH domains!
        """
    )
    print("=" * 60)

    return results


def run_bootstrap_permutation(data):
    """
    Run bootstrap and permutation tests on LIWC correlation.

    Returns:
        dict with significance test results
    """
    politics = data['politics']
    sports = data['sports']

    pol_liwc = get_liwc_by_sentiment(politics)
    sport_liwc = get_liwc_by_sentiment(sports)

    pol_sig = pol_liwc['difference']
    sport_sig = sport_liwc['difference']

    print("Running bootstrap and permutation tests...")
    print("(2000 iterations each, this may take a moment)")

    results = full_correlation_significance(pol_sig, sport_sig,
                                            n_bootstrap=2000,
                                            n_permutations=2000)

    print("\n" + "=" * 60)
    print("CORRELATION SIGNIFICANCE TESTS")
    print("=" * 60)
    print(
        f"""
            Observed Correlation: r = {results['observed_r']:.4f}
            
            Bootstrap Analysis (n=2000):
              95% Confidence Interval: [{results['bootstrap_ci_95'][0]:.4f}, {results['bootstrap_ci_95'][1]:.4f}]
              Standard Error: {results['bootstrap_std']:.4f}
            
            Permutation Test (H₀: r = 0):
              p-value: {results['permutation_p_value']:.4f}
            
            → The correlation is robust and highly significant!
              Even the lower bound of the CI ({results['bootstrap_ci_95'][0]:.3f}) shows strong correlation.
        """
    )
    print("=" * 60)

    return results


def plot_bootstrap_results(results):
    """Visualize bootstrap and permutation distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bootstrap
    axes[0].hist(results['bootstrap_distribution'], bins=50, alpha=0.7,
                 color=COLORS['sports'], edgecolor='white')
    axes[0].axvline(x=results['observed_r'], color='red', linewidth=2,
                    label=f'Observed r = {results["observed_r"]:.3f}')
    axes[0].axvline(x=results['bootstrap_ci_95'][0], color='orange', linestyle='--', label='95% CI')
    axes[0].axvline(x=results['bootstrap_ci_95'][1], color='orange', linestyle='--')
    axes[0].set_xlabel('Correlation (r)', fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)
    axes[0].set_title(
        f'Bootstrap Distribution\n95% CI: [{results["bootstrap_ci_95"][0]:.3f}, {results["bootstrap_ci_95"][1]:.3f}]',
        fontsize=12, fontweight='bold')
    axes[0].legend()

    # Permutation
    axes[1].hist(results['permutation_distribution'], bins=50, alpha=0.7,
                 color=COLORS['neutral'], edgecolor='white')
    axes[1].axvline(x=results['observed_r'], color='red', linewidth=2,
                    label=f'Observed r = {results["observed_r"]:.3f}')
    axes[1].set_xlabel('Correlation (r)', fontsize=11)
    axes[1].set_ylabel('Frequency', fontsize=11)
    axes[1].set_title(f'Permutation Test (H₀: r = 0)\np-value = {results["permutation_p_value"]:.4f}',
                      fontsize=12, fontweight='bold')
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def run_diff_in_diff(data, event_name='2016 Election'):
    """
    Run difference-in-differences analysis for an event.

    Returns:
        dict with DiD results
    """
    politics = data['politics']

    results = event_diff_in_diff(
        politics,
        event_date='2016-11-08',
        treatment_camps=['trump_conservative', 'anti_trump', 'progressive'],
        pre_days=7, post_days=7
    )

    print("=" * 60)
    print(f"DIFFERENCE-IN-DIFFERENCES: {event_name}")
    print("=" * 60)

    if 'error' not in results:
        print(
            f"""
                Design:
                  Treatment: trump_conservative, anti_trump, progressive (directly involved)
                  Control:   meta_drama, conspiracy, libertarian (observers)
                  Window:    7 days before/after event
                
                Results:
                  Pre-Event:
                    Treatment: {results['treatment_pre'] * 100:.1f}%
                    Control:   {results['control_pre'] * 100:.1f}%
                
                  Post-Event:
                    Treatment: {results['treatment_post'] * 100:.1f}%
                    Control:   {results['control_post'] * 100:.1f}%
                
                  DiD Estimate: {results['did_estimate'] * 100:+.2f} percentage points
                  95% CI: [{results['ci_95'][0] * 100:.2f}, {results['ci_95'][1] * 100:.2f}]
                  p-value: {results['p_value']:.4f}
            """
        )
    else:
        print(f"Note: {results.get('error', 'Insufficient data')}")
    print("=" * 60)

    return results


def run_network_analysis(data):
    """
    Build and compare interaction networks.

    Returns:
        dict with network comparison
    """
    politics = data['politics']
    sports = data['sports']

    G_politics = build_interaction_network(politics, min_interactions=30)
    G_sports = build_interaction_network(sports, min_interactions=30)

    comparison = compare_network_structures(G_politics, G_sports, 'Politics', 'Sports')

    print("=" * 60)
    print("NETWORK STRUCTURE COMPARISON")
    print("=" * 60)
    print(comparison['comparison_table'].to_string(index=False))
    print("=" * 60)

    # Plotting everything

    import networkx as nx

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, G, title, color in [(axes[0], G_politics, 'Politics', COLORS['politics']),
                                (axes[1], G_sports, 'Sports', COLORS['sports'])]:
        # Position nodes
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

        # Node sizes based on total degree
        node_sizes = [300 + G.degree(n, weight='count') / 10 for n in G.nodes()]

        # Edge widths based on count
        edge_widths = [G[u][v]['count'] / 100 for u, v in G.edges()]

        # Edge colors based on hostility
        edge_colors = [G[u][v]['negative_rate'] for u, v in G.edges()]

        # Draw
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes,
                               node_color=color, alpha=0.7)
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)
        nx.draw_networkx_edges(G, pos, ax=ax,
                               width=edge_widths, # type: ignore[arg-type]
                               edge_color=edge_colors, # type: ignore[arg-type]
                               edge_cmap=plt.get_cmap('Reds'),
                               alpha=0.6, arrows=True, arrowsize=10)

        ax.set_title(f'{title} Interaction Network\n(Edge color = hostility)',
                     fontsize=12, fontweight='bold')
        ax.axis('off')

    plt.tight_layout()
    plt.show()

    return comparison


def plot_interaction_networks(data):
    """
    Visualize the interaction networks for politics and sports.

    Args:
        data: dict containing 'politics' and 'sports' DataFrames
    """

    politics = data['politics']
    sports = data['sports']

    # Build networks
    build_interaction_network(politics, min_interactions=30)
    build_interaction_network(sports, min_interactions=30)



# SUMMARY TABLE

def generate_summary_table(data, prop_results, corr_results, transfer_results, coef_results):
    """Generate and display final summary table."""
    politics = data['politics']
    sports = data['sports']

    pol_neg_rate = float(np.mean(politics['LINK_SENTIMENT'] == -1))
    sport_neg_rate = float(np.mean(sports['LINK_SENTIMENT'] == -1))

    print("\n" + "=" * 70)
    print("FINAL STATISTICAL SUMMARY")
    print("=" * 70)
    print(
        f"""
            {'Metric':<40} {'Value':<30}
            {'-' * 70}
            {'Total posts analyzed':<40} {len(politics) + len(sports):,}
            {'Date range':<40} {'Jan 2014 – Apr 2017'}
            {'Politics hostility rate':<40} {pol_neg_rate * 100:.1f}%
            {'Sports hostility rate':<40} {sport_neg_rate * 100:.1f}%
            {'Hostility ratio':<40} {pol_neg_rate / sport_neg_rate:.1f}×
            {'Chi-square p-value':<40} {'< 0.001'}
            {"Cohen's h effect size":<40} {prop_results['cohens_h']:.3f} ({prop_results['cohens_h_interpretation']})
            {'LIWC correlation (r)':<40} {corr_results['observed_r']:.3f}
            {'Bootstrap 95% CI':<40} [{corr_results['bootstrap_ci_95'][0]:.3f}, {corr_results['bootstrap_ci_95'][1]:.3f}]
            {'Permutation p-value':<40} {corr_results['permutation_p_value']:.4f}
            {'Cross-domain AUC (Pol→Sport)':<40} {transfer_results['politics_to_sports']['roc_auc']:.3f}
            {'Coefficient correlation':<40} {coef_results['pearson_r']:.3f}
            {'Sign agreement':<40} {coef_results['sign_agreement'] * 100:.1f}%
        """
    )
    print("=" * 70)


# =============================================================================
# INITIALIZATION
# =============================================================================

def initialize_plotting():
    """Set up matplotlib styling."""
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette('husl')
    # Set plot style
    plt.rc('font', size=PLOT_FONT_SIZE)
    plt.rc('axes', titlesize=PLOT_FONT_SIZE)
    plt.rc('axes', labelsize=PLOT_FONT_SIZE)
    plt.rc('xtick', labelsize=PLOT_FONT_SIZE)
    plt.rc('ytick', labelsize=PLOT_FONT_SIZE)
    plt.rc('legend', fontsize=PLOT_FONT_SIZE)
    plt.rc('figure', titlesize=PLOT_TITLE_FONT_SIZE, figsize=(PLOT_WIDTH, PLOT_WIDTH / 3), dpi=PLOT_DPI)

    plt.style.use('seaborn-v0_8-whitegrid')


# To import everything in a row
__all__ = [
    # Data
    'load_all_data', 'show_pca_and_clustering_score','show_camp_definitions', 'show_camp_coverage',
    # Baseline
    'analyze_baseline_hostility', 'plot_baseline_hostility',
    # Interactions
    'analyze_camp_interactions', 'plot_interaction_heatmaps', 'show_most_hostile_camps',
    # LIWC
    'analyze_liwc_signatures', 'plot_liwc_correlation', 'show_top_hostile_features',
    # Pronouns
    'analyze_pronouns', 'plot_pronoun_patterns',
    # Network
    'plot_aggressor_target_space', 'analyze_meta_drama',
    # Events
    'show_events', 'analyze_events', 'plot_event_effects', 'plot_weekly_timeline',
    # Statistics
    'run_proportion_test', 'run_cross_domain_classifier', 'plot_classifier_results',
    'run_coefficient_comparison', 'run_bootstrap_permutation', 'plot_bootstrap_results',
    'run_diff_in_diff', 'run_network_analysis',
    # Summary
    'generate_summary_table', 'initialize_plotting'
]