"""
Validation script for "Digital Tribes" analysis.
Run this with your data to check if the proposed analysis is feasible.

Usage:
    from validation import run_all_validations
    results = run_all_validations(df_body, df_title)
"""

import pandas as pd
import numpy as np
from collections import defaultdict

# =============================================================================
# SUBREDDIT DEFINITIONS
# =============================================================================

POLITICS_CAMPS = {
    'conservative': [
        'the_donald', 'conservative', 'conservatives', 'republican', 'republicans',
        'tedcruz', 'teaparty', 'metacanada', 'asktrumpsupporters', 'mr_trump',
        'hillaryforprison', 'conservatives_r_us', 'tedcruzforpresident'
    ],
    'liberal': [
        'politics', 'political_revolution', 'sandersforpresident', 'democrats',
        'hillaryclinton', 'enoughtrumpspam', 'wayofthebern', 'kossacks_for_sanders',
        'stillsandersforpres', 'ourpresident'
    ],
    'libertarian': [
        'libertarian', 'anarcho_capitalism', 'goldandblack', 'garyjohnson',
        'libertarianmeme', 'shitstatistssay'
    ],
    'alt_right': [
        'european', 'whiterights', 'europeannationalism', 'altright',
        'northamerican', 'white_pride', 'farright'
    ],
    'news_alt': [
        'altnewz', 'uncensorednews', 'worldpolitics', 'descentintotyranny',
        'endlesswar', 'antiwar'
    ]
}

# NFL focused
SPORTS_CAMPS = {
    'patriots': ['patriots', 'patriots2'],
    'seahawks': ['seahawks'],
    'broncos': ['denverbroncos'],
    'falcons': ['falcons'],
    'panthers': ['panthers'],
    'packers': ['greenbaypackers'],
    'cowboys': ['cowboys'],
    'nfl_general': ['nfl', 'nfl_draft', 'nflroundtable'],
    'nba_general': ['nba', 'warriors', 'lakers', 'clevelandcavs', 'bostonceltics'],
    'hockey': ['hockey', 'habs', 'leafs', 'bostonbruins']
}


def flatten_camps(camps_dict):
    """Flatten camp dict to subreddit -> camp mapping."""
    mapping = {}
    all_subs = []
    for camp, subs in camps_dict.items():
        for sub in subs:
            mapping[sub] = camp
            all_subs.append(sub)
    return mapping, all_subs

POLITICS_MAP, POLITICS_SUBS = flatten_camps(POLITICS_CAMPS)
SPORTS_MAP, SPORTS_SUBS = flatten_camps(SPORTS_CAMPS)


# =============================================================================
# VALIDATION 1: Cross-post volume between rivals
# =============================================================================

def check_crosspost_volume(df, subreddit_list, subreddit_map, domain_name):
    """
    Check if there are enough cross-posts between communities.

    Parameters:
        df: DataFrame with SOURCE_SUBREDDIT, TARGET_SUBREDDIT columns
        subreddit_list: list of subreddits to filter on
        subreddit_map: dict mapping subreddit -> camp
        domain_name: str, for display purposes

    Returns:
        dict with volume statistics and camp-to-camp matrix
    """
    # Filter to relevant subreddits (source OR target in list)
    mask = (df['SOURCE_SUBREDDIT'].isin(subreddit_list) |
            df['TARGET_SUBREDDIT'].isin(subreddit_list))
    df_filtered = df[mask].copy()

    # Further filter to only rows where BOTH are in the list (for camp analysis)
    mask_both = (df_filtered['SOURCE_SUBREDDIT'].isin(subreddit_list) &
                 df_filtered['TARGET_SUBREDDIT'].isin(subreddit_list))
    df_internal = df_filtered[mask_both].copy()

    # Map to camps
    df_internal['source_camp'] = df_internal['SOURCE_SUBREDDIT'].map(subreddit_map)
    df_internal['target_camp'] = df_internal['TARGET_SUBREDDIT'].map(subreddit_map)

    # Build camp-to-camp volume matrix
    camp_matrix = df_internal.groupby(['source_camp', 'target_camp']).size().unstack(fill_value=0)

    # Top subreddit pairs
    pair_counts = df_internal.groupby(['SOURCE_SUBREDDIT', 'TARGET_SUBREDDIT']).size()
    top_pairs = pair_counts.sort_values(ascending=False).head(20)

    results = {
        'domain': domain_name,
        'total_posts_any': len(df_filtered),
        'total_posts_internal': len(df_internal),
        'unique_sources': df_internal['SOURCE_SUBREDDIT'].nunique(),
        'unique_targets': df_internal['TARGET_SUBREDDIT'].nunique(),
        'camp_matrix': camp_matrix,
        'top_pairs': top_pairs
    }

    return results


# =============================================================================
# VALIDATION 2: Sentiment variance
# =============================================================================

def check_sentiment_variance(df, subreddit_list, domain_name):
    """
    Parameters:
        df: DataFrame with SOURCE_SUBREDDIT, TARGET_SUBREDDIT, LINK_SENTIMENT columns
        subreddit_list: list of subreddits to filter on
        domain_name: str, for display purposes

    Returns:
        dict with sentiment distribution statistics
    """
    mask = (df['SOURCE_SUBREDDIT'].isin(subreddit_list) &
            df['TARGET_SUBREDDIT'].isin(subreddit_list))
    df_filtered = df[mask].copy()

    sentiment_counts = df_filtered['LINK_SENTIMENT'].value_counts()
    sentiment_pcts = df_filtered['LINK_SENTIMENT'].value_counts(normalize=True) * 100

    results = {
        'domain': domain_name,
        'total_posts': len(df_filtered),
        'sentiment_counts': sentiment_counts.to_dict(),
        'sentiment_pcts': sentiment_pcts.to_dict(),
        'has_variance': len(sentiment_counts) > 1 and sentiment_counts.min() > 100
    }

    return results


# =============================================================================
# VALIDATION 3: LIWC comparison between hostile posts
# =============================================================================

def check_liwc_patterns(df, subreddit_list, domain_name):
    """
    Compare LIWC features between positive and negative sentiment posts.

    Parameters:
        df: DataFrame with LINK_SENTIMENT and LIWC_* columns
        subreddit_list: list of subreddits to filter on
        domain_name: str, for display purposes

    Returns:
        dict with LIWC statistics for positive vs negative posts
    """
    mask = (df['SOURCE_SUBREDDIT'].isin(subreddit_list) &
            df['TARGET_SUBREDDIT'].isin(subreddit_list))
    df_filtered = df[mask].copy()

    # Identify LIWC columns
    liwc_cols = [col for col in df.columns if col.startswith('LIWC_')]

    if not liwc_cols:
        return {'domain': domain_name, 'error': 'No LIWC columns found'}

    # Split by sentiment
    df_positive = df_filtered[df_filtered['LINK_SENTIMENT'] == 1]
    df_negative = df_filtered[df_filtered['LINK_SENTIMENT'] == -1]

    # Calculate mean LIWC for each
    liwc_positive = df_positive[liwc_cols].mean()
    liwc_negative = df_negative[liwc_cols].mean()

    # Calculate difference (negative - positive) to see what's elevated in hostile posts
    liwc_diff = liwc_negative - liwc_positive

    # Top features elevated in hostile posts
    top_hostile_features = liwc_diff.sort_values(ascending=False).head(10)

    # Top features elevated in friendly posts
    top_friendly_features = liwc_diff.sort_values(ascending=True).head(10)

    results = {
        'domain': domain_name,
        'n_positive': len(df_positive),
        'n_negative': len(df_negative),
        'liwc_positive_mean': liwc_positive.to_dict(),
        'liwc_negative_mean': liwc_negative.to_dict(),
        'top_hostile_features': top_hostile_features.to_dict(),
        'top_friendly_features': top_friendly_features.to_dict()
    }

    return results


# =============================================================================
# VALIDATION 4: Cross-domain LIWC similarity
# =============================================================================

def compare_liwc_across_domains(politics_liwc, sports_liwc):
    """
    Compare LIWC patterns between politics and sports hostile posts.

    Parameters:
        politics_liwc: dict from check_liwc_patterns for politics
        sports_liwc: dict from check_liwc_patterns for sports

    Returns:
        dict with correlation and comparison statistics
    """
    if 'error' in politics_liwc or 'error' in sports_liwc:
        return {'error': 'LIWC data missing for one domain'}

    # Get the negative (hostile) means for both
    pol_neg = pd.Series(politics_liwc['liwc_negative_mean'])
    sports_neg = pd.Series(sports_liwc['liwc_negative_mean'])

    # Align indices
    common_features = pol_neg.index.intersection(sports_neg.index)
    pol_neg = pol_neg[common_features]
    sports_neg = sports_neg[common_features]

    # Correlation between hostile LIWC profiles
    correlation = pol_neg.corr(sports_neg)

    # Rank correlation (more robust)
    rank_corr = pol_neg.rank().corr(sports_neg.rank())

    # Top features that are elevated in BOTH domains' hostile posts
    pol_top = set(politics_liwc['top_hostile_features'].keys())
    sports_top = set(sports_liwc['top_hostile_features'].keys())
    shared_hostile = pol_top.intersection(sports_top)

    results = {
        'pearson_correlation': correlation,
        'spearman_correlation': rank_corr,
        'shared_hostile_features': list(shared_hostile),
        'n_shared': len(shared_hostile)
    }

    return results


# =============================================================================
# MASTER VALIDATION RUNNER
# =============================================================================

def run_all_validations(df_body, df_title=None):
    """
    Run all validation checks and print summary.

    Parameters:
        df_body: main DataFrame (body dataset)
        df_title: optional title DataFrame (will be combined if provided)

    Returns:
        dict with all validation results
    """
    # Combine if title provided
    if df_title is not None:
        df = pd.concat([df_body, df_title], ignore_index=True)
        print(f"Combined body + title: {len(df):,} total posts\n")
    else:
        df = df_body
        print(f"Using body only: {len(df):,} total posts\n")

    results = {}

    # --- VALIDATION 1: Volume ---
    print("=" * 60)
    print("VALIDATION 1: Cross-post Volume")
    print("=" * 60)

    vol_politics = check_crosspost_volume(df, POLITICS_SUBS, POLITICS_MAP, 'politics')
    vol_sports = check_crosspost_volume(df, SPORTS_SUBS, SPORTS_MAP, 'sports')

    for vol in [vol_politics, vol_sports]:
        print(f"\n{vol['domain'].upper()}:")
        print(f"  Posts involving these subs: {vol['total_posts_any']:,}")
        print(f"  Posts between these subs:   {vol['total_posts_internal']:,}")
        print(f"  Unique sources: {vol['unique_sources']}, Unique targets: {vol['unique_targets']}")
        print(f"\n  Top 10 pairs:")
        for (src, tgt), count in list(vol['top_pairs'].items())[:10]:
            print(f"    {src} -> {tgt}: {count}")

    results['volume_politics'] = vol_politics
    results['volume_sports'] = vol_sports

    # --- VALIDATION 2: Sentiment ---
    print("\n" + "=" * 60)
    print("VALIDATION 2: Sentiment Variance")
    print("=" * 60)

    sent_politics = check_sentiment_variance(df, POLITICS_SUBS, 'politics')
    sent_sports = check_sentiment_variance(df, SPORTS_SUBS, 'sports')

    for sent in [sent_politics, sent_sports]:
        print(f"\n{sent['domain'].upper()}:")
        print(f"  Total posts: {sent['total_posts']:,}")
        print(f"  Sentiment distribution:")
        for val, pct in sent['sentiment_pcts'].items():
            label = {1: 'Positive', 0: 'Neutral', -1: 'Negative'}.get(val, val)
            count = sent['sentiment_counts'][val]
            print(f"    {label}: {pct:.1f}% ({count:,})")
        print(f"  Sufficient variance: {'YES ✓' if sent['has_variance'] else 'NO ✗'}")

    results['sentiment_politics'] = sent_politics
    results['sentiment_sports'] = sent_sports

    # --- VALIDATION 3: LIWC Patterns ---
    print("\n" + "=" * 60)
    print("VALIDATION 3: LIWC Patterns in Hostile vs Friendly Posts")
    print("=" * 60)

    liwc_politics = check_liwc_patterns(df, POLITICS_SUBS, 'politics')
    liwc_sports = check_liwc_patterns(df, SPORTS_SUBS, 'sports')

    for liwc in [liwc_politics, liwc_sports]:
        if 'error' in liwc:
            print(f"\n{liwc['domain'].upper()}: {liwc['error']}")
            continue
        print(f"\n{liwc['domain'].upper()}:")
        print(f"  Positive posts: {liwc['n_positive']:,}")
        print(f"  Negative posts: {liwc['n_negative']:,}")
        print(f"\n  Top LIWC features in HOSTILE posts:")
        for feat, diff in list(liwc['top_hostile_features'].items())[:5]:
            print(f"    {feat}: +{diff:.4f}")
        print(f"\n  Top LIWC features in FRIENDLY posts:")
        for feat, diff in list(liwc['top_friendly_features'].items())[:5]:
            print(f"    {feat}: {diff:.4f}")

    results['liwc_politics'] = liwc_politics
    results['liwc_sports'] = liwc_sports

    # --- VALIDATION 4: Cross-domain comparison ---
    print("\n" + "=" * 60)
    print("VALIDATION 4: Cross-Domain LIWC Similarity")
    print("=" * 60)

    cross_domain = compare_liwc_across_domains(liwc_politics, liwc_sports)

    if 'error' in cross_domain:
        print(f"\n{cross_domain['error']}")
    else:
        print(f"\n  Correlation of hostile LIWC profiles:")
        print(f"    Pearson:  {cross_domain['pearson_correlation']:.3f}")
        print(f"    Spearman: {cross_domain['spearman_correlation']:.3f}")
        print(f"\n  Shared top hostile features: {cross_domain['n_shared']}")
        if cross_domain['shared_hostile_features']:
            print(f"    {', '.join(cross_domain['shared_hostile_features'])}")

    results['cross_domain'] = cross_domain

    # --- SUMMARY ---
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    checks = []

    # Volume check
    vol_ok = vol_politics['total_posts_internal'] > 1000 and vol_sports['total_posts_internal'] > 500
    checks.append(('Sufficient cross-post volume', vol_ok))

    # Sentiment check
    sent_ok = sent_politics['has_variance'] and sent_sports['has_variance']
    checks.append(('Sentiment variance exists', sent_ok))

    # LIWC check
    liwc_ok = (liwc_politics.get('n_negative', 0) > 100 and
               liwc_sports.get('n_negative', 0) > 50)
    checks.append(('Enough hostile posts for LIWC', liwc_ok))

    # Cross-domain check
    corr = cross_domain.get('spearman_correlation', 0)
    corr_ok = corr > 0.3
    checks.append(('Cross-domain LIWC similarity', corr_ok))

    print()
    all_pass = True
    for name, passed in checks:
        status = '✓ PASS' if passed else '✗ FAIL'
        print(f"  {status}: {name}")
        all_pass = all_pass and passed

    print()
    if all_pass:
        print("  → All validations passed! The analysis is feasible.")
    else:
        print("  → Some validations failed. Review the details above.")

    results['summary'] = {'all_pass': all_pass, 'checks': checks}

    return results


# =============================================================================
# QUICK PREVIEW FUNCTIONS
# =============================================================================

def preview_camp_sentiment_matrix(df, subreddit_list, subreddit_map, domain_name):
    """
    Generate camp-to-camp sentiment matrix (for visualization preview).

    Returns:
        DataFrame with mean sentiment from source_camp to target_camp
    """
    mask = (df['SOURCE_SUBREDDIT'].isin(subreddit_list) &
            df['TARGET_SUBREDDIT'].isin(subreddit_list))
    df_filtered = df[mask].copy()

    df_filtered['source_camp'] = df_filtered['SOURCE_SUBREDDIT'].map(subreddit_map)
    df_filtered['target_camp'] = df_filtered['TARGET_SUBREDDIT'].map(subreddit_map)

    # Mean sentiment matrix
    sentiment_matrix = df_filtered.groupby(
        ['source_camp', 'target_camp']
    )['LINK_SENTIMENT'].mean().unstack(fill_value=0)

    return sentiment_matrix



if __name__ == "__main__":
    print("Load your data and run:")
    # Load data

