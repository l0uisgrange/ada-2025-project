"""
Phase 3: Event-Based Analysis (The Stress Test)

This module analyzes how major events affect baseline interaction patterns.
Key questions:
1. Do events increase activity?
2. Do events increase hostility?
3. Do events change the LIWC signature?
4. Are effects similar across politics and sports?

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import timedelta


# EVENT ANALYSIS FUNCTIONS

def analyze_event(df, event_date, event_name, days_before=7, days_after=7):
    """
    Analysis of activity and sentiment around an event.

    Parameters:
        df: DataFrame with TIMESTAMP, LINK_SENTIMENT, LIWC columns
        event_date: datetime or str
        event_name: str for labeling
        days_before: int
        days_after: int

    Returns:
        dict with:
            - daily_stats: DataFrame of daily metrics
            - period_stats: dict with before/during/after aggregates
            - liwc_change: LIWC signature change during event
    """
    if isinstance(event_date, str):
        event_date = pd.to_datetime(event_date)

    # Define windows
    start = event_date - timedelta(days=days_before)
    end = event_date + timedelta(days=days_after)

    # Filter data
    mask = (df['TIMESTAMP'] >= start) & (df['TIMESTAMP'] <= end)
    event_data = df[mask].copy()
    event_data['date'] = event_data['TIMESTAMP'].dt.date
    event_data['days_from_event'] = (event_data['TIMESTAMP'].dt.date - event_date.date()).apply(lambda x: x.days)

    # Daily statistics
    daily = event_data.groupby('days_from_event').agg(
        count=('LINK_SENTIMENT', 'size'),
        negative_count=('LINK_SENTIMENT', lambda x: np.sum(x == -1)),
        mean_sentiment=('LINK_SENTIMENT', 'mean'),
        vader_neg=('VADER_NEG', 'mean'),
        vader_comp=('VADER_COMP', 'mean')
    ).reset_index()
    daily['negative_rate'] = daily['negative_count'] / daily['count']

    # Period statistics (before/during/after)
    before_mask = event_data['days_from_event'] < 0
    during_mask = event_data['days_from_event'] == 0
    after_mask = event_data['days_from_event'] > 0

    def period_stats(mask):
        subset = event_data[mask]
        if len(subset) == 0:
            return {'count': 0, 'negative_rate': 0, 'vader_neg': 0}
        return {
            'count': len(subset),
            'negative_rate': np.mean(subset['LINK_SENTIMENT'] == -1),
            'vader_neg': subset['VADER_NEG'].mean(),
            'vader_comp': subset['VADER_COMP'].mean()
        }

    periods = {
        'before': period_stats(before_mask),
        'during': period_stats(during_mask),
        'after': period_stats(after_mask)
    }

    # LIWC comparison (during vs baseline)
    liwc_cols = [c for c in df.columns if c.startswith('LIWC_')]
    baseline_liwc = event_data[~during_mask][liwc_cols].mean()
    during_liwc = event_data[during_mask][liwc_cols].mean()
    liwc_change = during_liwc - baseline_liwc

    return {
        'event_name': event_name,
        'event_date': event_date,
        'daily_stats': daily,
        'period_stats': periods,
        'liwc_change': liwc_change,
        'total_posts': len(event_data)
    }


def analyze_all_events(politics_df, sports_df, events_df, days_before=7, days_after=7):
    """
    Analyze all events in the events dataframe.

    Returns:
        dict mapping event_name -> analysis results
    """
    results = {}

    for _, event in events_df.iterrows():
        df = politics_df if event['domain'] == 'politics' else sports_df

        analysis = analyze_event(
            df,
            event['date'],
            event['name'],
            days_before=days_before,
            days_after=days_after
        )
        analysis['domain'] = event['domain']
        analysis['event_type'] = event['type']
        results[event['name']] = analysis

    return results


def calculate_event_effect(analysis):
    """
    Calculate change from baseline during event.

    Parameters:
        analysis: dict from analyze_event()

    Returns:
        dict with effect metrics
    """
    periods = analysis['period_stats']

    # Baseline is average of before and after
    baseline_neg = (periods['before']['negative_rate'] + periods['after']['negative_rate']) / 2
    during_neg = periods['during']['negative_rate']

    # Effect size
    effect = during_neg - baseline_neg
    pct_change = (effect / baseline_neg * 100) if baseline_neg > 0 else 0

    return {
        'event_name': analysis['event_name'],
        'baseline_neg_rate': baseline_neg,
        'during_neg_rate': during_neg,
        'absolute_effect': effect,
        'pct_change': pct_change,
        'during_count': periods['during']['count']
    }


def compare_event_effects(all_analyses):
    """
    Compare effects across all events.

    Returns:
        DataFrame with effect metrics for each event
    """
    effects = []
    for name, analysis in all_analyses.items():
        effect = calculate_event_effect(analysis)
        effect['domain'] = analysis['domain']
        effect['event_type'] = analysis['event_type']
        effects.append(effect)

    return pd.DataFrame(effects).sort_values('absolute_effect', ascending=False)


# TIME SERIES ANALYSIS

def get_weekly_trends(df, start_date=None, end_date=None):
    """
    Get weekly aggregated statistics.

    Returns:
        DataFrame with weekly metrics
    """
    df = df.copy()
    df['week'] = df['TIMESTAMP'].dt.to_period('W').apply(lambda x: x.start_time)

    if start_date:
        df = df[df['TIMESTAMP'] >= start_date]
    if end_date:
        df = df[df['TIMESTAMP'] <= end_date]

    weekly = df.groupby('week').agg(
        count=('LINK_SENTIMENT', 'size'),
        negative_rate=('LINK_SENTIMENT', lambda x: np.mean(x == -1)),
        vader_neg=('VADER_NEG', 'mean'),
        vader_comp=('VADER_COMP', 'mean')
    ).reset_index()

    return weekly


def get_monthly_trends(df):
    """
    Get monthly aggregated statistics.

    Returns:
        DataFrame with monthly metrics
    """
    df = df.copy()
    df['month'] = df['TIMESTAMP'].dt.to_period('M').apply(lambda x: x.start_time)

    monthly = df.groupby('month').agg(
        count=('LINK_SENTIMENT', 'size'),
        negative_rate=('LINK_SENTIMENT', lambda x: np.mean(x == -1)),
        vader_neg=('VADER_NEG', 'mean')
    ).reset_index()

    return monthly


# LIWC EVENT ANALYSIS

def compare_liwc_during_events(all_analyses, domain='politics'):
    """
    Compare LIWC changes across events of the same domain.

    Returns:
        DataFrame with LIWC feature changes for each event
    """
    domain_events = {k: v for k, v in all_analyses.items() if v['domain'] == domain}

    liwc_changes = {}
    for name, analysis in domain_events.items():
        liwc_changes[name] = analysis['liwc_change']

    return pd.DataFrame(liwc_changes)


def get_event_liwc_signature(all_analyses):
    """
    Get average LIWC change during events vs baseline.

    Returns:
        dict with 'politics' and 'sports' average LIWC changes
    """
    politics_changes = []
    sports_changes = []

    for name, analysis in all_analyses.items():
        if analysis['period_stats']['during']['count'] > 10:  # Min threshold
            if analysis['domain'] == 'politics':
                politics_changes.append(analysis['liwc_change'])
            else:
                sports_changes.append(analysis['liwc_change'])

    result = {}
    if politics_changes:
        result['politics'] = pd.concat(politics_changes, axis=1).mean(axis=1)
    if sports_changes:
        result['sports'] = pd.concat(sports_changes, axis=1).mean(axis=1)

    return result


# VISUALIZATION FUNCTIONS

def plot_event_timeline(analysis, figsize=(14, 5), save_path=None):
    """
    Plot daily activity and sentiment around an event.

    Returns:
        matplotlib figure
    """
    daily = analysis['daily_stats']
    event_name = analysis['event_name']

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Activity plot
    axes[0].bar(daily['days_from_event'], daily['count'], color='#3498db', alpha=0.7)
    axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Event Day')
    axes[0].set_xlabel('Days from Event')
    axes[0].set_ylabel('Number of Posts')
    axes[0].set_title(f'{event_name}\nActivity')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)

    # Sentiment plot
    axes[1].bar(daily['days_from_event'], daily['negative_rate'] * 100,
                color='#e74c3c', alpha=0.7)
    axes[1].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Event Day')
    axes[1].axhline(y=daily['negative_rate'].mean() * 100, color='gray',
                    linestyle=':', label='Average')
    axes[1].set_xlabel('Days from Event')
    axes[1].set_ylabel('Negative Rate (%)')
    axes[1].set_title(f'{event_name}\nHostility')
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_before_during_after(analyses, events_to_plot, figsize=(14, 6)):
    """
    Plot before/during/after comparison for multiple events.

    Parameters:
        analyses: dict from analyze_all_events
        events_to_plot: list of event names to include
        figsize: figure size

    Returns:
        matplotlib figure
    """
    data = []
    for event_name in events_to_plot:
        if event_name in analyses:
            periods = analyses[event_name]['period_stats']
            for period, stats in periods.items():
                data.append({
                    'event': event_name[:25] + '...' if len(event_name) > 25 else event_name,
                    'period': period,
                    'negative_rate': stats['negative_rate'] * 100,
                    'count': stats['count']
                })

    df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=figsize)

    # Create grouped bar chart
    events = df['event'].unique()
    x = np.arange(len(events))
    width = 0.25

    periods = ['before', 'during', 'after']
    colors = ['#3498db', '#e74c3c', '#2ecc71']

    for i, period in enumerate(periods):
        period_data = df[df['period'] == period]
        values = [period_data[period_data['event'] == e]['negative_rate'].values[0]
                  if len(period_data[period_data['event'] == e]) > 0 else 0
                  for e in events]
        ax.bar(x + i * width, values, width, label=period.capitalize(),
               color=colors[i], alpha=0.8)

    ax.set_xlabel('Event')
    ax.set_ylabel('Negative Rate (%)')
    ax.set_title('Hostility: Before vs During vs After Events', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(events, rotation=45, ha='right', fontsize=9)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    return fig


def plot_event_effects_comparison(effects_df, figsize=(12, 6), save_path=None):
    """
    Plot event effects comparison across domains.

    Returns:
        matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Sort by effect
    df = effects_df.sort_values('absolute_effect', ascending=True)

    y = np.arange(len(df))

    ax.set_yticks(y)
    ax.set_yticklabels(df['event_name'].str[:30], fontsize=9)
    ax.set_xlabel('Change in Negative Rate (percentage points)')
    ax.set_title('Event Effects on Hostility\n(During Event vs Baseline)',
                 fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.grid(axis='x', alpha=0.3)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#e74c3c', alpha=0.8, label='Politics'),
                       Patch(facecolor='#3498db', alpha=0.8, label='Sports')]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_yearly_timeline(politics_df, sports_df, events_df, figsize=(16, 8), save_path=None):
    """
    Plot full timeline with events marked.

    Returns:
        matplotlib figure
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # Politics
    pol_weekly = get_weekly_trends(politics_df)
    axes[0].fill_between(pol_weekly['week'], pol_weekly['negative_rate'] * 100,
                         alpha=0.3, color='#e74c3c')
    axes[0].plot(pol_weekly['week'], pol_weekly['negative_rate'] * 100,
                 color='#e74c3c', linewidth=1.5, label='Negative Rate')

    # Mark politics events
    pol_events = events_df[events_df['domain'] == 'politics']
    for _, event in pol_events.iterrows():
        axes[0].axvline(x=event['date'], color='gray', linestyle='--', alpha=0.7)
        axes[0].annotate(event['name'][:20], xy=(event['date'], axes[0].get_ylim()[1] * 0.9),
                         rotation=45, fontsize=7, ha='left')

    axes[0].set_ylabel('Negative Rate (%)')
    axes[0].set_title('Politics: Weekly Hostility Timeline', fontsize=12, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].set_ylim(0, 35)

    # Sports
    sport_weekly = get_weekly_trends(sports_df)
    axes[1].fill_between(sport_weekly['week'], sport_weekly['negative_rate'] * 100,
                         alpha=0.3, color='#3498db')
    axes[1].plot(sport_weekly['week'], sport_weekly['negative_rate'] * 100,
                 color='#3498db', linewidth=1.5, label='Negative Rate')

    # Mark sports events
    sport_events = events_df[events_df['domain'] == 'sports']
    for _, event in sport_events.iterrows():
        axes[1].axvline(x=event['date'], color='gray', linestyle='--', alpha=0.7)
        axes[1].annotate(event['name'][:20], xy=(event['date'], axes[1].get_ylim()[1] * 0.9),
                         rotation=45, fontsize=7, ha='left')

    axes[1].set_ylabel('Negative Rate (%)')
    axes[1].set_xlabel('Date')
    axes[1].set_title('Sports: Weekly Hostility Timeline', fontsize=12, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].set_ylim(0, 20)

    # Format x-axis
    axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_event_liwc_change(analysis, n_features=10, figsize=(12, 5), save_path=None):
    """
    Plot LIWC changes during a specific event.

    Returns:
        matplotlib figure
    """
    liwc_change = analysis['liwc_change'].dropna()

    # Get top changes (positive and negative)
    top_increase = liwc_change.sort_values(ascending=False).head(n_features)

    fig, ax = plt.subplots(figsize=figsize)

    colors = ['#e74c3c' if v > 0 else '#3498db' for v in top_increase.values]

    x = np.arange(len(top_increase))
    ax.bar(x, top_increase.values, color=colors, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([f.replace('LIWC_', '') for f in top_increase.index],
                       rotation=45, ha='right')
    ax.set_ylabel('Change (During - Baseline)')
    ax.set_title(f'{analysis["event_name"]}\nLIWC Changes During Event',
                 fontsize=12, fontweight='bold')
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_stress_test_summary(all_analyses, figsize=(14, 8), save_path=None):
    """
    Plot summary showing whether events amplify or change patterns.

    Returns:
        matplotlib figure
    """
    # Calculate effect for each event
    effects = []
    for name, analysis in all_analyses.items():
        if analysis['period_stats']['during']['count'] >= 10:
            effect = calculate_event_effect(analysis)
            effect['domain'] = analysis['domain']
            effects.append(effect)

    effects_df = pd.DataFrame(effects)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Left: Effect by domain
    for domain, color in [('politics', '#e74c3c'), ('sports', '#3498db')]:
        domain_effects = effects_df[effects_df['domain'] == domain]
        axes[0].scatter(domain_effects['baseline_neg_rate'] * 100,
                        domain_effects['during_neg_rate'] * 100,
                        c=color, s=100, alpha=0.7, label=domain.capitalize())

        # Add event labels
        for _, row in domain_effects.iterrows():
            axes[0].annotate(row['event_name'][:15],
                             (row['baseline_neg_rate'] * 100, row['during_neg_rate'] * 100),
                             fontsize=7, alpha=0.8)

    # Add diagonal
    lims = [0, max(axes[0].get_xlim()[1], axes[0].get_ylim()[1])]
    axes[0].plot(lims, lims, 'k:', alpha=0.5, label='No change')

    axes[0].set_xlabel('Baseline Negative Rate (%)')
    axes[0].set_ylabel('During Event Negative Rate (%)')
    axes[0].set_title('Event Day vs Baseline Hostility', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Right: Distribution of effects
    pol_effects = effects_df[effects_df['domain'] == 'politics']['absolute_effect'] * 100
    sport_effects = effects_df[effects_df['domain'] == 'sports']['absolute_effect'] * 100

    positions = [1, 2]
    bp = axes[1].boxplot([pol_effects, sport_effects], positions=positions, widths=0.6,
                         patch_artist=True)

    bp['boxes'][0].set_facecolor('#e74c3c')
    bp['boxes'][1].set_facecolor('#3498db')
    for box in bp['boxes']:
        box.set_alpha(0.7)

    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(['Politics', 'Sports'])
    axes[1].set_ylabel('Change in Negative Rate (pp)')
    axes[1].set_title('Distribution of Event Effects', fontsize=12, fontweight='bold')
    axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


# DEMO

if __name__ == "__main__":
    from data_prep import load_prepared_data, EVENTS

    print("Loading data...")
    politics, sports = load_prepared_data()

    print("Analyzing events...")
    all_analyses = analyze_all_events(politics, sports, EVENTS)

    print("\n" + "=" * 70)
    print("PHASE 3: EVENT ANALYSIS RESULTS")
    print("=" * 70)

    # Compare effects
    effects = compare_event_effects(all_analyses)

    print("\n1. EVENT EFFECTS ON HOSTILITY")
    print("-" * 70)
    print("\nEvents ranked by change in negative rate during event:")
    print(effects[['event_name', 'domain', 'baseline_neg_rate', 'during_neg_rate',
                   'absolute_effect', 'during_count']].to_string(index=False))

    # Summary by domain
    print("\n2. SUMMARY BY DOMAIN")
    print("-" * 70)

    for domain in ['politics', 'sports']:
        domain_effects = effects[effects['domain'] == domain]
        avg_effect = domain_effects['absolute_effect'].mean()
        n_increase = (domain_effects['absolute_effect'] > 0).sum()
        n_decrease = (domain_effects['absolute_effect'] < 0).sum()

        print(f"\n{domain.upper()}:")
        print(f"  Average effect: {avg_effect * 100:+.1f} percentage points")
        print(f"  Events with increased hostility: {n_increase}")
        print(f"  Events with decreased hostility: {n_decrease}")

    print("\n" + "=" * 70)
    print("PHASE 3 COMPLETE")
    print("=" * 70)