"""
Generate visualization images for the Digital Tribes data story.
FIXED VERSION: Works with files in archive/Daniel/ folder
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add the Daniel folder to Python path so we can import from there
daniel_path = Path(__file__).parent / 'archive' / 'Daniel'
sys.path.insert(0, str(daniel_path))

# Now import from the Daniel folder
from data_prep import load_prepared_data, EVENTS
from interaction_analysis import (
    get_liwc_by_sentiment,
    build_interaction_matrix,
    get_camp_hostility_profile
)
from event_analysis import analyze_all_events, compare_event_effects

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create output directory in root assets/img
output_dir = Path('assets/img')
output_dir.mkdir(parents=True, exist_ok=True)

print("Loading data from archive/Daniel/...")
print(f"Looking for data files in: {daniel_path}")

# Change to Daniel directory to load data with correct paths
import os
original_dir = os.getcwd()
os.chdir(daniel_path)

try:
    politics, sports = load_prepared_data()
    print("✓ Data loaded successfully!")
except Exception as e:
    print(f"❌ Error loading data: {e}")
    print(f"Make sure data files are in: {daniel_path}")
    sys.exit(1)

# Change back to original directory for saving images
os.chdir(original_dir)

# ============================================================================
# VISUALIZATION 1: LIWC Hostility Signature Comparison
# ============================================================================
print("\nGenerating LIWC comparison chart...")

pol_liwc = get_liwc_by_sentiment(politics)
sport_liwc = get_liwc_by_sentiment(sports)

# Get top 12 features from politics
top_features = pol_liwc['difference'].sort_values(ascending=False).head(12).index.tolist()

fig, ax = plt.subplots(figsize=(14, 7))

x = np.arange(len(top_features))
width = 0.35

bars1 = ax.bar(x - width/2, [pol_liwc['difference'][f] for f in top_features], 
               width, label='Politics', color='#e74c3c', alpha=0.8)
bars2 = ax.bar(x + width/2, [sport_liwc['difference'][f] for f in top_features], 
               width, label='Sports', color='#3498db', alpha=0.8)

ax.set_ylabel('Change in Hostile Posts\n(negative - positive)', fontsize=12, fontweight='bold')
ax.set_title('LIWC Hostility Signatures: Politics vs Sports', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels([f.replace('LIWC_', '') for f in top_features], 
                    rotation=45, ha='right', fontsize=10)
ax.legend(loc='upper right', fontsize=11)
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.grid(axis='y', alpha=0.3)

# Add correlation annotation
corr = pol_liwc['difference'].corr(sport_liwc['difference'])
ax.annotate(f'Correlation: r = {corr:.3f}', 
            xy=(0.02, 0.98), xycoords='axes fraction',
            ha='left', va='top', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig(output_dir / 'liwc_comparison.png', dpi=150, bbox_inches='tight')
print(f"✓ Saved: {output_dir / 'liwc_comparison.png'}")
plt.close()

# ============================================================================
# VISUALIZATION 2: Hostility Rates Bar Chart
# ============================================================================
print("Generating hostility comparison chart...")

fig, ax = plt.subplots(figsize=(10, 6))

domains = ['Politics', 'Sports']
hostility_rates = [17.3, 5.2]
colors = ['#e74c3c', '#3498db']

bars = ax.bar(domains, hostility_rates, color=colors, alpha=0.8, width=0.5)

# Add value labels on bars
for i, (bar, rate) in enumerate(zip(bars, hostility_rates)):
    ax.text(bar.get_x() + bar.get_width()/2, rate + 0.5,
            f'{rate}%', ha='center', va='bottom', 
            fontsize=16, fontweight='bold', color=colors[i])

ax.set_ylabel('Hostility Rate (%)', fontsize=12, fontweight='bold')
ax.set_title('Cross-Community Hostility: Politics vs Sports', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_ylim(0, 22)
ax.grid(axis='y', alpha=0.3)

# Add 3.3x annotation
ax.annotate('3.3x more hostile', xy=(0.5, 11), xytext=(0.7, 13),
            fontsize=14, fontweight='bold', color='#e67e22',
            arrowprops=dict(arrowstyle='->', color='#e67e22', lw=2))

plt.tight_layout()
plt.savefig(output_dir / 'hostility_comparison.png', dpi=150, bbox_inches='tight')
print(f"✓ Saved: {output_dir / 'hostility_comparison.png'}")
plt.close()

# ============================================================================
# VISUALIZATION 3: Event Effects Ranking
# ============================================================================
print("Generating event effects chart...")

all_analyses = analyze_all_events(politics, sports, EVENTS)
effects = compare_event_effects(all_analyses)

# Take top 8 events
effects_plot = effects.head(8).sort_values('absolute_effect', ascending=True)

fig, ax = plt.subplots(figsize=(12, 7))

colors = ['#e74c3c' if d == 'politics' else '#3498db' for d in effects_plot['domain']]
y = np.arange(len(effects_plot))

bars = ax.barh(y, effects_plot['absolute_effect'] * 100, color=colors, alpha=0.8)

# Add value labels
for i, (idx, row) in enumerate(effects_plot.iterrows()):
    value = row['absolute_effect'] * 100
    ax.text(value + 0.2, i, f'+{value:.1f}pp', 
            va='center', fontsize=10, fontweight='bold')

ax.set_yticks(y)
ax.set_yticklabels([name[:35] + '...' if len(name) > 35 else name 
                    for name in effects_plot['event_name']], fontsize=10)
ax.set_xlabel('Change in Hostility Rate (percentage points)', 
              fontsize=12, fontweight='bold')
ax.set_title('Event Effects on Hostility\n(During Event vs Baseline)', 
             fontsize=16, fontweight='bold', pad=20)
ax.axvline(x=0, color='black', linewidth=0.5)
ax.grid(axis='x', alpha=0.3)

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#e74c3c', alpha=0.8, label='Politics'),
    Patch(facecolor='#3498db', alpha=0.8, label='Sports')
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=11)

plt.tight_layout()
plt.savefig(output_dir / 'event_effects.png', dpi=150, bbox_inches='tight')
print(f"✓ Saved: {output_dir / 'event_effects.png'}")
plt.close()

# ============================================================================
# VISUALIZATION 4: Camp Hostility Profiles
# ============================================================================
print("Generating camp profiles chart...")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

for idx, (df, title, ax) in enumerate([
    (politics, 'Politics', axes[0]),
    (sports, 'Sports', axes[1])
]):
    profile = get_camp_hostility_profile(df)
    profile_plot = profile.sort_values('outgoing_neg_rate', ascending=True).head(8)
    
    y = np.arange(len(profile_plot))
    height = 0.35
    
    bars1 = ax.barh(y - height/2, profile_plot['outgoing_neg_rate'] * 100, 
                    height, label='Outgoing (as attacker)', 
                    color='#e74c3c', alpha=0.8)
    bars2 = ax.barh(y + height/2, profile_plot['incoming_neg_rate'] * 100, 
                    height, label='Incoming (as target)', 
                    color='#3498db', alpha=0.8)
    
    ax.set_yticks(y)
    ax.set_yticklabels(profile_plot['camp'], fontsize=9)
    ax.set_xlabel('Hostility Rate (%)', fontsize=11, fontweight='bold')
    ax.set_title(f'{title}: Camp Hostility Profiles', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'camp_profiles.png', dpi=150, bbox_inches='tight')
print(f"✓ Saved: {output_dir / 'camp_profiles.png'}")
plt.close()

# ============================================================================
# VISUALIZATION 5: Temporal Pattern (Before/During/After)
# ============================================================================
print("Generating temporal pattern chart...")

# Calculate average hostility for before/during/after across all events
periods_pol = []
periods_sport = []

for name, analysis in all_analyses.items():
    if analysis['period_stats']['during']['count'] >= 10:
        periods = analysis['period_stats']
        if analysis['domain'] == 'politics':
            periods_pol.append({
                'before': periods['before']['negative_rate'] * 100,
                'during': periods['during']['negative_rate'] * 100,
                'after': periods['after']['negative_rate'] * 100
            })
        else:
            periods_sport.append({
                'before': periods['before']['negative_rate'] * 100,
                'during': periods['during']['negative_rate'] * 100,
                'after': periods['after']['negative_rate'] * 100
            })

avg_pol = pd.DataFrame(periods_pol).mean()
avg_sport = pd.DataFrame(periods_sport).mean()

fig, ax = plt.subplots(figsize=(12, 7))

x = [0, 1, 2]
x_labels = ['Before', 'During Event', 'After']

# Politics line
ax.plot(x, avg_pol, 'o-', color='#e74c3c', linewidth=3, 
        markersize=12, label='Politics', alpha=0.8)
# Sports line
ax.plot(x, avg_sport, 'o-', color='#3498db', linewidth=3, 
        markersize=12, label='Sports', alpha=0.8)

# Add value labels
for i, (pol_val, sport_val) in enumerate(zip(avg_pol, avg_sport)):
    ax.text(x[i], pol_val + 0.8, f'{pol_val:.1f}%', 
            ha='center', fontsize=11, fontweight='bold', color='#e74c3c')
    ax.text(x[i], sport_val - 1.2, f'{sport_val:.1f}%', 
            ha='center', fontsize=11, fontweight='bold', color='#3498db')

ax.set_xticks(x)
ax.set_xticklabels(x_labels, fontsize=12, fontweight='bold')
ax.set_ylabel('Hostility Rate (%)', fontsize=12, fontweight='bold')
ax.set_title('The Pattern: Spikes Are Temporary\n(Average across all major events)', 
             fontsize=16, fontweight='bold', pad=20)
ax.legend(loc='upper right', fontsize=12)
ax.grid(alpha=0.3)
ax.set_ylim(0, 25)

# Add annotations
ax.annotate('Spike', xy=(1, avg_pol[1]), xytext=(1.3, avg_pol[1] + 2),
            arrowprops=dict(arrowstyle='->', color='gray', lw=1.5),
            fontsize=11, color='gray')
ax.annotate('Returns to baseline', xy=(2, avg_pol[2]), xytext=(1.5, avg_pol[2] - 3),
            arrowprops=dict(arrowstyle='->', color='gray', lw=1.5),
            fontsize=11, color='gray')

plt.tight_layout()
plt.savefig(output_dir / 'temporal_pattern.png', dpi=150, bbox_inches='tight')
print(f"✓ Saved: {output_dir / 'temporal_pattern.png'}")
plt.close()

# ============================================================================
# VISUALIZATION 6: Correlation Scatter Plot
# ============================================================================
print("Generating correlation scatter plot...")

pol_sig = pol_liwc['difference']
sport_sig = sport_liwc['difference']

# Combine data
comparison_df = pd.DataFrame({
    'politics': pol_sig,
    'sports': sport_sig
}).dropna()

fig, ax = plt.subplots(figsize=(10, 10))

# Scatter plot
ax.scatter(comparison_df['politics'], comparison_df['sports'], 
          s=100, alpha=0.6, color='#9b59b6', edgecolors='white', linewidth=1.5)

# Add regression line
z = np.polyfit(comparison_df['politics'], comparison_df['sports'], 1)
p = np.poly1d(z)
ax.plot(comparison_df['politics'], p(comparison_df['politics']), 
        "r--", alpha=0.8, linewidth=2, label='Linear fit')

# Labels for top features
top_to_label = comparison_df.nlargest(8, 'politics').index
for feature in top_to_label:
    ax.annotate(feature.replace('LIWC_', ''), 
                xy=(comparison_df.loc[feature, 'politics'], 
                    comparison_df.loc[feature, 'sports']),
                xytext=(5, 5), textcoords='offset points',
                fontsize=8, alpha=0.7)

ax.set_xlabel('Politics: LIWC Feature Change', fontsize=12, fontweight='bold')
ax.set_ylabel('Sports: LIWC Feature Change', fontsize=12, fontweight='bold')
ax.set_title('Cross-Domain LIWC Correlation\n(Hostility Signature Comparison)', 
             fontsize=16, fontweight='bold', pad=20)

# Add correlation text
corr = comparison_df['politics'].corr(comparison_df['sports'])
ax.text(0.05, 0.95, f'r = {corr:.3f}\n(n={len(comparison_df)} features)', 
        transform=ax.transAxes, fontsize=14, fontweight='bold',
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

ax.grid(alpha=0.3)
ax.legend(fontsize=11)

# Diagonal reference line
lims = [
    min(ax.get_xlim()[0], ax.get_ylim()[0]),
    max(ax.get_xlim()[1], ax.get_ylim()[1])
]
ax.plot(lims, lims, 'k:', alpha=0.5, zorder=0, linewidth=1)

plt.tight_layout()
plt.savefig(output_dir / 'correlation_scatter.png', dpi=150, bbox_inches='tight')
print(f"✓ Saved: {output_dir / 'correlation_scatter.png'}")
plt.close()

print("\n" + "="*70)
print("✓ ALL VISUALIZATIONS GENERATED SUCCESSFULLY")
print("="*70)
print(f"\nImages saved to: {output_dir.absolute()}")
print("\nGenerated files:")
print("  • liwc_comparison.png")
print("  • hostility_comparison.png")
print("  • event_effects.png")
print("  • camp_profiles.png")
print("  • temporal_pattern.png")
print("  • correlation_scatter.png")
print("\nNow you can:")
print("1. Copy these images to your pages branch")
print("2. Or commit them to this branch and merge to pages")
print("3. Push to GitHub to see them on your site!")
