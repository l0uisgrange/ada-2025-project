"""
Statistical Analysis Module for Digital Tribes Project

1. Cross-domain classifier transfer (supervised learning)
2. Logistic regression coefficient comparison
3. Bootstrap/permutation tests for correlation
4. Chi-square tests and effect sizes for proportions
5. Difference-in-differences for event analysis
6. Network analysis metrics
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report,
    confusion_matrix, f1_score
)
from sklearn.preprocessing import StandardScaler
import networkx as nx
import warnings

warnings.filterwarnings('ignore')


# =============================================================================
# 1. CROSS-DOMAIN CLASSIFIER TRANSFER
# =============================================================================

def prepare_classification_data(df, feature_cols):
    """
    Prepare data for classification.

    Parameters:
        df: DataFrame with LINK_SENTIMENT and feature columns
        feature_cols: list of feature column names

    Returns:
        X: feature matrix (scaled)
        y: binary target (1=hostile, 0=friendly)
        scaler: fitted StandardScaler
    """
    # Filter to available columns
    available_cols = [c for c in feature_cols if c in df.columns]

    X = df[available_cols].fillna(0).values
    y = np.asarray(df['LINK_SENTIMENT'] == -1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler, available_cols


def cross_domain_classifier_test(df_train, df_test, feature_cols,
                                 model_type='logistic', random_state=42):
    """
    Train a classifier on one domain and test on another.
    This tests whether hostility patterns transfer across domains.

    Parameters:
        df_train: DataFrame for training (e.g., politics)
        df_test: DataFrame for testing (e.g., sports)
        feature_cols: list of feature column names
        model_type: 'logistic' or 'random_forest'
        random_state: for reproducibility

    Returns:
        dict with metrics and model details
    """
    # Prepare training data
    X_train, y_train, scaler, used_cols = prepare_classification_data(df_train, feature_cols)

    # Prepare test data with same scaler
    available_cols = [c for c in used_cols if c in df_test.columns]
    X_test = df_test[available_cols].fillna(0).values
    y_test = np.asarray(df_test['LINK_SENTIMENT'] == -1)
    X_test_scaled = scaler.transform(X_test)

    # Initialize model
    if model_type == 'logistic':
        model = LogisticRegression(random_state=random_state, max_iter=1000,
                                   class_weight='balanced')
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=random_state,
                                       class_weight='balanced')

    # Train
    model.fit(X_train, y_train)

    # Predict on test domain
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    # Calculate metrics
    results = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_prob),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'n_train': len(y_train),
        'n_test': len(y_test),
        'train_hostile_rate': y_train.mean(),
        'test_hostile_rate': y_test.mean(),
        'features_used': used_cols,
        'model': model
    }

    return results


def same_domain_baseline(df, feature_cols, model_type='logistic',
                         test_size=0.3, random_state=42):
    """
    Train and test on the same domain (baseline comparison).

    Returns:
        dict with metrics
    """
    X, y, scaler, used_cols = prepare_classification_data(df, feature_cols)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    if model_type == 'logistic':
        model = LogisticRegression(random_state=random_state, max_iter=1000,
                                   class_weight='balanced')
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=random_state,
                                       class_weight='balanced')

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_prob),
        'n_test': len(y_test)
    }


def full_transfer_analysis(politics_df, sports_df, feature_cols):
    """
    Run complete cross-domain transfer analysis.

    Returns:
        dict with all results and comparison
    """
    results = {}

    # Same-domain baselines
    print("Computing same-domain baselines...")
    results['politics_baseline'] = same_domain_baseline(politics_df, feature_cols)
    results['sports_baseline'] = same_domain_baseline(sports_df, feature_cols)

    # Cross-domain transfer
    print("Testing politics → sports transfer...")
    results['politics_to_sports'] = cross_domain_classifier_test(
        politics_df, sports_df, feature_cols
    )

    print("Testing sports → politics transfer...")
    results['sports_to_politics'] = cross_domain_classifier_test(
        sports_df, politics_df, feature_cols
    )

    # Summary
    results['summary'] = pd.DataFrame({
        'Scenario': ['Politics (same-domain)', 'Sports (same-domain)',
                     'Politics → Sports', 'Sports → Politics'],
        'AUC-ROC': [
            results['politics_baseline']['roc_auc'],
            results['sports_baseline']['roc_auc'],
            results['politics_to_sports']['roc_auc'],
            results['sports_to_politics']['roc_auc']
        ],
        'Accuracy': [
            results['politics_baseline']['accuracy'],
            results['sports_baseline']['accuracy'],
            results['politics_to_sports']['accuracy'],
            results['sports_to_politics']['accuracy']
        ],
        'F1-Score': [
            results['politics_baseline']['f1_score'],
            results['sports_baseline']['f1_score'],
            results['politics_to_sports']['f1_score'],
            results['sports_to_politics']['f1_score']
        ]
    })

    return results


# =============================================================================
# 2. LOGISTIC REGRESSION COEFFICIENT COMPARISON
# =============================================================================

def fit_logistic_with_coefficients(df, feature_cols, random_state=42):
    """
    Fit logistic regression and extract coefficients.

    Returns:
        dict with model, coefficients, and statistics
    """
    X, y, scaler, used_cols = prepare_classification_data(df, feature_cols)

    model = LogisticRegression(random_state=random_state, max_iter=1000,
                               class_weight='balanced', penalty='l2', C=1.0)
    model.fit(X, y)

    # Extract coefficients
    coef_df = pd.DataFrame({
        'feature': used_cols,
        'coefficient': model.coef_[0],
        'abs_coefficient': np.abs(model.coef_[0])
    }).sort_values('abs_coefficient', ascending=False)

    # Cross-validation score
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')

    return {
        'model': model,
        'coefficients': coef_df,
        'cv_auc_mean': cv_scores.mean(),
        'cv_auc_std': cv_scores.std(),
        'feature_names': used_cols
    }


def compare_logistic_coefficients(politics_df, sports_df, feature_cols):
    """
    Compare logistic regression coefficients between domains.
    Tests whether the same features predict hostility in both domains.

    Returns:
        dict with comparison statistics
    """
    print("Fitting logistic regression for politics...")
    pol_results = fit_logistic_with_coefficients(politics_df, feature_cols)

    print("Fitting logistic regression for sports...")
    sport_results = fit_logistic_with_coefficients(sports_df, feature_cols)

    # Align coefficients
    pol_coef = pol_results['coefficients'].set_index('feature')['coefficient']
    sport_coef = sport_results['coefficients'].set_index('feature')['coefficient']

    common_features = pol_coef.index.intersection(sport_coef.index)
    pol_aligned = pol_coef[common_features]
    sport_aligned = sport_coef[common_features]

    # Calculate correlation of coefficients
    pearson_r, pearson_p = stats.pearsonr(pol_aligned, sport_aligned)
    spearman_r, spearman_p = stats.spearmanr(np.asarray(pol_aligned), np.asarray(sport_aligned)) # type: ignore[arg-type]

    # Sign agreement (do coefficients point the same direction?)
    sign_agreement = np.mean(np.sign(pol_aligned) == np.sign(sport_aligned))

    # Top features comparison
    pol_top10 = set(pol_results['coefficients'].head(10)['feature'])
    sport_top10 = set(sport_results['coefficients'].head(10)['feature'])
    top10_overlap = len(pol_top10 & sport_top10)

    comparison_df = pd.DataFrame({
        'feature': common_features,
        'politics_coef': pol_aligned.values,
        'sports_coef': sport_aligned.values,
        'same_sign': np.sign(pol_aligned.values) == np.sign(sport_aligned.values)
    }).sort_values('politics_coef', key=abs, ascending=False)

    return {
        'politics_results': pol_results,
        'sports_results': sport_results,
        'coefficient_comparison': comparison_df,
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'spearman_r': spearman_r,
        'spearman_p': spearman_p,
        'sign_agreement': sign_agreement,
        'top10_overlap': top10_overlap,
        'n_features': len(common_features)
    }


# =============================================================================
# 3. BOOTSTRAP AND PERMUTATION TESTS
# =============================================================================

def bootstrap_correlation(series1, series2, n_bootstrap=1000,
                          confidence_level=0.95, random_state=42):
    """
    Bootstrap confidence interval for correlation.

    Parameters:
        series1, series2: aligned Series to correlate
        n_bootstrap: number of bootstrap samples
        confidence_level: CI level (default 95%)

    Returns:
        dict with observed correlation and CI
    """
    np.random.seed(random_state)

    # Align series
    common_idx = series1.index.intersection(series2.index)
    s1 = series1[common_idx].values
    s2 = series2[common_idx].values
    n = len(s1)

    # Observed correlation
    observed_r = np.corrcoef(s1, s2)[0, 1]

    # Bootstrap
    bootstrap_correlations = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(n, size=n, replace=True)
        boot_r = np.corrcoef(s1[indices], s2[indices])[0, 1]
        bootstrap_correlations.append(boot_r)

    bootstrap_correlations = np.array(bootstrap_correlations)

    # Confidence interval
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_correlations, alpha / 2 * 100)
    ci_upper = np.percentile(bootstrap_correlations, (1 - alpha / 2) * 100)

    return {
        'observed_r': observed_r,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'confidence_level': confidence_level,
        'bootstrap_std': bootstrap_correlations.std(),
        'bootstrap_distribution': bootstrap_correlations,
        'n_features': n
    }


def permutation_test_correlation(series1, series2, n_permutations=1000,
                                 random_state=42):
    """
    Permutation test for correlation significance.
    Tests null hypothesis that correlation = 0.

    Returns:
        dict with observed correlation and p-value
    """
    np.random.seed(random_state)

    # Align series
    common_idx = series1.index.intersection(series2.index)
    s1 = series1[common_idx].values
    s2 = series2[common_idx].values

    # Observed correlation
    observed_r = np.corrcoef(s1, s2)[0, 1]

    # Permutation test
    permuted_correlations = []
    for _ in range(n_permutations):
        s2_permuted = np.random.permutation(s2)
        perm_r = np.corrcoef(s1, s2_permuted)[0, 1]
        permuted_correlations.append(perm_r)

    permuted_correlations = np.array(permuted_correlations)

    # Two-tailed p-value
    p_value = (np.abs(permuted_correlations) >= np.abs(observed_r)).mean()

    return {
        'observed_r': observed_r,
        'p_value': p_value,
        'permutation_distribution': permuted_correlations,
        'permutation_mean': permuted_correlations.mean(),
        'permutation_std': permuted_correlations.std()
    }


def full_correlation_significance(series1, series2, n_bootstrap=1000,
                                  n_permutations=1000):
    """
    Complete correlation significance analysis.
    Combines bootstrap CI and permutation test.

    Returns:
        dict with all results
    """
    print("Running bootstrap analysis...")
    bootstrap_results = bootstrap_correlation(series1, series2, n_bootstrap)

    print("Running permutation test...")
    permutation_results = permutation_test_correlation(series1, series2, n_permutations)

    # Standard parametric test for comparison
    common_idx = series1.index.intersection(series2.index)
    pearson_r, pearson_p = stats.pearsonr(series1[common_idx], series2[common_idx])

    return {
        'observed_r': bootstrap_results['observed_r'],
        'bootstrap_ci_95': (bootstrap_results['ci_lower'], bootstrap_results['ci_upper']),
        'bootstrap_std': bootstrap_results['bootstrap_std'],
        'permutation_p_value': permutation_results['p_value'],
        'parametric_p_value': pearson_p,
        'bootstrap_distribution': bootstrap_results['bootstrap_distribution'],
        'permutation_distribution': permutation_results['permutation_distribution'],
        'n_features': bootstrap_results['n_features']
    }


# =============================================================================
# 4. PROPORTION TESTS AND EFFECT SIZES
# =============================================================================

def test_proportion_difference(n1_success, n1_total, n2_success, n2_total):
    """
    Test difference between two proportions.

    Parameters:
        n1_success: number of successes in group 1
        n1_total: total in group 1
        n2_success: number of successes in group 2
        n2_total: total in group 2

    Returns:
        dict with test statistics and effect sizes
    """
    p1 = n1_success / n1_total
    p2 = n2_success / n2_total

    # Pooled proportion
    p_pooled = (n1_success + n2_success) / (n1_total + n2_total)

    # Standard error
    se = np.sqrt(p_pooled * (1 - p_pooled) * (1 / n1_total + 1 / n2_total))

    # Z-statistic
    z_stat = (p1 - p2) / se
    p_value = 2 * (1 - stats.norm.cdf(np.abs(z_stat)))

    # Chi-square test
    contingency_table = np.array([
        [n1_success, n1_total - n1_success],
        [n2_success, n2_total - n2_success]
    ])
    chi2, chi2_p, dof, expected = stats.chi2_contingency(contingency_table)

    # Effect size: Cohen's h
    phi1 = 2 * np.arcsin(np.sqrt(p1))
    phi2 = 2 * np.arcsin(np.sqrt(p2))
    cohens_h = phi1 - phi2

    # Risk ratio
    risk_ratio = p1 / p2 if p2 > 0 else np.inf

    # Odds ratio
    odds1 = p1 / (1 - p1) if p1 < 1 else np.inf
    odds2 = p2 / (1 - p2) if p2 < 1 else np.inf
    odds_ratio = odds1 / odds2 if odds2 > 0 else np.inf

    # 95% CI for difference
    se_diff = np.sqrt(p1 * (1 - p1) / n1_total + p2 * (1 - p2) / n2_total)
    ci_lower = (p1 - p2) - 1.96 * se_diff
    ci_upper = (p1 - p2) + 1.96 * se_diff

    return {
        'p1': p1,
        'p2': p2,
        'difference': p1 - p2,
        'difference_ci_95': (ci_lower, ci_upper),
        'z_statistic': z_stat,
        'z_p_value': p_value,
        'chi2_statistic': chi2,
        'chi2_p_value': chi2_p,
        'cohens_h': cohens_h,
        'cohens_h_interpretation': interpret_cohens_h(cohens_h),
        'risk_ratio': risk_ratio,
        'odds_ratio': odds_ratio
    }


def interpret_cohens_h(h):
    """Interpret Cohen's h effect size."""
    h_abs = abs(h)
    if h_abs < 0.2:
        return 'small'
    elif h_abs < 0.5:
        return 'medium'
    elif h_abs < 0.8:
        return 'large'
    else:
        return 'very large'


def compare_domain_hostility(politics_df, sports_df):
    """
    Statistical comparison of hostility between politics and sports.

    Returns:
        dict with proportion test results
    """
    pol_hostile = np.sum(politics_df['LINK_SENTIMENT'] == -1)
    pol_total = len(politics_df)

    sport_hostile = np.sum(sports_df['LINK_SENTIMENT'] == -1)
    sport_total = len(sports_df)

    results = test_proportion_difference(pol_hostile, pol_total,
                                         sport_hostile, sport_total)

    results['politics_n'] = pol_total
    results['politics_hostile'] = pol_hostile
    results['sports_n'] = sport_total
    results['sports_hostile'] = sport_hostile

    return results


# =============================================================================
# 5. DIFFERENCE-IN-DIFFERENCES FOR EVENT ANALYSIS
# =============================================================================

def event_diff_in_diff(df, event_date, treatment_camps,
                       pre_days=7, post_days=7):
    """
    Difference-in-differences analysis for event effects.

    Compares change in hostility for treatment camps vs control camps
    around an event.

    Parameters:
        df: DataFrame with TIMESTAMP, source_camp, LINK_SENTIMENT
        event_date: datetime of the event
        treatment_camps: list of camps expected to be affected
        pre_days: days before event for "pre" period
        post_days: days after event for "post" period

    Returns:
        dict with DiD estimate and statistics
    """
    if isinstance(event_date, str):
        event_date = pd.to_datetime(event_date)

    # Define time periods
    pre_start = event_date - pd.Timedelta(days=pre_days)
    post_end = event_date + pd.Timedelta(days=post_days)

    # Filter data
    df_filtered = df[
        (df['TIMESTAMP'] >= pre_start) &
        (df['TIMESTAMP'] <= post_end)
        ].copy()

    # Assign treatment/control and pre/post
    df_filtered['is_treatment'] = df_filtered['source_camp'].isin(treatment_camps)
    df_filtered['is_post'] = df_filtered['TIMESTAMP'] >= event_date
    df_filtered['is_hostile'] = np.asarray(df_filtered['LINK_SENTIMENT'] == -1)

    # Calculate group means
    groups = df_filtered.groupby(['is_treatment', 'is_post'])['is_hostile'].agg(['mean', 'count', 'sum'])

    try:
        # Pre-period
        treat_pre = groups.loc[(True, False), 'mean']
        control_pre = groups.loc[(False, False), 'mean']

        # Post-period
        treat_post = groups.loc[(True, True), 'mean']
        control_post = groups.loc[(False, True), 'mean']

        # DiD estimate
        treat_change = treat_post - treat_pre
        control_change = control_post - control_pre
        did_estimate = treat_change - control_change

        # Counts for SE calculation
        n_treat_pre = groups.loc[(True, False), 'count']
        n_treat_post = groups.loc[(True, True), 'count']
        n_control_pre = groups.loc[(False, False), 'count']
        n_control_post = groups.loc[(False, True), 'count']

        # Standard error (approximate)
        se_treat = np.sqrt(
            treat_pre * (1 - treat_pre) / n_treat_pre +
            treat_post * (1 - treat_post) / n_treat_post
        )
        se_control = np.sqrt(
            control_pre * (1 - control_pre) / n_control_pre +
            control_post * (1 - control_post) / n_control_post
        )
        se_did = np.sqrt(se_treat ** 2 + se_control ** 2)

        # t-statistic and p-value
        t_stat = did_estimate / se_did if se_did > 0 else 0
        p_value = 2 * (
                    1 - stats.t.cdf(np.abs(t_stat), df=n_treat_pre + n_treat_post + n_control_pre + n_control_post - 4))

        return {
            'did_estimate': did_estimate,
            'se': se_did,
            't_statistic': t_stat,
            'p_value': p_value,
            'ci_95': (did_estimate - 1.96 * se_did, did_estimate + 1.96 * se_did),
            'treatment_pre': treat_pre,
            'treatment_post': treat_post,
            'treatment_change': treat_change,
            'control_pre': control_pre,
            'control_post': control_post,
            'control_change': control_change,
            'n_treatment_pre': n_treat_pre,
            'n_treatment_post': n_treat_post,
            'n_control_pre': n_control_pre,
            'n_control_post': n_control_post,
            'groups': groups.reset_index()
        }

    except KeyError as e:
        return {
            'error': f'Insufficient data for some groups: {e}',
            'groups': groups.reset_index() if len(groups) > 0 else None
        }


def analyze_event_with_did(df, event_name, event_date, domain,
                           treatment_camps, control_camps):
    """
    Full DiD analysis for a specific event.

    Returns:
        dict with event name and DiD results
    """
    results = event_diff_in_diff(df, event_date, treatment_camps, control_camps)
    results['event_name'] = event_name
    results['event_date'] = event_date
    results['domain'] = domain
    results['treatment_camps'] = treatment_camps
    results['control_camps'] = control_camps

    return results


# =============================================================================
# 6. NETWORK ANALYSIS
# =============================================================================

def build_interaction_network(df, min_interactions=10, weight_col='count'):
    """
    Build directed network of cross-community interactions.

    Parameters:
        df: DataFrame with source_camp, target_camp, LINK_SENTIMENT
        min_interactions: minimum interactions to include edge
        weight_col: 'count' or 'negative_rate'

    Returns:
        networkx DiGraph
    """
    # Aggregate interactions
    df_filtered = df[(df['source_camp'] != 'other') & (df['target_camp'] != 'other')]

    agg = df_filtered.groupby(['source_camp', 'target_camp']).agg(
        count=('LINK_SENTIMENT', 'size'),
        negative_count=('LINK_SENTIMENT', lambda x: np.sum(x == -1))
    ).reset_index()

    agg['negative_rate'] = agg['negative_count'] / agg['count']

    # Filter by minimum interactions
    agg_filtered = agg[agg['count'] >= min_interactions]

    # Build graph
    G = nx.DiGraph()

    for _, row in agg_filtered.iterrows():
        weight = row[weight_col] if weight_col in row else row['count']
        G.add_edge(row['source_camp'], row['target_camp'],
                   weight=weight,
                   count=row['count'],
                   negative_rate=row['negative_rate'])

    return G


def calculate_network_metrics(G):
    """
    Calculate network metrics for the interaction graph.

    Returns:
        dict with various network metrics
    """
    metrics = {
        'n_nodes': G.number_of_nodes(),
        'n_edges': G.number_of_edges(),
        'density': nx.density(G)
    }

    # In-degree (how much attention a camp gets)
    in_degree = dict(G.in_degree(weight='count'))
    metrics['in_degree'] = in_degree
    metrics['top_targets'] = sorted(in_degree.items(), key=lambda x: x[1], reverse=True)[:5]

    # Out-degree (how much a camp talks about others)
    out_degree = dict(G.out_degree(weight='count'))
    metrics['out_degree'] = out_degree
    metrics['top_sources'] = sorted(out_degree.items(), key=lambda x: x[1], reverse=True)[:5]

    # Weighted in/out by hostility
    hostile_in = {}
    hostile_out = {}
    for node in G.nodes():
        hostile_in[node] = sum(G[u][node]['negative_rate'] * G[u][node]['count']
                               for u in G.predecessors(node))
        hostile_out[node] = sum(G[node][v]['negative_rate'] * G[node][v]['count']
                                for v in G.successors(node))

    metrics['hostile_in'] = hostile_in
    metrics['hostile_out'] = hostile_out
    metrics['top_hostile_targets'] = sorted(hostile_in.items(), key=lambda x: x[1], reverse=True)[:5]
    metrics['top_hostile_sources'] = sorted(hostile_out.items(), key=lambda x: x[1], reverse=True)[:5]

    # PageRank (importance in network)
    try:
        pagerank = nx.pagerank(G, weight='count')
        metrics['pagerank'] = pagerank
        metrics['top_pagerank'] = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:5]
    except:
        metrics['pagerank'] = None

    # Reciprocity
    metrics['reciprocity'] = nx.reciprocity(G)

    return metrics


def compare_network_structures(G1, G2, name1='Network 1', name2='Network 2'):
    """
    Compare two network structures.

    Returns:
        dict with comparison metrics
    """
    metrics1 = calculate_network_metrics(G1)
    metrics2 = calculate_network_metrics(G2)

    comparison = {
        'metric': ['Nodes', 'Edges', 'Density', 'Reciprocity'],
        name1: [metrics1['n_nodes'], metrics1['n_edges'],
                metrics1['density'], metrics1['reciprocity']],
        name2: [metrics2['n_nodes'], metrics2['n_edges'],
                metrics2['density'], metrics2['reciprocity']]
    }

    return {
        name1: metrics1,
        name2: metrics2,
        'comparison_table': pd.DataFrame(comparison)
    }


def get_network_edges_df(G):
    """
    Convert network edges to DataFrame for visualization.

    Returns:
        DataFrame with source, target, weight, negative_rate
    """
    edges = []
    for u, v, data in G.edges(data=True):
        edges.append({
            'source': u,
            'target': v,
            'count': data.get('count', 1),
            'negative_rate': data.get('negative_rate', 0),
            'weight': data.get('weight', 1)
        })

    return pd.DataFrame(edges).sort_values('count', ascending=False)


# =============================================================================
# SUMMARY FUNCTION
# =============================================================================

def run_all_statistical_tests(politics_df, sports_df, feature_cols):
    """
    Run all statistical tests and return comprehensive results.

    This is the main entry point for the statistical analysis.

    Returns:
        dict with all test results
    """
    results = {}

    print("=" * 60)
    print("RUNNING COMPREHENSIVE STATISTICAL ANALYSIS")
    print("=" * 60)

    # 1. Proportion test
    print("\n1. Testing proportion difference (politics vs sports hostility)...")
    results['proportion_test'] = compare_domain_hostility(politics_df, sports_df)

    # 2. Cross-domain classifier
    print("\n2. Running cross-domain classifier transfer test...")
    results['classifier_transfer'] = full_transfer_analysis(
        politics_df, sports_df, feature_cols
    )

    # 3. Logistic regression comparison
    print("\n3. Comparing logistic regression coefficients...")
    results['coefficient_comparison'] = compare_logistic_coefficients(
        politics_df, sports_df, feature_cols
    )

    # 4. Bootstrap correlation (need LIWC signatures)
    print("\n4. Running bootstrap/permutation tests for LIWC correlation...")
    from interaction_analysis import get_liwc_by_sentiment
    pol_sig = get_liwc_by_sentiment(politics_df)['difference']
    sport_sig = get_liwc_by_sentiment(sports_df)['difference']
    results['correlation_significance'] = full_correlation_significance(pol_sig, sport_sig)

    # 5. Network analysis
    print("\n5. Building and analyzing interaction networks...")
    G_pol = build_interaction_network(politics_df)
    G_sport = build_interaction_network(sports_df)
    results['network_analysis'] = compare_network_structures(
        G_pol, G_sport, 'Politics', 'Sports'
    )
    results['network_politics'] = G_pol
    results['network_sports'] = G_sport

    print("\n" + "=" * 60)
    print("STATISTICAL ANALYSIS COMPLETE")
    print("=" * 60)

    return results