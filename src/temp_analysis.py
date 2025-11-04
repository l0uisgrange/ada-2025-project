"""
Temporal analysis utilities for tracking Reddit activity over time.

This module provides functions for analyzing subreddit mention patterns,
activity spikes, and time-based aggregations.
"""

import pandas as pd
import numpy as np


def count_mentions_over_time(body_data, title_data, pattern, period='M'):
    """
    Count mentions of a subreddit over time (aggregated by period).
    
    Parameters:
    -----------
    body_data : pd.DataFrame
        Reddit hyperlinks from post bodies
    title_data : pd.DataFrame
        Reddit hyperlinks from post titles
    pattern : str
        Subreddit name to search for
    period : str, optional
        Pandas period string: 'D' (day), 'W' (week), 'M' (month), 'Y' (year)
        Default: 'M'
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with Title Mentions and Body Mentions columns, indexed by period
    """
    # Create copies to avoid modifying original data
    tf = title_data.copy()
    bf = body_data.copy()
    
    # Ensure timestamps are datetime
    tf['TIMESTAMP'] = pd.to_datetime(tf['TIMESTAMP'], errors='coerce')
    bf['TIMESTAMP'] = pd.to_datetime(bf['TIMESTAMP'], errors='coerce')
    
    # Flag relevant rows (subreddit appears as source or target)
    tf['is_relevant'] = (
        (tf['SOURCE_SUBREDDIT'] == pattern) |
        (tf['TARGET_SUBREDDIT'] == pattern)
    )
    bf['is_relevant'] = (
        (bf['SOURCE_SUBREDDIT'] == pattern) |
        (bf['TARGET_SUBREDDIT'] == pattern)
    )
    
    # Filter and assign periods
    df_title = tf[tf['is_relevant']].assign(PERIOD=lambda x: x['TIMESTAMP'].dt.to_period(period))
    df_body = bf[bf['is_relevant']].assign(PERIOD=lambda x: x['TIMESTAMP'].dt.to_period(period))
    
    # Count mentions per period
    counts_title = df_title.groupby('PERIOD').size()
    counts_body = df_body.groupby('PERIOD').size()
    
    # Combine into one DataFrame
    counts_df = pd.DataFrame({
        'Title Mentions': counts_title,
        'Body Mentions': counts_body
    }).fillna(0)
    
    return counts_df


def filter_by_date_range(data, subreddit, start_date, end_date, match_type='target'):
    """
    Filter hyperlinks by subreddit and date range.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Reddit hyperlink data (body or title)
    subreddit : str
        Subreddit name to filter
    start_date : str
        Start date (YYYY-MM-DD format)
    end_date : str
        End date (YYYY-MM-DD format)
    match_type : str, optional
        'target', 'source', or 'both' (default: 'target')
        
    Returns:
    --------
    pd.DataFrame
        Filtered DataFrame
    """
    # Ensure timestamp is datetime
    df = data.copy()
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], errors='coerce')
    
    # Create subreddit filter
    if match_type == 'target':
        subreddit_mask = df['TARGET_SUBREDDIT'] == subreddit
    elif match_type == 'source':
        subreddit_mask = df['SOURCE_SUBREDDIT'] == subreddit
    elif match_type == 'both':
        subreddit_mask = (
            (df['SOURCE_SUBREDDIT'] == subreddit) |
            (df['TARGET_SUBREDDIT'] == subreddit)
        )
    else:
        raise ValueError("match_type must be 'target', 'source', or 'both'")
    
    # Filter by subreddit and date range
    filtered = df[
        subreddit_mask &
        df['TIMESTAMP'].between(start_date, end_date)
    ]
    
    return filtered


def detect_activity_spikes(counts_df, threshold_std=2.0, column='Total'):
    """
    Detect activity spikes using statistical thresholds.
    
    Parameters:
    -----------
    counts_df : pd.DataFrame
        DataFrame with mention counts (from count_mentions_over_time)
    threshold_std : float, optional
        Number of standard deviations above mean to consider a spike (default: 2.0)
    column : str, optional
        Column to analyze: 'Title Mentions', 'Body Mentions', or 'Total' (default: 'Total')
        
    Returns:
    --------
    pd.DataFrame
        Rows where activity is considered a spike, with 'zscore' column added
    """
    df = counts_df.copy()
    
    # Calculate total if needed
    if column == 'Total':
        if 'Title Mentions' in df.columns and 'Body Mentions' in df.columns:
            df['Total'] = df['Title Mentions'] + df['Body Mentions']
        else:
            raise ValueError("counts_df must have 'Title Mentions' and 'Body Mentions' columns")
    
    # Calculate z-scores
    mean_val = df[column].mean()
    std_val = df[column].std()
    
    if std_val == 0:
        # No variation, no spikes
        return pd.DataFrame()
    
    df['zscore'] = (df[column] - mean_val) / std_val
    
    # Filter spikes
    spikes = df[df['zscore'] > threshold_std].copy()
    
    return spikes


def aggregate_activity_by_event(hyperlink_data, events_df, window_days=7):
    """
    Aggregate Reddit activity around event dates.
    
    Parameters:
    -----------
    hyperlink_data : pd.DataFrame
        Combined hyperlink data with TIMESTAMP column
    events_df : pd.DataFrame
        Events data with 'Date' column
    window_days : int, optional
        Number of days before/after event to include (default: 7)
        
    Returns:
    --------
    pd.DataFrame
        Activity counts around each event
    """
    results = []
    
    for _, event in events_df.iterrows():
        event_date = pd.to_datetime(event['Date'])
        start_window = event_date - pd.Timedelta(days=window_days)
        end_window = event_date + pd.Timedelta(days=window_days)
        
        # Filter activity in window
        activity_in_window = hyperlink_data[
            hyperlink_data['TIMESTAMP'].between(start_window, end_window)
        ]
        
        results.append({
            'event_name': event.get('Name', 'Unknown'),
            'event_date': event_date,
            'activity_count': len(activity_in_window),
            'window_start': start_window,
            'window_end': end_window
        })
    
    return pd.DataFrame(results)

