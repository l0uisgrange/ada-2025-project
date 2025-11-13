import pandas as pd

from .consts import MIN_ACTIVE_POSTS, DATA_START_DATE, DATA_END_DATE

def count_subreddit_activity(body_data, title_data, subreddits_data):
    """
    Count how many times each subreddit appears as SOURCE or TARGET.
    
    Args:
        body_data (pd.DataFrame): Reddit hyperlinks from post bodies
        title_data (pd.DataFrame): Reddit hyperlinks from post titles
        subreddits_data (pd.DataFrame): Subreddit embeddings data
        
    Returns:
        pd.DataFrame with columns: SUBREDDIT, POST_COUNT_SOURCE, POST_COUNT_TARGET, POST_COUNT_TOTAL
    """
    # ========================================
    # Count SOURCE appearances
    count_source_body = body_data['SOURCE_SUBREDDIT'].value_counts().reset_index()
    count_source_body.columns = ['SUBREDDIT', 'POST_COUNT_BODY']
    
    count_source_title = title_data['SOURCE_SUBREDDIT'].value_counts().reset_index()
    count_source_title.columns = ['SUBREDDIT', 'POST_COUNT_TITLE']
    
    combined_source = pd.merge(count_source_body, count_source_title,
                               on='SUBREDDIT', how='outer').fillna(0)
    
    combined_source['POST_COUNT_SOURCE'] = (combined_source['POST_COUNT_BODY'] +
                                            combined_source['POST_COUNT_TITLE'])
    
    # ========================================
    # Count TARGET appearances
    count_target_body = body_data['TARGET_SUBREDDIT'].value_counts().reset_index()
    count_target_body.columns = ['SUBREDDIT', 'POST_COUNT_BODY']
    
    count_target_title = title_data['TARGET_SUBREDDIT'].value_counts().reset_index()
    count_target_title.columns = ['SUBREDDIT', 'POST_COUNT_TITLE']
    
    combined_target = pd.merge(count_target_body, count_target_title, 
                               on='SUBREDDIT', how='outer').fillna(0)
    
    combined_target['POST_COUNT_TARGET'] = (combined_target['POST_COUNT_BODY'] + 
                                            combined_target['POST_COUNT_TITLE'])
    
    # ========================================
    # Merge SOURCE and TARGET counts
    source_count = pd.merge(subreddits_data[['SUBREDDIT']], combined_source,
                           on='SUBREDDIT', how='left').fillna(0)
    
    target_count = pd.merge(subreddits_data[['SUBREDDIT']], combined_target,
                           on='SUBREDDIT', how='left').fillna(0)
    
    # ========================================
    # Calculate total counts
    total_counts = pd.merge(source_count, target_count, 
                           on='SUBREDDIT', how='outer', 
                           suffixes=('_SOURCE', '_TARGET')).fillna(0)
    
    total_counts['POST_COUNT_TOTAL'] = (total_counts['POST_COUNT_SOURCE'] + 
                                        total_counts['POST_COUNT_TARGET'])
    
    return total_counts


def filter_active_subreddits(body_data, title_data, subreddits_data, 
                             min_posts=MIN_ACTIVE_POSTS):
    """
    Filter subreddits to only include active ones (>min_posts total appearances).
    
    Args:
        body_data (pd.DataFrame): Reddit hyperlinks from post bodies
        title_data (pd.DataFrame): Reddit hyperlinks from post titles
        subreddits_data (pd.DataFrame): Subreddit embeddings data
        min_posts (int): Minimum number of total posts (default: MIN_ACTIVE_POSTS from consts)
        
    Returns:
        tuple of (pd.DataFrame, pd.DataFrame)
            - active_subreddits: Filtered subreddit embeddings
            - total_counts: Activity counts for all subreddits
    """
    
    
    total_counts = count_subreddit_activity(body_data, title_data, subreddits_data)
    
    # Filter subreddits with sufficient activity
    active_mask = total_counts['POST_COUNT_TOTAL'] > min_posts
    active_subreddit_names = total_counts[active_mask]['SUBREDDIT']
    
    active_subreddits = subreddits_data[
        subreddits_data['SUBREDDIT'].isin(active_subreddit_names)
    ].copy()
    
    return active_subreddits, total_counts


def filter_events_by_date(events_data, start_date=None, end_date=None):
    """
    Filter events to a specific date range.
    
    Parameters:
    -----------
    events_data : pd.DataFrame
        Events data with 'Date' column
    start_date : str, optional
        Start date (default: DATA_START_DATE from consts)
    end_date : str, optional
        End date (default: DATA_END_DATE from consts)
        
    Returns:
    --------
    pd.DataFrame
        Filtered events within date range
    """
    if start_date is None:
        start_date = DATA_START_DATE
    if end_date is None:
        end_date = DATA_END_DATE
    
    filtered = events_data[
        (events_data['Date'] >= start_date) & 
        (events_data['Date'] <= end_date)
    ].copy()
    
    return filtered


def merge_hyperlink_data(body_data, title_data):
    """
    Merge body and title hyperlink datasets into a single DataFrame.
    
    Parameters:
    -----------
    body_data : pd.DataFrame
        Reddit hyperlinks from post bodies
    title_data : pd.DataFrame
        Reddit hyperlinks from post titles
        
    Returns:
    --------
    pd.DataFrame
        Combined hyperlink data with 'source' column indicating origin
    """
    body_copy = body_data.copy()
    title_copy = title_data.copy()
    
    body_copy['source'] = 'body'
    title_copy['source'] = 'title'
    
    combined = pd.concat([body_copy, title_copy], ignore_index=True)
    combined['TIMESTAMP'] = pd.to_datetime(combined['TIMESTAMP'], errors='coerce')
    
    return combined