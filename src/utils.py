# Utility functions (data loading etc..) - like a helper file
import pandas as pd
from src import config

def load_data(    
    body_path: str = config.BODY_HYPERLINKS_DATA,
    title_path: str = config.TITLE_HYPERLINKS_DATA,
    subreddits_path: str = config.SUBREDDITS_EMBEDDINGS_DATA,
    users_path: str = config.USERS_EMBEDDINGS_DATA
):
    """
    Load all required datasets as pandas DataFrames.

    Parameters:
        body_path (str): Path to the body hyperlinks data file (TSV).
        title_path (str): Path to the title hyperlinks data file (TSV).
        subreddits_path (str): Path to the subreddit embeddings data file (CSV).
        users_path (str): Path to the user embeddings data file (CSV).

    Returns:
        tuple: A tuple containing four pandas DataFrames in the following order:
            - body_data (pd.DataFrame): DataFrame for body hyperlinks.
            - title_data (pd.DataFrame): DataFrame for title hyperlinks.
            - subreddits_data (pd.DataFrame): DataFrame for subreddit embeddings.
            - users_data (pd.DataFrame): DataFrame for user embeddings.
    """
    body_data = pd.read_csv(body_path, sep='\t', parse_dates=['TIMESTAMP'])
    title_data = pd.read_csv(title_path, sep='\t', parse_dates=['TIMESTAMP'])
    subreddits_data = pd.read_csv(subreddits_path, names=['SUBREDDIT'] + [f'EMBEDDING_{i}' for i in range(300)])
    users_data = pd.read_csv(users_path, names=['USER'] + [f'EMBEDDING_{i}' for i in range(300)])

    return body_data, title_data, subreddits_data, users_data