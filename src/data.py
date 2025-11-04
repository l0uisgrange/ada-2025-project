import pandas as pd
import kagglehub
import os

EMBEDDING_DIM = 300
START_DATE = "2014-01-01"
END_DATE = "2017-04-30"

def download_data():
    """
    Downloads all required datasets from Kaggle and returns the path to the downloaded files.
    
    Returns:
        str: Path to the downloaded dataset directory
    """
    return kagglehub.dataset_download("fejwiehf3928uhcwa/ada-2025-project-bald")


def load_data():
    """
    Loads all required Reddit and event datasets as pandas DataFrames from Kaggle.
    
    Returns:
        tuple: (df_body, df_title, df_subreddits, df_users, df_events)
            - df_body: Reddit hyperlinks from post bodies
            - df_title: Reddit hyperlinks from post titles
            - df_subreddits: Subreddit embeddings (300-dimensional vectors)
            - df_users: User embeddings (300-dimensional vectors)
            - df_events: Global holidays filtered for 2014-2017
    """

    datasets_path = download_data()
    print("Loading datasets from", datasets_path)

    # Reddit body dataset
    df_body = pd.read_csv(
        os.path.join(datasets_path, "soc-redditHyperlinks-body.tsv"),
        sep="\t",
        parse_dates=["TIMESTAMP"],
    )

    # Reddit title dataset
    df_title = pd.read_csv(
        os.path.join(datasets_path, "soc-redditHyperlinks-title.tsv"),
        sep="\t",
        parse_dates=["TIMESTAMP"],
    )

    # Reddit subreddits embeddings
    df_subreddits = pd.read_csv(
        os.path.join(datasets_path, "web-redditEmbeddings-subreddits.csv"),
        names=["SUBREDDIT"] + [f"EMBEDDING_{i}" for i in range(EMBEDDING_DIM)],
    )

    # Reddit users embeddings
    df_users = pd.read_csv(
        os.path.join(datasets_path, "web-redditEmbeddings-users.csv"),
        names=["USER"] + [f"EMBEDDING_{i}" for i in range(EMBEDDING_DIM)],
    )

    # Events list
    df_events_raw = pd.read_csv(
        os.path.join(datasets_path, "global_holidays.csv"),
        parse_dates=["Date"],
    )

    # Only take relevant events
    df_events = df_events_raw.loc[(df_events_raw['Date'] >= START_DATE) & (df_events_raw['Date'] <= END_DATE)].copy()

    print("Completed")
    return df_body, df_title, df_subreddits, df_users, df_events
