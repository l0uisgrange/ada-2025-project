import pandas as pd
import kagglehub
import os

def download_data():
    """
    Downloads all required datasets from Kaggle and returns the path to the downloaded files.
    """

    return kagglehub.dataset_download("fejwiehf3928uhcwa/ada-2025-project-bald")

def load_data():
    """
    Loads all required datasets as pandas DataFrames from the Kaggle path.
    """

    datasets_path = download_data()
    print("Loading datasets from: ", datasets_path)

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
        names=["SUBREDDIT"] + [f"EMBEDDING_{i}" for i in range(300)],
    )

    # Reddit users embeddings
    df_users = pd.read_csv(
        os.path.join(datasets_path, "web-redditEmbeddings-users.csv"),
        names=["USER"] + [f"EMBEDDING_{i}" for i in range(300)],
    )

    # Events list
    """df_events = pd.read_csv(
        os.path.join(path, "web-redditEmbeddings-users.csv"),
        names=["USER"] + [f"EMBEDDING_{i}" for i in range(300)],
    )"""

    print("Completed")
    return df_body, df_title, df_subreddits, df_users
