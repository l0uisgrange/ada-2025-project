import pandas as pd
import kagglehub
import os

def load_data():
    """
    Downloads and loads all required datasets as pandas DataFrames.
    """

    # Downloads dataset from Kaggle
    destination_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data'))
    path = kagglehub.dataset_download("fejwiehf3928uhcwa/ada-2025-project-bald", force_download=True)

    # Moves the dataset to the data directory
    os.makedirs(destination_path, exist_ok=True)
    for filename in os.listdir(path):
        src_file = os.path.join(path, filename)
        dst_file = os.path.join(destination_path, filename)
        if os.path.isfile(src_file):
            os.rename(src_file, dst_file)
            print(f"Moved {filename} to {destination_path}")
    print(f"Creating dataframes")

    # Reddit body dataset
    df_body = pd.read_csv(
        os.path.join(destination_path, "soc-redditHyperlinks-body.tsv"),
        sep="\t",
        parse_dates=["TIMESTAMP"],
    )

    # Reddit title dataset
    df_title = pd.read_csv(
        os.path.join(destination_path, "soc-redditHyperlinks-title.tsv"),
        sep="\t",
        parse_dates=["TIMESTAMP"],
    )

    # Reddit subreddits embeddings
    df_subreddits = pd.read_csv(
        os.path.join(destination_path, "web-redditEmbeddings-subreddits.csv"),
        names=["SUBREDDIT"] + [f"EMBEDDING_{i}" for i in range(300)],
    )

    # Reddit users embeddings
    df_users = pd.read_csv(
        os.path.join(destination_path, "web-redditEmbeddings-users.csv"),
        names=["USER"] + [f"EMBEDDING_{i}" for i in range(300)],
    )

    # Events list
    """df_events = pd.read_csv(
        os.path.join(path, "web-redditEmbeddings-users.csv"),
        names=["USER"] + [f"EMBEDDING_{i}" for i in range(300)],
    )"""

    return df_body, df_title, df_subreddits, df_users
