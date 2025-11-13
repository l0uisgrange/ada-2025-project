import pandas as pd
import kagglehub
import os

from src.consts import *
from src.utils import parse_properties


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
            - df_english_pl
            - df_esports
            - df_superbowl
            - df_holidays: Global holidays filtered for 2014-2017
    """

    datasets_path = download_data()

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
        names=["SUBREDDIT"] + [f"EMBEDDING_{i}" for i in range(EMBEDDING_DIMENSIONS)],
    )


    # Premier League dataset
    df_english_pl = pd.read_csv(
        os.path.join(datasets_path, "english-premier-league-results_2014-01-2017-04.csv")
    )

    # E-Sports tournaments dataset
    df_esports = pd.read_csv(
        os.path.join(datasets_path, "esports_tournaments_2014_2017.csv")
    )

    # Superbowl dataset
    df_superbowl = pd.read_csv(
        os.path.join(datasets_path, "superbowl_2014_2017.csv")
    )

    df_body = parse_properties(df_body)
    df_title = parse_properties(df_title)

    return df_body, df_title, df_subreddits, df_english_pl, df_esports, df_superbowl
