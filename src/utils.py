import pandas as pd

from src.consts import PROPERTIES


def count_pattern_by_period(df_body, df_title: pd.DataFrame, pattern, period = "M"):
    """
    Find similar subreddits based on centroid of picked subreddits.

    All subreddits are equally weighted regardless of their proximity to each other.

    Returns:
        pd.DataFrame
            Top N similar subreddits with columns: SUBREDDIT, SIMILARITY, ORIGINAL
            :param period:
            :param pattern:
            :param df_body:
            :param df_title:
    """
    # Prepare data: filter mentions of the chosen subreddit
    title_mentions = df_title[
        (df_title["SOURCE_SUBREDDIT"] == pattern) |
        (df_title["TARGET_SUBREDDIT"] == pattern)
    ].copy()

    body_mentions = df_body[
        (df_body["SOURCE_SUBREDDIT"] == pattern) |
        (df_body["TARGET_SUBREDDIT"] == pattern)
    ].copy()

    # Convert timestamps
    title_mentions["TIMESTAMP"] = pd.to_datetime(title_mentions["TIMESTAMP"], errors="coerce")
    body_mentions["TIMESTAMP"] = pd.to_datetime(body_mentions["TIMESTAMP"], errors="coerce")

    # Assign period and count mentions
    title_mentions["PERIOD"] = title_mentions["TIMESTAMP"].dt.to_period(period)
    body_mentions["PERIOD"] = body_mentions["TIMESTAMP"].dt.to_period(period)

    counts_title = title_mentions.groupby("PERIOD").size()
    counts_body = body_mentions.groupby("PERIOD").size()

    # Combine into DataFrame
    return pd.DataFrame({ "TITLE_ACTIVITY": counts_title,"BODY_ACTIVITY": counts_body }).fillna(0)


def parse_properties(df: pd.DataFrame):
    """
    Parses properties column into multiple columns.
    """

    df_temp = df["PROPERTIES"].str.split(',', expand=True)
    df_temp = df_temp.astype(float)
    df_temp.columns = PROPERTIES.keys()
    df_resultat = pd.concat([df.drop(columns=["PROPERTIES"]), df_temp], axis=1)
    return df_resultat