import matplotlib.pyplot as plt
import pandas as pd

features = [
    "Number of characters",
    "Number of characters without counting white space",
    "Fraction of alphabetical characters",
    "Fraction of digits",
    "Fraction of uppercase characters",
    "Fraction of white spaces",
    "Fraction of special characters (e.g., comma, exclamation mark, etc.)",
    "Number of words",
    "Number of unique words",
    "Number of long words (at least 6 characters)",
    "Average word length",
    "Number of unique stopwords",
    "Fraction of stopwords",
    "Number of sentences",
    "Number of long sentences (at least 10 words)",
    "Average number of characters per sentence",
    "Average number of words per sentence",
    "Automated readability index",
    "Positive sentiment (VADER)",
    "Negative sentiment (VADER)",
    "Compound sentiment (VADER)",
    "LIWC_Funct",
    "LIWC_Pronoun",
    "LIWC_Ppron",
    "LIWC_I",
    "LIWC_We",
    "LIWC_You",
    "LIWC_SheHe",
    "LIWC_They",
    "LIWC_Ipron",
    "LIWC_Article",
    "LIWC_Verbs",
    "LIWC_AuxVb",
    "LIWC_Past",
    "LIWC_Present",
    "LIWC_Future",
    "LIWC_Adverbs",
    "LIWC_Prep",
    "LIWC_Conj",
    "LIWC_Negate",
    "LIWC_Quant",
    "LIWC_Numbers",
    "LIWC_Swear",
    "LIWC_Social",
    "LIWC_Family",
    "LIWC_Friends",
    "LIWC_Humans",
    "LIWC_Affect",
    "LIWC_Posemo",
    "LIWC_Negemo",
    "LIWC_Anx",
    "LIWC_Anger",
    "LIWC_Sad",
    "LIWC_CogMech",
    "LIWC_Insight",
    "LIWC_Cause",
    "LIWC_Discrep",
    "LIWC_Tentat",
    "LIWC_Certain",
    "LIWC_Inhib",
    "LIWC_Incl",
    "LIWC_Excl",
    "LIWC_Percept",
    "LIWC_See",
    "LIWC_Hear",
    "LIWC_Feel",
    "LIWC_Bio",
    "LIWC_Body",
    "LIWC_Health",
    "LIWC_Sexual",
    "LIWC_Ingest",
    "LIWC_Relativ",
    "LIWC_Motion",
    "LIWC_Space",
    "LIWC_Time",
    "LIWC_Work",
    "LIWC_Achiev",
    "LIWC_Leisure",
    "LIWC_Home",
    "LIWC_Money",
    "LIWC_Relig",
    "LIWC_Death",
    "LIWC_Assent",
    "LIWC_Dissent",
    "LIWC_Nonflu",
    "LIWC_Filler"
]

def get_dataframe_features(tf, tf2, subreddits_list):
    df_title_prop = tf[
        (
            tf["SOURCE_SUBREDDIT"].isin(subreddits_list) |
            tf["TARGET_SUBREDDIT"].isin(subreddits_list)
        )
    ]

    df_body_prop = tf2[
        (
            tf2["SOURCE_SUBREDDIT"].isin(subreddits_list) |
            tf2["TARGET_SUBREDDIT"].isin(subreddits_list)
        )
    ]
    raw_rows =[]
    for f in features[18:]:
        raw_rows.append((
            f,
            round(df_title_prop[f].mean(), 5),
            round(df_title_prop[f].max(), 2),
            round(df_title_prop[f].sum(), 2),
            round(df_title_prop[f].std(), 5),
            round(df_body_prop[f].mean(), 5),
            round(df_body_prop[f].max(), 2),
            round(df_body_prop[f].sum(), 2),
            round(df_body_prop[f].std(), 5),
        ))
    columns=["Feature", "Mean_Title", "Max_Title", "Sum_Title", "Std_Title", "Mean_Body", "Max_Body", "Sum_Body", "Std_Body"]
    df_features = pd.DataFrame(raw_rows, columns=columns)
    df_features_sentiment = df_features.loc[:2]
    df_features_liwc = df_features.loc[3:]
    return df_features_sentiment, df_features_liwc


def plot_subreddits_features_trend(tf, subreddits_list, start_time, end_time, features_list, period):

    period_map = { "D": "DAY", "W": "WEEK", "M": "MONTH", "Q": "QUARTER", "Y": "YEAR" }

    start_time = pd.Timestamp(start_time)
    end_time   = pd.Timestamp(end_time)
    
    # apply filters
    df = tf[
        (
            tf["SOURCE_SUBREDDIT"].isin(subreddits_list) |
            tf["TARGET_SUBREDDIT"].isin(subreddits_list)
        )
        & (tf["TIMESTAMP"] >= start_time)
        & (tf["TIMESTAMP"] <= end_time)
    ]
    df = df.assign(**{period_map[period]: df["TIMESTAMP"].dt.to_period(period)})


    period_stats = (
        df.groupby(period_map[period])[features_list]
        .sum()
        .assign(Count=df.groupby(period_map[period]).size())
    )
    period_stats.index = period_stats.index.to_timestamp()

    fig, ax1 = plt.subplots(figsize=(16,6))

    # Bar plot for post count
    ax1.bar(period_stats.index, period_stats["Count"], color="gray", alpha=0.3)
    ax1.set_ylabel("Post Count", color="gray")
    ax1.set_xlabel(period_map[period])
    ax1.tick_params(axis="y")

    # Line plot for each sentiment on the second axis
    ax2 = ax1.twinx()
    for f in features_list:
        ax2.plot(period_stats.index, period_stats[f], label=f, linewidth=2)
    ax2.set_ylabel("Sum of Feature Values")
    ax2.tick_params(axis="y")

    # Title and legend
    plt.title(f"Sentiment {features_list}\n{period_map[period]} Trends ({start_time.date()} – {end_time.date()})\nwithin {subreddits_list}")
    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes, title="Features")
    fig.tight_layout()
    plt.show()