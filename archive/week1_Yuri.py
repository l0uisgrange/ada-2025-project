import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

DATA_START_DATE = "2014-01-01"
DATA_END_DATE = "2017-04-30"

def get_dataframe_features(tf, tf2, features_list, subreddits_list):
    
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
    for f in features_list:
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
    # print(df_features)
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
    ax1.bar(period_stats.index, period_stats["Count"], color="black", alpha=0.3)
    ax1.set_ylabel("Post Count", color="gray")
    ax1.set_xlabel(period_map[period])
    ax1.tick_params(axis="y")
    print(period_stats["Count"].sum())
    # Line plot for each sentiment on the second axis
    ax2 = ax1.twinx()
    for f in features_list:
        ax2.plot(period_stats.index, period_stats[f], label=f, linewidth=2)
    ax2.set_ylabel("Sum of Feature Values")
    ax2.tick_params(axis="y")

    # Title and legend
    plt.title(f"Sentiment {features_list}\n{period_map[period]} Trends ({start_time.date()} – {end_time.date()})") #\nwithin {subreddits_list}")
    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes, title="Features")
    fig.tight_layout()
    plt.show()

def plot_liwc_heatmaps_for_features_list(
    ax,
    df,
    subreddits,
    properties_list,
    start_date=DATA_START_DATE,
    end_date=DATA_END_DATE,
    subreddit_col='TARGET_SUBREDDIT',
    title=None,
    vmin=0,
    vmax=None
):
    import numpy as np
    """
    Shows a heatmap of LIWC properties based on given subreddits for a given period of time.
    """

    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])
    df_time_filtered = df[
        (df["TIMESTAMP"] >= pd.to_datetime(start_date)) &
        (df["TIMESTAMP"] <= pd.to_datetime(end_date))
        ]
    df_redd_filtered = df_time_filtered[df_time_filtered[subreddit_col].isin(subreddits)].copy()
 
    # properties_list = list(PROPERTIES.keys())[start_properties:start_properties + size_properties]
    # labels_list = list(PROPERTIES.values())[start_properties:start_properties + size_properties]

    sum_by_source = df_redd_filtered.groupby(subreddit_col)[properties_list].sum()
    mat_values = sum_by_source.values
    ax.imshow(mat_values, aspect='auto', origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)

    rows, cols = mat_values.shape
    for i in range(rows):
        for j in range(cols):
            value_to_display = f"{mat_values[i, j]:.2f}"
            ax.text(j, i, value_to_display, ha="center", va="center", color='white')
    ax.set_xticks(np.arange(sum_by_source.shape[1]))

    ax.set_xticklabels(properties_list, rotation=45, ha='right')
    ax.set_yticks(np.arange(sum_by_source.shape[0]))
    ax.set_yticklabels(sum_by_source.index)
    ax.set_title(title)
    ax.grid(False)

    return mat_values