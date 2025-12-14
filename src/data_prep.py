"""
    This module takes care of all the data preparation. For clustering
    and for the later analysis done on it.
"""
import pandas as pd
import numpy as np
import kagglehub
import os
from src.consts import *

# =============================================================================
# CLUSTERING DATA
# =============================================================================

def download_data():
    """
    Downloads all required datasets from Kaggle and returns the path to the downloaded files.

    Returns:
        str: Path to the downloaded dataset directory
    """
    return kagglehub.dataset_download("fejwiehf3928uhcwa/ada-2025-project-bald")


def parse_properties(df: pd.DataFrame):
    """
    Parses properties column into multiple columns.
    """

    df_temp = df["PROPERTIES"].str.split(',', expand=True)
    df_temp = df_temp.astype(float)
    df_temp.columns = PROPERTIES.keys()
    df_resultat = pd.concat([df.drop(columns=["PROPERTIES"]), df_temp], axis=1)
    return df_resultat


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

    df_body = parse_properties(df_body)
    df_title = parse_properties(df_title)

    return df_body, df_title, df_subreddits


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
    # Source count
    count_source_body = body_data['SOURCE_SUBREDDIT'].value_counts().reset_index()
    count_source_body.columns = ['SUBREDDIT', 'POST_COUNT_BODY']

    count_source_title = title_data['SOURCE_SUBREDDIT'].value_counts().reset_index()
    count_source_title.columns = ['SUBREDDIT', 'POST_COUNT_TITLE']

    combined_source = pd.merge(count_source_body, count_source_title,
                               on='SUBREDDIT', how='outer').fillna(0)

    combined_source['POST_COUNT_SOURCE'] = (combined_source['POST_COUNT_BODY'] +
                                            combined_source['POST_COUNT_TITLE'])

    # Target count
    count_target_body = body_data['TARGET_SUBREDDIT'].value_counts().reset_index()
    count_target_body.columns = ['SUBREDDIT', 'POST_COUNT_BODY']

    count_target_title = title_data['TARGET_SUBREDDIT'].value_counts().reset_index()
    count_target_title.columns = ['SUBREDDIT', 'POST_COUNT_TITLE']

    combined_target = pd.merge(count_target_body, count_target_title,
                               on='SUBREDDIT', how='outer').fillna(0)

    combined_target['POST_COUNT_TARGET'] = (combined_target['POST_COUNT_BODY'] +
                                            combined_target['POST_COUNT_TITLE'])

    source_count = pd.merge(subreddits_data[['SUBREDDIT']], combined_source,
                            on='SUBREDDIT', how='left').fillna(0)

    target_count = pd.merge(subreddits_data[['SUBREDDIT']], combined_target,
                            on='SUBREDDIT', how='left').fillna(0)

    # Merge everything
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

# =============================================================================
# CAMP DEFINITIONS
# =============================================================================

POLITICS_CAMPS = {
    'trump_conservative': [
        'the_donald', 'conservative', 'republican', 'hillaryforprison',
        'asktrumpsupporters', 'shitpoliticssays', 'new_right'
    ],

    'libertarian': [
        'anarcho_capitalism', 'libertarian', 'goldandblack',
        'shitstatistssay', 'garyjohnson', 'enoughlibertarianspam'
    ],

    'progressive': [
        'sandersforpresident', 'wayofthebern', 'kossacks_for_sanders',
        'stillsandersforpres', 'socialism', 'anarchism', 'political_revolution'
    ],

    'anti_trump': [
        'enoughtrumpspam', 'enough_sanders_spam', 'shitrconservativesays',
        'circlebroke2', 'againsthatesubreddits', 'topmindsofreddit'
    ],

    'alt_right': [
        'european', 'whiterights', 'uncensorednews', 'postnationalist',
        'metacanada'
    ],

    'conspiracy': [
        'conspiracy', 'conspiratard', 'undelete'
    ],

    'news_mainstream': [
        'politics', 'worldpolitics', 'worldnews', 'news', 'politic',
        'canadapolitics', 'europe', 'canada'
    ],

    'gender_politics': [
        'mensrights', 'theredpill', 'againstmensrights'
    ],

    # (observers)
    'meta_drama': [
        'subredditdrama', 'drama', 'bestof', 'shitliberalssay',
        'bestofoutrageculture', 'panichistory', 'shitredditsays'
    ]
}

SPORTS_CAMPS = {
    'nfl_general': [
        'nfl', 'nfl_draft', 'fantasyfootball'
    ],

    'afc_east': [
        'patriots', 'buffalobills', 'miamidolphins', 'nyjets'
    ],

    'afc_north': [
        'browns', 'bengals', 'ravens', 'steelers'
    ],

    'afc_south': [
        'texans', 'tennesseetitans', 'colts', 'jaguars'
    ],

    'afc_west': [
        'oaklandraiders', 'denverbroncos', 'chargers', 'kansascitychiefs'
    ],

    'nfc_east': [
        'eagles', 'cowboys', 'redskins', 'nygiants'
    ],

    'nfc_north': [
        'greenbaypackers', 'chibears', 'minnesotavikings', 'detroitlions'
    ],

    'nfc_south': [
        'panthers', 'falcons', 'saints', 'buccaneers'
    ],

    'nfc_west': [
        'seahawks', '49ers', 'losangelesrams', 'azcardinals'
    ],

    'nba': [
        'nba', 'warriors', 'bostonceltics', 'nbaspurs', 'chicagobulls',
        'sixers', 'lakers', 'clevelandcavs', 'nbacirclejerk'
    ],

    'hockey': [
        'hockey', 'bluejackets', 'leafs', 'canucks', 'hockeycirclejerk'
    ],

    'baseball': [
        'baseball', 'baseballcirclejerk'
    ],

    'college': [
        'cfb', 'collegebasketball', 'longhornnation'
    ],

    # (observers)
    'meta_drama': [
        'subredditdrama', 'bestof'
    ]
}

# =============================================================================
# FEATURE GROUP DEFINITIONS
# =============================================================================

# Text structure features (how the post is written)
TEXT_STRUCTURE_COLS = [
    'N_CHARS',  # Number of characters
    'N_CHARS_NO_WS',  # Characters without whitespace
    'FRAC_ALPHA',  # Fraction alphabetical
    'FRAC_DIGIT',  # Fraction digits
    'FRAC_UPPER',  # Fraction uppercase (SHOUTING indicator)
    'FRAC_WS',  # Fraction whitespace
    'FRAC_SPECIAL',  # Fraction special chars (!?.,)
    'N_WORDS',  # Word count
    'N_UNIQUE_WORDS',  # Unique words
    'N_LONG_WORDS',  # Words with 6+ chars
    'AVG_WORD_LEN',  # Average word length
    'N_UNIQUE_STOP',  # Unique stopwords
    'FRAC_STOP',  # Fraction stopwords
    'N_SENT',  # Sentence count
    'N_LONG_SENT',  # Sentences with 10+ words
    'AVG_CHAR_SENT',  # Avg chars per sentence
    'AVG_WORD_SENT',  # Avg words per sentence
    'ARI'  # Automated Readability Index
]

# VADER sentiment scores (continuous, from text analysis)
VADER_COLS = [
    'VADER_POS',  # Positive sentiment score
    'VADER_NEG',  # Negative sentiment score
    'VADER_COMP'  # Compound sentiment (-1 to +1)
]

KEY_FEATURES = {
    'hostility_indicators': [
        'VADER_NEG', 'VADER_COMP', 'LIWC_NEGEMO', 'LIWC_ANGER', 'LIWC_SWEAR'
    ],
    'text_complexity': [
        'ARI', 'AVG_WORD_LEN', 'N_WORDS', 'N_SENT'
    ],
    'style_markers': [
        'FRAC_UPPER', 'FRAC_SPECIAL', 'FRAC_STOP'
    ]
}

# LIWC columns
LIWC_COLS = [
    'LIWC_FUNCT', 'LIWC_PRON', 'LIWC_PPRON', 'LIWC_I', 'LIWC_WE', 'LIWC_YOU',
    'LIWC_SHEHE', 'LIWC_THEY', 'LIWC_IPRON', 'LIWC_ART', 'LIWC_VERB', 'LIWC_AUXVB',
    'LIWC_PAST', 'LIWC_PRES', 'LIWC_FUT', 'LIWC_ADV', 'LIWC_PREP', 'LIWC_CONJ',
    'LIWC_NEG', 'LIWC_QUANT', 'LIWC_NUM', 'LIWC_SWEAR', 'LIWC_SOC', 'LIWC_FAM',
    'LIWC_FRIEND', 'LIWC_HUMAN', 'LIWC_AFFECT', 'LIWC_POSEMO', 'LIWC_NEGEMO',
    'LIWC_ANX', 'LIWC_ANGER', 'LIWC_SAD', 'LIWC_COG', 'LIWC_INSIGHT', 'LIWC_CAUSE',
    'LIWC_DISCREP', 'LIWC_TENT', 'LIWC_CERT', 'LIWC_INHIB', 'LIWC_INCL', 'LIWC_EXCL',
    'LIWC_PERCEPT', 'LIWC_SEE', 'LIWC_HEAR', 'LIWC_FEEL', 'LIWC_BIO', 'LIWC_BODY',
    'LIWC_HEALTH', 'LIWC_SEX', 'LIWC_INGEST', 'LIWC_RELAT', 'LIWC_MOTION',
    'LIWC_SPACE', 'LIWC_TIME', 'LIWC_WORK', 'LIWC_ACHIEV', 'LIWC_LEISURE',
    'LIWC_HOME', 'LIWC_MONEY', 'LIWC_RELIG', 'LIWC_DEATH', 'LIWC_ASSENT',
    'LIWC_DISSENT', 'LIWC_NONFLU', 'LIWC_FILLER'
]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _build_subreddit_to_camp(camps_dict):
    """
    Associates subreddits to their respective camps or teams.

    Returns:
        dict: {subreddit_name: camp_name}
    """
    mapping = {}
    for camp, subs in camps_dict.items():
        for sub in subs:
            mapping[sub] = camp
    return mapping


POLITICS_SUB_TO_CAMP = _build_subreddit_to_camp(POLITICS_CAMPS)
SPORTS_SUB_TO_CAMP = _build_subreddit_to_camp(SPORTS_CAMPS)


def _get_all_subreddits(camps_dict):
    """Returns flat list of all subreddits in camps."""
    return [sub for subs in camps_dict.values() for sub in subs]


POLITICS_ALL_SUBS = _get_all_subreddits(POLITICS_CAMPS)
SPORTS_ALL_SUBS = _get_all_subreddits(SPORTS_CAMPS)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_prepared_data(
        politics_body_path='../../data/politics_body.csv',
        politics_title_path='../../data/politics_title.csv',
        sports_body_path='../../data/sports_body.csv',
        sports_title_path='../../data/sports_title.csv'
):
    """
    Load and prepare politics and sports dataframes with camp labels.

    Returns:
        tuple: (politics_df, sports_df) with added columns:
            - source_camp: camp of source subreddit (or 'other')
            - target_camp: camp of target subreddit (or 'other')
            - source_type: 'body' or 'title'
            - TIMESTAMP: parsed as datetime
    """
    # Load all CSVs
    pol_body = pd.read_csv(politics_body_path)
    pol_title = pd.read_csv(politics_title_path)
    sport_body = pd.read_csv(sports_body_path)
    sport_title = pd.read_csv(sports_title_path)

    # Mark source type
    pol_body['source_type'] = 'body'
    pol_title['source_type'] = 'title'
    sport_body['source_type'] = 'body'
    sport_title['source_type'] = 'title'

    # Combine
    politics = pd.concat([pol_body, pol_title], ignore_index=True)
    sports = pd.concat([sport_body, sport_title], ignore_index=True)

    # Parse timestamps
    politics['TIMESTAMP'] = pd.to_datetime(politics['TIMESTAMP'])
    sports['TIMESTAMP'] = pd.to_datetime(sports['TIMESTAMP'])

    # Add camp labels
    politics['source_camp'] = politics['SOURCE_SUBREDDIT'].map(POLITICS_SUB_TO_CAMP).fillna('other')
    politics['target_camp'] = politics['TARGET_SUBREDDIT'].map(POLITICS_SUB_TO_CAMP).fillna('other')

    sports['source_camp'] = sports['SOURCE_SUBREDDIT'].map(SPORTS_SUB_TO_CAMP).fillna('other')
    sports['target_camp'] = sports['TARGET_SUBREDDIT'].map(SPORTS_SUB_TO_CAMP).fillna('other')

    return politics, sports


def get_camp_interactions(df):
    """
    Aggregate interactions by camp pairs.

    Parameters:
        df: DataFrame with source_camp, target_camp, LINK_SENTIMENT

    Returns:
        DataFrame with columns: source_camp, target_camp, count, mean_sentiment,
                                negative_rate, positive_rate
    """
    agg = df.groupby(['source_camp', 'target_camp']).agg(
        count=('LINK_SENTIMENT', 'size'),
        mean_sentiment=('LINK_SENTIMENT', 'mean'),
        negative_count=('LINK_SENTIMENT', lambda x: np.sum(x == -1)),
        positive_count=('LINK_SENTIMENT', lambda x: np.sum(x == 1))
    ).reset_index()

    agg['negative_rate'] = agg['negative_count'] / agg['count']
    agg['positive_rate'] = agg['positive_count'] / agg['count']

    return agg


def get_sentiment_matrix(df, metric='mean_sentiment'):
    """
    Create camp-to-camp matrix of sentiment.

    Parameters:
        df: DataFrame with source_camp, target_camp, LINK_SENTIMENT
        metric: 'mean_sentiment' or 'negative_rate'

    Returns:
        DataFrame pivot table (source_camp as rows, target_camp as columns)
    """
    interactions = get_camp_interactions(df)
    matrix = interactions.pivot(
        index='source_camp',
        columns='target_camp',
        values=metric
    )
    return matrix


def get_volume_matrix(df):
    """
    Create camp-to-camp matrix of interaction counts.

    Returns:
        DataFrame pivot table with counts
    """
    interactions = get_camp_interactions(df)
    matrix = interactions.pivot(
        index='source_camp',
        columns='target_camp',
        values='count'
    ).fillna(0).astype(int)
    return matrix


# =============================================================================
# EVENT DEFINITIONS
# =============================================================================

EVENTS = pd.DataFrame([
    # Political events
    {'date': '2016-11-08', 'name': '2016 US Election', 'domain': 'politics', 'type': 'election'},
    {'date': '2016-06-23', 'name': 'Brexit Referendum', 'domain': 'politics', 'type': 'election'},
    {'date': '2016-03-01', 'name': 'Super Tuesday 2016', 'domain': 'politics', 'type': 'primary'},
    {'date': '2016-07-18', 'name': 'RNC Convention Start', 'domain': 'politics', 'type': 'convention'},
    {'date': '2016-07-25', 'name': 'DNC Convention Start', 'domain': 'politics', 'type': 'convention'},
    {'date': '2016-09-26', 'name': 'First Presidential Debate', 'domain': 'politics', 'type': 'debate'},
    {'date': '2016-10-09', 'name': 'Second Presidential Debate', 'domain': 'politics', 'type': 'debate'},
    {'date': '2017-01-20', 'name': 'Trump Inauguration', 'domain': 'politics', 'type': 'inauguration'},

    # Sports events - Super Bowls
    {'date': '2017-02-05', 'name': 'Super Bowl LI (Patriots vs Falcons)', 'domain': 'sports', 'type': 'superbowl'},
    {'date': '2016-02-07', 'name': 'Super Bowl 50 (Broncos vs Panthers)', 'domain': 'sports', 'type': 'superbowl'},
    {'date': '2015-02-01', 'name': 'Super Bowl XLIX (Patriots vs Seahawks)', 'domain': 'sports', 'type': 'superbowl'},
    {'date': '2014-02-02', 'name': 'Super Bowl XLVIII (Seahawks vs Broncos)', 'domain': 'sports', 'type': 'superbowl'},

    # Sports events - NBA Finals
    {'date': '2016-06-19', 'name': 'NBA Finals 2016 Game 7 (Cavs vs Warriors)', 'domain': 'sports',
     'type': 'nba_finals'},
    {'date': '2015-06-16', 'name': 'NBA Finals 2015 Game 6 (Warriors win)', 'domain': 'sports', 'type': 'nba_finals'},
])

EVENTS['date'] = pd.to_datetime(EVENTS['date'])


def filter_around_event(df, event_date, days_before=7, days_after=7):
    """
    Filter dataframe to rows around an event.

    Parameters:
        df: DataFrame with TIMESTAMP column
        event_date: datetime or str
        days_before: int, days before event to include
        days_after: int, days after event to include

    Returns:
        DataFrame filtered to the time window, with added column:
            - days_from_event: integer days relative to event (negative = before)
    """
    if isinstance(event_date, str):
        event_date = pd.to_datetime(event_date)

    start = event_date - pd.Timedelta(days=days_before)
    end = event_date + pd.Timedelta(days=days_after)

    mask = (df['TIMESTAMP'] >= start) & (df['TIMESTAMP'] <= end)
    result = df[mask].copy()

    result['days_from_event'] = (result['TIMESTAMP'].dt.date - event_date.date()).apply(lambda x: x.days)

    return result


def get_before_during_after(df, event_date, before_days=7, after_days=7):
    """
    Split data into before/during/after periods.

    Parameters:
        df: DataFrame with TIMESTAMP column
        event_date: datetime or str
        before_days: how many days before to include in "before"
        after_days: how many days after to include in "after"

    Returns:
        dict with keys 'before', 'during', 'after', each containing filtered DataFrame
    """
    if isinstance(event_date, str):
        event_date = pd.to_datetime(event_date)

    event_day = event_date.date()

    before_start = event_date - pd.Timedelta(days=before_days)
    after_end = event_date + pd.Timedelta(days=after_days)

    before_mask = (df['TIMESTAMP'] >= before_start) & (df['TIMESTAMP'].dt.date < event_day)
    during_mask = df['TIMESTAMP'].dt.date == event_day
    after_mask = (df['TIMESTAMP'].dt.date > event_day) & (df['TIMESTAMP'] <= after_end)

    return {
        'before': df[before_mask].copy(),
        'during': df[during_mask].copy(),
        'after': df[after_mask].copy()
    }


# =============================================================================
# FEATURE ANALYSIS UTILITIES
# =============================================================================

def get_liwc_columns(df):
    """Returns list of LIWC column names in dataframe."""
    return [col for col in df.columns if col.startswith('LIWC_')]


def get_feature_profile(df, columns):
    """
    Calculate mean values for specified columns.

    Parameters:
        df: DataFrame
        columns: list of column names

    Returns:
        Series with feature means
    """
    available = [c for c in columns if c in df.columns]
    return df[available].mean()


def get_liwc_profile(df, liwc_cols=None):
    """
    Calculate mean LIWC values for a dataframe.

    Parameters:
        df: DataFrame containing LIWC columns
        liwc_cols: list of column names (auto-detected if None)

    Returns:
        Series with LIWC feature means
    """
    if liwc_cols is None:
        liwc_cols = get_liwc_columns(df)
    return df[liwc_cols].mean()


def get_hostility_signature(df, columns=None):
    """
    Calculate the "hostility signature" - difference between negative and positive posts.

    Parameters:
        df: DataFrame with LINK_SENTIMENT and feature columns
        columns: list of column names (auto-detects LIWC if None)

    Returns:
        Series with (negative_mean - positive_mean) for each feature
    """
    if columns is None:
        columns = get_liwc_columns(df)

    available = [c for c in columns if c in df.columns]
    pos = df[df['LINK_SENTIMENT'] == 1][available].mean()
    neg = df[df['LINK_SENTIMENT'] == -1][available].mean()

    return neg - pos


def get_full_hostility_signature(df):
    """
    Calculate hostility signature for ALL feature types.

    Parameters:
        df: DataFrame with LINK_SENTIMENT and all feature columns

    Returns:
        dict with keys 'liwc', 'text_structure', 'vader', each containing
        a Series of (negative_mean - positive_mean) values
    """
    return {
        'liwc': get_hostility_signature(df, LIWC_COLS),
        'text_structure': get_hostility_signature(df, TEXT_STRUCTURE_COLS),
        'vader': get_hostility_signature(df, VADER_COLS)
    }


def compare_hostility_signatures(sig1, sig2):
    """
    Calculate correlation between two hostility signatures.

    Parameters:
        sig1, sig2: Series with same index (feature names)

    Returns:
        dict with 'pearson' and 'spearman' correlations
    """
    from scipy import stats
    common = sig1.index.intersection(sig2.index)
    s1, s2 = sig1[common], sig2[common]

    return {
        'pearson': s1.corr(s2),
        'spearman': stats.spearmanr(s1, s2)[0],
        'n_features': len(common)
    }


def get_vader_sentiment_stats(df):
    """
    Get VADER sentiment statistics by LINK_SENTIMENT label.

    Returns:
        DataFrame with VADER stats for positive vs negative labeled posts
    """
    stats_list = []
    for label, name in [(1, 'positive'), (-1, 'negative')]:
        subset = df[df['LINK_SENTIMENT'] == label]
        stats_list.append({
            'label': name,
            'count': len(subset),
            'vader_neg_mean': subset['VADER_NEG'].mean(),
            'vader_pos_mean': subset['VADER_POS'].mean(),
            'vader_comp_mean': subset['VADER_COMP'].mean()
        })
    return pd.DataFrame(stats_list)


# =============================================================================
# SUMMARY STATISTICS
# =============================================================================

def print_data_summary(politics, sports):
    """Print summary statistics for prepared data."""
    print("=" * 60)
    print("DATA PREPARATION SUMMARY")
    print("=" * 60)

    for name, df in [('POLITICS', politics), ('SPORTS', sports)]:
        print(f"\n{name}:")
        print(f"  Total posts: {len(df):,}")
        print(f"  Date range: {df['TIMESTAMP'].min().date()} to {df['TIMESTAMP'].max().date()}")
        print(f"  Unique source subreddits: {df['SOURCE_SUBREDDIT'].nunique()}")
        print(f"  Unique target subreddits: {df['TARGET_SUBREDDIT'].nunique()}")

        # Camp coverage
        labeled_source = np.sum(df['source_camp'] != 'other')
        labeled_target = np.sum(df['target_camp'] != 'other')
        print(f"  Posts with labeled source camp: {labeled_source:,} ({labeled_source / len(df) * 100:.1f}%)")
        print(f"  Posts with labeled target camp: {labeled_target:,} ({labeled_target / len(df) * 100:.1f}%)")

        # Sentiment breakdown
        neg = np.sum(df['LINK_SENTIMENT'] == -1)
        pos = np.sum(df['LINK_SENTIMENT'] == 1)
        print(
            f"  Sentiment: {pos:,} positive ({pos / len(df) * 100:.1f}%), {neg:,} negative ({neg / len(df) * 100:.1f}%)")

        # Camp distribution
        print(f"\n  Posts by source camp:")
        camp_counts = df['source_camp'].value_counts()
        for camp, count in camp_counts.head(8).items():
            print(f"    {camp}: {count:,}")
