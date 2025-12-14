# ——————————————————————————————————————
# Dataset
# ——————————————————————————————————————

# Date ranges for analysis
DATA_START_DATE = "2014-01-01"
DATA_END_DATE = "2017-04-30"

# ——————————————————————————————————————
# Plot styles
# ——————————————————————————————————————

PLOT_DPI = 200
PLOT_FONT_SIZE = 12
PLOT_TITLE_FONT_SIZE = 16
PLOT_WIDTH = 15

# ——————————————————————————————————————
# Clustering
# ——————————————————————————————————————

# Parameter for clustering by similarity
SIMILARITY_N_CLUSTERS = 100

# Clustering parameters
OPTIMAL_K = 20
MIN_ACTIVE_POSTS = 100

# Seeds to use for clustering
THEMES_SEEDS = {
    'sports': [
        'nba', 'nfl', 'hockey', 'fantasyfootball', 'cfb', 'maddenultimateteam', 'baseball', 'golf', 'patriots',
        'collegebasketball', 'seahawks', 'minnesotavikings', 'madden', 'greenbaypackers', 'warriors', 'eagles',
        'denverbroncos', 'torontobluejays', '49ers', 'lakers', 'browns', 'oaklandraiders'
    ],
    'politics': [
        'the_donald', 'politic', 'conspiracy', 'conservative', 'worldpolitics', 'libertarian', 'mensrights',
        'new_right', 'theredpill', 'anarcho_capitalism', 'whiterights', 'european', 'politicalvideo', 'metacanada',
        'hillaryforprison', 'uncensorednews', 'asktrumpsupporters', 'altnewz', 'kossacks_for_sanders',
        'wayofthebern', 'republican', 'conservatives'
    ],
}

# ——————————————————————————————————————
# LIWC Properties
# ——————————————————————————————————————

# Embedding dimensions
EMBEDDING_DIMENSIONS = 300

# Properties
PROPERTIES = {
    "N_CHARS": "Number of characters", # Total number of characters
    "N_CHARS_NO_WS": "Number of characters without white space", # Characters excluding whitespace
    "FRAC_ALPHA": "Fraction alphabetic characters", # Fraction of alphabetic letters
    "FRAC_DIGIT": "Fraction digits", # Fraction of digits
    "FRAC_UPPER": "Fraction uppercase characters", # Fraction of uppercase letters
    "FRAC_WS": "Fraction white space", # Fraction of whitespace characters
    "FRAC_SPECIAL": "Fraction special characters", # Fraction of punctuation/special chars
    "N_WORDS": "Number of words", # Total word count
    "N_UNIQUE_WORDS": "Number of unique words", # Unique word count
    "N_LONG_WORDS": "Number of long words", # Words with at least 6 characters
    "AVG_WORD_LEN": "Average word length", # Mean word length
    "N_UNIQUE_STOP": "Number of unique stopwords", # Unique stopwords count
    "FRAC_STOP": "Fraction stopwords", # Fraction of tokens that are stopwords
    "N_SENT": "Number of sentences", # Sentence count
    "N_LONG_SENT": "Number of long sentences", # Sentences with at least 10 words
    "AVG_CHAR_SENT": "Average characters per sentence", # Mean characters per sentence
    "AVG_WORD_SENT": "Average words per sentence", # Mean words per sentence
    "ARI": "Automated Readability Index", # Readability score
    "VADER_POS": "VADER Positive", # Positive sentiment score (VADER)
    "VADER_NEG": "VADER Negative", # Negative sentiment score (VADER)
    "VADER_COMP": "VADER Compound", # Compound sentiment score (VADER)
    "LIWC_FUNCT": "LIWC Func. words", # Function words (articles, prepositions, conjunctions)
    "LIWC_PRON": "LIWC Pronouns", # All pronouns
    "LIWC_PPRON": "LIWC Pers. pronouns", # Personal pronouns (I, you, he, she, we, they)
    "LIWC_I": "LIWC I", # First-person singular "I"
    "LIWC_WE": "LIWC We", # First-person plural "we"
    "LIWC_YOU": "LIWC You", # Second-person "you"
    "LIWC_SHEHE": "LIWC SheHe", # Third-person singular (she, he)
    "LIWC_THEY": "LIWC They", # Third-person plural (they)
    "LIWC_IPRON": "LIWC Imp. pronouns", # Impersonal/indefinite pronouns (it, one)
    "LIWC_ART": "LIWC Articles", # Articles (a, an, the)
    "LIWC_VERB": "LIWC Verbs", # All verb forms
    "LIWC_AUXVB": "LIWC Aux. verbs", # Auxiliary verbs (be, have, do, modals)
    "LIWC_PAST": "LIWC Past tense", # Past tense verbs
    "LIWC_PRES": "LIWC Present tense", # Present tense verbs
    "LIWC_FUT": "LIWC Future", # Future markers (will, gonna)
    "LIWC_ADV": "LIWC Adverbs", # Adverbs
    "LIWC_PREP": "LIWC Prepositions", # Prepositions
    "LIWC_CONJ": "LIWC Conjunctions", # Conjunctions
    "LIWC_NEG": "LIWC Negation", # Negations (not, no, never)
    "LIWC_QUANT": "LIWC Quantifiers", # Quantifiers (many, few, much)
    "LIWC_NUM": "LIWC Numbers", # Numerals and number words
    "LIWC_SWEAR": "LIWC Swear", # Profanity / swear words
    "LIWC_SOC": "LIWC Social", # Social process words (talk, friend, share)
    "LIWC_FAM": "LIWC Family", # Family-related words
    "LIWC_FRIEND": "LIWC Friends", # Friends / peer-related words
    "LIWC_HUMAN": "LIWC Humans", # References to people / humans
    "LIWC_AFFECT": "LIWC Affect", # Emotion words
    "LIWC_POSEMO": "LIWC Pos. emotion", # Positive emotion words
    "LIWC_NEGEMO": "LIWC Neg. emotion", # Negative emotion words
    "LIWC_ANX": "LIWC Anxiety", # Anxiety-related words (nervous, afraid)
    "LIWC_ANGER": "LIWC Anger", # Anger-related words
    "LIWC_SAD": "LIWC Sadness", # Sadness-related words
    "LIWC_COG": "LIWC Cogn. processes", # Cognitive process words (think, know)
    "LIWC_INSIGHT": "LIWC Insight", # Insight words (realize, understand)
    "LIWC_CAUSE": "LIWC Cause", # Causal words (because, therefore)
    "LIWC_DISCREP": "LIWC Discrepancy", # Discrepancy words (should, would, could)
    "LIWC_TENT": "LIWC Tentative", # Tentative words (maybe, perhaps)
    "LIWC_CERT": "LIWC Certainty", # Certainty words (definitely, sure)
    "LIWC_INHIB": "LIWC Inhibition", # Inhibition words (stop, prevent, avoid)
    "LIWC_INCL": "LIWC Inclusion", # Inclusion words (include, with)
    "LIWC_EXCL": "LIWC Exclusion", # Exclusion words (exclude, without)
    "LIWC_PERCEPT": "LIWC Perceptual", # Perceptual processes (see, hear, feel)
    "LIWC_SEE": "LIWC See", # Vision-related words
    "LIWC_HEAR": "LIWC Hear", # Hearing-related words
    "LIWC_FEEL": "LIWC Feel", # Sensation / feeling words
    "LIWC_BIO": "LIWC Biological", # Biological process words (eat, sleep)
    "LIWC_BODY": "LIWC Body", # Body-part words (arm, head)
    "LIWC_HEALTH": "LIWC Health", # Health-related words
    "LIWC_SEX": "LIWC Sexual", # Sexual content words
    "LIWC_INGEST": "LIWC Ingest", # Eating / drinking words (eat, drink)
    "LIWC_RELAT": "LIWC Relativity", # Relativity / relational words (near, far)
    "LIWC_MOTION": "LIWC Motion", # Motion verbs (go, move, travel)
    "LIWC_SPACE": "LIWC Space", # Spatial / location words (left, above)
    "LIWC_TIME": "LIWC Time", # Temporal words (when, day, year)
    "LIWC_WORK": "LIWC Work", # Work and occupational words
    "LIWC_ACHIEV": "LIWC Achievement", # Achievement / goal words
    "LIWC_LEISURE": "LIWC Leisure", # Leisure and recreation words
    "LIWC_HOME": "LIWC Home", # Home / domestic words
    "LIWC_MONEY": "LIWC Money", # Money / finance words
    "LIWC_RELIG": "LIWC Religion", # Religion and spirituality words
    "LIWC_DEATH": "LIWC Death", # Death-related words
    "LIWC_ASSENT": "LIWC Assent", # Agreement / approval words (yes, okay)
    "LIWC_DISSENT": "LIWC Dissent", # Disagreement / opposition words (no, oppose)
    "LIWC_NONFLU": "LIWC Nonfluencies", # Non-fluencies (uh, um, repetitions)
    "LIWC_FILLER": "LIWC Filler", # Filler phrases (you know, I mean)
}