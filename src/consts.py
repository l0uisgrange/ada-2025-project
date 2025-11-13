# ——————————————————————————————————————
# Clustering
# ——————————————————————————————————————

# Parameter for clustering by similarity
SIMILARITY_N_CLUSTERS = 100

# Politics by subcategory
RIGHT_SUBREDDITS = [
    "conspiracy",
    "republican",
    "conservative",
    "conservatives",
    "conservatives_r_us",
    "whitenationalism",
    "white_pride",
    "the_donald",
    "hillaryforprison",
    "asktrumpsupporters",
    "mr_trump",
    "mensrights",
    "conservativesonly"
]

LEFT_SUBREDDITS = [
    "SandersForPresident",
    "hillaryclinton",
    "EnoughTrumpSpam",
    "LateStageCapitalism",
    "The_Mueller",
    "Political_Revolution",
    "democrats",
    "askhillarysupporters",
    "governmentoppression",
    "conflictofinterest",
]

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
    'gaming': [
        'gaming', 'pcmasterrace', 'ps4', 'xboxone', 'games', 'rainbow6', 'minecraft', 'fo4', 'darksouls3',
        'thedivision', 'fallout', 'dayz', 'halo', 'darksouls2', 'skyrim', 'grandtheftautov', 'oculus', 'witcher',
        'kerbalspaceprogram', 'battlefield_one', 'battlefield_4', 'globaloffensivetrade', 'fireteams',
        'rocketleagueexchange', 'dota2', 'leagueoflegends', 'globaloffensive', 'destinythegame', 'overwatch',
        'hearthstone', 'wow', '2007scape', 'pokemongo', 'smite', 'runescape'
    ],
    'soccer': [
        'soccer', 'football', 'premierleague', 'realmadrid', 'liverpoolfc', 'chelseafc', 'barca', 'fifa', 'coys', 'gunners', 'reddevils', 'soccerstreams', 'bayernmunich', 'fcbayern', 'juventus', 'soccerbetting', 'atletico', 'tottenham', 'epl', 'laliga'
    ]
}

# ——————————————————————————————————————
# LIWC Properties
# ——————————————————————————————————————

# Text statistics (0-17)
N_CHARS = 0
N_CHARS_NO_WS = 1
FRAC_ALPHA = 2
FRAC_DIGIT = 3
FRAC_UPPER = 4
FRAC_WS = 5
FRAC_SPECIAL = 6
N_WORDS = 7
N_UNIQUE_WORDS = 8
N_LONG_WORDS = 9
AVG_WORD_LEN = 10
N_UNIQUE_STOP = 11
FRAC_STOP = 12
N_SENT = 13
N_LONG_SENT = 14
AVG_CHAR_SENT = 15
AVG_WORD_SENT = 16
ARI = 17

# VADER sentiment (18-20)
VADER_POS = 18
VADER_NEG = 19
VADER_COMP = 20

# LIWC features (21-85)
LIWC_FUNCT = 21
LIWC_PRON = 22
LIWC_PPRON = 23
LIWC_I = 24
LIWC_WE = 25
LIWC_YOU = 26
LIWC_SHEHE = 27
LIWC_THEY = 28
LIWC_IPRON = 29
LIWC_ART = 30
LIWC_VERB = 31
LIWC_AUXVB = 32
LIWC_PAST = 33
LIWC_PRES = 34
LIWC_FUT = 35
LIWC_ADV = 36
LIWC_PREP = 37
LIWC_CONJ = 38
LIWC_NEG = 39
LIWC_QUANT = 40
LIWC_NUM = 41
LIWC_SWEAR = 42
LIWC_SOC = 43
LIWC_FAM = 44
LIWC_FRIEND = 45
LIWC_HUMAN = 46
LIWC_AFFECT = 47
LIWC_POSEMO = 48
LIWC_NEGEMO = 49
LIWC_ANX = 50
LIWC_ANGER = 51
LIWC_SAD = 52
LIWC_COG = 53
LIWC_INSIGHT = 54
LIWC_CAUSE = 55
LIWC_DISCREP = 56
LIWC_TENT = 57
LIWC_CERT = 58
LIWC_INHIB = 59
LIWC_INCL = 60
LIWC_EXCL = 61
LIWC_PERCEPT = 62
LIWC_SEE = 63
LIWC_HEAR = 64
LIWC_FEEL = 65
LIWC_BIO = 66
LIWC_BODY = 67
LIWC_HEALTH = 68
LIWC_SEX = 69
LIWC_INGEST = 70
LIWC_RELAT = 71
LIWC_MOTION = 72
LIWC_SPACE = 73
LIWC_TIME = 74
LIWC_WORK = 75
LIWC_ACHIEV = 76
LIWC_LEISURE = 77
LIWC_HOME = 78
LIWC_MONEY = 79
LIWC_RELIG = 80
LIWC_DEATH = 81
LIWC_ASSENT = 82
LIWC_DISSENT = 83
LIWC_NONFLU = 84
LIWC_FILLER = 85

# Embedding dimensions
EMBEDDING_DIMENSIONS = 300

# Clustering parameters
OPTIMAL_K = 20
MIN_ACTIVE_POSTS = 100

# Date ranges for analysis
DATA_START_DATE = "2014-01-01"
DATA_END_DATE = "2017-04-30"

# Visualization parameters
PLOT_FIGSIZE_MONTHLY = (14, 6)
PLOT_FIGSIZE_WEEKLY = (24, 8)

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