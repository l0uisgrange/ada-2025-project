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
    "european",
    "antiwar",
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
PROPERTIES = [
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
    "LIWC_Funct",  # Function words (articles, prepositions, conjunctions)
    "LIWC_Pronoun",  # All pronouns
    "LIWC_Ppron",  # Personal pronouns (I, you, he, she, we, they)
    "LIWC_I",  # First-person singular "I"
    "LIWC_We",  # First-person plural "we"
    "LIWC_You",  # Second-person "you"
    "LIWC_SheHe",  # Third-person singular (she, he)
    "LIWC_They",  # Third-person plural "they"
    "LIWC_Ipron",  # Impersonal/indefinite pronouns (it, one)
    "LIWC_Article",  # Articles (a, an, the)
    "LIWC_Verbs",  # All verb forms
    "LIWC_AuxVb",  # Auxiliary verbs (be, have, do, modals)
    "LIWC_Past",  # Past tense verbs
    "LIWC_Present",  # Present tense verbs
    "LIWC_Future",  # Future markers (will, gonna)
    "LIWC_Adverbs",  # Adverbs
    "LIWC_Prep",  # Prepositions
    "LIWC_Conj",  # Conjunctions
    "LIWC_Negate",  # Negations (not, no, never)
    "LIWC_Quant",  # Quantifiers (many, few, much)
    "LIWC_Numbers",  # Numerals and number words
    "LIWC_Swear",  # Profanity/swear words
    "LIWC_Social",  # Social process words (talk, friend, share)
    "LIWC_Family",  # Family-related words
    "LIWC_Friends",  # Friends/peer-related words
    "LIWC_Humans",  # References to people/humans
    "LIWC_Affect",  # Emotion words
    "LIWC_Posemo",  # Positive emotion words
    "LIWC_Negemo",  # Negative emotion words
    "LIWC_Anx",  # Anxiety-related words (nervous, afraid)
    "LIWC_Anger",  # Anger-related words
    "LIWC_Sad",  # Sadness-related words
    "LIWC_CogMech",  # Cognitive process words (think, know)
    "LIWC_Insight",  # Insight words (realize, understand)
    "LIWC_Cause",  # Causal words (because, therefore)
    "LIWC_Discrep",  # Discrepancy words (should, would, could)
    "LIWC_Tentat",  # Tentative words (maybe, perhaps)
    "LIWC_Certain",  # Certainty words (definitely, sure)
    "LIWC_Inhib",  # Inhibition words (stop, prevent, avoid)
    "LIWC_Incl",  # Inclusion words (include, with)
    "LIWC_Excl",  # Exclusion words (exclude, without)
    "LIWC_Percept",  # Perceptual processes (see, hear, feel)
    "LIWC_See",  # Vision-related words
    "LIWC_Hear",  # Hearing-related words
    "LIWC_Feel",  # Sensation/feeling words
    "LIWC_Bio",  # Biological process words (eat, sleep)
    "LIWC_Body",  # Body-part words (arm, head)
    "LIWC_Health",  # Health-related words
    "LIWC_Sexual",  # Sexual content words
    "LIWC_Ingest",  # Eating/drinking words (eat, drink)
    "LIWC_Relativ",  # Relativity/relational words (near, far)
    "LIWC_Motion",  # Motion verbs (go, move, travel)
    "LIWC_Space",  # Spatial/location words (left, above)
    "LIWC_Time",  # Temporal words (when, day, year)
    "LIWC_Work",  # Work and occupational words
    "LIWC_Achiev",  # Achievement/goal words
    "LIWC_Leisure",  # Leisure and recreation words
    "LIWC_Home",  # Home/domestic words
    "LIWC_Money",  # Money/finance words
    "LIWC_Relig",  # Religion and spirituality words
    "LIWC_Death",  # Death-related words
    "LIWC_Assent",  # Agreement/approval words (yes, okay)
    "LIWC_Dissent",  # Disagreement/opposition words (no, oppose)
    "LIWC_Nonflu",  # Non-fluencies (uh, um, repetitions)
    "LIWC_Filler"  # Filler phrases (you know, I mean)
]