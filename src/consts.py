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
