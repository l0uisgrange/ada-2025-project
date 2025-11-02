# <img src="https://epfl-ada.github.io/assets/img/ada.svg" width="40" /> The Social Fabric They Weave

Reddit is one of the largest social networks in the world, and its vast and diverse communities make it the perfect environment to analyze how people's behavior is affected by real world events.

This project consists of a statistical analysis to map users behavior across different key communities, specifically focusing on sports, politics, and holiday related subreddits. Our main goal is to **find and demonstrate measurable links between major world events** (like a football match or an election) **and immediate behavioral shifts** inside subreddits.

We investigate how users' actions change (participation, cooperation, conflict, ...) and how these events influence linguistic patterns (like the use of punctuation or specific casing). Finally, we explore how these fluctuations affect community dynamics: do some subreddits boom in popularity, become aggressive targets, or otherwise act as supportive allies to others? This project uncovers the hidden statistical patterns that weaves online social behavior to global events.

## ❓ Research Questions

Our research questions are split into 3 parts:

1. **Political events** (e.g. 2016 U.S. presidential election)
- How do major political events affect interaction patterns and topic trends on Reddit?
2. **Sports events** (e.g. English Premier League, the Super Bowl)
- How do major sports events and e-sports tournaments reflect in Reddit activity and community interactions?
3. **Holidays** (e.g. Christmas, New Year's Day, or Mother's day)
- How do holidays affect subreddits activity?

## 📊 Additional Datasets

For our project, we searched online a few datasets that would allow us to create more interesting statistics
- Embedding vectors of subreddits (communities on Reddit) [source](https://snap.stanford.edu/data/web-RedditEmbeddings.html)
- List of global holidays by country [source](https://www.kaggle.com/datasets/umerhaddii/global-holidays-and-travel-data/data)
- E-Sport tournaments with start, end times, and city where each event took place [source](https://www.kaggle.com/datasets/hbakker/esports-200-tournaments)
- English Premier League (EPL) results with match dates, home teams, and away teams [source](https://www.kaggle.com/datasets/irkaal/english-premier-league-results)
- Super bowl matches [source](https://www.kaggle.com/datasets/timoboz/superbowl-history-1967-2020)

As we had a lot of datasets, we used [Kaggle](https://www.kaggle.com/datasets/fejwiehf3928uhcwa/ada-2025-project-bald) so it's easier for everyone to install automatically. It was not possible to put it on GitHub as some datasets were very large (> 200MB).

## 🏗️ Methods

#### Data Preprocessing and Filtering

- Load Reddit Hyperlinks dataset
- Filter for commerce-related subreddits using LIWC categories
- Normalize subreddit names and handle missing or ambigous data
- Retain edges with valid linguistic features

Main outcome:
A cleaned and filtered dataset (edges_filtered.csv) focused on commerce-related subreddits with relevant linguistic features.

##### Daniel Proposal for Commerce-related Subreddits Filtering
- Manually label around 100 commerce subreddits vs non commerce
- Aggregate LIWC features per subreddit (mean scores)
- Train a classifier (Random Forest, SVM) to predict commerce vs non-commerce subredd
- Optional: Network analysis (find subreddits that link heavily to the manually labeled commerce subreddits)

#### Exploratory Data Analysis (EDA)

- Descriptive statistics:
  - Number of nodes and edges
  - Finding which subreddits are highly connected
  - Polarity and sentiment distributions
  - Temporal activity patterns
- Linguistic features:
  - Average LIWC scores per subreddit
  - Identify correlations between linguistic features and subreddit categories
- Visualizations:
  - Subreddit-level network graphs
  - Sentiment vs. link density scatter plots | scatter emotion vs connectivity
  - Heatmap of LIWC correlations

Main outcome:
A comprehensive EDA report with visualizations and insights into the dataset.
This allows validation and reveals initial patterns.

#### Consumer Persona Clustering

- Identify groups of subreddits with similar language and behavior to reveal archetypes
- Features per subreddit:
  - LIWC vector
  - Network structure embeddings (Node2Vec, DeepWalk)
  - Embeddings
- To do:
  - Standardize + concatenate features
  - Dimensionality reduction (PCA, t-SNE, UMAP)
  - Clustering algorithms (K-Means, DBSCAN, Hierarchical)
  - Cluster validation (silhouette score, Davies-Bouldin index)
  - Choose final persona groups

Main outcome:
Defined consumer personas with characteristic linguistic and network features.

#### Temporal Dynamics and event analysis
- See how each persone behaves over time
- Segment timeline by month/year
- For each persona cluster: count number of links, measure emotional language trends + compare around events 
(Black Friday, product launches)
- Methods: time series analysis, change point detection

Main outcome:
Understanding of how personas change behavior during key commerce events. Who reacts and who is there for market insight?

#### Influence and Diffusion Modeling
- Build influence networks to see how information spreads between personas
- Methods:
  - Directed graphs where edges represent influence (based on link direction and sentiment)
  - Centrality measures (PageRank, Betweenness)
  - Cascade sizes tracking (chains of consecutive cross-links) -> viral cross-subreddit threads
- Identify influential personas and their role in spreading sentiment

Main outcome:
Who mobilizes users + which personas are key influencers in shaping product narratives.

#### Interpretation and Storytelling

- Synthesize findings into a coherent narrative (Police Inverstigation Report style)
- Visualizations:
  - Persona profiles with cards with key features
  - Visualize influence pathways between persona clusters
  - Timeline of persona activity around key events and connect back to narrative
- Conclude with actionable insights for marketers on targeting personas and managing online product narratives.

## ⏰ Proposed Timeline

#### Week 0 (before Nov 5)

- Obtain, preprocess and clean the dataset
- Define commerce related starting points to cluster subreddits (for example: `tech`, `luxury`, `services`, `goods`, etc.)
- Quick analysis of clusters and embedding checks

#### Week 1 (before Nov 12)

- Exploratory Data Analysis (EDA)
- LIWC and tags descriptive statistics
- Initial visualizations

#### Week 2 (before Nov 19)

- Consumer persona clustering
- Subreddit feature engineering (LIWC + embeddings + degree stats)
- Cluster validation

#### Week 3 (before Nov 26)

- Temporal dynamics per persona using time-series trends around events
- Influence network analysis

#### Week 4 (before Nov 3)

- Integrate results and write analysis
- Refine figures and ensure reproducible pipeline
- Full results notebook (EDA → clustering → influence)

#### Week 5 (before Dec 10)

- Storytelling and polishing to have an interactive data story
- Prepare final presentation and submission

#### Week 6 (before Dec 17)

_Additional margin_

## 🤝 Team Organization

For weeks 0 to 2, we are assigning tasks on a weekly basis and update the timeline above accordingly. For weeks after (3 to 6), we will do the tasks together with meetings each week.

## ⚙️ Quickstart

1. Clone the repository
2. Open a terminal and execute `python -m venv /PATH/TO/PROJECT/.venv` to create a virtual environment
3. Execute `pip install -r requirements.txt` to install the project dependencies

_Note: the datasets will be downloaded automatically from [Kaggle](https://www.kaggle.com/datasets/fejwiehf3928uhcwa/ada-2025-project-bald) when running the Notebook._

## 🗄️ Project structure

```
├── ... cache             # Cached datasets files
├── archive               # Archive files 
├── src                   # Source code
│   ├── data
│   ├── models
│   ├── utils
│   └── scripts
├── results.ipynb         # Notebook showing the results
├── .gitignore
├── requirements.txt      # File for installing python dependencies
└── README.md
```


