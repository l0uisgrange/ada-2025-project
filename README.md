# <img src="https://epfl-ada.github.io/assets/img/ada.svg" width="40" /> The Social Fabric They Weave

Reddit is one of the largest social networks in the world, and its vast and diverse communities make it the perfect
environment to analyze how people's behavior is affected by real world events.

This project consists of a statistical analysis to map users' behavior across different key communities, specifically
focusing on sports, politics, and holiday-related subreddits. Our main goal is to **find and demonstrate measurable
links between major world events** (like a football match or an election) **and immediate behavioral shifts** inside
subreddits.

We investigate how users' actions change (participation, cooperation, conflict, ...) and how these events influence
linguistic patterns (like the use of punctuation or specific casing). Finally, we explore how these fluctuations affect
community dynamics: do some subreddits boom in popularity, become aggressive targets, or otherwise act as supportive
allies to others? This project uncovers the hidden statistical patterns that weave online social behavior to global
events.

## ❓ Research Questions

Our research questions are split into 3 parts:

1. **Political events** (e.g. 2016 U.S. presidential election)

- How do major political events affect interaction patterns and topic trends on Reddit?

2. **Sports events** (e.g. English Premier League, the Super Bowl)

- How do major sports events and e-sports tournaments reflect in Reddit activity and community interactions?

3. **Holidays** (e.g. Christmas, New Year's Day, or Mother's Day)

- How do holidays affect subreddits activity?

## 📊 Additional Datasets

For our project, we searched online a few datasets that would allow us to create more interesting statistics

- Embedding vectors of subreddits (communities on
  Reddit) [source](https://snap.stanford.edu/data/web-RedditEmbeddings.html)
- List of global holidays by
  country [source](https://www.kaggle.com/datasets/umerhaddii/global-holidays-and-travel-data/data)
- E-Sport tournaments with start, end times, and city where each event took
  place [source](https://www.kaggle.com/datasets/hbakker/esports-200-tournaments)
- English Premier League (EPL) results with match dates, home teams, and away
  teams [source](https://www.kaggle.com/datasets/irkaal/english-premier-league-results)
- Super bowl matches [source](https://www.kaggle.com/datasets/timoboz/superbowl-history-1967-2020)

As we had a lot of datasets, we used [Kaggle](https://www.kaggle.com/datasets/fejwiehf3928uhcwa/ada-2025-project-bald)
so it's easier for everyone to install automatically. It was not possible to put it on GitHub as some datasets were very
large (> 200MB).

## 🏗️ Methods

#### Data Preprocessing and Filtering

- Reddit Hyperlink Network: Load both the title and body hyperlink datasets (TSV format). Combine them into a unified
  directed, temporal, signed graph where nodes represent subreddits and edges represent cross-subreddit hyperlinks.
- Filtering and Normalization:
    - Normalize subreddit names (lowercase, remove “r/”).
    - Remove self-links and low-frequency subreddits.
    - Retain only edges that include valid sentiment annotations and LIWC features.
- External Datasets Integration:
    - Map external event datasets (political elections, major sports tournaments, and holidays) into a timeline of world
      events.
    - Align Reddit timestamps with these events (+/- few days/weeks) to measure pre- and post-event behavioral changes.
    - Merge embedding vectors for subreddits (from SNAP embeddings dataset) to enrich graph nodes with semantic
      features.

Outcome:
A clean, enriched, temporal Reddit graph ready for event-based and linguistic analysis.

#### Exploratory Data Analysis (EDA)

- Network Descriptives:
    - Compute the number of active subreddits, degree distributions, edge polarity ratios, and community structure using
      Louvain modularity.
    - Identify the most connected and influential subreddits (in-degree/out-degree centrality).
- Temporal Dynamics:
    - Visualize subreddit activity and sentiment distributions over time.
    - Identify spikes or anomalies aligned with major world events.
- Linguistic Analysis:
    - Aggregate LIWC features (affect, social, cognitive, punctuation, ...) per subreddit and time window.
    - Compare distributions across event types (politics, sports, holidays).

Outcome:
Understanding of the overall Reddit hyperlink network structure, its evolution, and the basic linguistic and emotional
characteristics of interactions.

#### Event-Based Behavioral Analysis

For each major event category (politics, sports, holidays):

1. Temporal Alignment:
   - Segment the network into time slices around the event.
   - Compare subreddit-level features before, during, and after the event.
2. Behavioral Metrics:
   - Volume of cross-links, sentiment shifts, and linguistic tone (via LIWC).
   - Community-level participation changes (number of active edges/nodes).
3. Statistical Testing:
   - Use paired tests or bootstrapping to detect significant changes in sentiment and connectivity.
   - Evaluate whether behavioral shifts are unique to specific event categories.

Outcome:
Quantitative evidence of behavioral fluctuations in Reddit communities tied to real-world events.

#### Community and Influence Analysis

- Graph Modeling:
    - Represent subreddit interactions as temporal signed graphs (positive vs. negative sentiment).
    - Compute centrality metrics (Betweenness, PageRank, Eigenvector) to identify “bridge” or “hub” subreddits that
      connect topic clusters.
- Diffusion Patterns:
    - Analyze information or sentiment “flows” between subreddits surrounding events.
    - Track cascades of hyperlinks over time (chains of consecutive links or mentions).
- Community Dynamics:
    - Detect emerging or dissolving clusters using dynamic community detection methods.
    - Compare linguistic similarity within vs. between clusters.

Outcome:
Insight into how influence, sentiment, and participation propagate through Reddit communities during key real-world
events.

#### Interpretation and Storytelling

- Combine quantitative results (EDA, event analysis, diffusion modeling) into a cohesive narrative about collective
  online behavior.
- Use visualizations such as:
    - Temporal sentiment timelines
    - Interactive network maps of subreddit interactions
    - Event impact comparison dashboards
- Summarize findings in a storytelling format that highlights how Reddit’s social fabric mirrors real-world dynamics.

Outcome:
A clear, engaging, and data-driven story showing how global events ripple through Reddit’s interconnected communities.

## ⏰ Proposed Timeline

#### Week 0 (before Nov 5)

- Obtain, preprocess, and clean the Reddit Hyperlink datasets (title + body).
- Integrate subreddit embeddings and external event datasets (political elections, sports tournaments, holidays).
- Quick exploration of network structure and embeddings to validate data.

#### Week 1 (before Nov 12)

- Perform Exploratory Data Analysis (EDA):
    - Network descriptives: nodes, edges, degree distributions, sentiment ratios, community structure.
    - Temporal dynamics: subreddit activity and sentiment over time.
    - Linguistic analysis: aggregate LIWC features per subreddit and time window.
- Generate initial visualizations for network, temporal, and linguistic patterns.

#### Week 2 (before Nov 19)

- Event-Based Behavioral Analysis:
    - Segment activity around political, sports, and holiday events.
    - Compute metrics: subreddit activity, cross-links, sentiment shifts, linguistic changes.
    - Conduct statistical testing to detect significant pre-event vs. post-event changes.
- Document insights and patterns identified for each event category.

#### Week 3 (before Nov 26)

- Community & Influence Analysis:
    - Detect subreddit communities/clusters using network structure and embeddings.
    - Compute centrality metrics (Betweenness, PageRank, Eigenvector) to identify hubs and bridges.
    - Track information and sentiment cascades across communities over time.

#### Week 4 (before Nov 3)

- Integrate results from EDA, event analysis, and community/influence analysis.
- Finalize results notebook with figures, tables, and summaries.
- Refine visualizations and ensure reproducibility of the analysis pipeline.

#### Week 5 (before Dec 10)

- Interpretation & Storytelling:
    - Combine quantitative results into a coherent narrative.
    - Prepare visualizations for the story: temporal sentiment timelines, network maps, event impact comparisons.
    - Prepare interactive dashboards or notebooks for the final data story.

#### Week 6 (before Dec 17)

- Final polishing of notebooks, README, and visualizations.
- Ensure GitHub repository is complete, organized, and fully reproducible.
- Make final adjustments to figures, text, or additional analyses as needed before submission.

## 🤝 Team Organization

For weeks 0 to 2, we are assigning tasks on a weekly basis and update the timeline above accordingly. For weeks after (3
to 6), we will do the tasks together with meetings each week.

## ⚙️ Quickstart

1. Clone the repository
2. Open a terminal and execute `python -m venv /PATH/TO/PROJECT/.venv` to create a virtual environment
3. Execute `pip install -r requirements.txt` to install the project dependencies

_Note: the datasets will be downloaded automatically
from [Kaggle](https://www.kaggle.com/datasets/fejwiehf3928uhcwa/ada-2025-project-bald) when running the Notebook._

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


