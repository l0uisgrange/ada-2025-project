# <img src="https://epfl-ada.github.io/assets/img/ada.svg" width="40" /> Digital tribes

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

[Data story](https://epfl-ada.github.io/ada-2025-project-baldy5/) — [Final Notebook](./results.ipynb)

## ❓ Research Questions

We focus on two domains where event-driven behavior is most prominent:

1. **Political subreddits**

- How do major political events (elections, debates, inaugurations) shift hostility, sentiment, and camp-to-camp interactions?
- Do hostility patterns generalize across camps, and which linguistic markers (LIWC/VADER) best predict hostile content?

2. **Sports subreddits**

- How do marquee sports events (e.g., Super Bowl, NBA Finals) reshape activity and rivalry dynamics between team camps?
- Are stress-test effects (before vs during vs after events) comparable to politics, and which features drive these changes?

## 📊 Additional Datasets

For our project, we searched online a few datasets that would allow us to create more interesting statistics, but after the EDA we finally decided that the following was enough.

- Embedding vectors of subreddits (communities on Reddit) [source](https://snap.stanford.edu/data/web-RedditEmbeddings.html)

For practical reasons, we used [Kaggle](https://www.kaggle.com/datasets/fejwiehf3928uhcwa/ada-2025-project-bald)
so it's easier for everyone to install the datasets automatically. It was not possible to put it on GitHub as the datasets were
huge (> 200MB).

## 🏗️ Methods

#### Data Preprocessing and Filtering

- Load all datasets into clean Pandas DataFrames.
- Filter the date period into the main dataset period (Jan 2014 to Jul 2017)
- Group the subreddits using clustering methods to prepare the analysis using the events datasets.
- Plot each external event dataset (political elections, major sports tournaments, and holidays) on a timeline of world
  events.

_Outcome: clean list of subreddit considered per part and filtered datasets._

#### Exploratory Data Analysis (EDA)

- Compute different LIWC attributes for each group to discover the most interesting ones.
- Visualize subreddit activity and sentiment distributions over time.
- Identify spikes or anomalies aligned with major world events.
- Aggregate LIWC features (affect, social, cognitive, punctuation, ...) per subreddit and time window.
- Compare distributions across event types (politics, sports, holidays).

_Outcome: understanding of the basic linguistic and properties around events._

#### Community Interactions and Linguistic Analysis

For each major event category (politics, sports, holidays)

- Identify the most active source subreddits
- Identify the most targeted subreddits
- Analyze interactions by using LIWC features, sentiment, and linguistic properties before, during and after events
- Identify massive interactions
- Plot graphs to highlight different behaviors

_Outcome: identify behavioral shifts based on events_

## ⏰ Proposed Timeline

#### Week 0 (before Nov 5)

- [x] Obtain, preprocess, and clean the main Reddit datasets (title and body).
- [x] Find, filter, and integrate subreddit embeddings and external event datasets (political elections, sports tournaments, holidays).
- [x] Clustering of the subreddits to identify the main groups for each category (political, sportive, and holidays)
- [x] Clean the `results.ipynb` for P2

#### Week 1 (before Nov 12)

- [x] Perform Exploratory Data Analysis EDA (see Methods section above)

#### Week 2 (before Nov 19)

- [x] Finish EDA (only if needed) and generate additional visualizations.
- [x] Identify massive interactions.
- [x] Identify the most active / targeted subreddits and quickly analyze interactions.

#### Week 3 (before Nov 26)

- [x] Analyze interactions by using LIWC features, sentiment, and linguistic properties using events.
- [x] Plot or prepare graphs to highlight different behaviors.
- [x] Have a complete analysis for each theme.

#### Week 4 (before Dec 3)

- [x] Clean code from the notebook and python files.
- [x] Finalize the `result.ipynb` notebook with explanations, figures, tables, and summaries.
- [x] Refine visualizations and structure to ensure readability.
- [x] Make the notebook interesting and easy to follow.

#### Week 5 (before Dec 10)

- [x] Final polishing of notebooks, README.md, and visualizations.
- [x] Ensure the GitHub repository is complete and organized.
- [x] Make final adjustments to figures, text, or additional analyses as needed before submission.

#### Week 6 (before Dec 17)

_Margin week_

## 🤝 Team Organization

For weeks 0 to 2, we are assigning tasks on a weekly basis and update the timeline above accordingly. For weeks after (3 to 6), we will do the tasks together with meetings each week.

- **Badr**

  - Interaction and LIWC analyses in `results_final.ipynb`.
  - Visualizations for reciprocity, camp profiles, and LIWC comparisons (`interaction_analysis.py`).

- **Daniel**

  - Statistical methods and event-test analysis (`statistical_analysis.py`, `event_analysis.py`).
  - Classifier transfer, coefficient comparison, bootstrap/permutation tests, and event timelines.
  - Data preparation utilities (`data_prep.py`).

- **Arnaud**

  - storyline

- **Louis**

  - First iteration work on `README.md`, EDA, and `results.ipynb` with Yuri
  - Datasets integration with Kaggle, cleaning and import (`data.py`)
  - Storyline (`pages` branch): plots, template, structure and part of the text

## ⚙️ Quickstart

1. Clone the repository
2. Open a terminal and execute `python -m venv /PATH/TO/PROJECT/.venv` to create a virtual environment
3. Execute `pip install -r requirements.txt` to install the project dependencies

_Note: the datasets will be downloaded automatically from [Kaggle](https://www.kaggle.com/datasets/fejwiehf3928uhcwa/ada-2025-project-bald) when running the Notebook._

## 🗄️ Project structure (to update)

```
├── ... cache             # Cached datasets files
├── archive               # Archive files
├── src                   # Source code
│   ├── clustering.py
│   ├── consts.py
│   ├── data.py
│   ├── preprocessing.py
│   ├── similarity.py
│   ├── utils.py
│   └── visualization.py
├── results.ipynb         # Notebook showing the results
├── .gitignore
├── requirements.txt      # File for installing python dependencies
└── README.md
```
