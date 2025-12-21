# <img src="https://epfl-ada.github.io/assets/img/ada.svg" width="40" /> Digital tribes

Reddit is one of the largest social networks in the world, and its vast and diverse communities make it the perfect
environment to analyze how tribal behavior manifests in online interactions.

This project presents a statistical analysis of cross-community hostility patterns on Reddit, specifically
comparing political and sports communities. Our main goal is to **discover whether online hostility follows universal
linguistic patterns across different domains**, examining how the *frequency* vs *vocabulary* of conflict varies between
political discourse and sports rivalry.

[Datastory](https://epfl-ada.github.io/ada-2025-project-baldy5/) — [Final notebook](./results.ipynb)

## ❓ Research questions

We focus on understanding the nature of tribal hostility across domains:

1. **Political vs Sports Hostility**

   - How do baseline hostility rates differ between political and sports communities?
   - Which camps (ideological groups or team fandoms) generate the most hostile interactions?
   - What role do "observer" communities (like r/subredditdrama) play in hostility patterns?

2. **Universal Hostility Signature**

   - Do hostile posts use the same linguistic markers across domains?
   - Can classifiers trained on one domain detect hostility in another?
   - What LIWC (Linguistic Inquiry and Word Count) features characterize hostile interactions?

3. **Event Impact Analysis**

   - How do major events (elections, championships) affect hostility patterns?
   - Are these effects temporary spikes or lasting behavioral changes?

## 📊 Additional datasets

For our project, we searched online a few datasets that would allow us to create more interesting statistics, but after the EDA we finally decided that the following was enough.

- Embedding vectors of subreddits (communities on Reddit) [source](https://snap.stanford.edu/data/web-RedditEmbeddings.html)

For practical reasons, we used [Kaggle](https://www.kaggle.com/datasets/fejwiehf3928uhcwa/ada-2025-project-bald)
so it's easier for everyone to install the datasets automatically. It was not possible to put it on GitHub as the datasets were
huge (> 200MB).

## 🗂️ Methods

### Data Preprocessing and Filtering

- Load all datasets into clean Pandas DataFrames (286,561 body posts, 571,927 title posts)
- Filter the date period into the main dataset period (Jan 2014 to Apr 2017)
- Extract 86 features per post: 65 LIWC psycholinguistic markers, 18 text structure features, 3 VADER sentiment scores
- Group subreddits into meaningful "camps" using embedding-based clustering

### Clustering and Camp Definitions

- Apply PCA to 300-dimensional subreddit embeddings (retaining 94% variance at 50 components)
- Evaluate K-means clustering using silhouette and Davies-Bouldin scores
- Combine data-driven clustering with domain expertise for interpretable camp definitions
- Define political camps: trump_conservative, anti_trump, progressive, alt_right, meta_drama, etc.
- Define sports camps: NFL divisions, NBA, MLB, NHL, soccer leagues, etc.

### Baseline Hostility Analysis

- Calculate negative interaction rates for each domain
- Compare hostility rates using chi-square tests and Cohen's h effect size
- Build camp-to-camp interaction matrices to identify hostile relationships
- Analyze which camps are aggressors vs targets

### LIWC Signature Analysis

- Compute hostility signatures: mean LIWC values in hostile posts minus friendly posts
- Correlate signatures across domains (Pearson and Spearman)
- Identify universal hostility markers: NEGEMO, ANGER, SWEAR, THEY pronouns
- Analyze pronoun patterns (we/they dynamics) in tribal communication

### Event Impact Analysis

- Define 14 major events: 8 political (2016 Election, Brexit, debates) and 6 sports (Super Bowls, NBA Finals)
- Compare hostility rates in 7-day windows before, during, and after events
- Visualize weekly hostility trends with event markers
- Run difference-in-differences analysis for causal inference

### Statistical Validation

- **Proportion tests**: Chi-square with Yates correction, Cohen's h effect sizes
- **Cross-domain transfer**: Train logistic classifiers on one domain, test on another (AUC evaluation)
- **Coefficient comparison**: Correlate logistic regression coefficients across domains
- **Bootstrap/Permutation tests**: Robust confidence intervals and null hypothesis testing
- **Network analysis**: Compare interaction network structures (density, reciprocity, clustering)

## 📈 Key results

| Finding | Evidence |
| :------- | :-------- |
| Politics is 3.3× more hostile than sports | 17.3% vs 5.2% negative interaction rate (Cohen's h = 0.40) |
| Hostility "sounds" the same across domains | r = 0.937 LIWC signature correlation |
| Cross-domain classifiers transfer successfully | AUC = 0.91 for politics→sports transfer |
| Same features predict hostility in both domains | 73.5% coefficient sign agreement |
| Events cause temporary spikes, not permanent change | Behavior returns to baseline within days |
| Observer communities are universally hostile | meta_drama shows ~25% hostility in both domains |

## ⏰ Proposed timeline

#### Week 0 (before Nov 5)

- [x] Obtain, preprocess, and clean the main Reddit datasets (title and body)
- [x] Find, filter, and integrate subreddit embeddings and external event datasets
- [x] Clustering of the subreddits to identify the main groups for each category
- [x] Clean the `results.ipynb` for P2

#### Week 1 (before Nov 12)

- [x] Perform Exploratory Data Analysis EDA

#### Week 2 (before Nov 19)

- [x] Generate additional visualizations
- [x] Identify massive interactions and camp-to-camp patterns
- [x] Analyze interaction matrices and hostility profiles

#### Week 3 (before Nov 26)

- [x] Analyze LIWC signatures and cross-domain correlations
- [x] Implement cross-domain classifier transfer experiments
- [x] Complete event impact analysis with statistical tests

#### Week 4 (before Dec 3)

- [x] Clean code from the notebook and python files
- [x] Implement statistical validation (bootstrap, permutation, DiD)
- [x] Network structure analysis and comparison

#### Week 5 (before Dec 10)

- [x] Final polishing of notebooks, README.md, and visualizations
- [x] Ensure the GitHub repository is complete and organized
- [x] Finalize the data story website

#### Week 6 (before Dec 17)

- [x] Final review and submission

## 🤝 Team organization

We would like to point out that for the P3 of this project, Yuri did not participate as she chosed to drop this course. Please take that into consideration upon grading this project.

- **Badr**

  - Mathematical sections of `results.ipynb`

- **Daniel**

  - End-to-end analysis pipeline from data preparation to statistical validation
  - Clustering and camp definitions (clustering.py, data_prep.py), full module
  - Interaction matrices and LIWC signature analysis (interaction_analysis.py), full module
  - Event impact analysis with before/during/after comparisons (event_analysis.py), full module
  - Statistical validation: classifier transfer, coefficient comparison, bootstrap, permutation, DiD (statistical_analysis.py), full module
  - Final results notebook and wrapper functions (results.ipynb, results_helpers.py), except mathematical markdown cells.

- **Arnaud**

  - Datastory (`pages` branch): shockwaves and universal signature
  - First EDA on Sports and `results.ipynb` with Yuri

- **Louis**

  - Writing of `README.md`
  - Datasets integration with Kaggle, cleaning and import (`data_prep.py`)
  - Datastory (`pages` branch): plots, template, structure and content
  - First EDA on Politics and `results.ipynb` with Yuri

## ⚙️ Quickstart

1. Clone the repository
2. Open a terminal and execute `python -m venv /PATH/TO/PROJECT/.venv` to create a virtual environment
3. Execute `pip install -r requirements.txt` to install the project dependencies

_Note: the datasets will be downloaded automatically from [Kaggle](https://www.kaggle.com/datasets/fejwiehf3928uhcwa/ada-2025-project-bald) when running the Notebook._

## 🗄️ Project structure

```
├── ... cache                   # Cached datasets files
├── src                         # Source code
│   ├── clustering.py
│   ├── consts.py
│   ├── data_prep.py
│   ├── event_analysis.py
│   ├── interaction_analysis.py
│   ├── results_helpers.py
│   └── statistical_analyisis.py
├── .gitignore
├── results.ipynb               # Notebook showing the results
├── requirements.txt            # File for installing python dependencies
└── README.md
```
