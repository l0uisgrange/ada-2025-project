# <img src="https://epfl-ada.github.io/assets/img/ada.svg" width="40" /> Consumer Personas, C.S.I

<!-- A 150 word description of the project idea and goals. What’s the motivation behind your project? What story would you like to tell, and why? -->

This project will treat Reddit's commerce communities as a digital crime scene, where clues
are just waiting to be uncovered. Our primary goal is to profile the distinct consumer personas
(Brand Loyalists, Skeptics, Early Adopters).
We will analyze their digital fingerprints (cross-linking patterns) and statements (linguistic features) to establish
a clear Modus Operandi (M.O.) for each.

In today's market, a product's success is often decided by anonymous online conversations. We’re motivated by a simple
question: who really controls a product's narrative?

Our story follows these "suspects" as we, the investigators, track their behavior during high-stakes "heists" like
Black Friday and new product launches. We'll identify the "shot-callers" that mobilize product discussions
and also see how intel (sentiment) gets laundered or distorted as it spreads.
The final story will reveal who holds the real influence in the digital marketplace.

## ❓ Research Questions

#### The Modus Operandi

Can we build behavioral profiles (M.O.s) for distinct consumer types based on their digital "fingerprints" (cross-linking) and recorded "statements" (linguistic features)?

#### Behavior During the "Big Job"

How do our suspects change their M.O. during high-stakes "heists" (Black Friday, product launches, Christmas)?

#### The "Shot-Callers

Which personas successfully "mobilize" other users into cross-community "meets" (discussions) about a product?

#### The "Broken Telephone" Investigation

Does the "intel" (positive or negative sentiment) about a product get distorted or "laundered" as it passes between different personas?

## 📊 Additional Datasets

- Embedding vectors of users on Reddit. [source](https://snap.stanford.edu/data/web-RedditEmbeddings.html)
- Embedding vectors of subreddits (communities on Reddit). [source](https://snap.stanford.edu/data/web-RedditEmbeddings.html)

<!-- List the additional dataset(s) you want to use (if any), and some ideas on how you expect to get, manage, process, and enrich it/them. Show us that you’ve read the docs and some examples, and that you have a clear idea on what to expect. Discuss data size and format if relevant. It is your responsibility to check that what you propose is feasible. -->

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

## ⏰ Proposed Timeline and Organization

#### Week 1 (Nov 6)

- Obtain, preprocess and clean the dataset
- Define commerce related starting points to cluster subreddits (for example: `tech`, `luxury`, `services`, `goods`, etc.)
- Quick analysis of clusters
- Check we have embeddings

#### Week 2 (Nov 13)

- Exploratory Data Analysis (EDA)
- LIWC and tags descriptive statistics
- Initial visualizations

#### Week 3 (Nov 20)

- Consumer persona clustering
- Subreddit feature engineering (LIWC + embeddings + degree stats)
- Cluster validation

#### Week 4 (Nov 27)

- Temporal dynamics per persona using time-series trends around events
- Influence network analysis

#### Week 5 (Dec 4)

- Integrate results and write analysis
- Refine figures and ensure reproducible pipeline
- Full results notebook (EDA → clustering → influence)

#### Week 6 (Dec 11)

- Storytelling and polishing to have an interactive data story
- Prepare final presentation and submission

<!-- _Content below must be kept_ -->

## 🤝 Team Organization

For weeks 1 to 3, we are assigning tasks on a weekly basis. For weeks after (4 to 6), we will do the tasks together with meetings each week.

## ⚙️ Quickstart

1. Clone the repository
2. Download the [dataset](https://snap.stanford.edu/data/soc-RedditHyperlinks.html)
3. Move the dataset files into the `data` folder
4. Open a terminal and execute `python -m venv /PATH/TO/ada-2025-project/.venv` to create a virtual environment
5. Execute `pip install -r requirements.txt` to install the project dependencies

## 🗄️ Project structure

```
ada-2025-project
├── data
│   ├── soc-redditHyperlinks-title.tsv
│   ├── soc-redditHyperlinks-body.tsv
│   ├── web-redditEmbeddings-subreddits.csv
│   └── web-redditEmbeddings-users.csv
├── src
│   └── main.ipynb <- File for installing python dependencies
└── README.md
```
