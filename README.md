# <img src="https://epfl-ada.github.io/assets/img/ada.svg" width="40" /> Consumer Personas, C.S.I

>[!IMPORTANT]
>README.md file containing the detailed project proposal (up to 1000 words). Your README.md should contain:
>- [ ] Title
>- [ ] Abstract: A 150 word description of the project idea and goals. What’s the motivation behind your project? What story would you like to tell, and why?
>- [ ] Research Questions: A list of research questions you would like to address during the project.
>- [ ] Proposed additional datasets (if any): List the additional dataset(s) you want to use (if any), and some ideas on how you expect to get, manage, process, and enrich it/them. Show us that you’ve read the docs and some examples, and that you have a clear idea on what to expect. Discuss data size and format if relevant. It is your responsibility to check that what you propose is feasible.
>- [ ] Methods
>- [ ] Proposed timeline
>- [ ] Organization within the team: A list of internal milestones up until project Milestone P3.
>- [ ] Questions for TAs (optional): Add here any questions you have for us related to the proposed project.

<!-- A 150 word description of the project idea and goals. What’s the motivation behind your project? What story would you like to tell, and why? -->

This project will treat Reddit's commerce communities as a digital crime scene, where clues 
are just waiting to be uncovered. Our primary goal is to run a data investigation to 
"profile the usual suspects"—the distinct consumer personas (the "Brand Loyalist," the "Skeptic," the "Early Adopter"). 
We will analyze their digital "fingerprints" (cross-linking patterns) and "statements" (linguistic features) to establish
a clear Modus Operandi (M.O.) for each.

Our story follows these "suspects" as we, the investigators, track their behavior during high-stakes "heists" like 
Black Friday and new product launches. We'll identify the "shot-callers" who successfully mobilize product discussions 
and run a "broken telephone" investigation to see how "intel" (sentiment) gets "laundered" or distorted as it spreads. 
The final story will reveal who holds the real influence in the digital marketplace.

## ❓ Research Questions

## 1 - The Modus Operandi
Can we build behavioral profiles (M.O.s) for distinct consumer types based on their digital 
"fingerprints" (cross-linking) and recorded "statements" (linguistic features)?

## 2 - Behavior During the "Big Job"
How do our suspects change their M.O. during high-stakes "heists" 
(Black Friday, product launches, Christmas)?

## 3 - The "Shot-Callers"
Which personas successfully "mobilize" other users into cross-community "meets" (discussions)
about a product?

## 4 - The "Broken Telephone" Investigation
Does the "intel" (positive or negative sentiment) about a product get distorted or "laundered" 
as it passes between different personas?

<!-- A list of research questions you would like to address during the project. -->

## 📊 Additional Datasets

<!-- List the additional dataset(s) you want to use (if any), and some ideas on how you expect to get, manage, process, and enrich it/them. Show us that you’ve read the docs and some examples, and that you have a clear idea on what to expect. Discuss data size and format if relevant. It is your responsibility to check that what you propose is feasible. -->

## 🏗️ Methods

## ⏰ Proposed Timeline

## 🤝 Team Organization

<!-- A list of internal milestones up until project Milestone P3. -->

<!-- _Content below must be kept_ -->

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
│   ├── soc-redditHyperlinks-body.tsv
│   └── soc-redditHyperlinks-title.tsv
├── src
│   └── main.ipynb <- File for installing python dependencies
└── README.md
```
