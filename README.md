

# <img src="https://epfl-ada.github.io/assets/img/ada.svg" width="60" /> ADA Semester Project

| SCIPER | Name |
| :-- | :-- |
| [356420](https://people.epfl.ch/badr.almahouri) | Badr Al Mahouri |
| [341237](https://people.epfl.ch/louis.grange) | Louis Grange |
| [340497](https://people.epfl.ch/daniel.alvesataide) | Daniel Ataíde |
| [329310](https://people.epfl.ch/arnaud.tadic) | Arnaud Jacques Yves Tadic |
| [308396](https://people.epfl.ch/yuri.cho) | Yuri Cho |

## ⚙️ Installation

1. Clone the repository
2. Download the [dataset](https://snap.stanford.edu/data/soc-RedditHyperlinks.html)
3. Move the dataset into the data folder, so you should have something like this
    ```
    ada-2025-project
    ├── data
    │   ├── soc-redditHyperlinks-body.tsv
    │   └── soc-redditHyperlinks-title.tsv
    ├── src
    │   └── main.ipynb
    └── README.md
    ```
4. Open a terminal on this folder, and execute `python -m venv /PATH/TO/ada-2025-project/.venv` to create a virtual environment

## 🗄️ Data structure

For each post (title and body), we have the following data structure:

- SOURCE_SUBREDDIT
- TARGET_SUBREDDIT
- TIMESTAMP
- POST_ID
- LINK_SENTIMENT 
- PROPERTIES (as an array of floats listed below)

### Content properties
1. Number of characters
2. Number of characters without counting white space
3. Fraction of alphabetical characters
4. Fraction of digits
5. Fraction of uppercase characters
6. Fraction of white spaces
7. Fraction of special characters, such as comma, exclamation mark, etc.
8. Number of words
9. Number of unique works
10. Number of long words (at least 6 characters)
11. Average word length
12. Number of unique stopwords
13. Fraction of stopwords
14. Number of sentences
15. Number of long sentences (at least 10 words)
16. Average number of characters per sentence
17. Average number of words per sentence
18. Automated readability index
19. Positive sentiment calculated by VADER
20. Negative sentiment calculated by VADER
21. Compound sentiment calculated by VADER
22. LIWC_Funct
23. LIWC_Pronoun
24. LIWC_Ppron
25. LIWC_I
26. LIWC_We
27. LIWC_You
28. LIWC_SheHe
29. LIWC_They
30. LIWC_Ipron
31. LIWC_Article
32. LIWC_Verbs
33. LIWC_AuxVb
34. LIWC_Past
35. LIWC_Present
36. LIWC_Future
37. LIWC_Adverbs
38. LIWC_Prep
39. LIWC_Conj
40. LIWC_Negate
41. LIWC_Quant
42. LIWC_Numbers
43. LIWC_Swear
44. LIWC_Social
45. LIWC_Family
46. LIWC_Friends
47. LIWC_Humans
48. LIWC_Affect
49. LIWC_Posemo
50. LIWC_Negemo
51. LIWC_Anx
52. LIWC_Anger
53. LIWC_Sad
54. LIWC_CogMech
55. LIWC_Insight
56. LIWC_Cause
57. LIWC_Discrep
58. LIWC_Tentat
59. LIWC_Certain
60. LIWC_Inhib
61. LIWC_Incl
62. LIWC_Excl
63. LIWC_Percept
64. LIWC_See
65. LIWC_Hear
66. LIWC_Feel
67. LIWC_Bio
68. LIWC_Body
69. LIWC_Health
70. LIWC_Sexual
71. LIWC_Ingest
72. LIWC_Relativ
73. LIWC_Motion
74. LIWC_Space
75. LIWC_Time
76. LIWC_Work
77. LIWC_Achiev
78. LIWC_Leisure
79. LIWC_Home
80. LIWC_Money
81. LIWC_Relig
82. LIWC_Death
83. LIWC_Assent
84. LIWC_Dissent
85. LIWC_Nonflu
86. LIWC_Filler
