# study_motivation

Version 0.1.0

This study focused on predicting university dropout by using text mining techniques with the aim of exhuming information contained in students’ written motivation. In a feature engineering step, we created new variables from the raw text data to predict dropout and enhance the already available set of predictive student characteristics. Support Vector Machines (SVMs) were then trained in the classification step on a dataset of 7,060 motivation statements of students enrolling in a non-selective bachelor at a Dutch university during years 2014 and 2015. We used various combinations of input resulting in six different models. Input to the models consisted of a set of student characteristics, bag-of-words features from text data, topic modeling-based features, and extracted cognitive and non-cognitive features from text. Although the combination of text and student characteristics did not improve the prediction of dropout, results showed that text analysis alone predicted dropout similarly well as a set of student characteristics.


## Project organization

```
.
├── .gitignore
├── CITATION.md
├── LICENSE.md
├── README.md
├── requirements.txt
├── bin                <- Compiled and external code, ignored by git (PG)
│   └── external       <- Any external source code, ignored by git (RO)
├── config             <- Configuration files (HW)
├── data               <- All project data, ignored by git
│   ├── processed      <- The final, canonical data sets for modeling. (PG)
│   ├── raw            <- The original, immutable data dump. (RO)
│   └── temp           <- Intermediate data that has been transformed. (PG)
├── docs               <- Documentation notebook for users (HW)
│   ├── manuscript     <- Manuscript source, e.g., LaTeX, Markdown, etc. (HW)
│   └── reports        <- Other project reports and notebooks (e.g. Jupyter, .Rmd) (HW)
├── results
│   ├── figures        <- Figures for the manuscript or reports (PG)
│   └── output         <- Other output for the manuscript or reports (PG)
└── src                <- Source code for this project (HW)

```


## License

This project is licensed under the terms of the [MIT License](/LICENSE.md)

## Citation

Please [cite this project as described here](/CITATION.md).
