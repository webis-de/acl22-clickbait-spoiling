# Spoiler Type Classification

The code located here aims to automatically decide whether a clickbait post requires a short phrase spoiler or a longer passage spoiler.
This classification is the first step of our approach.

We provide the following:

- Feature-based classificators
  - Naive Bayes, SVM and Logistic Regression classifiers in [GridSearcher_merged_sepselect.py](GridSearcher_merged_sepselect.py)
- Transformer-based classificators
  - BERT based classifiers in [classification-bert.ipynb](classification-bert.ipynb) and [classification-concat-bert.ipynb](classification-concat-bert.ipynb)
  - DeBERTa based classifiers in [classification-deberta.ipynb](classification-deberta.ipynb) and [classification-concat-deberta.ipynb](classification-concatdeberta.ipynb)
  - RoBERTa based classifiers in [classification-roberta.ipynb](classification-roberta.ipynb) and [classification-concat-roberta.ipynb]

