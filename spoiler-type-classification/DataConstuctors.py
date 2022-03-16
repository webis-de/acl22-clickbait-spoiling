import pandas
from nltk import ngrams
from sklearn.base import BaseEstimator, TransformerMixin

class NgramTransformer(BaseEstimator, TransformerMixin):
    n = 0
    tokens = pandas.DataFrame()

    def __init__(self, n):
        self.n = n

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        self.tokens = pandas.DataFrame()
        self.tokens['tokens'] = X
        if self.n > 1:
            self.tokens['tokens'] = self.tokens['tokens'].apply(lambda x: list(ngrams(x, self.n)))

        print('NgramTrans tokens:', self.tokens.head())
        return self.tokens['tokens']

class NgramColTransformer(BaseEstimator, TransformerMixin):
    n = 0
    tokens = pandas.DataFrame()

    def __init__(self, n, col):
        self.n = n
        self.col = col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        self.tokens = pandas.DataFrame()
        self.tokens['tokens'] = X[self.col]
        if self.n > 1:
            self.tokens['tokens'] = self.tokens['tokens'].apply(lambda x: list(ngrams(x, self.n)))

        # print('NgramTrans tokens:', self.tokens.head())
        return self.tokens['tokens']

class ColLengthExtractor(BaseEstimator, TransformerMixin):
    lengths = pandas.DataFrame()

    def __init__(self, col):
        self.col = col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        self.lengths = pandas.DataFrame()
        self.lengths['len'] = X[self.col].apply(lambda x: len(x))

        return self.lengths