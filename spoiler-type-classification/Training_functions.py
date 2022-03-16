import argparse
import os
import pickle
import tarfile

import numpy
import pandas
# import xgboost
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif, VarianceThreshold
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from CalculateIDF import get_lemtokens
from Dataloader import loadDataSplit
from DataConstuctors import *

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning)

# dummy 'tokenizer' to just pass tokens to Vectorizers
def dummy(doc):
    return doc

def make_ngrams(tokens, n):
    ngram_tokens = pandas.DataFrame()
    if n > 1:
        ngram_tokens['tokens'] = tokens['tokens'].apply(lambda x: list(ngrams(x, n)))
        ngram_tokens['tags'] = tokens['tags'].apply(lambda x: list(ngrams(x, n)))
    else:
        return tokens

    return ngram_tokens

def get_cvs_idftrans(idfsdir, n_grams):
    print('--------------Loading N-gram idfs')
    # Get vocabularies and idfs for different n-grams of words and pos
    cvs, idftranss = [], []
    pos_cvs, pos_idftranss = [], []
    all_feature_names = []
    for i in range(1, n_grams + 1):
        for pos in [False, True]:
            vocab, idfs = get_vocab_idfs(idfsdir, pos=pos, n_grams=i)

            count_vect = CountVectorizer(encoding='utf-8', tokenizer=dummy, preprocessor=dummy, vocabulary=vocab)
            tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
            tfidf_transformer.idf_ = idfs

            all_feature_names += count_vect.get_feature_names()

            if pos:
                pos_cvs.append(count_vect)
                pos_idftranss.append(tfidf_transformer)
            else:
                cvs.append(count_vect)
                idftranss.append(tfidf_transformer)

            if pos:
                print('POS', i, '-grams loaded')
            else:
                print(i, '-grams loaded')

    return cvs, idftranss, pos_cvs, pos_idftranss, all_feature_names

# Extract vocabulary and precalculated idfs from file
def get_vocab_idfs(idfsdir, pos=False, n_grams=1):
    if pos:
        posstr = 'PoS-'
    else:
        posstr = ''

    with tarfile.open(idfsdir) as tar:#
        # print('tarfile: ', tar.name)
        for member in tar.getmembers():
            # print('member: ', member.name)
            if member.name == posstr + str(n_grams) + 'gram_CVvocab.pickle':
                file = tar.extractfile(member)
                vocab = pickle.load(file)

            if member.name == posstr + str(n_grams) + 'gram_idfs.pickle':
                file = tar.extractfile(member)
                idfs = pickle.load(file)

    return vocab, idfs

# Calculate Tfidfs, select best k features with chi2
def get_tfidf(vocab, idfs, train, postag=False):

    count_vect = CountVectorizer(encoding='utf-8', tokenizer=dummy, preprocessor=dummy, vocabulary=vocab)
    tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True) # False for only tf representation, scaled
    tfidf_transformer.idf_ = idfs

    if postag:
        count_vector = count_vect.transform(train['tags'])
    else:
        count_vector = count_vect.transform(train['tokens'])
    tf_idf_vector = tfidf_transformer.transform(count_vector)
    print('tfidf_vector: ', tf_idf_vector.shape)

    return tf_idf_vector, count_vect.get_feature_names()

def select_features_variance(train_vector, test_vector, fnames, threshold=0.0):

    selector = VarianceThreshold(threshold=threshold)

    kbest_vector = selector.fit_transform(train_vector)
    kbest_test_vector = selector.transform(test_vector)

    print('variance_vector: ', kbest_vector.shape)

    features_mask = selector.get_support()
    features_names = fnames

    kfeature_names = []
    for bool, name in zip(features_mask, features_names):
        if bool:
            kfeature_names.append(name)

    looki = pandas.DataFrame(selector.variances_, index=features_names, columns=['variance'])
    looki.sort_values(by=["variance"], ascending=False, inplace=True)
    print(looki.head(n=20))

    return kbest_vector, kbest_test_vector, kfeature_names

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', help="Path to train .json", required=True)
    args = parser.parse_args()