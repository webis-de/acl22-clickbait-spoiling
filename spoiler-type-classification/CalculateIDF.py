import os, tarfile, argparse
import pickle

import pandas, numpy
from collections import Counter

from nltk import ngrams
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from CB_Tokenizers import LemmaPosTokenizer
from Dataloader import loadDataSplit

# dummy 'tokenizer' to just pass tokens to Vectorizers
from ParallelTokenizer import do_preprocessing


def dummy(doc):
    return doc

def get_lemtokens(dataframe, lem=True, stop=True, lower=True):
    print('--------------Lemmatizing and Removing Stopwords')

    lemmatizer = LemmaPosTokenizer(stop=stop, lower=lower)

    data = dataframe

    if lem:
        data['tokentags'] = list(zip(data['tokens'], data['tags']))
        data['tokentags'] = data['tokentags'].apply(lambda x: list(zip(x[0], x[1])))
        data = data.drop(labels=['tokens', 'tags'], axis=1)

        data['tokentags'] = data['tokentags'].apply(lambda x: lemmatizer(x))

        # delete empty rows, cause of text passages contaning only e.g. punctuation
        # which were reduced to empty
        data['tokentags'] = data['tokentags'].apply(lambda x: numpy.nan if not x else x)
        data.dropna(subset=['tokentags'], inplace=True)

        data['tokentags'] = data['tokentags'].apply(lambda x: list(zip(*x)))
        data[['tokens', 'tags']] = pandas.DataFrame(data['tokentags'].tolist(), index=data.index)
        data.drop(['tokentags'], axis=1, inplace=True)

    else:
        if lower:
            data['tokens'] = data['tokens'].apply(lambda x: [t.lower() for t in x])

    return data

def vectorize(dataframe, count_vect, j, pos=False):
    print('--------------Transforming Data '+str(j)+'-grams')
    # count words in documents
    if pos:
        filename = 'PoS-'
    else:
        filename = ''
    word_count_vector = count_vect.transform(dataframe)

    # calculate idf for documents
    tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
    tfidf_transformer.fit(word_count_vector)

    print('--------------Writing CountVectorizer Vocab ' + str(j) + '-grams')
    print('length: ', len(count_vect.vocabulary_))
    with open(filename + str(j) + 'gram_CVvocab.pickle', 'wb') as cfile:
        pickle.dump(count_vect.vocabulary_, cfile)

    print('--------------Writing TfidfTransformer idfs ' + str(j) + '-grams')
    with open(filename + str(j) + 'gram_idfs.pickle', 'wb') as tfile:
        pickle.dump(tfidf_transformer.idf_, tfile)

def buildopen(Path, trainpath, lem=True, stop=True, n_grams=1, lower=True, parts=20):
    print('--------------Building OpenWebText')

    # move into folder with processed data
    os.chdir(Path)

    docDataDF = pandas.DataFrame(columns=['tokens', 'tags'])

    print('--------------Reading in Tokens and Tags')
    # OpenWebText was splitted into 20 subsets
    for j in range(1, parts+1):
        print(str(j)+'/'+str(parts), end='\r')
        # raise exception if tarfile with tokens .json is missing
        if not os.path.exists('OpenWebText_PoSTokens'+str(j)+'.tar.bz2'):
            raise Exception('Tarfile of the tokens '+str(j)+' .json file is missing')

        # read file with pretokenized data (words, PoS)
        with tarfile.open('OpenWebText_PoSTokens'+str(j)+'.tar.bz2') as tar:
            for member in tar.getmembers():
                if member.name == 'OpenWebText_PoSTokens'+str(j)+'.json':
                    file = tar.extractfile(member)
                    tmpDF = pandas.read_json(file, encoding='utf-8')

        # Append snipets
        docDataDF = docDataDF.append(tmpDF, ignore_index=True)

    # DF for vocabulary building of train dataset
    vocabDF = loadDataSplit(trainpath)
    vocabDataDF = pandas.DataFrame()
    vocabDataDF['all'] = vocabDF[['clickbait', 'title', 'text']].apply(lambda x: ' - '.join(x.values.astype(str)), axis=1)
    vocabDF = do_preprocessing(vocabDataDF, 'all', npools=2)

    # Create and move to folder
    if not os.path.exists('Vectorizer_storage'):
        os.mkdir('Vectorizer_storage')
    os.chdir('Vectorizer_storage')

    datafilename = 'Ngrams-Data'
    if lem:
        datafilename += '_lem'
    if lower:
        datafilename += '_low'
    if stop:
        datafilename += '_stop'
    datafilename += '.tar.bz2'

    with tarfile.open(datafilename, "w:bz2") as archive:

        tokensDF = get_lemtokens(docDataDF, lem=lem, stop=stop, lower=lower)
        vocabtokensDF = get_lemtokens(vocabDF['all'], lem=lem, stop=stop, lower=lower)
        print('--------------Calculating N-Grams')
        for j in range(1, n_grams + 1):

            ngramtokens = pandas.DataFrame()
            v_ngramtokens = pandas.DataFrame()
            if j > 1:
                # Corpus Data
                ngramtokens['tokens'] = tokensDF['tokens'].apply(lambda x: list(ngrams(x, j)))
                ngramtokens['tags'] = tokensDF['tags'].apply(lambda x: list(ngrams(x, j)))

                # Train Data
                v_ngramtokens['tokens'] = vocabtokensDF['tokens'].apply(lambda x: list(ngrams(x, j)))
                v_ngramtokens['tags'] = vocabtokensDF['tags'].apply(lambda x: list(ngrams(x, j)))
            else:
                ngramtokens['tokens'] = tokensDF['tokens']
                ngramtokens['tags'] = tokensDF['tags']

                v_ngramtokens['tokens'] = vocabtokensDF['tokens']
                v_ngramtokens['tags'] = vocabtokensDF['tags']

            # Compute Vocabulary distribution of OpenWebText dataset
            print('--------------' + str(j) + '-Grams')
            print(ngramtokens.head())

            cv1 = CountVectorizer(encoding='utf-8', tokenizer=dummy, preprocessor=dummy)
            cv2 = CountVectorizer(encoding='utf-8', tokenizer=dummy, preprocessor=dummy)
            cv1.fit(v_ngramtokens['tokens'])
            cv2.fit(v_ngramtokens['tags'])

            vectorize(ngramtokens['tokens'], cv1, j)
            vectorize(ngramtokens['tags'], cv2, j, pos=True)

            # adding files to tar, removing added files
            archive.add(str(j) + 'gram_CVvocab.pickle')
            archive.add(str(j) + 'gram_idfs.pickle')
            archive.add('PoS-' + str(j) + 'gram_CVvocab.pickle')
            archive.add('PoS-' + str(j) + 'gram_idfs.pickle')

            os.remove(str(j) + 'gram_CVvocab.pickle')
            os.remove(str(j) + 'gram_idfs.pickle')
            os.remove('PoS-' + str(j) + 'gram_CVvocab.pickle')
            os.remove('PoS-' + str(j) + 'gram_idfs.pickle')

    # move up in directory out of 'Vectorizer_storage' folder
    path_parent = os.path.dirname(os.getcwd())
    os.chdir(path_parent)

def buildcbs(Path, trainpath, name, lem=True, stop=True, lower=True, n_grams=1):
    print('--------------Building CBS')
    # move into folder with processed data
    os.chdir(Path)

    docDataDF = pandas.DataFrame(columns=['tokens', 'tags'])

    print('--------------Reading in Tokens and Tags')
    # raise exception if tarfile with tokens .json is missing
    if not os.path.exists(name + '_PoSTokens.tar.bz2'):
        raise Exception('Tarfile of the tokens .json is missing')

    # read file with pretokenized data (words, POS)
    with tarfile.open(name + '_PoSTokens.tar.bz2') as tar:
        for member in tar.getmembers():
            if member.name == name + '_PoSTokens.json':
                file = tar.extractfile(member)
                tmpDF = pandas.read_json(file, encoding='utf-8')

    # Append snipets
    docDataDF = docDataDF.append(tmpDF, ignore_index=True)

    # DF for vocabulary building of train dataset
    vocabDF = loadDataSplit(trainpath)
    vocabDataDF = pandas.DataFrame()
    vocabDataDF['all'] = vocabDF[['clickbait', 'title', 'text']].apply(lambda x: ' - '.join(x.values.astype(str)), axis=1)
    vocabDF = do_preprocessing(vocabDataDF, 'all', npools=2)

    if not os.path.exists('Vectorizer_storage'):
        os.mkdir('Vectorizer_storage')
    os.chdir('Vectorizer_storage')

    datafilename = 'Ngrams-Data'
    if lem:
        datafilename += '_lem'
    if lower:
        datafilename += '_low'
    if stop:
        datafilename += '_stop'
    datafilename += '.tar.bz2'

    with tarfile.open(datafilename, "w:bz2") as archive:

        tokensDF = get_lemtokens(docDataDF, lem=lem, stop=stop, lower=lower)
        vocabtokensDF = get_lemtokens(vocabDF, lem=lem, stop=stop, lower=lower)
        print('--------------Calculating N-Grams')
        for j in range(1, n_grams+1):

            ngramtokens = pandas.DataFrame()
            v_ngramtokens = pandas.DataFrame()
            if j > 1:
                # Corpus Data
                ngramtokens['tokens'] = tokensDF['tokens'].apply(lambda x: list(ngrams(x, j)))
                ngramtokens['tags'] = tokensDF['tags'].apply(lambda x: list(ngrams(x, j)))

                # Train Data
                v_ngramtokens['tokens'] = vocabtokensDF['tokens'].apply(lambda x: list(ngrams(x, j)))
                v_ngramtokens['tags'] = vocabtokensDF['tags'].apply(lambda x: list(ngrams(x, j)))
            else:
                ngramtokens['tokens'] = tokensDF['tokens']
                ngramtokens['tags'] = tokensDF['tags']

                v_ngramtokens['tokens'] = vocabtokensDF['tokens']
                v_ngramtokens['tags'] = vocabtokensDF['tags']

            # Compute Vocabulary distribution of OpenWebText dataset
            print('--------------'+str(j)+'-Grams')
            print(ngramtokens.head())

            cv1 = CountVectorizer(encoding='utf-8', tokenizer=dummy, preprocessor=dummy)
            cv2 = CountVectorizer(encoding='utf-8', tokenizer=dummy, preprocessor=dummy)
            cv1.fit(v_ngramtokens['tokens'])
            cv2.fit(v_ngramtokens['tags'])

            vectorize(ngramtokens['tokens'], cv1, j)
            vectorize(ngramtokens['tags'], cv2, j, pos=True)

            # adding files to tar, removing added files
            archive.add(str(j) + 'gram_CVvocab.pickle')
            archive.add(str(j) + 'gram_idfs.pickle')
            archive.add('PoS-' + str(j) + 'gram_CVvocab.pickle')
            archive.add('PoS-' + str(j) + 'gram_idfs.pickle')

            os.remove(str(j) + 'gram_CVvocab.pickle')
            os.remove(str(j) + 'gram_idfs.pickle')
            os.remove('PoS-' + str(j) + 'gram_CVvocab.pickle')
            os.remove('PoS-' + str(j) + 'gram_idfs.pickle')

    # move up in directory out of 'Vectorizer_storage' folder
    path_parent = os.path.dirname(os.getcwd())
    os.chdir(path_parent)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', required=True,
                        help="Dir with tokens .tar.bz2")
    parser.add_argument('--train', required=True,
                        help="Path to train .jsonl")
    parser.add_argument('--ngrams',
                        help="Up to how many ngrams should be calculated", default=4)
    parser.add_argument('--lemma', action='store_true', default=False,
                        help="Lemmatizing words")
    parser.add_argument('--lower', action='store_true', default=False,
                        help="Lowercasing words")
    parser.add_argument('--stop', action='store_true', default=False,
                        help="Clickbait specific stopwords filtering")
    names = parser.add_argument('--cname', choices=['OpenWebText', 'CBS20'],
                                help='Name of the corpus folder with precalculated tokens')
    args = parser.parse_args()

    if 'OpenWebText' == str(args.cname):
        buildopen(args.corpus, args.train, lem=bool(args.lemma), stop=bool(args.stop), n_grams=int(args.ngrams), parts=6,
                  lower=args.lower)

    elif 'CBS20' == str(args.cname):
        buildcbs(args.corpus, args.train, args.cname, lem=args.lemma, stop=args.stop, n_grams=int(args.ngrams),
                 lower=args.lower)

    else:
        tokens = pandas.DataFrame()
        raise Exception('Invalid name given. Choose from: ', names.choices)