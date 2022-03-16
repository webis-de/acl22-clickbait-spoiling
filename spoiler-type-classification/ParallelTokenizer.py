import os, argparse
import pandas, numpy
import tarfile
from multiprocessing import Pool

from nltk import pos_tag, word_tokenize

import Dataloader


def write_tokens_pos(doc):
    return pos_tag(word_tokenize(doc))

def do_preprocessing(vocabDataDF, col, npools=2):
    tokens = pandas.DataFrame()
    tokens['uuid'] = vocabDataDF['uuid']
    # automatically uses mp.cpu_count() as number of workers
    # mp.cpu_count() is 4 -> use 4 jobs
    with Pool(npools) as pool:
        tokens_list = []
        tags_list = []

        print('--------------Tokenizing')
        tokentuples = pool.map(write_tokens_pos, vocabDataDF[col].values.tolist())

        for token in tokentuples:
            if token:
                words, tags = zip(*token)
                tokens_list.append(words)
                tags_list.append(tags)
        tokens['tokens'] = tokens_list
        tokens['tags'] = tags_list

    return tokens

def tokenize_openwebtext(docfreq, npools=2):
    tars = []
    for root, dirs, files in os.walk(docfreq, topdown=False):
        for file in files:
            if ('urlsf_subset' in file) and file.endswith('.tar'):
                tars.append(os.path.join(root,file))

    for i in range(0,len(tars)):
        vocabDataDF = Dataloader.loadOpenWebTextData(tars[i], 'tmp', i)

        tokens = do_preprocessing(vocabDataDF, 'text', npools=npools)

        print('--------------Writing to File ', i)
        tokens.to_json(args.name + '_PoSTokens' + str(i) + '.json')

        with tarfile.open(args.name + "_PoSTokens"+str(i)+".tar.bz2", "w:bz2") as archive:
            archive.add(args.name + '_PoSTokens'+str(i)+'.json')

        os.remove(args.name + '_PoSTokens'+str(i)+'.json')

def tokenize_cbs(docfreq, npools=2):
    vocabDataDF = Dataloader.loadDataSplit(docfreq + 'clickbait-spoiling-21.jsonl')#clickbait-spoiling-corpus-20-raw9.jsonl

    idf_columns = ['clickbait', 'title', 'text']
    vocabDataDF['all'] = vocabDataDF[idf_columns].apply(lambda x: ' '.join(x.values.astype(str)), axis=1)

    print('First text:')
    print(vocabDataDF['all'].iat[0])

    tokens = do_preprocessing(vocabDataDF, 'all', npools=npools)

    return tokens


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', help='Directory of data to tokenize (e.g. OpenWebText)', required=True)
    parser.add_argument('--npools', default=2, type=int, help='Number of processes to use while tokenizing')
    names = parser.add_argument('--name', choices=['OpenWebText', 'CBS20'], help='Name of corpus', required=True)
    args = parser.parse_args()

    # Make directory for OpenWebText files
    if not os.path.exists(args.name + '_forIDF'):
        os.mkdir(args.name + '_forIDF')
    os.chdir(args.name + '_forIDF')

    if 'OpenWebText' == str(args.name):
        tokens = tokenize_openwebtext(args.dir, npools=args.npools)
    elif 'CBS20' == str(args.name):
        tokens = tokenize_cbs(args.dir, npools=args.npools)
    else:
        tokens = pandas.DataFrame()
        raise Exception('Invalid name given. Choose from: ', names.choices)

    print('--------------Writing to File ')
    tokens.to_json(args.name + '_PoSTokens.json')

    with tarfile.open(args.name + "_PoSTokens.tar.bz2", "w:bz2") as archive:
        archive.add(args.name + '_PoSTokens.json')

    os.remove(args.name + '_PoSTokens.json')
