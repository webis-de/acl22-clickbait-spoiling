import sys, os, json, pandas
import numpy
import tarfile


def loadGloveModel(File:str):
    print("--------------Loading Glove Model")
    f = open(File, 'r')
    gloveModel = {}
    for i, line in enumerate(f):
        if i % 100000 == 0:
            print(i, end='\r')

        splitLines = line.split(' ')
        word = splitLines[0]
        wordEmbedding = numpy.asarray(splitLines[1:], dtype='float32')
        gloveModel[word] = wordEmbedding
    print(len(gloveModel), " words loaded!")
    return gloveModel


def load_json_asdict(path_file, key):
    dictionary = {}
    with open(path_file, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)

            dictionary[obj[key]] = obj

    return dictionary

def loadDataSplit(File:str):
    print('--------------Loading Clickbait Data')
    uuids, labels, titles, texts, clickbaits, spoilers = [], [], [], [], [], []

    entries = [json.loads(line) for line in open(File, 'r', encoding='utf-8')]

    for index, cb in enumerate(entries):
        uuids.append(cb['uuid'])
        titles.append(cb['targetTitle'])
        texts.append(' '.join(cb['targetParagraphs']))
        clickbaits.append(cb['postText'][0])
        spoilers.append(cb['spoiler'])

        if 'short' in cb['tags'] or 'phrase' in cb['tags']:
            labels.append("phrase")
        elif 'multi' in cb['tags']:
             labels.append("multi")
        else:
            labels.append("passage")

    dataDF = pandas.DataFrame()
    dataDF['uuid'] = uuids
    dataDF['label'] = labels
    dataDF['clickbait'] = clickbaits
    dataDF['title'] = titles
    dataDF['text'] = texts
    dataDF['spoiler'] = spoilers

    return dataDF

def loadOpenWebTextData(tarpath:str, tmpdir:str, curstep=0):
    print('--------------Loading Web Texts for document frequency, step: ', curstep)
    texts = []

    # make "temporary" file directory for extractions
    if not os.path.exists(tmpdir):
        os.mkdir(tmpdir)
    os.chdir(tmpdir)

    if curstep < 10:
        step = '0'+str(curstep)
    else:
        step = str(curstep)

    if tarpath.endswith('.tar'):
        print('extracting .tar: ', tarpath)

        tar = tarfile.open(tarpath)

        for i, member in enumerate(tar.getmembers()):
            tar.extract(member)

            txt_tar = tarfile.open(member.name)

            for txt_member in txt_tar.getmembers():
                text = ''
                txt = txt_tar.extractfile(txt_member)
                for line in txt.readlines():
                    if line.strip():
                        string = line.decode('utf-8')
                        if text:
                            text += ' ' + string
                        else:
                            text += string
                    else:
                        if text:
                            texts.append(text)
                        text = ''
                texts.append(text)

            if i % 300 == 0:
                print(i,member.name, end='\r')
            os.remove(txt_tar.name)
            txt_tar.close()

        tar.close()

    else:
        raise Exception('Not a .tar file: '+ tarpath)

    # move up in directory out of 'tmp' folder
    path_parent = os.path.dirname(os.getcwd())
    os.chdir(path_parent)

    dataDF = pandas.DataFrame()
    dataDF['text'] = texts
    dataDF['textlength'] = dataDF['text'].apply(len)

    return dataDF