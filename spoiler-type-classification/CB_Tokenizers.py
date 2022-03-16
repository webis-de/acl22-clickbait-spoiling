import string
from nltk import word_tokenize, pos_tag, WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from sklearn.base import BaseEstimator, TransformerMixin

custom_stop = ["'s", "could", "might", "would", "sha", "ha", "n't", "'ve", "'d", "'ll", "'re", '’', "..."] \
              + stopwords.words('english') \
              + [str for str in string.punctuation]
cb_stop = set(custom_stop) - {"why", "how", "where", "not", "this", "they", "who", "he", "she", "these",
                              "here", "there", "when", "that", "?"}

# Convert Penn Treebank tags to WordNet tags, just for Lemmatization Process
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''


class LemmaPosTokenizer:
    def __init__(self, stop=False, lower=True):
        self.wnl = WordNetLemmatizer()
        self.stop = stop
        self.lower = lower

    def __call__(self, tokenstags, **kw):
        lem_tokens = []
        res = []

        tokens, tags = zip(*tokenstags)
        for i in range(0, len(tokenstags)):
            token = tokens[i]
            tag = tags[i]

            # Lemmatize lowercased word (necessary for WordNetLemmatizer)
            wn_tag = get_wordnet_pos(tag)
            if wn_tag:
                lemmatized = self.wnl.lemmatize(token.lower(), get_wordnet_pos(tag))
            else:
                lemmatized = self.wnl.lemmatize(token.lower())

            # Capitalize words back
            if not self.lower and token.istitle():
                word = lemmatized.capitalize()
            else:
                word = lemmatized

            tagged_token = [word, tag]
            lem_tokens.append(tagged_token)

        # Clean out stopwords
        if self.stop:
            stopped_tokens = [tok for tok in lem_tokens
                              if (not tok[0].lower() in cb_stop)]# and ((len(tok[0])>1) or (tok[0] in '?')))
            res = stopped_tokens
        else:
            res = lem_tokens

        # Build result tokens and tags
        tokens = [pos[0] for pos in res]
        tags = [pos[1] for pos in res]


        if kw.get('get_tokens'):
            return tokens
        elif kw.get('get_tags'):
            return tags
        else:
            return res


class LemmaTokenizer:
    def __init__(self, stop=False, lower=True):
        self.wnl = WordNetLemmatizer()
        self.stop = stop
        self.lower = lower

    def __call__(self, doc, get_tag=False):
        tagged_tokens = pos_tag(word_tokenize(doc))
        lem_tokens = []
        res = []

        for tagged_token in tagged_tokens:
            token = tagged_token[0]
            tag = tagged_token[1]

            # Lemmatize lowercased word (necessary for WordNetLemmatizer)
            wn_tag = get_wordnet_pos(tag)
            if wn_tag:
                lemmatized = self.wnl.lemmatize(token.lower(), get_wordnet_pos(tag))
            else:
                lemmatized = self.wnl.lemmatize(token.lower())

            # Capitalize words back
            if not self.lower and token.istitle():
                word = lemmatized.capitalize()
            else:
                word = lemmatized

            tagged_token = [word, tag]
            lem_tokens.append(tagged_token)

        # Clean out stopwords
        if self.stop:
            stopped_tokens = [tok for tok in lem_tokens if not tok[0].lower() in cb_stop]
            res, tags = list(zip(*stopped_tokens))
        else:
            res, tags = list(zip(*lem_tokens))

        if get_tag:
            return tags
        else:
            return res

class LemmaTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key
        self.tokenizer = LemmaTokenizer(stop=True)

    def transform(self, X, n, tags=False):
        """The workhorse of this feature extractor"""
        res = self.tokenizer(X)
        return res

    def fit(self, df, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self