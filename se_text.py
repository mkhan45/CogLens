import re, string, math
import numpy as np
from collections import Counter
from gensim.models.keyedvectors import KeyedVectors as KV
from typing import List

def se_text(caption: str, glove, idf: np.ndarray, vocab: List[str]):
    """
    Returns an embedded representation for a string of text.

    Parameters
    ----------
    caption: str
        The text that you want an embedding for.
    glove:
        The loaded glove database.
    idf: np.array([])
        The vector of idfs for each caption in the database.
    vocab: list(str)
        A list of all captions (vocab) in the database, sorted alphabetically. (use to_vocab() for this)

    Returns
    -------
    np.array([])
        The embedded representation for the string passed in.
    """

    caption = strip_punc(caption).lower().split()
    embedding = np.zeros((1, 50))

    for word in caption:
        word_idf = idf[vocab.index(word)]
        embedding += glove50[word]*word_idf

    embedding = normalize(embedding/len(caption))

    return embedding

def normalize(vector):
    """
    Normalize a vector of numbers to have unit length.

    Parameters
    ----------
    Vector: np.array([])
        The vector that you want normalized.

    Returns
    -------
    np.array([])
        The normalized vector.
    """
    return(vector/np.mean(vector) - 1)

def to_idf(counters, vocab):
    """ 
    Given the vocabulary, and the word-counts for each document, computes
    the inverse document frequency (IDF) for each term in the vocabulary.
    
    Parameters
    ----------
    counters : Iterable[collections.Counter]
        An iterable containing {word -> count} counters for respective
        documents.
    vocab: list(str)
        A sorted list of all captions (vocab) in the database.
    
    Returns
    -------
    numpy.ndarray
        An array whose entries correspond to those in `vocab`, storing
        the IDF for each term `t`: 
                           log10(N / nt)
        Where `N` is the number of documents, and `nt` is the number of 
        documents in which the term `t` occurs.
    """
    idf = list()
    total_counter = Counter()
    for counter in counters:
        total_counter.update(set(counter)) #makes sure there's only one of each word per doc at most
    for word in vocab:
        idf.append(math.log(len(counters)/total_counter[word], 10))
    return np.array(idf)

def to_vocab(counters, k=None, stop_words=None):
    """ 
    [word, word, ...] -> sorted list of top-k unique words
    Excludes words included in `stop_words`
    
    Parameters
    ----------
    counters : Iterable[Iterable[str]]
    
    k : Optional[int]
        If specified, only the top-k words are returned
    
    stop_words : Optional[Collection[str]]
        A collection of words to be ignored when populating the vocabulary

    Returns
    -------
    List
        A list of all unique words in the counter, sorted alphabetically.
    """
    vocab_set = set()
    if stop_words is None:
        stop_words = []

    total_counter = Counter()
    for counter in counters:
        total_counter.update(dict((key, value) for key, value in counter.items() if key not in stop_words))

    if k is not None:
        total_counter = total_counter.most_common(k)
        vocab_set.update(pair[0] for pair in total_counter)
    else:
        vocab_set.update(key for key in total_counter.keys())
        
    return sorted(vocab_set)

def to_counters(all_captions):
    """
    Creates a count of each word in the captions.

    Parameters
    ----------
    captions: Iterable(strings)
        All captions in the database.

    Returns
    -------
    counters : Iterable[collections.Counter]
        An iterable containing {word -> count} counters for respective
        documents.
    """
    counters = []
    for caption in all_captions:
        counters.append(Counter(strip_punc(caption).lower().split()))
    return counters

def strip_punc(corpus: str):
    """ Removes all punctuation from a string.

        Parameters
        ----------
        corpus : str

        Returns
        -------
        str
            the corpus with all punctuation removed"""

    punc_regex = re.compile('[{}]'.format(re.escape(string.punctuation)))
    return punc_regex.sub("", corpus)

with open("/Users/crystal/repositories/Student_Week3/bag_of_words/dat/stopwords.txt", 'r') as r:
    stops = []
    for line in r:
        stops += [i.strip() for i in line.split('\t')]

glove50 = KV.load_word2vec_format('/Users/crystal/Desktop/python-workspace/CogWorks2019/glove.6B.50d.txt.w2v', binary=False)

all_captions = ["cat", "dog", "bird eating", "mouse", "mouse flying"]
counters = to_counters(all_captions)
vocab = to_vocab(counters, stop_words=stops)
idf = to_idf(counters, vocab)
embedding = se_text("mouse flying", glove50, idf, vocab)
print(embedding)