# helper_functions.py
# Created: Tal Daniel (August 2019)
# Updates: Ron Amit (January 2021)

# imports
import random, re
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from collections import Counter
import os, functools
from concurrent.futures import ProcessPoolExecutor

######################################################################################################


class EmailToWords(BaseEstimator, TransformerMixin):

    def __init__(self, lowercaseConversion=True, onlyAlphabet=True, removeURLs=True, max_word_len=10, min_word_len=3):
        self.lowercaseConversion = lowercaseConversion  # - Convert to lowercase
        self.onlyAlphabet = onlyAlphabet  # - Remove non letters
        self.removeURLs = removeURLs
        self.max_word_len = max_word_len
        self.min_word_len = min_word_len
 
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """"
        Transforms list of texts/emails to a list of word count dictionaries 
        + optional pre-processing 
        """
        word_counts_per_mail = []
        
        # run on all mails in the dataset
        for text in X:           
            
            if text is None or not isinstance(text, str):
                text = 'empty'
                
            # Remove URLs 
            if self.removeURLs:
                text = re.sub(r"http\S+", "", text)
                
            # convert to lowercase
            if self.lowercaseConversion:
                text = text.lower()
                
            # remove non-letters
            if self.onlyAlphabet:
                text = ''.join(ch for ch in text if ch == ' ' or ch.isalpha())

            # create a word counter dict
            word_counts = Counter(text.split())           
                
            # remove too long\short words:
            word_counts = Counter({word: count for (word, count) in word_counts.items()
                                   if self.min_word_len <= len(word) <= self.max_word_len})
                
            word_counts_per_mail.append(word_counts)
        return word_counts_per_mail
######################################################################################################

 
class WordCountToVector(BaseEstimator, TransformerMixin):
    def __init__(self, vocabulary_size=300):
        self.vocabulary_size = vocabulary_size
        self.most_common = None
        self.vocabulary_ = None

    def fit(self, word_counts_per_mail, y=None):
        """"
        Creates a vocabulary: word dictionary of the most common words (key==word, value==index of word in the vocabulary)
        """
        total_word_count = Counter()

        # run on all mails in the dataset:
        for word_count in word_counts_per_mail:
            for word, count in word_count.items():
                total_word_count[word] += count
        self.most_common = total_word_count.most_common(self.vocabulary_size)
        self.vocabulary_ = {word: index for index, (word, count) in enumerate(self.most_common)}
        return self

    def transform(self, word_counts_per_mail, y=None):
        """"
        Transform a list of word counts per email into a "spare matrix" with dimensions #emails X vocabulary_size.
        The entries of the matrix count the number of times a given word appear in a given email.
        """
        n_mails = len(word_counts_per_mail)
        rows = []
        cols = []
        data = []
        for mail_idx, word_count in enumerate(word_counts_per_mail):
            for word, count in word_count.items():
                if word in self.vocabulary_:
                    rows.append(mail_idx)
                    cols.append(self.vocabulary_[word])
                    data.append(count)
        # create a sparse matrix:
        return csr_matrix((data, (rows, cols)), shape=(n_mails, self.vocabulary_size))
######################################################################################################


email_pipeline = Pipeline([
    ("Email to Words", EmailToWords()),
    ("Wordcount to Vector", WordCountToVector()),
])



######################################################################################################

# 
# Functions for multiprocessing
#
# based on https://medium.com/@grvsinghal/speed-up-your-python-code-using-multiprocessing-on-windows-and-jupyter-or-ipython-2714b49d6fac

######################################################################################################


def execute_repeats(run_inds, func, param):
    """"
    Runs several repetitions of a function, returns a list of outputs
    """
    outs = []
    for i_run in run_inds:
        # ensure a special random seed for each run:
        random.seed(i_run)
        np.random.seed(i_run)
        # apply function
        out_single = func(*param)
        outs.append(out_single)
    return outs


def get_chunks(n_runs):
    """"
    Splits range(n_runs) to a list of n_chunks segments, as evenly sized as possible
    """
    run_inds = list(range(n_runs))
    n_chunks = os.cpu_count()
    remainder = n_runs % n_chunks
    division = n_runs // n_chunks
    chunk_sizes = [division + 1] * remainder + [division] * (n_chunks - remainder)
    chunks = [run_inds[sum(chunk_sizes[:i]):sum(chunk_sizes[:(i + 1)])] for i in range(n_chunks)]
    return chunks


# auxiliary function for multiprocessing
def run_parallel_aux(func, param, n_repeats):
    chunks = get_chunks(n_repeats)
    with ProcessPoolExecutor(max_workers=len(chunks)) as executor:
        outs = executor.map(functools.partial(execute_repeats, func=func, param=param), chunks)
    return sum(outs, [])
######################################################################################################
