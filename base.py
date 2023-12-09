# -*- coding: utf-8 -*-
# sentiment trainer for Telugu

import string

from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures


class BaseNLTKUtil(object):

    def __int__(self):
        self.name = None

    def get_word_features(self, words):
        """
        Extracts features from a list of words.

        Args:
            words: A list of words.

        Returns:
            A dictionary representing the features of the words.
        """

        return dict([(word.strip(string.punctuation), True) for word in words])

    def get_bigrams(self, words, score_fn=BigramAssocMeasures.chi_sq, n=200):
        """
        Extracts bigram features from a list of words.

        Args:
            words: A list of words.
            score_fn: The scoring function for bigram association.
            n: The number of bigrams to extract.

        Returns:
            A dictionary representing the bigram features of the words.
        """
        bigram_finder = BigramCollocationFinder.from_words(words)
        bigrams = bigram_finder.nbest(score_fn, n)
        return dict([(str(bigram), True) for bigram in bigrams], **self.get_word_features(words))
