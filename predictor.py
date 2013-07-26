# -*- coding: utf-8 -*-

# Sentiment Analyzer for telugu

import string
import cPickle as pickle
from nltk import Text
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures


class Sentiment(object):

    def __int__(self):
        self.name = None

    def loadSOClsssifier(self):
        fModel = open('SubjObjModel.pickle', "rb")
        subjobjclassifier = pickle.load(fModel)
        fModel.close()
        return subjobjclassifier

    def loadPNClsssifier(self):
        fModel = open('PosNegModel.pickle', "rb")
        posnegclassifier = pickle.load(fModel)
        fModel.close()
        return posnegclassifier

    def getSubjObj(self, text):
        words = Text(text.split(" "))
        bigrams = self.getBigrams(words)
        subjclassifier = self.loadSOClsssifier()
        posnegclassifier = self.loadPNClsssifier()

        subj_or_obj = SklearnClassifier.classify(subjclassifier, bigrams)
        if subj_or_obj == "objective":
            return "neutral"

        pos_or_neg = SklearnClassifier.classify(posnegclassifier, bigrams)

        if pos_or_neg == "negative":
            return "negative"
        else:
            return "positive"

    def getBigrams(self, words, score_fn=BigramAssocMeasures.chi_sq, n=200):
        bigram_finder = BigramCollocationFinder.from_words(words)
        bigram = bigram_finder.nbest(score_fn, n)
        bigramdict = dict([(big, True) for big in bigram])
        bigramdict.update(self.getWordFeats(words))
        return bigramdict

    def getWordFeats(self, words):
        return dict([(word.strip(string.punctuation), True) for word in words])


sentiment = Sentiment()
telugu_text = "అయితే సిద్దార్ధలోనూ, దర్శకుడులోనూ ఏదో కొత్తగా \
                చేయాలనే తపన ముచ్చటపడేటట్లు చేస్తుంది. "
print sentiment.getSubjObj(telugu_text)
