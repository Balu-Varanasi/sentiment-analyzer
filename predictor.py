# Sentiment Analyzer for telugu

import pickle

from nltk import Text
from nltk.classify.scikitlearn import SklearnClassifier

from base import BaseNLTKUtil


class Sentiment(BaseNLTKUtil):

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
        bigrams = self.get_bigrams(words)
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


sentiment = Sentiment()
telugu_text = "అయితే సిద్దార్ధలోనూ, దర్శకుడులోనూ ఏదో కొత్తగా చేయాలనే తపన ముచ్చటపడేటట్లు చేస్తుంది. "
print(sentiment.getSubjObj(telugu_text))

