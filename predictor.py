# Sentiment Analyzer for telugu

import pickle

from nltk import Text
from nltk.classify.scikitlearn import SklearnClassifier

from base import BaseNLTKUtil


class Sentiment(BaseNLTKUtil):

    def __int__(self):
        self.name = None

    @staticmethod
    def load_subjective_objective_classifier():
        model_file = open('SubjObjModel.pickle', "rb")
        subjobjclassifier = pickle.load(model_file)
        model_file.close()
        return subjobjclassifier

    @staticmethod
    def load_positive_negative_classifier():
        model_file = open('PosNegModel.pickle', "rb")
        positive_negative_classifier = pickle.load(model_file)
        model_file.close()
        return positive_negative_classifier

    def predict(self, text):
        words = Text(text.split(" "))
        bigrams = self.get_bigrams(words)
        subjective_objective_classifier = self.load_subjective_objective_classifier()
        positive_negative_classifier = self.load_positive_negative_classifier()

        subj_or_obj = SklearnClassifier.classify(subjective_objective_classifier, bigrams)
        if subj_or_obj == "objective":
            return "neutral"

        pos_or_neg = SklearnClassifier.classify(positive_negative_classifier, bigrams)

        if pos_or_neg == "negative":
            return "negative"
        else:
            return "positive"


if __name__ == '__main__':
    sentiment = Sentiment()
    telugu_text = "అయితే సిద్దార్ధలోనూ, దర్శకుడులోనూ ఏదో కొత్తగా చేయాలనే తపన ముచ్చటపడేటట్లు చేస్తుంది. "
    print(sentiment.predict(telugu_text))

