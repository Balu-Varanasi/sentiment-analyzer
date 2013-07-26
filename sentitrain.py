#sentiment trainer for Telugu

import string
import cPickle as pickle

from nltk.corpus import PlaintextCorpusReader as ptr
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures

from sklearn.svm import LinearSVC


class NltkTrainer(object):

    def __int__(self):
        self.name = None

    def trainSubObj(self):
        subjective = "./subjective"
        objective = "./objective"
        sub_files = ptr(subjective, '.*')
        obj_files = ptr(objective, '.*')
        sub_all_words = [sub_files.raw(fileid).split(" ")
                         for fileid in sub_files.fileids()]
        obj_all_words = [obj_files.raw(fileid).split(" ")
                         for fileid in obj_files.fileids()]
        sub_splited_words = [(self.getBigrams(words), 'subjective')
                             for words in sub_all_words]
        obj_splited_words = [(self.getBigrams(words), 'objective')
                             for words in obj_all_words]
        sub_obj_trainfeats = sub_splited_words[:] + obj_splited_words[:]

        # SVM with a Linear Kernel and default parameters
        sub_obj_classifier = SklearnClassifier(LinearSVC())
        sub_obj_classifier.train(sub_obj_trainfeats)
        return sub_obj_classifier

    def trainPosNeg(self):
        positive = "./positive"
        negative = "./negative"
        pos_files = ptr(positive, '.*')
        neg_files = ptr(negative, '.*')
        pos_all_words = [pos_files.raw(fileid).split(" ")
                         for fileid in pos_files.fileids()]
        neg_all_words = [neg_files.raw(fileid).split(" ")
                         for fileid in neg_files.fileids()]
        pos_splited_words = [(self.getBigrams(words), 'positive')
                             for words in pos_all_words]
        neg_splited_words = [(self.getBigrams(words), 'negative')
                             for words in neg_all_words]
        pos_neg_trainfeats = pos_splited_words[:] + neg_splited_words[:]
        pos_neg_classifier = SklearnClassifier(LinearSVC())
        pos_neg_classifier.train(pos_neg_trainfeats)
        return pos_neg_classifier

    def pickleClassifier(self, subj_obj_classifier, pos_neg_classifier):
        subj_obj_picklefile = open('SubjObjModel.pickle', 'wb')
        pos_neg_picklefile = open('PosNegModel.pickle', 'wb')

        pickle.dump(subj_obj_classifier, subj_obj_picklefile, 1)
        subj_obj_picklefile.close()

        pickle.dump(pos_neg_classifier, pos_neg_picklefile, 1)
        pos_neg_picklefile.close()

    def getWordFeats(self, words):
        return dict([(word.strip(string.punctuation), True) for word in words])

    def getBigrams(self, words, score_fn=BigramAssocMeasures.chi_sq, n=200):
        bigram_finder = BigramCollocationFinder.from_words(words)
        bigram = bigram_finder.nbest(score_fn, n)
        bigramdict = dict([(big, True) for big in bigram])
        bigramdict.update(self.getWordFeats(words))
        return bigramdict


trainer = NltkTrainer()
trainer.pickleClassifier(trainer.trainSubObj(), trainer.trainPosNeg())
