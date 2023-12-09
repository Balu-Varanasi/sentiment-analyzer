# sentiment trainer for Telugu

import pickle

from nltk.corpus import PlaintextCorpusReader as ptr
from nltk.classify.scikitlearn import SklearnClassifier

from sklearn.svm import LinearSVC

from base import BaseNLTKUtil


class NltkTrainer(BaseNLTKUtil):

    def __int__(self):
        self.name = None

    def train_subjective_objective_classifier(self):
        subjective = "./subjective"
        objective = "./objective"
        sub_files = ptr(subjective, '.*')
        obj_files = ptr(objective, '.*')
        sub_all_words = [sub_files.raw(fileid).split(" ")
                         for fileid in sub_files.fileids()]
        obj_all_words = [obj_files.raw(fileid).split(" ")
                         for fileid in obj_files.fileids()]
        sub_splited_words = [(self.get_bigrams(words), 'subjective')
                             for words in sub_all_words]
        obj_splited_words = [(self.get_bigrams(words), 'objective')
                             for words in obj_all_words]
        sub_obj_trainfeats = sub_splited_words[:] + obj_splited_words[:]

        # SVM with a Linear Kernel and default parameters
        classifier = SklearnClassifier(LinearSVC())
        classifier.train(sub_obj_trainfeats)
        return classifier

    def train_positive_negative_classifier(self):
        positive = "./positive"
        negative = "./negative"
        pos_files = ptr(positive, '.*')
        neg_files = ptr(negative, '.*')
        pos_all_words = [pos_files.raw(fileid).split(" ")
                         for fileid in pos_files.fileids()]
        neg_all_words = [neg_files.raw(fileid).split(" ")
                         for fileid in neg_files.fileids()]
        pos_splited_words = [(self.get_bigrams(words), 'positive')
                             for words in pos_all_words]
        neg_splited_words = [(self.get_bigrams(words), 'negative')
                             for words in neg_all_words]
        pos_neg_trainfeats = pos_splited_words[:] + neg_splited_words[:]
        classifier = SklearnClassifier(LinearSVC())
        classifier.train(pos_neg_trainfeats)
        return classifier

    @staticmethod
    def pickle_classifiers(subj_obj_classifier, pos_neg_classifier):
        subj_obj_picklefile = open('SubjObjModel.pickle', 'wb')
        pos_neg_picklefile = open('PosNegModel.pickle', 'wb')

        pickle.dump(subj_obj_classifier, subj_obj_picklefile, 1)
        subj_obj_picklefile.close()

        pickle.dump(pos_neg_classifier, pos_neg_picklefile, 1)
        pos_neg_picklefile.close()


if __name__ == '__main__':
    trainer = NltkTrainer()
    trainer.pickle_classifiers(
        trainer.train_subjective_objective_classifier(),
        trainer.train_positive_negative_classifier()
    )
