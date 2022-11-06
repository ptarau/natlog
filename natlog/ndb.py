import os
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import numpy as np
import os
import pickle

from .db import *


# simple multi-layer percdeptron
def neural_learner():
    return MLPClassifier(
        hidden_layer_sizes=(16, 32, 16),
        random_state=1234,
        verbose=1,
        # activation='relu',
        max_iter=2000
    )


def rf_learner():
    return RandomForestClassifier(random_state=1234)  # alternative


def set2bits(n, xs):
    """
    turns set into 1 hot encoded bitlist
    """
    return [1 if x in xs else 0 for x in range(n)]


def bits2set(bs):
    """
    turns bitslist into set of natural nunbers
    """
    return [i for i, b in enumerate(bs) if b == 1]


def seq2nums(xs):
    """
    turns symbol set into set of natural numbers
    """
    d, i = dict(), 0
    for x in xs:
        if x not in d:
            d[x] = i
            i += 1
    return d


def exists_file(fname):
    return os.path.exists(fname)


def to_pickle(obj, fname):
    """
    serializes an object to a .pickle file
    """
    with open(fname, "wb") as outf:
        pickle.dump(obj, outf)


def from_pickle(fname):
    """
    deserializes an object from a pickle file
    """
    with open(fname, "rb") as inf:
        return pickle.load(inf)


class Ndb(Db):
    """
    replaces indexing in Db with machine-learned equivalent
    """

    def __init__(self, learner=neural_learner):
        super().__init__()
        self.learner_name=learner.__name__
        self.learner = learner()

    def to_model_name(self, fname):
        return fname + "."+self.learner_name+".pickle"

    def load(self, fname):
        """
        overrides loading mechanism to fit learner
        """
        model_name = self.to_model_name(fname)
        if exists_file(model_name):
            self.learner, self.db_const_dict, self.css = from_pickle(model_name)
            return

        super().load(fname)
        db_const_dict = seq2nums(self.index)  # assuming dict ordered
        # create diagonal numpy matrix, one row for each constant
        X = np.eye(len(db_const_dict), dtype=int)
        val_count = len(self.css)
        y = np.array([set2bits(val_count, xs) for xs in self.index.values()])
        print('X:', X.shape, '\n', X)
        print('\ny:', y.shape, '\n', y, '\n')
        self.learner.fit(X, y)
        self.db_const_dict = db_const_dict
        to_pickle((self.learner, db_const_dict, self.css), model_name)

    def ground_match_of(self, query_tuple):
        """
        overrides database matching with learned predictions
        """
        query_consts = path_of(query_tuple)
        query_consts_nums = \
            [self.db_const_dict[c] for c in query_consts if c in self.db_const_dict]
        db_const_count = len(self.db_const_dict)
        qs = np.array([set2bits(db_const_count, query_consts_nums)])
        rs = self.learner.predict(qs)
        matches = bits2set(list(rs[0]))
        #print('!!!!!!:',matches,self.css)
        return matches
