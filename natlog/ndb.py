from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import numpy as np

from .db import *

# simple multi-layer percdeptron
neural_learner = MLPClassifier(
    hidden_layer_sizes=(16, 16),
    random_state=1234,
    verbose=1,
    # activation='relu',
    max_iter=10000
)

rf_learaner = RandomForestClassifier(random_state=1234)  # alternative


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


class Ndb(Db):
    """
    replaces indexing in Db with machine-learned equivalent
    """

    def load(self, fname, learner=neural_learner):
        """
        overrides loading mechanism to fit learner
        """
        super().load(fname)
        db_const_dict = seq2nums(self.index)  # assuming dict ordered
        # create diagonal numpy matrix, one row for each constant
        X = np.eye(len(db_const_dict), dtype=int)
        val_count = len(self.css)
        y = np.array([set2bits(val_count, xs) for xs in self.index.values()])
        print('X:', X.shape, '\n', X)
        print('\ny:', y.shape, '\n', y, '\n')
        learner.fit(X, y)
        self.learner, self.db_const_dict = learner, db_const_dict

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
        #print('!!!!!!:',matches)
        return matches
