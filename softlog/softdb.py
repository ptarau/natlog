from natlog.db import *
from sentence_store.main import Embedder

class SoftDB(Db):
    """
    specializes to db derived from text
    """
    def __init__(self,
                 min_knn_dist=0.6,
                 max_knn_count=3
                 ):
        super().__init__()
        self.min_knn_dist = min_knn_dist
        self.max_knn_count = max_knn_count
        self.abduced_clauses = []

    def __repr__(self):
        return 'SoftDB'

    def initalize_store(self,cache_name):
        self.emb = Embedder(cache_name)
        self.emb.clear()
        self.abduced_clauses=[]

    def digest(self, text):
        self.initalize_store(cache_name="soft_db_cache")
        self.emb.store_text(text)

    def load_txt(self, doc_name, doc_type='txt'):
        # can be 'url', 'wikipage', 'txt', 'pdf'
        cache_name="".join(c for c in doc_name if c.isalpha())
        self.initalize_store(cache_name=cache_name)

        self.emb.store_doc(doc_type, doc_name)

    def unify_with_fact(self, h, trail):
        # pairs of the form i=sent index,r=confidence
        print('<<<',h)
        h=h[0]
        _knn_pairs, answers = self.emb.knn_query(h, self.max_knn_count)
        for sent, dist in answers:
            print('>>>',sent,dist)
            if dist <= self.min_knn_dist:
                print('PASSING!')
                self.abduced_clauses.append((h,sent,dist))
                yield sent


def test_softdb():
    sents = [
        "The cat sits on the mat.",
        "The dog barks to the moon.",
        "The pirate travels the oceans.",
        "The phone rings with a musical tone.",
        "The man watches the bright moon."
    ]
    quest = "Who barks out there?"
    text = "\n".join(sents)
    sdb = SoftDB(min_knn_dist=0.99,
                 max_knn_count=8)
    sdb.digest(text)
    for a in sdb.unify_with_fact(quest, []):
        #print(a)
        pass
    for c in sdb.abduced_clauses:
      print(c)


if __name__ == "__main__":
    test_softdb()

"""
Matches are found with knn against the fact base.
A min threshold is needed for a match to be accepted.
We use sentence-store as backend.

To rember the result as a Natlog program,
each time a match t is found for head h,
the clause h:-t is generated.

That also gives a log for the interaction with the sofdb.
"""
