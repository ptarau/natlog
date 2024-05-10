from natlog.db import *
from sentence_store.main import Embedder

class SoftDB(Db):
    """
    specializes to db derived from text
    """

    def __repr__(self):
        return 'SoftDB'

    def initalize_store(self,cache_name):
        self.emb = Embedder(cache_name)
        self.emb.clear()
        self.abduced_clauses=dict()

    def digest(self, text):

        self.initalize_store(cache_name="soft_db_cache")
        self.emb.store_text(text, clean=False)

    def load_txt(self, doc_name, doc_type='txt'):
        # can be 'url', 'wikipage', 'txt', 'pdf'
        cache_name="".join(c for c in doc_name if c.isalpha())
        self.initalize_store(cache_name=cache_name)

        self.emb.store_doc(doc_type,doc_name,clean=True)

    def unify_with_fact(self, hs, trail):
        # pairs of the form i=sent index,r=confidence
        # print('<<<',hs)
        assert len(hs)==4,hs
        h=hs[0]
        k=hs[1]
        d=hs[2]
        v=hs[3]
        k=int(k)
        d=float(d)/100
        _knn_pairs, answers = self.emb.knn_query(h, k)
        for sent, dist in answers:
            #print('>>>',sent,dist)
            if dist <= d: # self.min_knn_dist:
                self.abduced_clauses[(h,sent)]=dist
                u = unify(v, sent, trail)
                yield u


def test_softdb():
    sents = [
        "The cat sits on the mat.",
        "The dog barks at a cat.",
        "The dog barks at the moon.",
        "The pirate travels the oceans.",
        "The phone rings with a musical tone.",
        "The man watches the bright moon."
    ]
    v=Var()
    quest = ('Who barks out there', 3, 99, v)
    text = "\n".join(sents)
    sdb = SoftDB()
    sdb.digest(text)
    for _ in sdb.unify_with_fact(quest, []):
        #print(a, '-->', v)
        pass
    print('THE ABDUCED CLAUSES aRE:')
    for (h,b),r in sdb.abduced_clauses.items():
      if b.endswith('.'):b=b[0:-1]
      print(f"'{h}' : '{b}'. % {r}")


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
