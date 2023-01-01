import sentence_transformers as st
from sentence_transformers.util import semantic_search, cos_sim
from sklearn.neighbors import NearestNeighbors
import torch
import numpy as np
from nltk.tokenize import sent_tokenize
import networkx as nx


def file2text(fname):
    """
      turns a text file into a string
    """
    with open(fname, 'rt') as f:
        return f.read()


class SentEncoder:
    """
    creates callable sentence encoder instance that
    given a list of sentences returns a 2D tensor of embeddings
    """
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.device = 'gpu' if torch.cuda.is_available() else 'cpu'

        self.model = st.SentenceTransformer(model_name, device=self.device)

    def __call__(self, sents):
        """
        when called, given a list of sentences, returns a 2D tensor of embeddings
        """
        x = self.model.encode(sents, show_progress_bar=True,
                              convert_to_tensor=True, device=self.device)
        return x


def knn_pairs(encs, k=3):
    """
    extracts edges of the directed knn-graph
    associating k closest neighbors to each node
    representing a sentence via its embedding
    """
    cos_scores = cos_sim(encs, encs)
    top_results = torch.topk(cos_scores, k=k + 1)
    m = top_results[1]  # indices
    r = top_results[0]  # similarity rsnks
    s = m.size()

    es = []
    for i in range(s[0]):
        for j in range(1, s[1] - 1):
            e = i, int(m[i, j]), r[i, j]
            es.append(e)

    return es


def summarize(encs, sents, k=3, length=3, reverse=True):
    """
    summarizes a document given as a set of sentences
    and teir embeddings
    """
    es = knn_pairs(encs, k=k)
    g = nx.DiGraph()
    for f,t,r in es:
        if reverse:
            f,t=t,f
        g.add_edge(f,t,weight=r)

    #print(g)

    rs = nx.pagerank(g)  # uses also similarity weights
    rs = sorted(rs.items(), reverse=True, key=lambda x: x[1])

    ns = [n for (n, r) in rs]
    # ensure natural order (as in the text)
    ns = sorted(ns[0:length])
    return [sents[n] for n in ns]


def semsearch(ts, qs, k=3):
    """
    searches closest matches for a list (actualy a tensor) of
    embeddings of queries qs in the similar embeddings of a document ts
    """
    r = semantic_search(qs, ts, query_chunk_size=100, corpus_chunk_size=500000, top_k=k)
    return r


def run_semsearch(fname, queries):
    """
      given a text file fname and a text representing query sentences
      it summarizes the file and answers each query
    """
    text = file2text(fname)
    enc = SentEncoder()

    sents = sent_tokenize(text)
    queries = sent_tokenize(queries)

    ts = enc(sents)
    qs = enc(queries)

    sumsents = summarize(ts, sents)

    rss = semsearch(ts, qs)

    answers = []
    for q, rs in zip(qs, rss):
        answer = []
        # ensure natural order of answer sentences (as in the text)
        rs=sorted(rs,key=lambda x:x['corpus_id'])
        for r in rs:
            i = r['corpus_id']
            # print('R:', r)
            a = sents[i]
            answer.append(a)
        answers.append(answer)

    return sumsents, queries, answers


def test_semsearch(fname='gpt.txt', queries='Which AI company developed GPT3? What architecture GPT3 uses?'):
    """
    tester showng summary and answers to each question about a given
    text document
    """
    ss, qs, rs = run_semsearch(fname, queries)

    print('\nSUMMARY:')
    for s in ss:
        print(s)
    print()

    for (q, ans) in zip(qs, rs):
        print('Q:', q)
        for a in ans:
           print('A:', a)
        print()


if __name__ == "__main__":
    test_semsearch()
    test_semsearch('sf.txt','Will Jake get back to the real world?')
    test_semsearch('star.txt', 'Who would want to fly on a superluminal spaceship?')
    test_semsearch('kafka.txt', 'What did the meeting with the priest mean? Did K. meet the Italian?')
