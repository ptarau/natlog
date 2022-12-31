import sentence_transformers as st
from sentence_transformers.util import semantic_search, cos_sim
from sklearn.neighbors import NearestNeighbors
import torch
import numpy as np
from nltk.tokenize import sent_tokenize
import networkx as nx


def file2text(fname):
    with open(fname, 'rt') as f:
        return f.read()


class SentEncoder:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.device = 'gpu' if torch.cuda.is_available() else 'cpu'

        self.model = st.SentenceTransformer(model_name, device=self.device)

    def __call__(self, sents):
        x = self.model.encode(sents, show_progress_bar=True,
                              convert_to_tensor=True, device=self.device)
        return x


def knn_pairs(encs, k=3):
    cos_scores = cos_sim(encs, encs)
    top_results = torch.topk(cos_scores, k=k + 1)
    m = top_results[1]  # indices
    s = m.size()

    print('MSHAPE:', m.size())

    es = []
    for i in range(s[0]):

        for j in range(1, s[1] - 1):
            e = i, int(m[i, j])
            es.append(e)

    return es


def summarize(encs, sents, k=3, l=3):
    es = knn_pairs(encs, k=k)
    g = nx.DiGraph(es)
    print(g)
    rs = nx.pagerank(g)  # TODO: add similarity weights
    rs = sorted(rs.items(), reverse=True, key=lambda x: x[1])
    # print('EDGES:',rs)
    ns = [n for (n, r) in rs]
    ns = sorted(ns[0:l])
    return [sents[n] for n in ns]


def semsearch(ts, qs):
    r = semantic_search(qs, ts, query_chunk_size=100, corpus_chunk_size=500000, top_k=2)
    return r


def run_semsearch(fname, query):
    text = file2text(fname)
    enc = SentEncoder()

    sents = sent_tokenize(text)
    queries = sent_tokenize(query)

    ts = enc(sents)
    qs = enc(queries)

    sumsents = summarize(ts, sents)

    rss = semsearch(ts, qs)

    answers = []
    for q, rs in zip(qs, rss):
        answer = []
        for r in rs:
            i = r['corpus_id']
            # print('R:', r)
            a = sents[i]
            answer.append(a)
        answers.append(answer)

    return sumsents, queries, answers


def test_semsearch(fname='gpt.txt', query='Which AI company developed GPT3? What architecture it uses?'):
    ss, qs, rs = run_semsearch(fname, query)

    print('\nSUMMARY:')
    for s in ss:
        print(s)
    print()

    for (q, a) in zip(qs, rs):
        print('Q:', q)
        print('A:', a)
        print()


if __name__ == "__main__":
    test_semsearch()
