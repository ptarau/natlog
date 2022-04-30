# machine learns indexing - ready to answer
# queries with logic variables
import pandas as pd

def load_df(fname, delimiter='\t'):
    df = pd.read_csv(fname, header=None, delimiter=delimiter)
    print(df[8])


def test_selfsup():
    load_df('../natprogs/elements.tsv')


if __name__=="__main__":
    test_selfsup()


