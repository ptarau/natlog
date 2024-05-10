from natlog.natlog import Natlog,natprogs
from natlog.textual_natlog import TextualNatlog
import stanza

def to_sents(text, lang='en'):
    nlp = stanza.Pipeline(lang=lang, processors='tokenize')
    doc = nlp(text)
    sents = []
    for sent in doc.sentences:
        toks = []
        for token in sent.tokens:
            toks.append(token.text)
        sent_text = " ".join(toks)
        sents.append(sent_text)
    return sents  # "\n".join(sents)

def standardize_txt(fname,lang='en'):

    with open(fname,'r') as f:
        text=f.read()
    sents=to_sents(text, lang=lang)
    text="\n".join(sents)
    with open(fname,'w') as g:
        g.write(text)


def start():
  nname ="textual.nat"
  dname = natprogs()+'../docs/prolog50.txt'
  standardize_txt(dname)

  n = TextualNatlog(file_name=nname, db_name=dname)
  n.repl()

if __name__=="__main__":
    start()
