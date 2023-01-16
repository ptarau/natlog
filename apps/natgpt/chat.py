import os
import webbrowser
import openai
from natlog import Natlog, natprogs, lconsult

# openai.api_key = os.getenv("OPENAI_API_KEY")

shared = dict()


def share(f):
    shared[f.__name__] = f
    return f


@share
def question():
    quest = input('Question: ')
    if quest:
        return 'the', quest
    else:
        return 'no'


@share
def answer(a):
    print('Answer:', a)


@share
def ask(quest, temp=0.4, toks=100):
    quest = quest.strip(' ')
    prompt = f'if you would ask me {quest} I would say that'
    answer = query(prompt, temp=temp, toks=toks)
    return ('the', answer) if answer else 'no'


def query(prompt, temp=0.4, toks=200):
    answer = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=temp,
        max_tokens=toks,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=["\n"]
    )

    answer = answer['choices']
    if not answer: return None
    answer = answer[0]['text']
    if not answer: return None
    answer = answer.strip(' ')

    return answer


@share
def to_rels(qs):
    pref = 'If you would ask me what are the subject, verb and object in '
    suf = ' I would say subject: '

    def quote(qs):
        return '"' + qs + '"'
        # return qs

    r = 'subject: ' + query(pref + quote(qs) + suf)
    print('Sent:', qs)
    print()

    print('RElS:', r)
    print()

    r = r.replace(';', '')

    r = r.split(', verb: ')

    s = r[0][len('subject: '):]
    v, o = r[1].split(', object: ')
    if o[-1] == '.': o = o[0:-1]

    print('SVO:', [s, v, o])
    print()
    return r


@share
def paint(prompt):
    response = openai.Image.create(
        prompt=prompt,
        n=1,
        size="1024x1024"
    )
    image_url = response['data'][0]['url']
    return image_url


@share
def embed(text):
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    embedding = response['data'][0]['embedding']
    return embedding


def edit(how,what):
    r=openai.Edit.create(
        model="text-davinci-edit-001",
        input=what,
        instruction=how,
        n=1,
        temperature=0.1
    )
    print('RES:\n',(r))
    return r['choices'][0]['text']


def edit_test():
    instr= "Say in a simpler way"
    r=edit(
        instr,
        "Red colored ethyl alcohol drinks originating from grapes can cause dangerous driving."
    )
    print(r)

@share
def browse(url):
    print('BROWSING:', url)
    return webbrowser.open(url)


def share_syms():
    for n, f in globals().items():
        if n not in {'add'}:
            shared[n] = f
    return shared


def run_natlog(natprog="chat.nat"):
    n = Natlog(file_name=natprog,
               with_lib=natprogs() + "lib.nat", callables=share_syms())
    next(n.solve('initialize.'))
    n.repl()


def test_chat(q='where is Dallas located'):
    r = ask(q)
    print('\nQ:', q)
    if r == 'no':
        print('A: I do not know.')
    else:
        print('A:', r[1], '\n')


if __name__ == "__main__":
    #run_natlog()
    edit_test()
