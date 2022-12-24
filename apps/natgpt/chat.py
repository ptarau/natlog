import os
import openai
from natlog import Natlog, natprogs

openai.api_key = os.getenv("OPENAI_API_KEY")

shared = dict()


def share(f):
    shared[f.__name__] = f
    return f


def question():
    return input('Question: ')

def answer(a):
    print('Answer:',a)

@share
def ask(quest, temp=0.4, toks=100):
    quest = quest.strip(' ')
    prompt = f'if you would ask me {quest} I would say that'
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
    if not answer: return 'no'
    answer = answer[0]['text']
    if not answer: return 'no'
    answer = answer.strip(' ')
    return 'the', answer


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
    test_chat()
    run_natlog()
