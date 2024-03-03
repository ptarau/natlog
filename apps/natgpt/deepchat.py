import os
import webbrowser
import openai
import tiktoken
from natlog import Natlog, natprogs, lconsult
import json

MAX_TOKENS = 1 << 14  # make shorter if needed e.g. 300

API_KEY = [os.getenv("OPENAI_API_KEY")]
API_BASE = ["https://api.openai.com/v1"]


def ask_llm(model=None, mes=None, temperature=None, n=None):
    assert None not in (model, mes, temperature, n), (model, mes, temperature, n)

    def llm_res(r, i):
        return r.choices[i].message.content.strip()

    client = openai.OpenAI(
        api_key=API_KEY[0],
        base_url=API_BASE[0]
    )

    r = client.chat.completions.create(
        messages=mes,
        model=model,
        temperature=temperature,
        n=n
    )

    pt = r.usage.prompt_tokens
    ct = r.usage.completion_tokens

    answers = [llm_res(r, i) for i in range(n)]

    return answers, pt, ct


class ChatMind:
    def __init__(self, max_toks=MAX_TOKENS, avatar=None):
        self.model = "gpt-4-turbo-preview"

        self.max_toks = max_toks
        self.avatar = avatar

        if not self.resume():
            self.short_mem = dict()
            self.long_mem = dict()
            self.toks = []

    def to_message(self, quest):
        mes = []
        for (q, a) in self.short_mem.items():
            qd = dict(role='user', content=q)
            ad = dict(role='assistant', content=a)
            mes.extend([qd, ad])
        mes.append(dict(role='user', content=quest))
        return mes

    def get_state(self):
        return dict(long_mem=self.long_mem,
                    short_mem=self.short_mem,
                    toks=self.toks)

    def set_state(self, state):
        self.long_mem = state['long_mem']
        self.short_mem = state['short_mem']
        self.toks = state['toks']

    def store_name(self):
        if self.avatar is None: return None
        return 'states/' + self.avatar + '.json'

    def resume(self):
        fname = self.store_name()
        if fname is not None and exists_file(fname):
            state = from_json(fname)
            # print('!!!!',json.dumps(state,indent=4))
            self.set_state(state)
            return True
        return False

    def persist(self):
        fname = self.store_name()
        if fname is None: return
        state = self.get_state()
        to_json(state, fname)

    def already_answered(self, quest):
        answer = self.short_mem.get(quest, None)
        if answer is not None: return answer
        answer = self.long_mem.get(quest, None)
        return answer

    def __repr__(self):
        return json.dumps(self.get_state(), indent=2)

    def ask(self, quest):
        if quest == 'quit':
            self.persist()
            return 'bye'

        answered = self.already_answered(quest)
        if answered is not None:
            return answered

        self.trim_context(quest)

        assert len(self.short_mem) == len(self.toks)

        mes = self.to_message(quest)

        answers, pt, ct = ask_llm(model=self.model, mes=mes, temperature=0.2, n=1)
        answer = answers[0]

        self.toks.append(pt + ct)

        self.short_mem[quest] = answer

        print('LEN SHORT TERM:',len(self.short_mem))

        return answer

    def trim_context(self, quest):
        if len(self.toks) == 0: return

        total_toks = sum(self.toks)
        avg_toks = total_toks / len(self.toks)

        quest_toks = count_toks(quest)
        tok_estimate = total_toks + quest_toks + 2 * avg_toks  # conservative ...

        if tok_estimate > self.max_toks:
            k, v = dict_trim(self.short_mem)
            self.long_mem[k] = v
            self.toks = self.toks[1:]


# tools

def count_toks(text):
    enc = tiktoken.get_encoding("gpt2")
    toks = enc.encode(text)
    return len(toks)


def exists_file(fname):
    return os.path.exists(fname)


def ensure_path(fname):
    """
    makes sure path to directory and directory exist
    """
    d, _ = os.path.split(fname)
    os.makedirs(d, exist_ok=True)


def to_json(obj, fname, indent=2):
    """
    serializes an object to a json file
    assumes object made of array and dicts
    """
    ensure_path(fname)
    with open(fname, "w") as outf:
        json.dump(obj, outf, indent=indent)


def from_json(fname):
    """
    deserializes an object from a json file
    """
    with open(fname, "rt") as inf:
        obj = json.load(inf)
        return obj


def dict_trim(d):
    k = next(iter(d))
    v = d.pop(k)
    return k, v



def run_natlog(natprog="deepchat.nat"):
    n = Natlog(file_name=natprog,
               with_lib=natprogs() + "lib.nat", callables=globals())
    next(n.solve('initialize.'))
    # n.repl()
    for x in n.solve('chat.'):
        print()


def test_deepchat():
    cm = ChatMind(avatar='you')

    answer = cm.ask('What was the warmest temperature measured on Mars?')
    answer = cm.ask('The same question for planet Venus?')
    answer = cm.ask('And what about dwarf planet Ceres?')
    answer = cm.ask('Is it much colder on Pluto?')
    print(f'ANSWER: {answer}\n')
    answer = cm.ask('quit')
    print(answer, '\n')

    print('STATE:\n', cm)


if __name__ == "__main__":
    # test_deepchat()
    run_natlog()
