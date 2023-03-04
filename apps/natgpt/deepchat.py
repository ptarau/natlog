import os
import webbrowser
import openai
import tiktoken
from natlog import Natlog, natprogs, lconsult

MAX_TOKENS = 1 << 12  # make shorter if needed

openai.api_key = os.getenv("OPENAI_API_KEY")

def count_toks(text):
    enc = tiktoken.get_encoding("gpt2")
    toks = enc.encode(text)
    return len(toks)

class ChatMind:
    def __init__(self):
        self.model = "gpt-3.5-turbo"
        self.short_mem = []
        self.long_mem = dict()
        self.answers = []
        self.sys_prompt = [dict(role='system', content='You are a helpful assistant.')]
        self.toks = []

    def add_content(self, role, content):
        if role != 'system':
            d = dict(role=role, content=content)
            self.short_mem.append(d)

    def __repr__(self):
        return 'LONG: ' + str(self.long_mem) + "\n\n" + 'SHORT: ' + str(self.short_mem) + "\n\n" \
            + 'ANSWERS: ' + str(self.answers) + "\n\n" + f'TOKENS: {self.toks}\n'

    def ask(self, quest):
        self.trim_context(quest)

        # print(len(self.answers), len(self.short_mem), len(self.toks))

        assert len(self.answers) == len(self.short_mem) == len(self.toks)

        self.add_content('user', quest)

        r = openai.ChatCompletion.create(
            model=self.model,
            messages=self.sys_prompt + self.short_mem
        )
        result = r['choices'][0]['message']['content']
        result = result.split('\n')[-1]
        self.answers.append(result)
        t = r['usage']['total_tokens']
        self.toks.append(t)

        return result

    def trim_context(self, quest):
        if len(self.toks) == 0: return

        total_toks = sum(self.toks)
        avg_toks = total_toks / len(self.toks)

        quest_toks = count_toks(quest)
        tok_estimate = total_toks + quest_toks + 2 * avg_toks  # conservative ...

        if tok_estimate > MAX_TOKENS:
            self.long_mem[self.short_mem[0]['content']] = self.answers[0]

            self.toks = self.toks[1:]
            self.short_mem = self.short_mem[1:]
            self.answers = self.answers[1:]

def run_natlog(natprog="deepchat.nat"):
    n = Natlog(file_name=natprog,
               with_lib=natprogs() + "lib.nat",callables=globals())
    next(n.solve('initialize.'))
    #n.repl()
    for x in n.solve('chat.'):
        print()


def test_deepchat():
    cm = ChatMind()

    answer = cm.ask('What was the warmest temperature measured on Mars?')
    answer = cm.ask('The same question for planet Venus?')
    answer = cm.ask('And what about dwarf planet Ceres?')
    answer = cm.ask('Is it much colder on Pluto?')

    print(cm)


if __name__ == "__main__":
    #test_deepchat()
    run_natlog()
